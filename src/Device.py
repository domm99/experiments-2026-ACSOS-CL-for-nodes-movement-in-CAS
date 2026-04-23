from phyelds.calculus import aggregate
from phyelds.libraries.time import local_time
from torch.utils.data import ConcatDataset
from CustomLeaderElection import elect_leaders
from phyelds.libraries.spreading import broadcast
from phyelds.libraries.collect import collect_with
from src import SIMULATION_STEPS, CHANGE_AREA_EACH
from phyelds.simulator.render import RenderMonitor
from phyelds.libraries.spreading import distance_to
from phyelds.libraries.device import local_id, store
from phyelds.simulator.deployments import deformed_lattice
from phyelds.calculus import aggregate, neighbors, remember
from phyelds.libraries.distances import neighbors_distances
from src.learning import local_training, local_distillation, model_evaluation, average_weights, initialize_model

impulsesEvery = 5
distillationEpochs = 1
distillationAlpha = 0.5
distillationTemperature = 2.0

@aggregate
def device(
    data,
    initial_model_weights,
    learning_device,
    seed,
    number_of_subareas,
    partitioning,
    moving=False,
    training_strategy='normal',
    distill_on_area_entry=True,
    enable_replay=True,
):

    ### Getting data
    dataset_name = data.dataset_name
    all_train_data = data.train_data
    all_validation_data = data.val_data
    all_test_data = data.test_data

    ### Hyperparams for exporting results
    hyperparams = f'seed-{seed}_subareas-{number_of_subareas}_dataset-{dataset_name}_partitioning-{partitioning}'

    ### Local training
    set_value, stored_info = remember((initial_model_weights, 0, 0))
    local_model_weights, tick, previous_area = stored_info

    if local_id() == 0:
        print(f'Doing tick: {tick}')

    local_model = load_from_weights(local_model_weights, dataset_name)

    current_area = (tick // CHANGE_AREA_EACH) % len(all_train_data) if moving else 0
    area_changed = moving and tick > 0 and current_area != previous_area

    if moving and enable_replay:
        # Add replay data from previous areas
        train_data = ConcatDataset(all_train_data[:current_area+1])
    else:
        train_data = all_train_data[current_area]

    trained_model, _ = local_training(local_model, 2, train_data, 128, learning_device)

    ### SCR
    distances = neighbors_distances()
    (am_i_leader, leader_id) = elect_leaders(20, distances)
    potential = distance_to(am_i_leader, distances)
    models = collect_with(potential, [trained_model], lambda x, y: x + y)
    aggregated_model = average_weights(models, [1.0 for _ in models])
    area_model = broadcast(am_i_leader, aggregated_model, distances)

    should_merge = (tick % impulsesEvery == 0) or (distill_on_area_entry and area_changed)
    if should_merge:
        if training_strategy == "distillation":
            # Distillation
            trained_model, _ = local_distillation(
                trained_model,
                area_model,
                train_data,
                128,
                learning_device,
                dataset_name,
                epochs=distillationEpochs,
                alpha=distillationAlpha,
                temperature=distillationTemperature,
            )
        elif training_strategy == "normal":
            trained_model = average_weights([trained_model, area_model], [0.1, 0.9])
        else:
            raise ValueError(f"Unknown training strategy: {training_strategy}")

    ### Moving node validation
    if moving:
        for area_id, validation_data in enumerate(all_validation_data):
            validation_accuracy, _ = model_evaluation(trained_model, validation_data, 128, learning_device, dataset_name)
            store(f'accuracy-area-{area_id}', validation_accuracy)

    set_value((trained_model, tick + 1, current_area))

    store('final_model', trained_model)
    store('test_data', all_test_data)
    store('hyperparams', hyperparams)

    return leader_id


def log(train_loss, validation_loss, validation_accuracy):
    store('TrainLoss', train_loss)
    store('ValidationLoss', validation_loss)
    store('ValidationAccuracy', validation_accuracy)


def load_from_weights(weights, dataset_name):
    model = initialize_model(dataset_name)
    model.load_state_dict(weights)
    return model