from phyelds.calculus import aggregate
from phyelds.libraries.time import local_time
from torch.utils.data import ConcatDataset, Subset
from phyelds.libraries.collect import count_nodes
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

impulsesEvery = 4
distillationEpochs = 1
distillationAlpha = 0.5
distillationTemperature = 2.0
local_sample_percentage = 0.05

@aggregate
def device_simple():
    distances = neighbors_distances()
    (am_i_leader, leader_id) = elect_leaders(20, distances)
    potential = distance_to(am_i_leader, distances)
    nodes = count_nodes(potential)
    if am_i_leader:
        print(f"Leader elected: {leader_id} - nodes: {nodes} - time: {local_time()}")
    area_value = broadcast(am_i_leader, nodes, distances)
    return area_value

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
    distill_on_area_entry=False,
    enable_replay=True,
    adaptable_area_weight=False,
    area_weight=0.9,
    min_area_weight=0.1,
    max_area_weight=0.9,
    alpha=0.5,
    min_current_alpha=0.1,
    max_current_alpha=0.9,
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
    #if local_id() == 0:
    #    print(f'Doing tick: {tick}')
    
    local_model = load_from_weights(local_model_weights, dataset_name)

    current_area = (tick // CHANGE_AREA_EACH) if moving else 0
    #if local_id() == 0:
    #    print(f"Node {local_id()} - Tick {tick} - Current area: {current_area} - {(tick // CHANGE_AREA_EACH)}")
    area_changed = moving and tick > 0 and current_area != previous_area
    sampled = int(len(all_train_data[current_area]) * local_sample_percentage)
    if moving and enable_replay:
        # Add replay data from previous areas
        train_data = ConcatDataset([Subset(ds, range(int(len(ds) * local_sample_percentage))) for ds in all_train_data[:current_area+1]])
        # Take the current area data for distillation
    else:
        train_data = Subset(all_train_data[current_area], range(sampled))
    trained_model, _ = local_training(local_model, 2, train_data, 32, learning_device)
    distilled_data = Subset(all_train_data[current_area], range(sampled))
    
    # statistics
    # print(f"Node {local_id()} - Tick {tick} - Area {current_area} - Train samples: {len(train_data)}")
    ### SCR
    distances = neighbors_distances()
    (am_i_leader, leader_id) = elect_leaders(20, distances)
    potential = distance_to(am_i_leader, distances)
    models = collect_with(potential, [trained_model], lambda x, y: x + y)
    aggregated_model = average_weights(models, [1.0 for _ in models])
    area_model = broadcast(am_i_leader, aggregated_model, distances)

    if moving and adaptable_area_weight:
        time_in_area = tick % CHANGE_AREA_EACH
        area_weight = time_in_area / CHANGE_AREA_EACH
        area_weight = max(min_area_weight, min(max_area_weight, area_weight))

    should_merge = (tick % impulsesEvery == 0) or (distill_on_area_entry and area_changed)
    if should_merge:
        if training_strategy == "distillation":
            if moving:
                current_alpha = min_current_alpha + (max_current_alpha - min_current_alpha) * (tick % CHANGE_AREA_EACH) / CHANGE_AREA_EACH
            else:
                current_alpha = alpha
            
            # Distillation
            trained_model, _ = local_distillation(
                trained_model,
                area_model,
                distilled_data,
                32,
                learning_device,
                dataset_name,
                epochs=5,
                alpha=current_alpha,
                temperature=distillationTemperature,
            )
            trained_model, _ = local_training(load_from_weights(trained_model, dataset_name), 1, train_data, 16, learning_device)
        
        elif training_strategy == "normal":
            if moving and adaptable_area_weight:
                time_in_area = tick % CHANGE_AREA_EACH
                current_area_weight = time_in_area / CHANGE_AREA_EACH
                current_area_weight = max(min_area_weight, min(max_area_weight, current_area_weight))
            else:
                current_area_weight = area_weight
            local_weight = 1.0 - area_weight
            trained_model = average_weights([trained_model, area_model], [local_weight, current_area_weight])
            trained_model, _ = local_training(load_from_weights(trained_model, dataset_name), 1, train_data, 16, learning_device)
        elif training_strategy == "no_merge":
            pass
        else:
            raise ValueError(f"Unknown training strategy: {training_strategy}")


    if am_i_leader:
        print(f"Node {local_id()} is a leader at tick {tick} with potential {potential} and collected {len(models)} models.")
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