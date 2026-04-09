from typing import Protocol

import torch
from torch import nn
from torch.utils.data import Subset

from src import SIMULATION_STEPS
from phyelds.calculus import aggregate
from phyelds.libraries.time import local_time
from CustomLeaderElection import elect_leaders
from phyelds.libraries.spreading import broadcast
from phyelds.libraries.collect import collect_with
from phyelds.simulator.render import RenderMonitor
from phyelds.libraries.spreading import distance_to
from phyelds.libraries.device import local_id, store
from phyelds.simulator.deployments import deformed_lattice
from phyelds.calculus import aggregate, neighbors, remember
from phyelds.libraries.distances import neighbors_distances
from src.experiment_utils import make_experiment_suffix
from src.learning import local_training, model_evaluation, average_weights, initialize_model

impulsesEvery = 5
StateDict = dict[str, torch.Tensor]


# The same data structure found in the main (only used fields)
class DeviceInputData(Protocol):
    dataset_name: str
    train_data: tuple[Subset, Subset]
    test_data: tuple[Subset, Subset]


@aggregate
def device(
    data: DeviceInputData,
    initial_model_weights: StateDict,
    learning_device: str,
    seed: int,
    number_of_subareas: int,
    partitioning: str,
    traveler_enabled: bool = False,
    traveler_holdout_ratio: float = 0.05,
) -> int:

    ### Getting data
    dataset_name = data.dataset_name
    train_data, val_data = data.train_data
    test_data = data.test_data[0]

    ### Hyperparams for exporting results
    hyperparams = make_experiment_suffix(
        seed=seed,
        number_of_regions=number_of_subareas,
        dataset_name=dataset_name,
        partitioning_method=partitioning,
        traveler_enabled=traveler_enabled,
        traveler_holdout_ratio=traveler_holdout_ratio,
    )

    ### Local training
    set_value, stored_info = remember((initial_model_weights, 0))
    local_model_weights, tick = stored_info

    local_model = load_from_weights(local_model_weights, dataset_name)

    trained_model, training_loss = local_training(local_model, 2, train_data, 128, learning_device)
    validation_accuracy, validation_loss = model_evaluation(trained_model, val_data, 128, learning_device, dataset_name)

    log(training_loss, validation_loss, validation_accuracy)  # Metrics logging

    ### SCR
    distances = neighbors_distances()
    (am_i_leader, leader_id) = elect_leaders(20, distances)
    potential = distance_to(am_i_leader, distances)
    models = collect_with(potential, [trained_model], lambda x, y: x + y)
    aggregated_model = average_weights(models, [1.0 for _ in models])
    area_model = broadcast(am_i_leader, aggregated_model, distances)

    if tick % impulsesEvery == 0:
        avg = average_weights([trained_model, area_model], [0.1, 0.9])
        set_value((avg, tick + 1))
    else:
        set_value((trained_model, tick + 1))

    store('final_model', trained_model)
    store('final_area_model', area_model)
    store('test_data', test_data)
    store('hyperparams', hyperparams)

    return leader_id


def log(train_loss: float, validation_loss: float, validation_accuracy: float) -> None:
    store('TrainLoss', train_loss)
    store('ValidationLoss', validation_loss)
    store('ValidationAccuracy', validation_accuracy)


def load_from_weights(weights: StateDict, dataset_name: str) -> nn.Module:
    model = initialize_model(dataset_name)
    model.load_state_dict(weights)
    return model
