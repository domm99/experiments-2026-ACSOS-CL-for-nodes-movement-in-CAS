import torch
import random
import numpy as np
from src.Device import device
from src import SIMULATION_STEPS
from src.experiment_utils import make_experiment_name
from src.traveler.datasets import split_dataset_holdout
from src.traveler.runner import run_traveler
from dataclasses import dataclass
from torch.utils.data import Subset
from CustomDeployments import multi_grid
from src.learning import initialize_model
from phyelds.simulator import Simulator
from CustomRenderMonitor import CustomRenderMonitor
from CustomDrawings import CustomDrawNodes, CustomDrawEdges
from src.TestSetEvaluationMonitor import TestSetEvalMonitor
from phyelds.simulator.runner import aggregate_program_runner
from phyelds.simulator.neighborhood import radius_neighborhood
from phyelds.simulator.exporter import csv_exporter, ExporterConfig
from phyelds.simulator.effects import DrawNodes, DrawEdges, RenderConfig, RenderMode
from ProFed import download_dataset, split_train_validation, partition_to_subregions


@dataclass
class DeviceData:
    dataset_name: str
    area_id: int
    train_data: tuple[Subset, Subset]
    test_data: tuple[Subset, Subset]


def get_current_learning_device():
    learning_device: str = 'cpu'
    if torch.accelerator.is_available():
        current_accelerator = torch.accelerator.current_accelerator()
        if current_accelerator is not None:
            learning_device = current_accelerator.type
    return learning_device


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

def run_simulation(
    dataset_name: str,
    partitioning_method: str,
    number_of_regions: int,
    seed: int = 42,
    traveler_enabled: bool = False,
    traveler_holdout_ratio: float = 0.05,
    traveler_verbose: bool = True,
) -> None:
    seed_everything(seed)
    learning_device = get_current_learning_device()
    simulator = Simulator()

    ## Nodes deployment
    simulator.environment.set_neighborhood_function(radius_neighborhood(40))
    mapping_area_nodes = multi_grid(simulator, [(0, 0, 7, 6, 2), (0, 30, 7, 6, 2), (30, 0, 7, 6, 2), (30, 30, 7, 6, 2)], 42)
    nodes_per_subarea = len(mapping_area_nodes[0])

    ## Data split and distribution
    train_data, test_data = download_dataset(dataset_name)
    train_data, validation_data = split_train_validation(train_data, 0.7)
    test_data, _ = split_train_validation(test_data, 1.0)

    train_environment = partition_to_subregions(
        train_data,
        validation_data,
        dataset_name,
        partitioning_method,
        number_of_regions,
        seed,
    )

    test_environment = partition_to_subregions(
        test_data,
        test_data,
        dataset_name,
        partitioning_method,
        number_of_regions,
        seed,
    )

    # Held out data for the traveling device
    traveler_area_train_datasets: dict[int, Subset] = {}   
    if traveler_enabled:
        for area_id, region in enumerate(train_environment.regions):
            static_area_train, traveler_area_train_subset = split_dataset_holdout(
                region.training_data,
                traveler_holdout_ratio,
                seed,
                area_id,
            )
            region.training_data = static_area_train
            traveler_area_train_datasets[area_id] = traveler_area_train_subset

    all_data = {}

    for area_id in range(number_of_regions):
        train_mapping_device_data = train_environment.from_subregion_to_devices(
            area_id,
            nodes_per_subarea,
        )
        test_mapping_device_data = test_environment.from_subregion_to_devices(
            area_id,
            nodes_per_subarea,
        )
        all_data[area_id] = (train_mapping_device_data, test_mapping_device_data)

    device_data = {}
    for area_id in mapping_area_nodes.keys():
        ids = mapping_area_nodes[area_id]
        for index in ids:
            train_data_mapping, test_data_mapping = all_data[area_id]
            train_data_mapping_entry = train_data_mapping[index % nodes_per_subarea]
            test_data_mapping_entry = test_data_mapping[index % nodes_per_subarea]

            device_data[index] = DeviceData(
                dataset_name,
                area_id,
                train_data_mapping_entry,
                test_data_mapping_entry,
            )

    initial_model_weights = initialize_model(dataset_name).state_dict()

    # schedule the main function
    for node in simulator.environment.nodes.values():
        simulator.schedule_event(
            random.random() / 100,
            aggregate_program_runner,
            simulator,
            1.1,
            node,
            device,
            data=device_data[node.id],
            initial_model_weights=initial_model_weights,
            learning_device=learning_device,
            seed=seed,
            number_of_subareas=number_of_regions,
            partitioning=partitioning_method,
            traveler_enabled=traveler_enabled,
            traveler_holdout_ratio=traveler_holdout_ratio,
        )

    # render
    # CustomRenderMonitor(
    #     simulator,
    #     RenderConfig(
    #         effects=[CustomDrawEdges(), CustomDrawNodes(color_from="result")],
    #         mode=RenderMode.SAVE,
    #         save_as="scr.mp4",
    #         dt=0.1,
    #     )
    # )

    # Exporting training data
    experiment_name = make_experiment_name(
        seed,
        number_of_regions,
        dataset_name,
        partitioning_method,
        traveler_enabled,
        traveler_holdout_ratio,
    )

    config = ExporterConfig(
        'data/',
        experiment_name,
        ['TrainLoss', 'ValidationLoss', 'ValidationAccuracy'],
        ['mean', 'std', 'min', 'max'],
        3,
    )
    simulator.schedule_event(1.0, csv_exporter, simulator, 1.0, config)
    simulator.add_monitor(TestSetEvalMonitor(simulator, learning_device, dataset_name))

    # Run simulation
    simulator.run(SIMULATION_STEPS)

    if traveler_enabled:
        run_traveler(
            simulator=simulator,
            device_data=device_data,
            traveler_area_train_datasets=traveler_area_train_datasets,
            test_environment=test_environment,
            dataset_name=dataset_name,
            learning_device=learning_device,
            seed=seed,
            experiment_name=experiment_name,
            verbose=traveler_verbose,
        )


def main():
    seeds = [42]
    dataset_names = ['EMNIST']
    partitioning_methods = ['Hard']
    number_of_subareas = 4
    traveler_enabled = True
    traveler_holdout_ratio = 0.05
    traveler_verbose = True

    for seed in seeds:
        for dataset_name in dataset_names:
            for partitioning_method in partitioning_methods:
                run_simulation(
                    dataset_name,
                    partitioning_method,
                    number_of_subareas,
                    seed,
                    traveler_enabled=traveler_enabled,
                    traveler_holdout_ratio=traveler_holdout_ratio,
                    traveler_verbose=traveler_verbose,
                )


if __name__ == '__main__':
    main()
