import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import torch
import random
import time
import numpy as np
from typing import Literal
from src.Device import device
from dataclasses import dataclass
from torch.utils.data import Subset
from CustomDeployments import multi_grid, grid_from
from src.learning import initialize_model
from phyelds.simulator import Simulator, Node
from src import SIMULATION_STEPS, CHANGE_AREA_EACH
from CustomRenderMonitor import CustomRenderMonitor
from CustomDrawings import CustomDrawNodes, CustomDrawEdges
from src.TestSetEvaluationMonitor import TestSetEvalMonitor
from phyelds.simulator.runner import aggregate_program_runner
from phyelds.simulator.neighborhood import radius_neighborhood
from phyelds.simulator.exporter import csv_exporter, ExporterConfig
from src.custom_evaluation_exporter import subareas_evaluation_csv_exporter
from phyelds.simulator.effects import DrawNodes, DrawEdges, RenderConfig, RenderMode
from ProFed import download_dataset, split_train_validation, partition_to_subregions



@dataclass
class DeviceData:
    dataset_name: str
    train_data: list[Subset]
    val_data: list[Subset]
    test_data: list[Subset]


def get_current_learning_device(preferred_learning_device: str | None) -> str:
    dev_mod = torch.get_device_module(preferred_learning_device)
    return dev_mod.__name__.removeprefix("torch.")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def move_node(simulator: Simulator, time_delta: float, node: Node, i: int, **kwargs) -> None:
    possible_positions = [(0,0), (30.5, 0.5), (30.5, 30.5), (0.5, 30.5)]
    node.update(new_position = possible_positions[i])
    simulator.schedule_event(time_delta, move_node, simulator, time_delta, node, (i+1)%4, **kwargs)


def run_simulation(
    dataset_name: str,
    partitioning_method: str,
    number_of_regions: int,
    preferred_learning_device: str | None,
    training_strategy: Literal["normal", "distillation", "no_merge"] = 'distillation',
    distill_on_area_entry: bool = True,
    enable_replay: bool = True,
    seed: int = 42,
) -> None:
    seed_everything(seed)
    print("Starting simulation with seed:", seed)
    start_time = time.time()
    learning_device = get_current_learning_device(preferred_learning_device)
    simulator = Simulator()

    if training_strategy not in {'normal', 'distillation', 'no_merge'}:
        raise ValueError(f'Unknown training strategy: {training_strategy}')

    ## Nodes deployment
    simulator.environment.set_neighborhood_function(radius_neighborhood(40))
    mapping_area_nodes = multi_grid(simulator, [
        grid_from(xs=0, ys=0, width=7, height=6, spacing=2),  
        grid_from(xs=30, ys=0, width=7, height=6, spacing=2), 
        grid_from(xs=30, ys=30, width=7, height=6, spacing=2), 
        grid_from(xs=0, ys=30, width=7, height=6, spacing=2)], 
        42
    )
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

    all_data = {}

    for area_id in range(number_of_subareas):
        train_mapping_device_data = train_environment.from_subregion_to_devices(
            area_id,
            nodes_per_subarea + (0 if area_id == 0 else 1),
        )
        test_mapping_device_data = test_environment.from_subregion_to_devices(
            area_id,
            nodes_per_subarea + (0 if area_id == 0 else 1),
        )
        all_data[area_id] = (train_mapping_device_data, test_mapping_device_data)

    device_data = {}

    for area_id in mapping_area_nodes.keys():
        ids = mapping_area_nodes[area_id]
        for index in ids:
            train_data_mapping, test_data_mapping = all_data[area_id]
            train_data = train_data_mapping[index % nodes_per_subarea]
            test_data = test_data_mapping[index % nodes_per_subarea]

            if index == 0:
                other_data = []
                for next_area_id in range(1, number_of_subareas):
                    train_data_mapping, test_data_mapping = all_data[next_area_id]
                    next_train_data = train_data_mapping[nodes_per_subarea]
                    next_test_data = test_data_mapping[nodes_per_subarea]
                    other_data.append((next_train_data[0], next_train_data[1], next_test_data[0]))
            else:
                other_data = []

            train, val = train_data
            all_train = [train] + [o[0] for o in other_data]
            all_val = [val] + [o[1] for o in other_data]
            all_test = [test_data] + [o[2] for o in other_data]

            device_data[index] = DeviceData(dataset_name, all_train, all_val, all_test)

    initial_model_weights = initialize_model(dataset_name).state_dict()

    # schedule the main function
    for node in simulator.environment.nodes.values():
        moving = node.id == 0
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
            moving=moving,
            training_strategy=training_strategy,
            distill_on_area_entry=distill_on_area_entry,
            enable_replay=enable_replay,
        )

    moving_node = list(simulator.environment.nodes.values())[0]
    simulator.schedule_event(0.1, move_node, simulator, CHANGE_AREA_EACH, moving_node, 1)

    # render
    CustomRenderMonitor(
        simulator,
        RenderConfig(
            effects=[CustomDrawEdges(), CustomDrawNodes(color_from="result")],
            mode=RenderMode.SAVE,
            save_as="scr.mp4",
            dt=0.1
        )
    )

    # Exporting training data

    config = ExporterConfig(
        'data/',
        f'experiment_seed-{seed}_subareas-{number_of_regions}_dataset-{dataset_name}_partitioning-{partitioning_method}',
        [],
        [],
        3,
    )

    #simulator.schedule_event(1.0, csv_exporter, simulator, 1.0, config)
    simulator.schedule_event(1.0, subareas_evaluation_csv_exporter, simulator, 1.0, config, number_of_regions, 0)
    #simulator.add_monitor(TestSetEvalMonitor(simulator, learning_device, dataset_name))

    # Run simulation
    simulator.run(SIMULATION_STEPS)
    print(f"Simulation finished in {time.time() - start_time:.2f} seconds")


if __name__ == '__main__':

    seeds = [42]
    dataset_names = ['EMNIST']
    partitioning_methods = ['Hard']
    number_of_subareas = 4
    preferred_learning_device = None
    training_strategy = 'normal'
    distill_on_area_entry = True
    enable_replay = True

    for seed in seeds:
        for dataset_name in dataset_names:
            for partitioning_method in partitioning_methods:
                run_simulation(
                    dataset_name,
                    partitioning_method,
                    number_of_subareas,
                    preferred_learning_device,
                    training_strategy,
                    distill_on_area_entry,
                    enable_replay,
                    seed,
                )
