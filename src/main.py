import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import argparse
import torch
import random
import time
import numpy as np
import multiprocessing
from typing import Literal
from src.Device import device, device_simple
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


MOVING_NODE_FRACTION = 0.30


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


def move_node(simulator: Simulator, time_delta: float, node: Node, home_area: int, step: int, node_offset: tuple[float, float], **kwargs) -> None:
    possible_positions = [(0, 0), (30.5, 0.5), (30.5, 30.5), (0.5, 30.5)]
    current_position_index = (home_area + step) % 4
    base = possible_positions[current_position_index]
    node.update(new_position=(base[0] + node_offset[0], base[1] + node_offset[1]))
    simulator.schedule_event(time_delta, move_node, simulator, time_delta, node, home_area, step + 1, node_offset, **kwargs)


def run_simulation(
    experiment_name:str,
    dataset_name: str,
    partitioning_method: str,
    number_of_regions: int,
    preferred_learning_device: str | None,
    training_strategy: Literal["normal", "distillation", "no_merge"] = 'distillation',
    distill_on_area_entry: bool = True,
    enable_replay: bool = True,
    adaptable_area_weight: bool = True,
    area_weight: float = 0.4,
    min_area_weight: float = 0.05,
    max_area_weight: float = 0.3,
    alpha: float = 0.5,
    min_current_alpha: float = 0.1,
    max_current_alpha: float = 0.9,
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
    simulator.environment.set_neighborhood_function(radius_neighborhood(10))
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

    for area_id in range(number_of_regions):
        train_mapping_device_data = train_environment.from_subregion_to_devices(
            area_id,
            nodes_per_subarea + 1,
        )
        test_mapping_device_data = test_environment.from_subregion_to_devices(
            area_id,
            nodes_per_subarea + 1,
        )
        all_data[area_id] = (train_mapping_device_data, test_mapping_device_data)

    total_nodes = sum(len(ids) for ids in mapping_area_nodes.values())
    total_moving = max(1, int(total_nodes * MOVING_NODE_FRACTION))
    nodes_per_area = {area_id: len(ids) for area_id, ids in mapping_area_nodes.items()}
    moving_per_area = {}
    remaining = total_moving
    for area_id in sorted(nodes_per_area.keys()):
        alloc = max(1, round(nodes_per_area[area_id] / total_nodes * total_moving)) if remaining > 1 else remaining
        moving_per_area[area_id] = alloc
        remaining -= alloc

    moving_node_ids = []
    for area_id in mapping_area_nodes.keys():
        area_node_list = mapping_area_nodes[area_id]
        n_moving = moving_per_area[area_id]
        step = max(1, len(area_node_list) // (n_moving + 1))
        selected = [area_node_list[i * step] for i in range(n_moving)]
        moving_node_ids.extend(selected)

    moving_node_ids_set = set(moving_node_ids)
    node_home_area = {}
    for area_id, ids in mapping_area_nodes.items():
        for nid in ids:
            if nid in moving_node_ids_set:
                node_home_area[nid] = area_id

    print(f"Moving nodes ({len(moving_node_ids)}/{total_nodes}, {MOVING_NODE_FRACTION*100:.0f}%): {moving_node_ids}")
    for nid in moving_node_ids:
        print(f"  Node {nid} - Home area: {node_home_area[nid]}")

    device_data = {}

    for area_id in mapping_area_nodes.keys():
        ids = mapping_area_nodes[area_id]
        for index in ids:
            train_data_mapping, test_data_mapping = all_data[area_id]
            train_data = train_data_mapping[index % nodes_per_subarea]
            test_data = test_data_mapping[index % nodes_per_subarea]

            if index in moving_node_ids_set:
                all_train = [None] * number_of_regions
                all_val = [None] * number_of_regions
                all_test = [None] * number_of_regions
                all_train[area_id] = train_data[0]
                all_val[area_id] = train_data[1]
                all_test[area_id] = test_data
                for other_area_id in range(number_of_regions):
                    if other_area_id == area_id:
                        continue
                    other_mapping_train, other_mapping_test = all_data[other_area_id]
                    other_train_data = other_mapping_train[nodes_per_subarea]
                    other_test_data = other_mapping_test[nodes_per_subarea]
                    all_train[other_area_id] = other_train_data[0]
                    all_val[other_area_id] = other_train_data[1]
                    all_test[other_area_id] = other_test_data
            else:
                train, val = train_data
                all_train = [train]
                all_val = [val]
                all_test = [test_data]

            device_data[index] = DeviceData(dataset_name, all_train, all_val, all_test)

    initial_model_weights = initialize_model(dataset_name).state_dict()

    # schedule the main function
    for node in simulator.environment.nodes.values():
        moving = node.id in moving_node_ids_set
        home_area = node_home_area.get(node.id, 0)
        simulator.schedule_event(
            random.random() / 100,
            aggregate_program_runner,
            simulator,
            1.0,
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
            adaptable_area_weight=adaptable_area_weight,
            area_weight=area_weight,
            min_area_weight=min_area_weight,
            max_area_weight=max_area_weight,
            alpha=alpha,
            min_current_alpha=min_current_alpha,
            max_current_alpha=max_current_alpha,
            home_area=home_area,
        )

    for moving_id in moving_node_ids:
        moving_node = simulator.environment.nodes[moving_id]
        home_area = node_home_area[moving_id]
        node_offset = (random.uniform(-1.0, 1.0), random.uniform(-1.0, 1.0))
        simulator.schedule_event(0.1, move_node, simulator, CHANGE_AREA_EACH, moving_node, home_area, 0, node_offset)

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
        f'experiment_seed-{seed}_{experiment_name}',
        [],
        [],
        3,
    )

    #simulator.schedule_event(1.0, csv_exporter, simulator, 1.0, config)
    simulator.schedule_event(1.0, subareas_evaluation_csv_exporter, simulator, 1.0, config, number_of_regions, moving_node_ids)
    #simulator.add_monitor(TestSetEvalMonitor(simulator, learning_device, dataset_name))

    # Run simulation
    simulator.run(SIMULATION_STEPS)
    print(f"Simulation finished in {time.time() - start_time:.2f} seconds")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run simulation experiment')
    parser.add_argument('--experiment_name', type=str, required=True)
    parser.add_argument('--dataset_name', type=str, required=True)
    parser.add_argument('--partitioning_method', type=str, required=True)
    parser.add_argument('--number_of_regions', type=int, required=True)
    parser.add_argument('--preferred_learning_device', type=str, default='cpu')
    parser.add_argument('--training_strategy', type=str, default='distillation', choices=['normal', 'distillation', 'no_merge'])
    parser.add_argument('--distill_on_area_entry', type=lambda x: x.lower() == 'true', default=True)
    parser.add_argument('--enable_replay', type=lambda x: x.lower() == 'true', default=True)
    parser.add_argument('--adaptable_area_weight', type=lambda x: x.lower() == 'true', default=True)
    parser.add_argument('--area_weight', type=float, default=0.9)
    parser.add_argument('--min_area_weight', type=float, default=0.1)
    parser.add_argument('--max_area_weight', type=float, default=0.9)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--min_current_alpha', type=float, default=0.1)
    parser.add_argument('--max_current_alpha', type=float, default=0.9)
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    run_simulation(
        args.experiment_name,
        args.dataset_name,
        args.partitioning_method,
        args.number_of_regions,
        args.preferred_learning_device,
        args.training_strategy,
        args.distill_on_area_entry,
        args.enable_replay,
        args.adaptable_area_weight,
        args.area_weight,
        args.min_area_weight,
        args.max_area_weight,
        args.alpha,
        args.min_current_alpha,
        args.max_current_alpha,
        args.seed,
    )
