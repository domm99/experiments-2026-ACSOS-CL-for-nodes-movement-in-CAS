import random
from Device import device
from dataclasses import dataclass
from torch.utils.data import Subset
from CustomDeployments import multi_grid
from src.learning import initialize_model
from phyelds.simulator import Simulator, Node
from CustomRenderMonitor import CustomRenderMonitor
from CustomDrawings import CustomDrawNodes, CustomDrawEdges
from phyelds.simulator.runner import aggregate_program_runner
from phyelds.simulator.neighborhood import radius_neighborhood
from phyelds.simulator.effects import DrawNodes, DrawEdges, RenderConfig, RenderMode
from ProFed import download_dataset, split_train_validation, partition_to_subregions


SIMULATION_STEPS = 40


@dataclass
class DeviceData:
    dataset_name: str
    train_data: Subset
    test_data: Subset
    other_data: list[tuple[Subset, Subset]]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    # TODO - seed also torch


def move_node(simulator: Simulator, time_delta: float, node: Node, i: int, **kwargs) -> None:
    possible_positions = [(0,0), (30.5, 0.5), (30.5, 30.5), (0.5, 30.5)]
    node.update(new_position = possible_positions[i])
    simulator.schedule_event(time_delta, move_node, simulator, time_delta, node, (i+1)%4, **kwargs)


def run_simulation(dataset_name: str, partitioning_method: str, number_of_regions: int, seed: int = 42) -> None:
    seed_everything(seed)

    simulator = Simulator()

    ## Nodes deployment
    simulator.environment.set_neighborhood_function(radius_neighborhood(40))
    mapping_area_nodes = multi_grid(simulator, [(0, 0, 7, 6, 2), (0, 30, 7, 6, 2), (30, 0, 7, 6, 2), (30, 30, 7, 6, 2)], 42)
    nodes_per_subarea = len(mapping_area_nodes[0])

    print(f'Nodes per subarea: {nodes_per_subarea}')

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
        print(f'Area: {area_id} -- Number of devices: {len(train_mapping_device_data)} -- ids {train_mapping_device_data.keys()}')

    device_data = {}

    print(all_data.keys())
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
                other_data = None

            device_data[index] = DeviceData(dataset_name, train_data, test_data, other_data)

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
            learning_device=device,
            seed=seed,
            number_of_subareas=number_of_subareas,
            partitioning=partitioning_method,
            moving=moving,
        )

    moving_node = list(simulator.environment.nodes.values())[0]
    #simulator.schedule_event(0.1, move_node, simulator, 30.0, moving_node, 1)

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
    simulator.run(SIMULATION_STEPS)

if __name__ == '__main__':

    seeds = [42]
    dataset_names = ['EMNIST']
    partitioning_methods = ['Hard']
    number_of_subareas = 4

    for seed in seeds:
        for dataset_name in dataset_names:
            for partitioning_method in partitioning_methods:
                run_simulation(dataset_name, partitioning_method, number_of_subareas, seed)