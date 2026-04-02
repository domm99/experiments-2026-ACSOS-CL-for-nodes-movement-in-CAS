import random
from phyelds.simulator import Simulator

def multi_gaussian(simulator: Simulator, nodes_per_gaussian: int, gaussian_statistics: list, random_seed: int) -> None:
    random.seed(random_seed)
    node_id = 0
    for (centerX, centerY, stddev) in gaussian_statistics:
        for _ in range(nodes_per_gaussian):
            x = random.gauss(centerX, stddev)
            y = random.gauss(centerY, stddev)
            simulator.create_node((x,y), None, node_id)
            node_id += 1

def multi_grid(simulator: Simulator, grid_statistics: list, random_seed: int) -> dict[int, list[int]]:
    random.seed(random_seed)
    node_id = 0
    area_id = 0
    mapping_area_nodes = {}
    for (xs, ys, width, height, spacing) in grid_statistics:
        mapping_area_nodes[area_id] = []
        for x in range (xs, xs + width, spacing):
            for y in range (ys, ys + height, spacing):
                simulator.create_node((x,y), None, node_id)
                mapping_area_nodes[area_id].append(node_id)
                node_id += 1
        area_id += 1
    return mapping_area_nodes