import random
from phyelds.calculus import aggregate
from phyelds.simulator import Simulator
from phyelds.libraries.time import local_time
from phyelds.libraries.collect import count_nodes
from phyelds.libraries.spreading import broadcast
from phyelds.simulator.render import RenderMonitor
from phyelds.libraries.spreading import distance_to
from CustomRenderMonitor import CustomRenderMonitor
from CustomDeployments import multi_gaussian, multi_grid
from phyelds.simulator.deployments import deformed_lattice
from phyelds.libraries.distances import neighbors_distances
from phyelds.libraries.leader_election import elect_leaders
from CustomDrawings import CustomDrawNodes, CustomDrawEdges
from phyelds.simulator.runner import aggregate_program_runner
from phyelds.simulator.neighborhood import radius_neighborhood
from phyelds.simulator.effects import DrawNodes, DrawEdges, RenderConfig, RenderMode


@aggregate
def main():
    """
    Example to use the phyelds library to create a simple simulation
    :return:
    """
    distances = neighbors_distances()
    leader = elect_leaders(4, distances)
    potential = distance_to(leader, distances)
    nodes = count_nodes(potential)
    area_value = broadcast(leader, nodes, distances)
    return area_value


if __name__ == '__main__':

    random.seed(42)

    simulator = Simulator()
    # deformed lattice
    simulator.environment.set_neighborhood_function(radius_neighborhood(1.15))
    multi_grid(simulator, [(0, 0, 5, 5, 1), (0, 10, 5, 5, 1), (10, 0, 5, 5, 1), (10, 10, 5, 5, 1)], 42)
    # put source
    for node in simulator.environment.nodes.values():
        node.data = {"source": False, "target": False}
    # put a source in the first node
    simulator.environment.node_list()[0].data["source"] = True
    target = simulator.environment.node_list()[-1]
    target.data["target"] = True
    # schedule the main function
    for node in simulator.environment.nodes.values():
        simulator.schedule_event(random.random() / 100, aggregate_program_runner, simulator, 1.1, node, main)
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
    simulator.run(200)