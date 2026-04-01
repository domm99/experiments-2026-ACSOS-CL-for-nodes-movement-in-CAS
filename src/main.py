import random
from phyelds.calculus import aggregate
from CustomDeployments import multi_grid
from phyelds.simulator import Simulator, Node
from phyelds.libraries.time import local_time
from phyelds.libraries.collect import count_nodes
from phyelds.libraries.spreading import broadcast
from phyelds.simulator.render import RenderMonitor
from phyelds.libraries.spreading import distance_to
from CustomRenderMonitor import CustomRenderMonitor
from phyelds.simulator.deployments import deformed_lattice
from phyelds.libraries.distances import neighbors_distances
# from phyelds.libraries.leader_election import elect_leaders
from CustomLeaderElection import elect_leaders
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
    (am_i_leader, leader_id) = elect_leaders(20, distances)
    # potential = distance_to(am_i_leader, distances)
    # nodes = count_nodes(potential)
    # area_value = broadcast(leader, nodes, distances)
    return leader_id


def move_node(simulator: Simulator, time_delta: float, node: Node, i: int, **kwargs):
    possible_positions = [(0,0), (30.5, 0.5), (30.5, 30.5), (0.5, 30.5)]
    node.update(new_position = possible_positions[i])
    simulator.schedule_event(time_delta, move_node, simulator, time_delta, node, (i+1)%4, **kwargs)


if __name__ == '__main__':

    random.seed(42)

    simulator = Simulator()
    # deformed lattice
    simulator.environment.set_neighborhood_function(radius_neighborhood(40))
    multi_grid(simulator, [(0, 0, 5, 5, 1), (0, 30, 5, 5, 1), (30, 0, 5, 5, 1), (30, 30, 5, 5, 1)], 42)

    # schedule the main function
    for node in simulator.environment.nodes.values():
        simulator.schedule_event(random.random() / 100, aggregate_program_runner, simulator, 1.1, node, main)

    moving_node = list(simulator.environment.nodes.values())[0]
    simulator.schedule_event(0.1, move_node, simulator, 30.0, moving_node, 1)

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
    simulator.run(100)