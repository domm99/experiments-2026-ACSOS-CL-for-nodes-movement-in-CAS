from phyelds.calculus import aggregate
from phyelds.libraries.time import local_time
from CustomLeaderElection import elect_leaders
from phyelds.libraries.collect import count_nodes
from phyelds.libraries.spreading import broadcast
from phyelds.simulator.render import RenderMonitor
from phyelds.libraries.spreading import distance_to
from phyelds.simulator.deployments import deformed_lattice
from phyelds.libraries.distances import neighbors_distances

@aggregate
def device():
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
