from phyelds.calculus import aggregate
from phyelds.libraries.device import local_id
from phyelds.libraries.time import local_time
from CustomLeaderElection import elect_leaders
from phyelds.libraries.collect import count_nodes
from phyelds.libraries.spreading import broadcast
from phyelds.simulator.render import RenderMonitor
from phyelds.libraries.spreading import distance_to
from phyelds.simulator.deployments import deformed_lattice
from phyelds.libraries.distances import neighbors_distances

@aggregate
def device(data, moving=False):

    train_data, val_data = data.train_data
    test_data = data.test_data[0]
    other_data = {i + 1: d for i, d in enumerate(data.other_data)} if local_id() == 0 else {}

    distances = neighbors_distances()
    (am_i_leader, leader_id) = elect_leaders(20, distances)
    potential = distance_to(am_i_leader, distances)
    nodes = count_nodes(potential)
    area_value = broadcast(leader, nodes, distances)
    return leader_id
