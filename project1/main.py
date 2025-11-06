# Mini Project 1: Additive Secret Sharing-based Distributed Average Consensus

from network import (
    Topology, Node, 
    RING_NODES, RING_EDGES, 
    MESH_NODES, MESH_EDGES, 
    STAR_NODES, STAR_EDGES, 
    FC_NODES, FC_EDGES,
    EX_NODES, EX_EDGES, 
    create_network
)

nodes, edges = create_network('ring', 6, 20, 30)
topology = Topology(nodes, edges)
print(topology.w_matrix)