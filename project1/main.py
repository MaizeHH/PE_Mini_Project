# Mini Project 1: Additive Secret Sharing-based Distributed Average Consensus

from network import (
    Topology, Node, 
    RING_NODES, RING_EDGES, 
    MESH_NODES, MESH_EDGES, 
    STAR_NODES, STAR_EDGES, 
    FC_NODES, FC_EDGES,
    create_network
)

nodes, edges = create_network('mesh', 6, 20, 30)
topology = Topology(nodes, edges)
print(nodes)
print(edges)
print("-----------------------------------------")
print(topology.build_matrices())