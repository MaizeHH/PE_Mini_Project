# Networks based on: https://simple.wikipedia.org/wiki/Network_topology#/media/File:NetworkTopologies.svg
from typing import List, Tuple

RING_NODES = [(1, 22), (2, 26), (3, 25), (4, 34), (5, 21), (6, 30)]
RING_EDGES = [(1, 2), (1, 6), (2, 3), (3, 4), (4, 5), (5, 6)]
MESH_NODES = [(1, 22), (2, 26), (3, 25), (4, 34), (5, 21), (6, 30)]
MESH_EDGES = [(1, 2), (1, 3), (1, 5), (2, 4), (3, 5), (4, 5), (5, 6)]
STAR_NODES = [(1, 22), (2, 26), (3, 25), (4, 34), (5, 21), (6, 30)]
STAR_EDGES = [(1, 6), (2, 6), (3, 6), (4, 6), (5, 6)]
FC_NODES = [(1, 22), (2, 26), (3, 25), (4, 34), (5, 21), (6, 30)]
FC_EDGES = [(1, 2), (1, 3), (1, 4), (1, 5), (1, 6), 
            (2, 3), (2, 4), (2, 5), (2, 6), 
            (3, 4), (3, 5), (3, 6), 
            (4, 5), (4, 6), 
            (5, 6)]

class Node:
    def __init__(self, id: int, data: int):
        self.id = id
        self.data = data
        self.neighbors = {}
    
    def add_neighbor(self, neighbor: 'Node'):
        self.neighbors[neighbor.id] = neighbor

class Topology:
    def __init__(self, nodes_list: List[Tuple[int, int]], edges: List[Tuple[int, int]]):
        self.nodes_list = nodes_list
        self.edges = edges
        self.nodes = {}
    
    def create_nodes(self):
        for id, data in self.nodes:
            self.nodes[id] = Node(id, data)
    
    def connect_nodes(self):
        for id1, id2 in self.edges:
            node1 = self.nodes.get(id1)
            node2 = self.nodes.get(id2)
            node1.add_neighbor(node2)
            node2.add_neighbor(node1)

    def build_matrices(self) -> Tuple[List[List[int]], List[List[int]], List[List[int]]]:
        n = len(self.nodes_list)
        adj_matrix = [[0]*n for _ in range(n)]
        deg_matrix = [[0]*n for _ in range(n)]
        inc_matrix = [[0]*len(self.edges) for _ in range(n)]
        neighbors = [0]*n

        for i, (id1, id2) in enumerate(self.edges):
            adj_matrix[id1-1][id2-1] = 1
            adj_matrix[id2-1][id1-1] = 1
            inc_matrix[id1-1][i] = 1
            inc_matrix[id2-1][i] = 1
            neighbors[id1-1] += 1
            neighbors[id2-1] += 1

        for i in range(n):
            deg_matrix[i][i] = neighbors[i]
        
        return adj_matrix, deg_matrix, inc_matrix