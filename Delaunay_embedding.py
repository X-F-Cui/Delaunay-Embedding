import networkx as nx
import torch
import math
import numpy as np

tree = nx.Graph()
#tree.add_edges_from([(0, 1, {'weight': 10}), (1, 2, {'weight': 21}), 
  #                   (1, 3, {'weight': 31}), (0, 4, {'weight': 40}),
  #                   (4, 5, {'weight': 54})])

tree.add_edges_from([(0, 1, {'weight': 10}), (0, 2, {'weight': 21}), 
                     (0, 3, {'weight': 31}), (0, 4, {'weight': 40}),
                     (0, 5, {'weight': 54})])

#print(tree[0][1]['weight'])

def embedding (tree, k, epsilon):
    
    eps = pow(10, -10)
    
    degrees = [tree.degree(i) for i in range(tree.number_of_nodes())]
    # max degree
    d = max(degrees)
    
    #cone separation angle beta < pi / d
    beta = eps
    
    #angle for cones 
    #alpha = 2*pi/d - 2*beta
    alpha = 2*math.pi / d - 2*beta
    
    nu = -2 * k * math.log(math.tan(beta/2))

    #Compute for each edge the minimum required length
    #L(v_i, v_j) = -2*k*ln(tan(alpha/2))
    #min length is the same for all edges
    min_length = -2 * k * math.log(math.tan(alpha / 2))
    
    #Compute for each edge the minimum scaling factor
    #eta_vi_vj = L(v_i, v_j) / w(v_i, v_j)
    etas = []
    for v_i, v_j, weight in tree.edges(data=True):
        etas.append(min_length / (tree[v_i][v_j]['weight']))
                    





print(embedding(tree, -1, pow(10, -10)))
