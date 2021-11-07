import networkx as nx
import math
import numpy as np
from numpy.linalg import norm

tree = nx.DiGraph()
#tree.add_edges_from([(0, 1, {'weight': 10}), (1, 2, {'weight': 21}), 
  #                   (1, 3, {'weight': 31}), (0, 4, {'weight': 40}),
  #                   (4, 5, {'weight': 54})])

tree.add_edges_from([(0, 1, {'weight': 10}), (0, 2, {'weight': 21}), 
                     (0, 3, {'weight': 31}), (0, 4, {'weight': 40}),
                     (0, 5, {'weight': 54})])

#print(tree[0][1]['weight'])

def compute_tau (tree, k, epsilon, is_weighted):
    eps = pow(10, -10)
    n = tree.number_of_nodes()
    degrees = [tree.degree(i) for i in range(n)]
    d_max = max(degrees)
    
    #cone separation angle beta < pi / d
    beta = math.pi / (1.2 * d_max)
    #angle for cones 
    #alpha = 2*pi/d - 2*beta
    alpha = 2*math.pi / d_max - 2*beta
    nu = -2 * k * math.log(math.tan(beta/2))
    #Compute for each edge the minimum required length
    #L(v_i, v_j) = -2*k*ln(tan(alpha/2))
    #min length is the same for all edges
    min_length = -2 * k * math.log(math.tan(alpha / 2))
    #Compute for each edge the minimum scaling factor
    #eta_vi_vj = L(v_i, v_j) / w(v_i, v_j)
    if is_weighted:
        weights = []
        for v_i, v_j, weight in tree.edges(data=True):
            weights.append(tree[v_i][v_j]['weight'])
        min_weight = min(weights)
        if min_weight == float("inf"):
            min_weight = 1
    else: 
        min_weight = 1
    eta_max = min_length / min_weight
    #Select tau > eta_max such that all edges are longer than nu*(1+epsilon)/epsilon
    #min weight of all edges
    tau = nu / min_weight * (1+epsilon)/ epsilon + eps
    
    #if tau <= eta_max, set tau > eta_max
    if tau <= eta_max:
        tau = eta_max + eps
        
    return tau

def map_to_zero (mu, x):
    a = mu / norm(mu)**2
    r2 = norm(a, 2)**2 - 1
    return r2/norm(x - a, 2)**2 * (x-a) + a

def add_children (p, x, edge_lengths):
    p0 = map_to_zero(x, p)
    x0 = map_to_zero(x, x)
    c = len(edge_lengths)
    q = norm(p0, 2)
    p_angle = math.acos(p0[0]/q)
    if p0[1] < 0:
        p_angle = 2*math.pi-p_angle
