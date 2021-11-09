import networkx as nx
import math
import numpy as np
from numpy.linalg import norm

tree = nx.DiGraph()
#tree.add_edges_from([(0, 1, {'weight': 10}), (1, 2, {'weight': 21}), 
               #     (1, 3, {'weight': 31}), (0, 4, {'weight': 40}),
               #     (4, 5, {'weight': 54})])

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
    c = len(edge_lengths)
    q = norm(p0, 2)
    p_angle = math.acos(p0[0]/q)
    if p0[1] < 0:
        p_angle = 2*math.pi-p_angle
        
    alpha = 2*math.pi / (c+1)
    points0 = np.zeros((c+1, 2))
    
    #place child nodes of x
    for k in range(1, c+1):
        angle = p_angle + alpha * k
        points0[k][0] = edge_lengths[k-1] * math.cos(angle)
        points0[k][1] = edge_lengths[k-1] * math.sin(angle)
        
    #reflect all neighboring nodes by mapping x to (0, 0)
    for k in range(c+1):
        points0[k, :] = map_to_zero(x, points0[k, :])
        
    return points0
    
# Express a hyperbolic distance in the unit disk
def hyp_to_euc_dist(x):
    return 1
    #overflow
    #return math.sqrt((math.cosh(x)-1)/(math.cosh(x)+1))

def hyp_embedding (tree, k, epsilon, is_weighted):  
    n = tree.number_of_nodes()
    coords = np.zeros((n, 2))
    
    root_children = list(tree.successors(0))
    d = len(root_children)   
    tau = compute_tau(tree, k, epsilon, is_weighted)
    
    #lengths of unweighted edges
    edge_lengths = list(map(hyp_to_euc_dist, list(map(lambda x: x * tau, np.ones(d)))))
    
    #lengths of weighted edges
    if is_weighted:
        k = 0
        for child in root_children:
            weight = tree[0][child]['weight']
            edge_lengths[k] = hyp_to_euc_dist(tau * weight)
            k += 1
    
    # queue containing the nodes whose children we're placing
    q = []
    
    #place children of the root
    for i in range(d):
        coords[root_children[i]][0] = edge_lengths[i] * math.cos(i * 2 * math.pi / d)
        coords[root_children[i]][1] = edge_lengths[i] * math.sin(i * 2 * math.pi / d)
                    
        q.append(root_children[i])
    
    node_idx = 0
    while len(q) > 0:
        #pop the node whose children we're placing off the queue
        h = q.pop(0)
        node_idx += 1
        
        children = list(tree.successors(h))
        parent = list(tree.predecessors(h))[0]
        num_children = len(children)
        
        #lengths of unweighted edges
        edge_lengths = list(map(hyp_to_euc_dist, list(map(lambda x: x * tau, np.ones(d)))))
        
        #lengths of weighted edges
        if is_weighted:
            k = 0
            for child in children:
                weight = tree[h][child]['weight']
                edge_lengths[k] = hyp_to_euc_dist(tau * weight)
                k += 1
    
        if num_children > 0:
            R = add_children(coords[parent, :], coords[h, :], edge_lengths)
            for i in range(num_children):
                coords[children[i], :] = R[i, :]
                
    return coords
            

print(hyp_embedding(tree, 1, pow(10, -10), True))
