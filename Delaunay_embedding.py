from mpmath import *
import networkx as nx
import matplotlib.pyplot as plt

tree = nx.DiGraph()
tree.add_edges_from([(0, 1, {'weight': 10}), (1, 2, {'weight': 21}), 
                    (1, 3, {'weight': 31}), (0, 4, {'weight': 40}),
                    (4, 5, {'weight': 54}), (4, 6, {'weight': 40})])

#tree.add_edges_from([(0, 1, {'weight': 10}), (0, 2, {'weight': 21}), 
                 #    (0, 3, {'weight': 31}), (0, 4, {'weight': 40}),
                 #    (0, 5, {'weight': 54})])

def compute_tau (tree, k, epsilon, is_weighted):    
    mp.dps = 50
    
    eps = power(10, -10)
    n = tree.number_of_nodes()
    degrees = [tree.degree(i) for i in range(n)]
    d_max = max(degrees)
    
    #cone separation angle beta < pi / d
    beta = fdiv(pi, fmul('1.2', d_max))
    #angle for cones 
    alpha = fsub(fdiv(fmul(2, pi), d_max), fmul(2, beta))
    nu = fmul(fmul(fneg(2), k), log(tan(fdiv(beta, 2))))
    #Compute for each edge the minimum required length
    #L(v_i, v_j) = -2*k*ln(tan(alpha/2))
    #min length is the same for all edges
    min_length = fmul(fmul(fneg(2), k), log(tan(fdiv(alpha, 2))))
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
        
    eta_max = fdiv(min_length, min_weight)    
    #Select tau > eta_max such that all edges are longer than nu*(1+epsilon)/epsilon
    #min weight of all edges
    tau = fadd(fdiv(fmul(fdiv(nu, min_weight), fadd(1, epsilon)), epsilon), eps)
    
    #if tau <= eta_max, set tau > eta_max
    if tau <= eta_max:
        tau = fadd(eta_max, eps)
        
    return tau

def map_to_zero (mu, x):
    mp.dps = 50
    
    z0 = mu[0] + mu[1]*j
    z = x[0] + x[1]*j
    result = fdiv(fsub(z0, z), fsub(1, fmul(conj(z0), z)))  #transformation function
    x[0] = re(result)
    x[1] = im(result)
    return x 
    
def add_children (p, x, edge_lengths):
    mp.dps = 50
    
    #map x to (0, 0)
    p0 = map_to_zero(x, p)
    c = len(edge_lengths)
    q = norm(p0)
    p_angle = acos(fdiv(p0[0], q))
    if p0[1] < 0:
        p_angle = fsub(fmul(2, pi), p_angle)
        
    alpha = fdiv(fmul(2, pi), (c+1))
    points0 = zeros(c, 2)
    
    #place child nodes of x
    for k in range(c):
        angle = fadd(p_angle, fmul(alpha, (k+1)))

        points0[k, 0] = fmul(edge_lengths[k], cos(angle))
        points0[k, 1] = fmul(edge_lengths[k], sin(angle))
        
        #reflect all neighboring nodes by mapping x back to its actual coordinates
        points0[k, :] = map_to_zero(x, points0[k, :])
     
    return points0
    
# Express a hyperbolic distance in the unit disk
def hyp_to_euc_dist(x):
    mp.dps = 50
    
    return sqrt(fdiv(fsub(cosh(x), 1), fadd(cosh(x), 1)))

def hyp_embedding (tree, k, epsilon, is_weighted):  
    mp.dps = 50
    
    n = tree.number_of_nodes()
    coords = zeros(n, 2)
    
    root_children = list(tree.successors(0))
    d = len(root_children)   
    tau = compute_tau(tree, k, epsilon, is_weighted)
    
    #lengths of unweighted edges
    edge_lengths = list(map(hyp_to_euc_dist, ones(d, 1) * tau))
       
    #lengths of weighted edges
    if is_weighted:
        k = 0
        for child in root_children:
            weight = tree[0][child]['weight']
            edge_lengths[k] = hyp_to_euc_dist(fmul(tau, weight))
            k += 1
    # queue containing the nodes whose children we're placing
    q = []
    
    #place children of the root
    for i in range(d):
        coords[root_children[i], 0] = fmul(edge_lengths[i], cos(i * 2 * pi / d))
        coords[root_children[i], 1] = fmul(edge_lengths[i], sin(i * 2 * pi / d))
                        
        q.append(root_children[i])
    
    node_idx = 0
    while len(q) > 0:
        #pop the node whose children we're placing off the queue
        h = q.pop(0)
        node_idx += 1
        
        children = list(tree.successors(h))
        parent = list(tree.predecessors(h))[0]
        num_children = len(children)
        
        for child in children:
            q.append(child)
        
        #lengths of unweighted edges
        edge_lengths = list(map(hyp_to_euc_dist, ones(d, 1) * tau))
        
        #lengths of weighted edges
        if is_weighted:
            k = 0
            for child in children:
                weight = tree[h][child]['weight']
                edge_lengths[k] = hyp_to_euc_dist(fmul(tau, weight))
                k += 1
    
        if num_children > 0:
            R = add_children(coords[parent, :], coords[h, :], edge_lengths)
            for i in range(num_children):
                coords[children[i], :] = R[i, :]
                
    return coords

def draw_edges (tree, x, embeddings):
    children = list(tree.successors(x))
    for child in children:
        x_values = [embeddings[x, 0], embeddings[child, 0]]
        y_values = [embeddings[x, 1], embeddings[child, 1]]
        plt.plot(x_values, y_values)

def visualize (tree, embeddings):
    for i in range(tree.number_of_nodes()):
        x_coord = embeddings[i, 0]
        y_coord = embeddings[i, 1]
        plt.scatter(x_coord, y_coord, label = str(i))
        #draw edges in the tree
        draw_edges(tree, i, embeddings)
        
    plt.legend()
     
    return embeddings

print(visualize(tree, hyp_embedding(tree, 1, 0.5, False)))


