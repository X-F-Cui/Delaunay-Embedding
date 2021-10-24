
import torch
import math

#tree is represented as edges and weights in the form [[v_i, v_j, w(v_i, v_j)], ...]
tree = torch.tensor([[0,1,6.0], [1,2,9.0], [1,3,10.0], [0,4,8.0], [4,5,7.0]])

def embedding(tree, k, epsilon):
    
    eps = pow(10, -10)
    n_vertices = list(tree.shape)[0] + 1
    
    #v: node index
    def child_node(v):  
        
        child_node = []
        
        for i in range(n_vertices-1):
            
            if tree[i][0].item() == v:              
                child_node.append(tree[i][1].item())
        
        return child_node
    
    degrees = torch.zeros(n_vertices)
    #a list of lists of child nodes for each vertex
    child_nodes = []
    
    for i in range(n_vertices):
        
        child_nodes.append(child_node(i))
        
        if i == 0:
            #root node doesn't have parent node
            degrees[i] = len(child_nodes[i])
        else:
            degrees[i] = len(child_nodes[i]) + 1
        
    #d: max degree of any node
    max_degree = torch.max(degrees).item()
    
    #cone separation angle beta < pi / d
    beta = eps
    
    #angle for cones 
    #alpha = 2*pi/d - 2*beta
    alpha = 2*math.pi / max_degree - 2*beta
    
    nu = -2 * k * math.log(math.tan(beta/2))
    
    #Compute for each edge the minimum required length
    #L(v_i, v_j) = -2*k*ln(tan(alpha/2))
    min_lengths = torch.clone(tree)
    
    min_length = -2 * k * math.log(math.tan(alpha / 2))
    
    for i in range(n_vertices-1):        
        min_lengths[i, 2] = min_length

    #Compute for each edge the minimum scaling factor
    #eta_vi_vj = L(v_i, v_j) / w(v_i, v_j)
    min_scaling_factor = torch.clone(tree)
    
    for i in range(n_vertices-1):
        min_scaling_factor[i, 2] = min_lengths[i, 2]/ tree[i, 2]
        
    #Compute the max value of eta over all edges
    etas = min_scaling_factor[:, 2]
    eta_max = torch.max(etas).item()
    
    #Select tau > eta_max such that all edges are longer than nu*(1+epsilon)/epsilon
    #min weight of all edges
    min_weight = torch.min(tree[:, 2]).item()
    tau = nu / min_weight * (1+epsilon)/ epsilon +eps
    
    #if tau <= eta_max, set tau > eta_max
    if tau <= eta_max:
        tau = eta_max + eps
        
    #return tau
        
    #embed edges
    coordinates = torch.zeros(n_vertices+1, 2)
    
    #v: vertex number
    #z: coordinates in hyperbolic space represented as complex number x+yi
    
    def embed_vertex(v, z):
        
        if len(child_nodes[v]) != 0:
            
            #for the ith child in the child nodes of v
            for i in range(len(child_nodes[v])): 
                
                #find the weight of embedded edge
                for j in range(n_vertices-1):                  
                    if (tree[j][0].item() == v) and (tree[j][1].item() == child_nodes[v][i]):
                        weight = tree[j][2].item()
                #embedded length
                r = tau * weight
                #cone angle
                theta = alpha * i
       
       
       
       
       
       
       
       
       
        
