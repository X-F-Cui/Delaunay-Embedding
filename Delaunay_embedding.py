
import torch
import math

#tree is represented as edges and weights in the form [[v_i, v_j, w(v_i, v_j)], ...]
tree = torch.tensor([[0,1,6.0], [1,2,9.0], [1,3,10.0], [0,4,8.0], [4,5,7.0]])

def eta_max(tree, k):
    
    epsilon = pow(10, -7)
    
    n_vertices = list(tree.shape)[0] + 1
    max_cone_angles = torch.zeros(n_vertices)
    
    #v: node index
    def vertex_degree(v):
    
        degree = 0  
        
        for i in range(n_vertices-1):
            
            if (tree[i][0].item() == v):              
                degree += 1
                
            if (tree[i][1].item() == v):               
                degree += 1
        
        return degree
    
    #Select for each vertex a maximum cone angle 
    #miu(v_i) < 2*pi/d(v_i)
    for i in range(n_vertices):
       max_cone_angles[i] = 2*math.pi / vertex_degree(i) - epsilon
        
    #Select for each edge (v_i,v_j) the maximum cone angle as 
    #alpha_ij = min(miu(v_i), miu(v_j))
    edges_max_cone_angles = torch.clone(tree)
    
    for counter in range(n_vertices-1):
        
        i = int(tree[counter][0].item())
        j = int(tree[counter][1].item())
        
        edges_max_cone_angles[counter, 2] = min(max_cone_angles[i], max_cone_angles[j])
        
    #Compute for each edge the minimum required length
    #L(v_i, v_j) = -2*k*ln(tan(alpha_ij/2))
    min_length = torch.clone(tree)
    
    for i in range(n_vertices-1):        
        min_length[i, 2] = -2 * k * math.log(math.tan(edges_max_cone_angles[i, 2] / 2))

    #Compute for each edge the minimum scaling factor
    #eta_vi_vj = L(v_i, v_j) / w(v_i, v_j)
    min_scaling_factor = torch.clone(tree)
    
    for i in range(n_vertices-1):
        min_scaling_factor[i, 2] = min_length[i, 2]/ tree[i, 2]
        
    #Compute the max value of eta over all edges
    etas = min_scaling_factor[:, 2]
    max_eta = torch.max(etas).item()
    
    return max_eta



print (eta_max(tree, -1))
       
       
       
       
       
       
       
       
       
        
