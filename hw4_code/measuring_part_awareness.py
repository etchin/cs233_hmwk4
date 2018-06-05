'''
CS233 Stanford, HW4

Note: Code tested with Python 2.7.

Written by Panos Achlioptas, 2018.
'''

import numpy as np
import os.path as osp
from collections import defaultdict

def adjacency_matrix(X):
    n = X.shape[0]
    adj_m = np.zeros((n,n))
    for i in range(n):
        for j in range(i,n):
            if i == j:
                adj_m[i,j] = np.inf
            else:
                d = np.linalg.norm(X[i,]-X[j,])
                adj_m[j,i] = adj_m[i,j] = d
    return(adj_m)

def find_nearest_neighbors(X, k=10):
    #X is a N x p matrix of latent vectors
    #N is the number of chairs, P is the dimension of each latent vector
    #We use euclidean distance to find each vector's nearest neighbor
    #returns a dictionary of nearest neighbors with distance
    adj_m = adjacency_matrix(X)
    nn_dict = dict()
    for i in range(adj_m.shape[0]):
        nn_dict[i] = [(m,adj_m[i,m]) for m in np.argsort(adj_m[i,])[:k]]
    return(nn_dict)

def model_stats(latent_vectors,shape_names,part_dist):
    knn = find_nearest_neighbors(latent_vectors,k=1)
    cum_dist = 0
    shared_parts = np.zeros(len(knn))
    matched_dist = 0
    for key in sorted(knn.keys()):
        sn_id = shape_names[key]
        neighbor_id = shape_names[knn[key][0][0]]
        m_dist = knn[key][0][1]
        matched_dist += m_dist
        x_loc = id_to_part_loc[sn_id][part_id]
        for i,part in enumerate(sn_id_to_parts[sn_id]):
            if part in sn_id_to_parts[neighbor_id]:
                cum_dist += part_dist[id_to_part_loc[sn_id][part],
                                       id_to_part_loc[neighbor_id][part]]
                shared_parts[key] += 1
            else:
                cum_dist += max(part_dist[id_to_part_loc[sn_id][part],
                              list(id_to_part_loc[neighbor_id].values())])
    return({'cumulative distance': cum_dist,
           'average latent distance': 1.0*matched_dist/len(knn),
           'average shared parts': np.mean(shared_parts)})
        
    
vanilla_ae_emb_file = '../data/out/Neural_nets/vanilla_ae/latent_codes.npz'
part_ae_emb_file = '../data/out/Neural_nets/pc_aware_ae/latent_codes.npz'
golden_part_dist_file = '../data/in/golden_dists.npz'

golden_data = np.load(golden_part_dist_file)
golden_part_dist = golden_data['golden_part_dist']
golden_names = golden_data['golden_names']

# To load vanilla-AE-embeddings (if False will open those of the 2-branch AE).
vanilla = True

#  Students. Extract shape-net model ids of golden, map them to their parts.
sn_id_to_parts = defaultdict(list)

# Students. Map shape-net model id and part_id to location in distance matrix.
id_to_part_loc = dict()

for i, gname in enumerate(golden_names):
    g_parts = gname.split('__')
    if g_parts[0] not in id_to_part_loc:
        id_to_part_loc[g_parts[0]] = dict()
    id_to_part_loc[g_parts[0]][g_parts[1]] = i
    sn_id_to_parts[g_parts[0]].append(g_parts[1])

if vanilla:
    in_d = np.load(vanilla_ae_emb_file)    # Students: this assumes that you used np.savez 
    latent_codes = in_d['l_codes']         # in the end of establishing_aes.py to save the
    test_names = in_d['test_names']        # latent_codes/test_names.
else:
    in_d = np.load(part_ae_emb_file)
    latent_codes = in_d['l_codes']
    test_names = in_d['test_names']


# Students. Implement Question [g].
#latent_codes must be Nxp where N is the number of samples. may need to transpose depending on output
model_performance = model_stats(latent_codes,test_names,golden_part_dist)

print('Average Holistic distance:',model_performance['average latent distance'] )
print('Average Aggrement: ', model_performance['average shared parts'])
print('Average Cumulative Dists:', model_performance['cumulative distance'])