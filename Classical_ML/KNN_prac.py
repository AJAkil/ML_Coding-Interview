import numpy as np

def knn(points, query_point, k):
    if k > points:
        k = len(points)

    points_ = np.array(points) #(m, D)
    query_ = np.array(query_point) #(q, D)

    diff = points_[:, np.newaxis, :] - query_ # (m,1,D) - (q,D) = (m,q,D)

    dist = np.linalg.norm(diff, axis=2) # along feature dim (m,q)

    # this is where the differences start to pop up
    nearest_indices = np.argsort(dist, axis=0)[:k, :] #(k, Q)
    nearest_indices = nearest_indices.T # (Q, k)

    results = []

    for nei_indices in nearest_indices:
        # I have k neighbors to choose from 
        temp = []

        for idx in nei_indices:
            temp.append(points[idx])
        
        results.append(temp)
    
    return results