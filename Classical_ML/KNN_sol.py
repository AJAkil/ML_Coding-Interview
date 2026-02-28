'''Given a list of points in n-dimensional space represented as tuples and a query point, implement a function to find the k nearest neighbors to the query point using Euclidean distance.

Requirements
Calculate the Euclidean distance from the query point to each point in the list
Return the k points with the smallest distances as a list of tuples
When distances are equal (ties), maintain the original order from the input list (i.e., the point appearing earlier in the input list should appear first in the result)

'''
import numpy as np

def k_nearest_neighbors(points, query_point, k):
    """
    Find k nearest neighbors to a query point
    
    Args:
        points: List of tuples representing points [(x1, y1), (x2, y2), ...]
        query_point: Tuple representing query point (x, y)
        k: Number of nearest neighbors to return
    
    Returns:
        List of k nearest neighbor points as tuples
        When distances are tied, points appearing earlier in the input list come first.
    """
    if k > len(points):
        k = len(points)

    points_array = np.array(points) #(M,D)
    query_array = np.array(query_point) #(1,D) or (Q,D)

    # (M,D) -(M,1,D) - (Q,D) -> (M,Q,D)
    diff = points_array[:, np.newaxis, :] - query_array
    distances = np.linalg.norm(diff, axis=2) # along feature axis - (M,Q)

    nearest_indices = np.argsort(distances, axis=0)[:k, :] #(k, Q)
    nearest_indices = nearest_indices.T

    results = []
    for nei_indices in nearest_indices:
        temp = []
        for idx in nei_indices:
            #print(idx)
            temp.append(points[idx])
        results.append(temp)

    # print(points_array.shape)
    # print(query_array.shape)
    return results