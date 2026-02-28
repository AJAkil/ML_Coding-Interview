import numpy as np
def k_means_clustering(points: list[tuple[float, ...]], k: int, initial_centroids: list[tuple[float, ...]], max_iterations: int) -> list[tuple[float, ...]]:
	# Your code here
	pts = np.array(points, dtype=np.float64) #(m,D)
	centroids = np.array(initial_centroids, dtype=np.float64) #(k,D)

	# print(pts.shape)
	# print(centroids.shape)

	for _ in range(max_iterations):
		distances = np.linalg.norm(pts[:, np.newaxis] - centroids, axis=2) # (m,D)->(m,1,D) - (k,D) = (m,k,D) -> (m,k)

		labels = np.argmin(distances, axis=1) #(m,k) -> m

		new_centroids = np.zeros_like(centroids) #(k,D)

		for i in range(k):
			cluster_points = pts[labels == i] # grabbing the labeled points (z, D)

			if len(cluster_points) > 0:
				new_centroids[i] = cluster_points.mean(axis=0)
			else:
				new_centroids[i] = centroids[i]
		
		# convergence check
		if np.allclose(new_centroids, centroids):
			break

		centroids = new_centroids

	# Convert each coordinate to a native python float after rounding to 4 decimals
	return [tuple   (round(coord, 4) for coord in centroid) for centroid in centroids.tolist()]