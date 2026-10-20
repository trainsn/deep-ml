import numpy as np

def k_means_clustering(points: list[tuple[float, ...]], k: int, initial_centroids: list[tuple[float, ...]], max_iterations: int) -> list[tuple[float, ...]]:
	# Your code here
	points = np.array(points)
	centroids = np.array(initial_centroids, dtype=float)

	for _ in range(max_iterations):
		distances = np.linalg.norm(points[:, np.newaxis] - centroids, axis=2)
		labels = np.argmin(distances, axis=1)

		new_centroids = np.zeros_like(centroids)
		for i in range(k):
			cluster_points = points[labels == i]
			if len(cluster_points) > 0:
				new_centroids[i] = cluster_points.mean(axis=0)
			else:
				new_centroids[i] = centroids[i]

		if np.allclose(new_centroids, centroids):
			centroids = new_centroids
			break

		centroids = new_centroids
	
	final_centroids = [tuple(c) for c in centroids]		

	return final_centroids