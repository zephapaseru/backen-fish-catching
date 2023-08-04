import numpy as np
import matplotlib.pyplot as plt

def initialize_membership_matrix(num_data_points, num_clusters):
    membership_matrix = np.random.rand(num_data_points, num_clusters)
    membership_matrix /= np.sum(membership_matrix, axis=1, keepdims=True)
    return membership_matrix

def update_cluster_centers(data_points, membership_matrix, fuzziness):
    num_clusters = membership_matrix.shape[1]
    membership_exp = membership_matrix ** fuzziness
    cluster_centers = np.dot(data_points.T, membership_exp) / np.sum(membership_exp, axis=0, keepdims=True)
    return cluster_centers.T

def update_membership_matrix(data_points, cluster_centers, fuzziness):
    num_data_points = data_points.shape[0]
    num_clusters = cluster_centers.shape[0]
    distance_matrix = np.zeros((num_data_points, num_clusters))

    for j in range(num_clusters):
        diff = data_points - cluster_centers[j]
        distance_matrix[:, j] = np.linalg.norm(diff, axis=1)

    membership_matrix = 1.0 / distance_matrix**(2 / (fuzziness - 1))
    membership_matrix /= np.sum(membership_matrix, axis=1, keepdims=True)
    return membership_matrix

def fuzzy_c_means(data_points, num_clusters, fuzziness, max_iterations=100, tolerance=1e-6):
    membership_matrix = initialize_membership_matrix(data_points.shape[0], num_clusters)
    prev_cluster_centers = np.zeros((num_clusters, data_points.shape[1]))

    for _ in range(max_iterations):
        cluster_centers = update_cluster_centers(data_points, membership_matrix, fuzziness)
        membership_matrix = update_membership_matrix(data_points, cluster_centers, fuzziness)

        center_shift = np.linalg.norm(cluster_centers - prev_cluster_centers)
        if center_shift < tolerance:
            break

        prev_cluster_centers = cluster_centers.copy()

    return cluster_centers, membership_matrix

def fuzzy_partition_coefficient(membership_matrix):
    return np.sum(membership_matrix**2) / membership_matrix.shape[0]

def plot_clusters(data_points, clusters, cluster_centers):
    plt.figure(figsize=(8, 6))
    num_clusters = cluster_centers.shape[0]

    for i in range(num_clusters):
        cluster_data = data_points[clusters == i]
        plt.scatter(cluster_data[:, 0], cluster_data[:, 1], label=f'Cluster {i+1}', alpha=0.7)

    plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], marker='x', s=100, c='black', label='Cluster Centers')
    plt.xlabel('Wave')
    plt.ylabel('Wind')
    plt.title('Fuzzy C-Means Clustering')
    plt.legend()
    plt.grid(True)
    plt.show()
