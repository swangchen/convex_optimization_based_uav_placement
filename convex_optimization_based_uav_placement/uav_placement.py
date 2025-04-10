import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from generate_data import generate_data
from optimal_points import optimal_points, calculate_capacity

def optimize_pow_height_cluster(cluster_points, centroid, power_threshold,
                              height_threshold, alpha, chan_capacity_thresh,
                              bw_uav, var_n):
    """Optimize power and height for a cluster"""
    # For this example, we'll use fixed values
    # In a real implementation, this would use convex optimization
    optimal_power = power_threshold
    optimal_height = height_threshold
    optimal_capacity = calculate_capacity(
        np.mean(np.sqrt(np.sum((cluster_points - centroid)**2, axis=1))),
        optimal_power, bw_uav, optimal_height, var_n
    )
    return (optimal_power, optimal_height, optimal_capacity, 
            centroid[0], centroid[1])

def main():
    # Parameters
    num_of_clusters = 40
    start_range_mean = -40
    end_range_mean = 40
    start_range_var = 0
    end_range_var = 10
    data_points_per_cluster = 100
    no_of_users = num_of_clusters * data_points_per_cluster

    # Generate user distribution data
    data = generate_data(num_of_clusters, start_range_mean, end_range_mean,
                        start_range_var, end_range_var, data_points_per_cluster)
    X = data[:, 0]
    Y = data[:, 1]

    # Plot Gaussian clusters
    plt.figure(figsize=(12, 8))
    for i in range(num_of_clusters):
        plt.plot(data[i*100:(i+1)*100, 0], data[i*100:(i+1)*100, 1], '.')
    plt.title('Gaussian Distributions')
    plt.xlabel('X Distance')
    plt.ylabel('Y Distance')
    plt.show()

    # Perform K-means clustering
    kmeans = KMeans(n_clusters=num_of_clusters, random_state=42)
    idx = kmeans.fit_predict(data)
    centroids = kmeans.cluster_centers_

    # Create clusters
    k_means_clusters = [data[idx == i] for i in range(num_of_clusters)]

    # Plot K-means results
    plt.figure(figsize=(12, 8))
    plt.scatter(X, Y, c=idx)
    plt.plot(centroids[:, 0], centroids[:, 1], 'kx', markersize=15, 
             linewidth=3, label='Centroids')
    plt.title('K-means Clustering Results')
    plt.xlabel('X Distance')
    plt.ylabel('Y Distance')
    plt.legend()
    plt.show()

    # Optimization parameters
    power_threshold = 10
    height_threshold = 0.5
    bw_uav = 5
    alpha = 0.5
    chan_capacity_thresh = 1
    var_n = 0.5

    # Optimize for each cluster
    optimal_data = np.zeros((num_of_clusters, 5))
    for i in range(num_of_clusters):
        optimal_data[i] = optimize_pow_height_cluster(
            k_means_clusters[i], centroids[i], power_threshold, height_threshold,
            alpha, chan_capacity_thresh, bw_uav, var_n
        )

    # Base station parameters
    x_bs = np.mean(centroids[:, 0])
    y_bs = np.mean(centroids[:, 1])
    P_bs = 50
    P_uav = power_threshold
    bw_bs = 10
    h_relay = 1
    h_bs = 0.1

    # Calculate optimal UAV positions
    uav_1 = []
    uav_2 = []
    for i in range(num_of_clusters):
        points = optimal_points(
            x_bs, y_bs, centroids[i, 0], centroids[i, 1],
            P_bs, P_uav, bw_bs, bw_uav, optimal_data[i, 1],
            h_bs, h_relay, chan_capacity_thresh, var_n
        )
        uav_1.append(points[0])
        uav_2.append(points[1])

    uav_1 = np.array(uav_1)
    uav_2 = np.array(uav_2)

    # Plot final results
    plt.figure(figsize=(12, 8))
    plt.scatter(X, Y, c='lightblue', alpha=0.5, label='Users')
    plt.plot(centroids[:, 0], centroids[:, 1], 'kx', markersize=10,
             linewidth=3, label='Centroids')
    plt.scatter(uav_1[:, 0], uav_1[:, 1], c='red', marker='^',
                s=100, label='UAV 1')
    plt.scatter(uav_2[:, 0], uav_2[:, 1], c='green', marker='^',
                s=100, label='UAV 2')
    plt.plot(x_bs, y_bs, 'ks', markersize=10, label='Base Station')
    
    # Plot communication ranges
    r_bs = np.sqrt(
        (2**(chan_capacity_thresh/bw_bs) - 1) * var_n / P_bs - h_bs**2
    )
    theta = np.linspace(0, 2*np.pi, 100)
    plt.plot(x_bs + r_bs*np.cos(theta), y_bs + r_bs*np.sin(theta),
             'k--', alpha=0.5)

    plt.title('Optimal UAV Placement')
    plt.xlabel('X Distance')
    plt.ylabel('Y Distance')
    plt.legend()
    plt.axis('equal')
    plt.show()

if __name__ == "__main__":
    main() 