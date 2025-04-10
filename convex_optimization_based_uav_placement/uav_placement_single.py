import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.optimize import minimize
import pandas as pd
plt.rcParams["font.sans-serif"]=["SimHei"] #设置字体
plt.rcParams["axes.unicode_minus"]=False #该语句解决图像中的“-”负号的乱码问题

def generate_data(num_of_clusters, start_range_mean, end_range_mean, start_range_var, end_range_var,
                  data_points_per_cluster, bs_range=50, min_samples_per_cluster=10):
    # 增加初始数据点，确保过滤后仍有足够样本
    total_points = max(num_of_clusters * data_points_per_cluster, num_of_clusters * min_samples_per_cluster * 2)
    means_x = np.random.uniform(start_range_mean, end_range_mean, num_of_clusters)
    means_y = np.random.uniform(start_range_mean, end_range_mean, num_of_clusters)
    variances = np.random.uniform(start_range_var, end_range_var, num_of_clusters)
    data = np.zeros((total_points, 2))
    densities = np.random.uniform(0.5, 1.5, num_of_clusters)
    points_per_cluster = total_points // num_of_clusters
    for i in range(num_of_clusters):
        start_idx = i * points_per_cluster
        end_idx = (i + 1) * points_per_cluster if i < num_of_clusters - 1 else total_points
        data[start_idx:end_idx, 0] = np.random.normal(means_x[i], np.sqrt(variances[i]), end_idx - start_idx)
        data[start_idx:end_idx, 1] = np.random.normal(means_y[i], np.sqrt(variances[i]), end_idx - start_idx)
    # 排除基站覆盖范围内的用户
    distances = np.sqrt(np.sum(data**2, axis=1))
    outside_bs = distances > bs_range
    filtered_data = data[outside_bs]
    # 如果过滤后样本不足，则扩展范围重新生成
    while len(filtered_data) < num_of_clusters * min_samples_per_cluster:
        start_range_mean *= 1.5
        end_range_mean *= 1.5
        means_x = np.random.uniform(start_range_mean, end_range_mean, num_of_clusters)
        means_y = np.random.uniform(start_range_mean, end_range_mean, num_of_clusters)
        data = np.zeros((total_points, 2))
        for i in range(num_of_clusters):
            start_idx = i * points_per_cluster
            end_idx = (i + 1) * points_per_cluster if i < num_of_clusters - 1 else total_points
            data[start_idx:end_idx, 0] = np.random.normal(means_x[i], np.sqrt(variances[i]), end_idx - start_idx)
            data[start_idx:end_idx, 1] = np.random.normal(means_y[i], np.sqrt(variances[i]), end_idx - start_idx)
        distances = np.sqrt(np.sum(data**2, axis=1))
        outside_bs = distances > bs_range
        filtered_data = data[outside_bs]
    return filtered_data, densities

def calculate_capacity(distance, power, bandwidth, height, var_n):
    total_distance = np.sqrt(distance ** 2 + height ** 2)
    return bandwidth * np.log2(1 + power / (total_distance ** 2 * var_n))

def optimize_vertical_height(cluster_data, centroid, density, power, bandwidth, var_n, video_bw_thresh=2.0):
    def objective(h):
        distances = np.sqrt(np.sum((cluster_data - centroid) ** 2, axis=1))
        total_loss = np.sum(power / (distances ** 2 + h ** 2) * var_n) / density
        return total_loss

    def constraint(h):
        distances = np.sqrt(np.sum((cluster_data - centroid) ** 2, axis=1))
        capacity = calculate_capacity(np.mean(distances), power, bandwidth, h, var_n)
        return capacity - video_bw_thresh

    result = minimize(objective, x0=1.0, bounds=[(0.1, 5.0)], constraints={'type': 'ineq', 'fun': constraint},
                      options={'maxiter': 1000})
    return result.x[0] if result.success else 1.0

def simulated_annealing_trajectory(cluster_centroids, x_bs, y_bs, P_uav, bw_uav, h_relay, var_n):
    def objective(positions):
        distances = np.sqrt(np.sum(np.diff(positions, axis=0) ** 2, axis=1))
        return np.sum(distances) if len(distances) > 0 else 0

    positions = np.copy(cluster_centroids)
    temp, alpha, iterations = 1000, 0.95, 100
    for _ in range(iterations):
        new_positions = positions + np.random.normal(0, 1, positions.shape)
        delta = objective(new_positions) - objective(positions)
        if delta < 0 or np.random.rand() < np.exp(-delta / temp):
            positions = new_positions
        temp *= alpha
    flight_distance = objective(positions)
    return positions, flight_distance

def calculate_video_coverage_rate(data, cluster_centroids, relay_uavs, h_cluster, power, bandwidth, var_n,
                                  video_bw_thresh=2.0):
    cluster_distances_full = np.sqrt(np.sum((data[:, None, :] - cluster_centroids) ** 2, axis=2))
    cluster_assignments = np.argmin(cluster_distances_full, axis=1)
    cluster_distances = cluster_distances_full[np.arange(len(data)), cluster_assignments]
    cluster_capacity = calculate_capacity(cluster_distances, power, bandwidth, h_cluster, var_n)

    relay_distances_full = np.sqrt(np.sum((cluster_centroids[:, None, :] - relay_uavs) ** 2, axis=2))
    relay_distances = np.min(relay_distances_full, axis=1)
    relay_capacity = calculate_capacity(relay_distances, power, bandwidth, h_cluster, var_n)
    relay_capacity_mapped = relay_capacity[cluster_assignments]

    covered = (cluster_capacity >= video_bw_thresh) & (relay_capacity_mapped >= video_bw_thresh)
    return np.mean(covered)

def calculate_power_loss(data, centroids, power, bandwidth, height, var_n):
    distances = np.min(np.sqrt(np.sum((data[:, None, :] - centroids) ** 2, axis=2)), axis=1)
    total_distances = np.sqrt(distances ** 2 + height ** 2)
    capacity = calculate_capacity(total_distances, power, bandwidth, height, var_n)
    return np.sum(power / (capacity + 1e-6))

def calculate_resource_efficiency(data, centroids, bandwidth, height, var_n):
    distances = np.min(np.sqrt(np.sum((data[:, None, :] - centroids) ** 2, axis=2)), axis=1)
    total_distances = np.sqrt(distances ** 2 + height ** 2)
    capacity = calculate_capacity(total_distances, 1, bandwidth, height, var_n)
    return np.mean(capacity) / bandwidth

def main():
    start_range_mean, end_range_mean = -40, 40
    start_range_var, end_range_var = 0, 10
    data_points_per_cluster = 100
    P_uav, bw_uav, var_n, h_relay = 15, 5, 0.5, 1.0
    video_bw_thresh = 2.0
    x_bs, y_bs = 0, 0

    # Figure 1: Video Coverage Rate vs. Number of UAVs
    uav_numbers = [10, 20, 30, 40, 50]
    coverage_rates = {'KMeans': [], 'Centroid-Joint': []}
    for n in uav_numbers:
        data, densities = generate_data(n, start_range_mean, end_range_mean, start_range_var, end_range_var,
                                        data_points_per_cluster)
        kmeans = KMeans(n_clusters=n)
        labels = kmeans.fit_predict(data)
        cluster_centroids = kmeans.cluster_centers_
        coverage_rates['KMeans'].append(
            calculate_video_coverage_rate(data, cluster_centroids, cluster_centroids, h_relay, P_uav, bw_uav, var_n))

        h_clusters = [optimize_vertical_height(data[labels == i], cluster_centroids[i], densities[min(i, len(densities) - 1)],
                                               P_uav, bw_uav, var_n) for i in range(n)]
        relay_positions, _ = simulated_annealing_trajectory(cluster_centroids, x_bs, y_bs, P_uav, bw_uav, h_relay, var_n)
        coverage_rates['Centroid-Joint'].append(
            calculate_video_coverage_rate(data, cluster_centroids, relay_positions,
                                          np.mean(h_clusters) if h_clusters else h_relay, P_uav, bw_uav, var_n))

    plt.figure(figsize=(10, 6))
    plt.plot(uav_numbers, coverage_rates['KMeans'], marker='o', label='KMeans', color='blue')
    plt.plot(uav_numbers, coverage_rates['Centroid-Joint'], marker='o', label='Centroid-Joint', color='orange')
    plt.fill_between(uav_numbers, coverage_rates['KMeans'], alpha=0.2, color='blue')
    plt.title('视频流覆盖率 vs. 无人机数量 / Video Coverage Rate vs. Number of UAVs')
    plt.xlabel('无人机数量 / Number of UAVs')
    plt.ylabel('覆盖率 / Coverage Rate')
    plt.legend()
    plt.grid(True)
    plt.savefig('fig1_coverage_rate.png')
    plt.show()
    print("\n表1：不同无人机数量下的视频流覆盖率")
    df1 = pd.DataFrame(coverage_rates, index=uav_numbers)
    print(df1)

    # Figure 2: Power Loss vs. Bandwidth Settings
    num_of_clusters = 40
    data, densities = generate_data(num_of_clusters, start_range_mean, end_range_mean, start_range_var, end_range_var,
                                    data_points_per_cluster)
    bandwidths = [2, 4, 6, 8, 10]
    power_losses = {'KMeans': [], 'Centroid-Joint': []}
    kmeans = KMeans(n_clusters=num_of_clusters)
    labels = kmeans.fit_predict(data)
    cluster_centroids = kmeans.cluster_centers_
    h_clusters = [optimize_vertical_height(data[labels == i], cluster_centroids[i], densities[min(i, len(densities) - 1)],
                                           P_uav, bw_uav, var_n) for i in range(num_of_clusters)]
    for bw in bandwidths:
        power_losses['KMeans'].append(calculate_power_loss(data, cluster_centroids, P_uav, bw, h_relay, var_n))
        relay_positions, _ = simulated_annealing_trajectory(cluster_centroids, x_bs, y_bs, P_uav, bw, h_relay, var_n)
        power_losses['Centroid-Joint'].append(
            calculate_power_loss(data, cluster_centroids, P_uav, bw, np.mean(h_clusters) if h_clusters else h_relay, var_n))

    plt.figure(figsize=(10, 6))
    plt.plot(bandwidths, power_losses['KMeans'], marker='o', label='KMeans', color='blue')
    plt.plot(bandwidths, power_losses['Centroid-Joint'], marker='o', label='Centroid-Joint', color='orange')
    plt.fill_between(bandwidths, power_losses['KMeans'], alpha=0.2, color='blue')
    plt.title('功率损耗 vs. 带宽设置 / Power Loss vs. Bandwidth Settings')
    plt.xlabel('带宽 (MHz) / Bandwidth (MHz)')
    plt.ylabel('功率损耗 (W) / Power Loss (W)')
    plt.legend()
    plt.grid(True)
    plt.savefig('fig2_power_loss.png')
    plt.show()
    print("\n表2：不同带宽设置下的功率损耗")
    df2 = pd.DataFrame(power_losses, index=bandwidths)
    print(df2)

    # Figure 3: Video Coverage Rate vs. Cluster UAV Altitude (Heatmap)
    heights = [0.2, 0.5, 1.0, 1.5, 2.0]
    coverage_rates_height = np.zeros((len(heights), 2))
    for i, h in enumerate(heights):
        coverage_rates_height[i, 0] = calculate_video_coverage_rate(data, cluster_centroids, cluster_centroids, h, P_uav, bw_uav, var_n)
        relay_positions, _ = simulated_annealing_trajectory(cluster_centroids, x_bs, y_bs, P_uav, bw_uav, h, var_n)
        coverage_rates_height[i, 1] = calculate_video_coverage_rate(data, cluster_centroids, relay_positions, h, P_uav, bw_uav, var_n)

    plt.figure(figsize=(10, 6))
    plt.imshow(coverage_rates_height, cmap='viridis', extent=[0, 1, min(heights), max(heights)], aspect='auto')
    plt.colorbar(label='覆盖率 / Coverage Rate')
    plt.xticks([0, 1], ['KMeans', 'Centroid-Joint'])
    plt.title('不同高度下的视频流覆盖率 / Video Coverage Rate at Different Heights')
    plt.xlabel('方法 / Method')
    plt.ylabel('高度 (km) / Height (km)')
    plt.savefig('fig3_coverage_rate_height.png')
    plt.show()
    print("\n表3：不同高度下的视频流覆盖率")
    df3 = pd.DataFrame(coverage_rates_height, index=heights, columns=['KMeans', 'Centroid-Joint'])
    print(df3)

    # Figure 4: Relay UAV Flight Distance vs. Scenarios
    scenarios = ['Low Noise', 'Medium Noise', 'High Noise', 'Dense', 'Sparse']
    scenario_params = [(0.1, 40), (0.5, 40), (1.0, 40), (0.5, 80), (0.5, 20)]
    flight_distances = {'KMeans': [], 'Centroid-Joint': []}
    dense_data, dense_relay_positions, dense_centroids = None, None, None
    for i, (var_n_sc, n_clusters_sc) in enumerate(scenario_params):
        data_sc, densities_sc = generate_data(n_clusters_sc, start_range_mean, end_range_mean, start_range_var,
                                              end_range_var, data_points_per_cluster)
        kmeans_sc = KMeans(n_clusters=n_clusters_sc)
        labels_sc = kmeans_sc.fit_predict(data_sc)
        cluster_centroids_sc = kmeans_sc.cluster_centers_
        flight_distances['KMeans'].append(np.sum(np.sqrt(np.sum((cluster_centroids_sc - [x_bs, y_bs]) ** 2, axis=1))))
        h_clusters_sc = [optimize_vertical_height(data_sc[labels_sc == i], cluster_centroids_sc[i],
                                                  densities_sc[min(i, len(densities_sc) - 1)], P_uav, bw_uav, var_n_sc)
                         for i in range(n_clusters_sc)]
        relay_positions_sc, flight_distance = simulated_annealing_trajectory(cluster_centroids_sc, x_bs, y_bs, P_uav, bw_uav, h_relay, var_n_sc)
        flight_distances['Centroid-Joint'].append(flight_distance)
        if scenarios[i] == 'Dense':
            dense_data, dense_relay_positions, dense_centroids = data_sc, relay_positions_sc, cluster_centroids_sc

    plt.figure(figsize=(10, 6))
    plt.plot(scenarios, flight_distances['KMeans'], marker='o', label='KMeans', color='blue')
    plt.plot(scenarios, flight_distances['Centroid-Joint'], marker='o', label='Centroid-Joint', color='orange')
    plt.fill_between(scenarios, flight_distances['KMeans'], alpha=0.2, color='blue')
    plt.title('中继无人机飞行距离 vs. 场景 / Relay UAV Flight Distance vs. Scenarios')
    plt.xlabel('场景 / Scenarios')
    plt.ylabel('飞行距离 (m) / Flight Distance (m)')
    plt.legend()
    plt.grid(True)
    plt.savefig('fig4_flight_distance.png')
    plt.show()
    print("\n表4：不同场景下的中继无人机飞行距离")
    df4 = pd.DataFrame(flight_distances, index=scenarios)
    print(df4)

    # Figure 5: Resource Allocation Efficiency
    resource_efficiency = {'KMeans': [], 'Centroid-Joint': []}
    for i, (var_n_sc, _) in enumerate(scenario_params):
        resource_efficiency['KMeans'].append(calculate_resource_efficiency(data_sc, cluster_centroids_sc, bw_uav, h_relay, var_n_sc))
        resource_efficiency['Centroid-Joint'].append(calculate_resource_efficiency(data_sc, relay_positions_sc, bw_uav, h_relay, var_n_sc))

    plt.figure(figsize=(10, 6))
    x = np.arange(len(scenarios))
    plt.bar(x - 0.2, resource_efficiency['KMeans'], 0.4, label='KMeans', color='blue')
    plt.bar(x + 0.2, resource_efficiency['Centroid-Joint'], 0.4, label='Centroid-Joint', color='orange')
    plt.xticks(x, scenarios)
    plt.title('资源分配效率 vs. 场景 / Resource Allocation Efficiency vs. Scenarios')
    plt.xlabel('场景 / Scenarios')
    plt.ylabel('资源分配效率 / Resource Allocation Efficiency')
    plt.legend()
    plt.grid(True)
    plt.savefig('fig5_resource_efficiency.png')
    plt.show()
    print("\n表5：不同场景下的资源分配效率")
    df5 = pd.DataFrame(resource_efficiency, index=scenarios)
    print(df5)

    # Figure 6: Enhanced Deployment Visualization
    plt.figure(figsize=(12, 8))
    plt.scatter(dense_data[:, 0], dense_data[:, 1], c='blue', alpha=0.1, label='数据点 / Data Points')
    plt.scatter(dense_centroids[:, 0], dense_centroids[:, 1], c='orange', marker='s', s=100, label='集群中心 / Cluster Centroids')
    plt.scatter(dense_relay_positions[:, 0], dense_relay_positions[:, 1], c='red', marker='x', s=100, label='中继无人机 / Relay UAVs')
    plt.plot(dense_relay_positions[:, 0], dense_relay_positions[:, 1], 'r--', label='无人机轨迹 / UAV Trajectory')
    plt.scatter([x_bs], [y_bs], c='green', marker='o', s=200, label='基站 / Base Station')
    for pos in dense_relay_positions:
        circle = plt.Circle(pos, 10, color='red', alpha=0.1)
        plt.gca().add_patch(circle)
    plt.title('密集场景下的无人机部署与覆盖范围 / UAV Deployment in Dense Scenario with Coverage')
    plt.xlabel('X坐标 (m) / X Coordinate (m)')
    plt.ylabel('Y坐标 (m) / Y Coordinate (m)')
    plt.legend()
    plt.grid(True)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig('fig6_dense_deployment.png')
    plt.show()

if __name__ == "__main__":
    main()