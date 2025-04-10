import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.optimize import minimize
import pandas as pd
import os
plt.rcParams["font.sans-serif"]=["SimHei"] #设置字体
plt.rcParams["axes.unicode_minus"]=False #该语句解决图像中的“-”负号的乱码问题
# 创建 output 文件夹（如果不存在）
output_dir = 'output'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def generate_data(num_of_clusters, start_range_mean, end_range_mean, start_range_var, end_range_var,
                  data_points_per_cluster, bs_range=50, min_samples_per_cluster=10):
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
    distances = np.sqrt(np.sum(data**2, axis=1))
    outside_bs = distances > bs_range
    filtered_data = data[outside_bs]
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

def pso_trajectory(cluster_centroids, x_bs, y_bs, P_uav, bw_uav, h_relay, var_n):
    def objective(positions):
        distances = np.sqrt(np.sum(np.diff(positions, axis=0) ** 2, axis=1))
        return np.sum(distances) if len(distances) > 0 else 0

    n_particles = 20
    positions = np.tile(cluster_centroids, (n_particles, 1, 1)) + np.random.normal(0, 5, (n_particles, *cluster_centroids.shape))
    velocities = np.zeros((n_particles, *cluster_centroids.shape))
    pbest = positions.copy()
    pbest_values = np.array([objective(p) for p in pbest])
    gbest = pbest[np.argmin(pbest_values)]
    c1, c2, w = 2, 2, 0.7
    for _ in range(50):
        r1, r2 = np.random.rand(n_particles, *cluster_centroids.shape), np.random.rand(n_particles, *cluster_centroids.shape)
        velocities = (w * velocities + c1 * r1 * (pbest - positions) + c2 * r2 * (gbest - positions))
        positions += velocities
        values = np.array([objective(p) for p in positions])
        improved = values < pbest_values
        pbest[improved] = positions[improved]
        pbest_values[improved] = values[improved]
        gbest = pbest[np.argmin(pbest_values)]
    flight_distance = objective(gbest)
    return gbest, flight_distance

def calculate_video_coverage_rate(data, cluster_centroids, relay_uavs, h_cluster, power, bandwidth, var_n,
                                  video_bw_thresh=2.0):
    cluster_distances_full = np.sqrt(np.sum((data[:, None, :] - cluster_centroids) ** 2, axis=2))
    cluster_assignments = np.argmin(cluster_distances_full, axis=1)
    cluster_distances = cluster_distances_full[np.arange(len(data)), cluster_assignments]
    cluster_capacity = calculate_capacity(cluster_distances, power, bandwidth, h_cluster, var_n)

    relay_distances_full = np.sqrt(np.sum((data[:, None, :] - relay_uavs) ** 2, axis=2))
    relay_distances = np.min(relay_distances_full, axis=1)
    relay_capacity = calculate_capacity(relay_distances, power, bandwidth, h_cluster, var_n)

    covered = (cluster_capacity >= video_bw_thresh) & (relay_capacity >= video_bw_thresh)
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
    coverage_rates = {'KMeans': [], 'Centroid-Joint': [], 'PSO': []}
    for n in uav_numbers:
        data, densities = generate_data(n, start_range_mean, end_range_mean, start_range_var, end_range_var,
                                        data_points_per_cluster)
        kmeans = KMeans(n_clusters=n)
        labels = kmeans.fit_predict(data)
        cluster_centroids = kmeans.cluster_centers_
        h_clusters = [optimize_vertical_height(data[labels == i], cluster_centroids[i], densities[min(i, len(densities) - 1)],
                                               P_uav, bw_uav, var_n) for i in range(n)]
        coverage_rates['KMeans'].append(
            calculate_video_coverage_rate(data, cluster_centroids, cluster_centroids, np.mean(h_clusters), P_uav, bw_uav, var_n))
        relay_positions_sa, _ = simulated_annealing_trajectory(cluster_centroids, x_bs, y_bs, P_uav, bw_uav, h_relay, var_n)
        coverage_rates['Centroid-Joint'].append(
            calculate_video_coverage_rate(data, cluster_centroids, relay_positions_sa, h_relay, P_uav, bw_uav, var_n))
        relay_positions_pso, _ = pso_trajectory(cluster_centroids, x_bs, y_bs, P_uav, bw_uav, h_relay, var_n)
        coverage_rates['PSO'].append(
            calculate_video_coverage_rate(data, cluster_centroids, relay_positions_pso, h_relay, P_uav, bw_uav, var_n))

    # 中文版本
    plt.figure(figsize=(10, 6))
    plt.plot(uav_numbers, coverage_rates['KMeans'], marker='o', label='KMeans + Convex', color='blue')
    plt.plot(uav_numbers, coverage_rates['PSO'], marker='o', label='PSO', color='green')
    plt.plot(uav_numbers, coverage_rates['Centroid-Joint'], marker='o', label='Centroid-Joint', color='orange')
    plt.fill_between(uav_numbers, coverage_rates['KMeans'], alpha=0.2, color='blue')
    plt.xlabel('无人机数量')
    plt.ylabel('覆盖率')
    plt.legend(loc='upper right')
    plt.tick_params(direction='in')  # 刻度向内
    plt.savefig(os.path.join(output_dir, 'fig1_coverage_rate_cn.png'))
    plt.close()

    # 英文版本
    plt.figure(figsize=(10, 6))
    plt.plot(uav_numbers, coverage_rates['KMeans'], marker='o', label='KMeans + Convex', color='blue')
    plt.plot(uav_numbers, coverage_rates['PSO'], marker='o', label='PSO', color='green')
    plt.plot(uav_numbers, coverage_rates['Centroid-Joint'], marker='o', label='Centroid-Joint', color='orange')
    plt.fill_between(uav_numbers, coverage_rates['KMeans'], alpha=0.2, color='blue')
    plt.xlabel('Number of UAVs')
    plt.ylabel('Coverage Rate')
    plt.legend(loc='upper right')
    plt.tick_params(direction='in')  # 刻度向内
    plt.savefig(os.path.join(output_dir, 'fig1_coverage_rate_en.png'))
    plt.close()

    print("\n表1：不同无人机数量下的视频流覆盖率")
    df1 = pd.DataFrame(coverage_rates, index=uav_numbers)
    print(df1)

    # Figure 2: Power Loss vs. Bandwidth Settings
    num_of_clusters = 40
    data, densities = generate_data(num_of_clusters, start_range_mean, end_range_mean, start_range_var, end_range_var,
                                    data_points_per_cluster)
    bandwidths = [2, 4, 6, 8, 10]
    power_losses = {'KMeans': [], 'Centroid-Joint': [], 'PSO': []}
    kmeans = KMeans(n_clusters=num_of_clusters)
    labels = kmeans.fit_predict(data)
    cluster_centroids = kmeans.cluster_centers_
    h_clusters = [optimize_vertical_height(data[labels == i], cluster_centroids[i], densities[min(i, len(densities) - 1)],
                                           P_uav, bw_uav, var_n) for i in range(num_of_clusters)]
    relay_positions_sa, _ = simulated_annealing_trajectory(cluster_centroids, x_bs, y_bs, P_uav, bw_uav, h_relay, var_n)
    relay_positions_pso, _ = pso_trajectory(cluster_centroids, x_bs, y_bs, P_uav, bw_uav, h_relay, var_n)
    for bw in bandwidths:
        power_losses['KMeans'].append(calculate_power_loss(data, cluster_centroids, P_uav, bw, np.mean(h_clusters), var_n))
        power_losses['Centroid-Joint'].append(calculate_power_loss(data, relay_positions_sa, P_uav, bw, h_relay, var_n))
        power_losses['PSO'].append(calculate_power_loss(data, relay_positions_pso, P_uav, bw, h_relay, var_n))

    # 中文版本
    plt.figure(figsize=(10, 6))
    plt.plot(bandwidths, power_losses['KMeans'], marker='o', label='KMeans + Convex', color='blue')
    plt.plot(bandwidths, power_losses['PSO'], marker='o', label='PSO', color='green')
    plt.plot(bandwidths, power_losses['Centroid-Joint'], marker='o', label='Centroid-Joint', color='orange')
    plt.fill_between(bandwidths, power_losses['KMeans'], alpha=0.2, color='blue')
    plt.xlabel('带宽/MHz')
    plt.ylabel('功率损耗/W')
    plt.legend(loc='upper right')
    plt.tick_params(direction='in')  # 刻度向内
    plt.savefig(os.path.join(output_dir, 'fig2_power_loss_cn.png'))
    plt.close()

    # 英文版本
    plt.figure(figsize=(10, 6))
    plt.plot(bandwidths, power_losses['KMeans'], marker='o', label='KMeans + Convex', color='blue')
    plt.plot(bandwidths, power_losses['PSO'], marker='o', label='PSO', color='green')
    plt.plot(bandwidths, power_losses['Centroid-Joint'], marker='o', label='Centroid-Joint', color='orange')
    plt.fill_between(bandwidths, power_losses['KMeans'], alpha=0.2, color='blue')
    plt.xlabel('Bandwidth/MHz')
    plt.ylabel('Power Loss/W')
    plt.legend(loc='upper right')
    plt.tick_params(direction='in')  # 刻度向内
    plt.savefig(os.path.join(output_dir, 'fig2_power_loss_en.png'))
    plt.close()

    print("\n表2：不同带宽设置下的功率损耗")
    df2 = pd.DataFrame(power_losses, index=bandwidths)
    print(df2)

    # Figure 3: Video Coverage Rate vs. Cluster UAV Altitude (Three Subplots)
    heights = [0.2, 0.5, 1.0, 1.5, 2.0]
    coverage_no_opt = [calculate_video_coverage_rate(data, cluster_centroids, cluster_centroids, h, P_uav, bw_uav, var_n)
                       for h in heights]
    coverage_kmeans = [calculate_video_coverage_rate(data, cluster_centroids, cluster_centroids, np.mean(h_clusters), P_uav, bw_uav, var_n)
                       for _ in heights]
    coverage_pso = [calculate_video_coverage_rate(data, cluster_centroids, relay_positions_pso, h, P_uav, bw_uav, var_n)
                    for h in heights]
    coverage_centroid = [calculate_video_coverage_rate(data, cluster_centroids, relay_positions_sa, h, P_uav, bw_uav, var_n)
                         for h in heights]

    # 中文版本
    plt.figure(figsize=(18, 5))
    plt.subplot(1, 3, 1)
    plt.plot(heights, coverage_no_opt, marker='o', label='无优化', color='gray')
    plt.xlabel('高度/km')
    plt.ylabel('覆盖率')
    plt.legend(loc='upper right')
    plt.tick_params(direction='in')
    plt.subplot(1, 3, 2)
    plt.plot(heights, coverage_kmeans, marker='o', label='KMeans + Convex', color='blue')
    plt.plot(heights, coverage_pso, marker='o', label='PSO', color='green')
    plt.plot(heights, coverage_centroid, marker='o', label='Centroid-Joint', color='orange')
    plt.xlabel('高度/km')
    plt.ylabel('覆盖率')
    plt.legend(loc='upper right')
    plt.tick_params(direction='in')
    plt.subplot(1, 3, 3)
    plt.plot(heights, coverage_no_opt, marker='o', label='无优化', color='gray')
    plt.plot(heights, coverage_kmeans, marker='o', label='KMeans + Convex', color='blue')
    plt.plot(heights, coverage_pso, marker='o', label='PSO', color='green')
    plt.plot(heights, coverage_centroid, marker='o', label='Centroid-Joint', color='orange')
    plt.xlabel('高度/km')
    plt.ylabel('覆盖率')
    plt.legend(loc='upper right')
    plt.tick_params(direction='in')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig3_coverage_rate_height_cn.png'))
    plt.close()

    # 英文版本
    plt.figure(figsize=(18, 5))
    plt.subplot(1, 3, 1)
    plt.plot(heights, coverage_no_opt, marker='o', label='No Optimization', color='gray')
    plt.xlabel('Height/km')
    plt.ylabel('Coverage Rate')
    plt.legend(loc='upper right')
    plt.tick_params(direction='in')
    plt.subplot(1, 3, 2)
    plt.plot(heights, coverage_kmeans, marker='o', label='KMeans + Convex', color='blue')
    plt.plot(heights, coverage_pso, marker='o', label='PSO', color='green')
    plt.plot(heights, coverage_centroid, marker='o', label='Centroid-Joint', color='orange')
    plt.xlabel('Height/km')
    plt.ylabel('Coverage Rate')
    plt.legend(loc='upper right')
    plt.tick_params(direction='in')
    plt.subplot(1, 3, 3)
    plt.plot(heights, coverage_no_opt, marker='o', label='No Optimization', color='gray')
    plt.plot(heights, coverage_kmeans, marker='o', label='KMeans + Convex', color='blue')
    plt.plot(heights, coverage_pso, marker='o', label='PSO', color='green')
    plt.plot(heights, coverage_centroid, marker='o', label='Centroid-Joint', color='orange')
    plt.xlabel('Height/km')
    plt.ylabel('Coverage Rate')
    plt.legend(loc='upper right')
    plt.tick_params(direction='in')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig3_coverage_rate_height_en.png'))
    plt.close()

    print("\n表3：不同高度下的视频流覆盖率（优化后）")
    df3 = pd.DataFrame({'KMeans': coverage_kmeans, 'PSO': coverage_pso, 'Centroid-Joint': coverage_centroid}, index=heights)
    print(df3)

    # Figure 4: Relay UAV Flight Distance vs. Scenarios
    scenarios = ['低噪声', '中噪声', '高噪声', '密集', '稀疏']  # 中文场景名
    scenarios_en = ['Low Noise', 'Medium Noise', 'High Noise', 'Dense', 'Sparse']  # 英文场景名
    scenario_params = [(0.1, 40), (0.5, 40), (1.0, 40), (0.5, 80), (0.5, 20)]
    flight_distances = {'KMeans': [], 'Centroid-Joint': [], 'PSO': []}
    dense_data, dense_relay_positions, dense_centroids, dense_h_clusters = None, None, None, None
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
        relay_positions_sa, flight_distance_sa = simulated_annealing_trajectory(cluster_centroids_sc, x_bs, y_bs, P_uav, bw_uav, h_relay, var_n_sc)
        relay_positions_pso, flight_distance_pso = pso_trajectory(cluster_centroids_sc, x_bs, y_bs, P_uav, bw_uav, h_relay, var_n_sc)
        flight_distances['Centroid-Joint'].append(flight_distance_sa)
        flight_distances['PSO'].append(flight_distance_pso)
        if scenarios[i] == '密集':
            dense_data, dense_relay_positions, dense_centroids, dense_h_clusters = data_sc, relay_positions_sa, cluster_centroids_sc, h_clusters_sc

    # 中文版本
    plt.figure(figsize=(10, 6))
    plt.plot(scenarios, flight_distances['KMeans'], marker='o', label='KMeans + Convex', color='blue')
    plt.plot(scenarios, flight_distances['PSO'], marker='o', label='PSO', color='green')
    plt.plot(scenarios, flight_distances['Centroid-Joint'], marker='o', label='Centroid-Joint', color='orange')
    plt.fill_between(scenarios, flight_distances['KMeans'], alpha=0.2, color='blue')
    plt.xlabel('场景')
    plt.ylabel('飞行距离/m')
    plt.legend(loc='upper right')
    plt.tick_params(direction='in')
    plt.savefig(os.path.join(output_dir, 'fig4_flight_distance_cn.png'))
    plt.close()

    # 英文版本
    plt.figure(figsize=(10, 6))
    plt.plot(scenarios_en, flight_distances['KMeans'], marker='o', label='KMeans + Convex', color='blue')
    plt.plot(scenarios_en, flight_distances['PSO'], marker='o', label='PSO', color='green')
    plt.plot(scenarios_en, flight_distances['Centroid-Joint'], marker='o', label='Centroid-Joint', color='orange')
    plt.fill_between(scenarios_en, flight_distances['KMeans'], alpha=0.2, color='blue')
    plt.xlabel('Scenarios')
    plt.ylabel('Flight Distance/m')
    plt.legend(loc='upper right')
    plt.tick_params(direction='in')
    plt.savefig(os.path.join(output_dir, 'fig4_flight_distance_en.png'))
    plt.close()

    print("\n表4：不同场景下的中继无人机飞行距离")
    df4 = pd.DataFrame(flight_distances, index=scenarios_en)
    print(df4)

    # Figure 5: Resource Allocation Efficiency
    resource_efficiency = {'KMeans': [], 'Centroid-Joint': [], 'PSO': []}
    for i, (var_n_sc, _) in enumerate(scenario_params):
        resource_efficiency['KMeans'].append(calculate_resource_efficiency(data_sc, cluster_centroids_sc, bw_uav, np.mean(h_clusters_sc), var_n_sc))
        resource_efficiency['Centroid-Joint'].append(calculate_resource_efficiency(data_sc, relay_positions_sa, bw_uav, h_relay, var_n_sc))
        resource_efficiency['PSO'].append(calculate_resource_efficiency(data_sc, relay_positions_pso, bw_uav, h_relay, var_n_sc))

    # 中文版本
    plt.figure(figsize=(10, 6))
    x = np.arange(len(scenarios))
    plt.bar(x - 0.3, resource_efficiency['KMeans'], 0.3, label='KMeans + Convex', color='blue')
    plt.bar(x, resource_efficiency['PSO'], 0.3, label='PSO', color='green')
    plt.bar(x + 0.3, resource_efficiency['Centroid-Joint'], 0.3, label='Centroid-Joint', color='orange')
    plt.xticks(x, scenarios)
    plt.xlabel('场景')
    plt.ylabel('资源分配效率')
    plt.legend(loc='upper right')
    plt.tick_params(direction='in')
    plt.savefig(os.path.join(output_dir, 'fig5_resource_efficiency_cn.png'))
    plt.close()

    # 英文版本
    plt.figure(figsize=(10, 6))
    x = np.arange(len(scenarios_en))
    plt.bar(x - 0.3, resource_efficiency['KMeans'], 0.3, label='KMeans + Convex', color='blue')
    plt.bar(x, resource_efficiency['PSO'], 0.3, label='PSO', color='green')
    plt.bar(x + 0.3, resource_efficiency['Centroid-Joint'], 0.3, label='Centroid-Joint', color='orange')
    plt.xticks(x, scenarios_en)
    plt.xlabel('Scenarios')
    plt.ylabel('Resource Allocation Efficiency')
    plt.legend(loc='upper right')
    plt.tick_params(direction='in')
    plt.savefig(os.path.join(output_dir, 'fig5_resource_efficiency_en.png'))
    plt.close()

    print("\n表5：不同场景下的资源分配效率")
    df5 = pd.DataFrame(resource_efficiency, index=scenarios_en)
    print(df5)

    # Figure 6: Enhanced Deployment Visualization
    # 中文版本
    plt.figure(figsize=(12, 8))
    plt.scatter(dense_data[:, 0], dense_data[:, 1], c='gray', alpha=0.1, label='数据点')
    plt.scatter(dense_centroids[:, 0], dense_centroids[:, 1], c='blue', marker='^', s=100, label='集群无人机')
    plt.scatter(dense_relay_positions[:, 0], dense_relay_positions[:, 1], c='green', marker='^', s=100, label='中继无人机')
    plt.plot(dense_relay_positions[:, 0], dense_relay_positions[:, 1], 'g--', label='中继无人机轨迹')
    plt.scatter([x_bs], [y_bs], c='red', marker='o', s=200, label='基站')
    base_circle = plt.Circle((x_bs, y_bs), 50, color='red', alpha=0.1, label='基站覆盖')
    plt.gca().add_patch(base_circle)
    for pos in dense_relay_positions:
        circle = plt.Circle(pos, 10, color='green', alpha=0.1)
        plt.gca().add_patch(circle)
    plt.xlabel('X坐标/m')
    plt.ylabel('Y坐标/m')
    plt.legend(loc='upper right')
    plt.tick_params(direction='in')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig(os.path.join(output_dir, 'fig6_dense_deployment_cn.png'))
    plt.close()

    # 英文版本
    plt.figure(figsize=(12, 8))
    plt.scatter(dense_data[:, 0], dense_data[:, 1], c='gray', alpha=0.1, label='Data Points')
    plt.scatter(dense_centroids[:, 0], dense_centroids[:, 1], c='blue', marker='^', s=100, label='Cluster UAVs')
    plt.scatter(dense_relay_positions[:, 0], dense_relay_positions[:, 1], c='green', marker='^', s=100, label='Relay UAVs')
    plt.plot(dense_relay_positions[:, 0], dense_relay_positions[:, 1], 'g--', label='Relay UAV Trajectory')
    plt.scatter([x_bs], [y_bs], c='red', marker='o', s=200, label='Base Station')
    base_circle = plt.Circle((x_bs, y_bs), 50, color='red', alpha=0.1, label='BS Coverage')
    plt.gca().add_patch(base_circle)
    for pos in dense_relay_positions:
        circle = plt.Circle(pos, 10, color='green', alpha=0.1)
        plt.gca().add_patch(circle)
    plt.xlabel('X Coordinate/m')
    plt.ylabel('Y Coordinate/m')
    plt.legend(loc='upper right')
    plt.tick_params(direction='in')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig(os.path.join(output_dir, 'fig6_dense_deployment_en.png'))
    plt.close()

    # Figure 7: Convex Optimization Effect on Power Loss
    heights = [0.2, 0.5, 1.0, 1.5, 2.0]
    power_no_convex = [calculate_power_loss(data, cluster_centroids, P_uav, bw_uav, h, var_n)
                       for h in heights]
    power_with_convex = [calculate_power_loss(data, cluster_centroids, P_uav, bw_uav, np.mean(h_clusters), var_n)
                         for _ in heights]

    # 中文版本
    plt.figure(figsize=(10, 6))
    plt.plot(heights, power_no_convex, marker='o', label='仅KMeans (无凸优化)', color='orange')
    plt.plot(heights, power_with_convex, marker='o', label='KMeans + Convex', color='blue')
    plt.fill_between(heights, power_with_convex, power_no_convex, alpha=0.2, color='blue')
    plt.xlabel('高度/km')
    plt.ylabel('功率损耗/W')
    plt.legend(loc='upper right')
    plt.tick_params(direction='in')
    plt.savefig(os.path.join(output_dir, 'fig7_convex_power_loss_cn.png'))
    plt.close()

    # 英文版本
    plt.figure(figsize=(10, 6))
    plt.plot(heights, power_no_convex, marker='o', label='KMeans Only (No Convex)', color='orange')
    plt.plot(heights, power_with_convex, marker='o', label='KMeans + Convex', color='blue')
    plt.fill_between(heights, power_with_convex, power_no_convex, alpha=0.2, color='blue')
    plt.xlabel('Height/km')
    plt.ylabel('Power Loss/W')
    plt.legend(loc='upper right')
    plt.tick_params(direction='in')
    plt.savefig(os.path.join(output_dir, 'fig7_convex_power_loss_en.png'))
    plt.close()

    print("\n表7：凸优化对功率损耗的影响")
    df7 = pd.DataFrame({'KMeans Only': power_no_convex, 'KMeans + Convex': power_with_convex}, index=heights)
    print(df7)

if __name__ == "__main__":
    main()