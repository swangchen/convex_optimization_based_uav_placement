import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import cvxpy as cp
from matplotlib.animation import FuncAnimation

# 数据生成函数（保持不变）
def generate_data(num_of_clusters, start_range_mean, end_range_mean, start_range_var, end_range_var, data_points_per_cluster):
    means_x = np.random.uniform(start_range_mean, end_range_mean, num_of_clusters)
    means_y = np.random.uniform(start_range_mean, end_range_mean, num_of_clusters)
    variances = np.random.uniform(start_range_var, end_range_var, num_of_clusters)
    data = np.zeros((num_of_clusters * data_points_per_cluster, 2))
    for i in range(num_of_clusters):
        start_idx = i * data_points_per_cluster
        end_idx = (i + 1) * data_points_per_cluster
        data[start_idx:end_idx, 0] = np.random.normal(means_x[i], np.sqrt(variances[i]), data_points_per_cluster)
        data[start_idx:end_idx, 1] = np.random.normal(means_y[i], np.sqrt(variances[i]), data_points_per_cluster)
    return data

# 信道容量计算（保持不变）
def calculate_capacity(distance, power, bandwidth, height, var_n):
    return bandwidth * np.log2(1 + power / ((distance ** 2 + height ** 2) * var_n))

# 高度优化
def optimize_height(cluster_points, centroid, power_threshold, bw_uav, var_n, chan_capacity_thresh):
    distances = np.sqrt(np.sum((cluster_points - centroid) ** 2, axis=1))
    mean_distance = np.mean(distances)
    h = cp.Variable()
    objective = cp.Minimize(h)
    constraints = [
        bw_uav * cp.log2(1 + power_threshold / ((mean_distance ** 2 + h ** 2) * var_n)) >= chan_capacity_thresh,
        h >= 0.1, h <= 10
    ]
    problem = cp.Problem(objective, constraints)
    problem.solve()
    return h.value

# 中继无人机轨迹优化
def optimize_relay_trajectory(x_bs, y_bs, centroids, max_iter=50):
    n_clusters = len(centroids)
    positions = np.zeros((max_iter, 2))
    positions[0] = [x_bs, y_bs]
    velocity = np.zeros(2)
    for t in range(1, max_iter):
        target = centroids[t % n_clusters]  # 循环访问每个集群
        r1, r2 = np.random.rand(2)
        velocity = 0.7 * velocity + 2 * r1 * (target - positions[t-1])
        positions[t] = positions[t-1] + velocity
    return positions

# 计算覆盖率
def calculate_coverage_rate(data, centroids, threshold=10):
    distances = np.min(np.linalg.norm(data[:, None, :] - centroids, axis=2), axis=1)
    return np.mean(distances < threshold)

# 主函数
def main():
    # 参数设置
    num_of_clusters = 40
    data = generate_data(num_of_clusters, -40, 40, 0, 10, 100)
    X, Y = data[:, 0], data[:, 1]
    power_threshold, bw_uav, var_n, chan_capacity_thresh = 10, 5, 0.5, 1
    P_bs, bw_bs, h_bs = 50, 10, 0.1

    # K-means聚类
    kmeans = KMeans(n_clusters=num_of_clusters, random_state=42)
    kmeans_labels = kmeans.fit_predict(data)
    kmeans_centroids = kmeans.cluster_centers_
    k_means_clusters = [data[kmeans_labels == i] for i in range(num_of_clusters)]

    # 高度优化
    optimal_heights = [optimize_height(k_means_clusters[i], kmeans_centroids[i], power_threshold, bw_uav, var_n, chan_capacity_thresh) for i in range(num_of_clusters)]
    optimal_heights = np.array(optimal_heights)

    # 中继无人机位置和轨迹
    x_bs, y_bs = np.mean(kmeans_centroids[:, 0]), np.mean(kmeans_centroids[:, 1])
    relay_trajectory = optimize_relay_trajectory(x_bs, y_bs, kmeans_centroids)

    # 图1：用户分布
    plt.figure(figsize=(10, 8))
    plt.scatter(X, Y, s=20, alpha=0.6, c='blue', edgecolors='grey')
    plt.title('用户分布 / Ground User Distribution', fontsize=16)
    plt.xlabel('X 距离 (m) / X Distance (m)', fontsize=14)
    plt.ylabel('Y 距离 (m) / Y Distance (m)', fontsize=14)
    plt.grid(True)
    plt.savefig('fig1_user_distribution.png', dpi=300)
    plt.show()

    # 图2：K-means聚类
    plt.figure(figsize=(10, 8))
    plt.scatter(X, Y, c=kmeans_labels, cmap='viridis', s=20, alpha=0.6)
    plt.scatter(kmeans_centroids[:, 0], kmeans_centroids[:, 1], c='red', marker='x', s=150)
    plt.title('K-means聚类结果 / K-means Clustering Results', fontsize=16)
    plt.xlabel('X 距离 (m) / X Distance (m)', fontsize=14)
    plt.ylabel('Y 距离 (m) / Y Distance (m)', fontsize=14)
    plt.grid(True)
    plt.savefig('fig2_kmeans_clustering.png', dpi=300)
    plt.show()

    # 图3：高度优化
    plt.figure(figsize=(10, 8))
    plt.plot(range(num_of_clusters), optimal_heights, 'bo-', label='高度 / Height')
    plt.title('高度优化结果 / Height Optimization Results', fontsize=16)
    plt.xlabel('集群编号 / Cluster ID', fontsize=14)
    plt.ylabel('高度 (m) / Height (m)', fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.savefig('fig3_height_optimization.png', dpi=300)
    plt.show()

    # 图4：中继无人机位置
    plt.figure(figsize=(10, 8))
    plt.scatter(X, Y, c='lightblue', s=20, alpha=0.5)
    plt.scatter(kmeans_centroids[:, 0], kmeans_centroids[:, 1], c='black', marker='x', s=150, label='集群无人机 / Cluster UAVs')
    plt.scatter(x_bs, y_bs, c='blue', marker='s', s=150, label='基站 / Base Station')
    plt.title('中继无人机位置 / Relay UAV Positions', fontsize=16)
    plt.xlabel('X 距离 (m) / X Distance (m)', fontsize=14)
    plt.ylabel('Y 距离 (m) / Y Distance (m)', fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.savefig('fig4_relay_positions.png', dpi=300)
    plt.show()

    # 图5：中继无人机轨迹
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(X, Y, c='lightblue', s=20, alpha=0.5)
    ax.scatter(x_bs, y_bs, c='blue', marker='s', s=150, label='基站 / Base Station')
    line, = ax.plot([], [], 'r-', label='轨迹 / Trajectory')
    ax.legend()
    ax.set_title('中继无人机轨迹 / Relay UAV Trajectory', fontsize=16)
    ax.set_xlabel('X 距离 (m) / X Distance (m)', fontsize=14)
    ax.set_ylabel('Y 距离 (m) / Y Distance (m)', fontsize=14)
    ax.grid(True)
    def update(frame):
        line.set_data(relay_trajectory[:frame, 0], relay_trajectory[:frame, 1])
        return line,
    ani = FuncAnimation(fig, update, frames=range(len(relay_trajectory)), interval=100)
    ani.save('fig5_relay_trajectory.gif', dpi=300)
    plt.show()

    # 图6：优化前覆盖范围
    plt.figure(figsize=(10, 8))
    plt.scatter(X, Y, c='lightblue', s=20, alpha=0.5)
    r_bs = np.sqrt((2 ** (chan_capacity_thresh / bw_bs) - 1) * var_n / P_bs - h_bs ** 2)
    theta = np.linspace(0, 2 * np.pi, 100)
    plt.plot(x_bs + r_bs * np.cos(theta), y_bs + r_bs * np.sin(theta), 'k--', label='基站覆盖 / BS Coverage')
    plt.title('优化前覆盖范围 / Pre-Optimization Coverage', fontsize=16)
    plt.xlabel('X 距离 (m) / X Distance (m)', fontsize=14)
    plt.ylabel('Y 距离 (m) / Y Distance (m)', fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.savefig('fig6_pre_coverage.png', dpi=300)
    plt.show()

    # 图7：优化后覆盖范围
    plt.figure(figsize=(10, 8))
    plt.scatter(X, Y, c='lightblue', s=20, alpha=0.5)
    plt.scatter(kmeans_centroids[:, 0], kmeans_centroids[:, 1], c='black', marker='x', s=150, label='集群无人机 / Cluster UAVs')
    plt.scatter(x_bs, y_bs, c='blue', marker='s', s=150, label='基站 / Base Station')
    for i in range(num_of_clusters):
        r_uav = np.sqrt((2 ** (chan_capacity_thresh / bw_uav) - 1) * var_n / power_threshold - optimal_heights[i] ** 2)
        plt.plot(kmeans_centroids[i, 0] + r_uav * np.cos(theta), kmeans_centroids[i, 1] + r_uav * np.sin(theta), 'g--', alpha=0.3)
    plt.plot(x_bs + r_bs * np.cos(theta), y_bs + r_bs * np.sin(theta), 'k--')
    plt.title('优化后覆盖范围 / Post-Optimization Coverage', fontsize=16)
    plt.xlabel('X 距离 (m) / X Distance (m)', fontsize=14)
    plt.ylabel('Y 距离 (m) / Y Distance (m)', fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.savefig('fig7_post_coverage.png', dpi=300)
    plt.show()

    # 图8：功率分配
    powers = [power_threshold] * num_of_clusters
    plt.figure(figsize=(10, 6))
    plt.bar(range(num_of_clusters), powers, color='orange', label='功率 / Power')
    plt.title('功率分配对比 / Power Allocation Comparison', fontsize=16)
    plt.xlabel('无人机编号 / UAV ID', fontsize=14)
    plt.ylabel('功率 (W) / Power (W)', fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.savefig('fig8_power_allocation.png', dpi=300)
    plt.show()

    # 图9：带宽利用
    capacities = [calculate_capacity(np.mean(np.sqrt(np.sum((k_means_clusters[i] - kmeans_centroids[i]) ** 2, axis=1))), power_threshold, bw_uav, optimal_heights[i], var_n) for i in range(num_of_clusters)]
    plt.figure(figsize=(10, 6))
    plt.plot(range(num_of_clusters), capacities, 'go-', label='带宽利用 / Bandwidth Utilization')
    plt.title('带宽利用率 / Bandwidth Utilization', fontsize=16)
    plt.xlabel('无人机编号 / UAV ID', fontsize=14)
    plt.ylabel('容量 (bps/Hz) / Capacity (bps/Hz)', fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.savefig('fig9_bandwidth_utilization.png', dpi=300)
    plt.show()

    # 图10：覆盖率对比
    uav_numbers = [10, 20, 30, 40, 50]
    coverage_rates = []
    for n in uav_numbers:
        kmeans = KMeans(n_clusters=n, random_state=42)
        kmeans_labels = kmeans.fit_predict(data)
        coverage_rates.append(calculate_coverage_rate(data, kmeans.cluster_centers_))
    plt.figure(figsize=(10, 6))
    plt.plot(uav_numbers, coverage_rates, 'bo-', label='覆盖率 / Coverage Rate')
    plt.title('覆盖率对比 / Coverage Rate Comparison', fontsize=16)
    plt.xlabel('无人机数量 / Number of UAVs', fontsize=14)
    plt.ylabel('覆盖率 / Coverage Rate', fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.savefig('fig10_coverage_rate.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    main()