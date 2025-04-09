
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from config.constants import (
    ELECTRODE_X, ELECTRODE_Y, COORDS,
    NUM_SAMPLES, GRID_RESOLUTION, X_MIN, X_MAX, Y_MIN, Y_MAX, POINTS
)

# === 📌 加载预测数据 ===
predicted_maps = np.load("predicted_activation_maps_real.npy")        # (N, 100, 100)
predicted_times = np.load("predicted_activation_times_real.npy")      # (N, 19)
test_electrograms = np.load("test_electrograms_real.npy")             # (N, 19, 250)

print("✅ 预测数据加载成功！")

def compute_direction_vectors_numpy(activation_times, coords, neighbor_radius=5.0):
    N = activation_times.shape[0]
    direction_vectors = np.zeros((N, 2))

    for i in range(N):
        vectors = []
        for j in range(N):
            if i == j:
                continue
            dx = coords[j] - coords[i]  # 注意这里是 j - i
            dist = np.linalg.norm(dx)
            if dist < neighbor_radius and dist > 0:
                dt = activation_times[j] - activation_times[i]  # 从早 → 晚
                vec = (dt / (dist + 1e-8)) * dx
                vectors.append(vec)
        if vectors:
            direction_vectors[i] = np.mean(vectors, axis=0)
    return direction_vectors


# === 📌 可视化多个样本 ===
num_visualizations = 100
for i in range(min(num_visualizations, len(predicted_maps))):
    print(f"\n🔍 Sample {i}")

    # === 🎧 Electrogram 可视化 ===
    plt.figure(figsize=(10, 5))
    for j in range(19):
        signal = test_electrograms[i, j]
        plt.plot(2 * signal + j * 2, label=f'Ch{j+1}')
    plt.xlabel("Time Steps")
    plt.ylabel("Amplitude")
    plt.title(f"Electrograms (Sample {i})")
    plt.tight_layout()
    plt.grid(True)
    plt.legend()
    plt.show()

    # === 📡 Activation Map 可视化 ===
    plt.figure(figsize=(6, 6))
    plt.imshow(predicted_maps[i], cmap='jet', extent=[X_MIN, X_MAX, Y_MIN, Y_MAX], origin='lower')
    plt.colorbar(label="Activation Time")
    plt.scatter(ELECTRODE_X, ELECTRODE_Y, c='black', s=50, edgecolors='k', label="Electrodes")
    for idx, (x, y) in enumerate(zip(ELECTRODE_X, ELECTRODE_Y), start=1):
        plt.text(x + 0.5, y, str(idx), color='black', fontsize=9, ha='left', va='center')
    plt.title(f"Predicted Activation Map - Sample {i}")
    plt.xlabel("X (mm)")
    plt.ylabel("Y (mm)")
    plt.legend(loc='upper right')
    plt.grid(False)
    plt.tight_layout()
    plt.show()

    # === 🕒 激活时间柱状图 ===
    times_pred = predicted_times[i]
    plt.figure(figsize=(10, 4))
    index = np.arange(1, 20)
    plt.bar(index, times_pred, width=0.6, label='Predicted')
    plt.xlabel("Electrode Channel")
    plt.ylabel("Activation Time")
    plt.title(f"Predicted Activation Times (Sample {i})")
    plt.xticks(index)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

    # === 🨝 方向向量可视化 ===
    pred_dirs = compute_direction_vectors_numpy(times_pred, COORDS)

    # 归一化方向向量以便可视化（单位向量）
    pred_dirs_norm = pred_dirs / (np.linalg.norm(pred_dirs, axis=1, keepdims=True) + 1e-6)

    plt.figure(figsize=(6, 6))
    plt.scatter(ELECTRODE_X, ELECTRODE_Y, c='black', s=50)
    for idx, (x, y) in enumerate(zip(ELECTRODE_X, ELECTRODE_Y), start=1):
        plt.text(x + 0.5, y, str(idx), color='black', fontsize=9)

    plt.quiver(ELECTRODE_X, ELECTRODE_Y,
               pred_dirs_norm[:, 0], pred_dirs_norm[:, 1],
               color='red', angles='xy', scale_units='xy', scale=0.5, label='Predicted')

    plt.title(f"Direction Vectors (Predicted) - Sample {i}")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.xlim(X_MIN, X_MAX)
    plt.ylim(Y_MIN, Y_MAX)
    plt.gca().set_aspect('equal')
    plt.legend()
    plt.tight_layout()
    plt.show()

print("✅ Real Data Visualization Complete!")


















