

# # -*- coding: utf-8 -*-
# """
# Visualization of AFNetCNN Predictions on Real Test Data
# Includes predicted activation maps, predicted labels, and activation times.
# """

# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd

# # === 📌 读取电极坐标 ===
# points = pd.read_csv('triElectrodeFlat_Points.csv', header=None).values
# electrode_x, electrode_y = points[:, 0], points[:, 1]
# x_min, x_max, y_min, y_max = -10, 10, -10, 10

# # === 📌 加载预测数据 ===
# predicted_maps = np.load("predicted_activation_maps.npy")        # (N, 100, 100)
# predicted_labels = np.load("predicted_labels.npy")               # (N, 3)
# predicted_times = np.load("predicted_activation_times.npy")      # (N, 19)
# test_electrograms = np.load("test_electrograms.npy")             # (N, 19, 400)

# print("✅ 预测数据加载成功！")

# # === 📌 类别映射 ===
# class_names = ["Focal", "Rotor", "Planar"]

# # 兼容概率/标签格式
# if predicted_labels.ndim == 1:
#     predicted_classes = predicted_labels.astype(int)
#     pred_probs_all = None
# else:
#     predicted_classes = np.argmax(predicted_labels, axis=1)
#     pred_probs_all = predicted_labels

# # === 📌 可视化多个样本 ===
# num_visualizations = 50
# for i in range(min(num_visualizations, len(predicted_maps))):
#     pred_class = predicted_classes[i]
#     class_pred_name = class_names[pred_class]

#     print(f"\n🔍 Sample {i} | Predicted: {class_pred_name}", end='')

#     if pred_probs_all is not None:
#         pred_probs = pred_probs_all[i]
#         print(f" | Probs → Focal: {pred_probs[0]:.2f}, Rotor: {pred_probs[1]:.2f}, Planar: {pred_probs[2]:.2f}")
#     else:
#         print()

#     # === 🎧 Electrogram 可视化 ===
#     plt.figure(figsize=(10, 5))
#     for j in range(19):
#         signal = test_electrograms[i, j]
#         plt.plot(2 * signal + j * 2, label=f'Ch{j+1}')
#     plt.xlabel("Time Steps")
#     plt.ylabel("Amplitude")
#     plt.title(f"Electrograms (Sample {i})")
#     plt.tight_layout()
#     plt.grid(True)
#     plt.legend()
#     plt.show()

#     # === 📡 Activation Map 可视化 ===
#     plt.figure(figsize=(6, 6))
#     plt.imshow(predicted_maps[i], cmap='jet', extent=[x_min, x_max, y_min, y_max], origin='lower')
#     plt.colorbar(label="Activation Time")
#     plt.scatter(electrode_x, electrode_y, c='black', s=50, edgecolors='k', label="Electrodes")
#     for idx, (x, y) in enumerate(zip(electrode_x, electrode_y), start=1):
#         plt.text(x + 0.5, y, str(idx), color='black', fontsize=9, ha='left', va='center')
#     plt.title(f"Predicted Activation Map - Sample {i} ({class_pred_name})")
#     plt.xlabel("X (mm)")
#     plt.ylabel("Y (mm)")
#     plt.legend(loc='upper right')
#     plt.grid(False)
#     plt.tight_layout()
#     plt.show()

#     # === 🕒 仅显示 Predicted Activation Time 柱状图 ===
#     times_pred = predicted_times[i]
#     plt.figure(figsize=(10, 4))
#     index = np.arange(1, 20)
#     plt.bar(index, times_pred, width=0.6, label='Predicted')
#     plt.xlabel("Electrode Channel")
#     plt.ylabel("Activation Time")
#     plt.title(f"Predicted Activation Times (Sample {i})")
#     plt.xticks(index)
#     plt.grid(True, linestyle='--', alpha=0.5)
#     plt.tight_layout()
#     plt.show()

# print("✅ Real Data Visualization Complete!")








# """Planar-Only"""


# # visualize_real_planar_only.py

# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd

# # === 📌 读取电极坐标 ===
# points = pd.read_csv('triElectrodeFlat_Points.csv', header=None).values
# electrode_x, electrode_y = points[:, 0], points[:, 1]
# x_min, x_max, y_min, y_max = -10, 10, -10, 10

# # === 📌 加载预测数据 ===
# predicted_maps = np.load("predicted_activation_maps_real.npy")        # (N, 100, 100)
# predicted_times = np.load("predicted_activation_times_real.npy")      # (N, 19)
# test_electrograms = np.load("test_electrograms_real.npy")             # (N, 19, 250)

# print("✅ 预测数据加载成功！")

# # === 📌 可视化多个样本 ===
# num_visualizations = 100
# for i in range(min(num_visualizations, len(predicted_maps))):
#     print(f"\n🔍 Sample {i}")

#     # === 🎧 Electrogram 可视化 ===
#     plt.figure(figsize=(10, 5))
#     for j in range(19):
#         signal = test_electrograms[i, j]
#         plt.plot(2 * signal + j * 2, label=f'Ch{j+1}')
#     plt.xlabel("Time Steps")
#     plt.ylabel("Amplitude")
#     plt.title(f"Electrograms (Sample {i})")
#     plt.tight_layout()
#     plt.grid(True)
#     plt.legend()
#     plt.show()

#     # === 📡 Activation Map 可视化 ===
#     plt.figure(figsize=(6, 6))
#     plt.imshow(predicted_maps[i], cmap='jet', extent=[x_min, x_max, y_min, y_max], origin='lower')
#     plt.colorbar(label="Activation Time")
#     plt.scatter(electrode_x, electrode_y, c='black', s=50, edgecolors='k', label="Electrodes")
#     for idx, (x, y) in enumerate(zip(electrode_x, electrode_y), start=1):
#         plt.text(x + 0.5, y, str(idx), color='black', fontsize=9, ha='left', va='center')
#     plt.title(f"Predicted Activation Map - Sample {i}")
#     plt.xlabel("X (mm)")
#     plt.ylabel("Y (mm)")
#     plt.legend(loc='upper right')
#     plt.grid(False)
#     plt.tight_layout()
#     plt.show()

#     # === 🕒 激活时间柱状图 ===
#     times_pred = predicted_times[i]
#     plt.figure(figsize=(10, 4))
#     index = np.arange(1, 20)
#     plt.bar(index, times_pred, width=0.6, label='Predicted')
#     plt.xlabel("Electrode Channel")
#     plt.ylabel("Activation Time")
#     plt.title(f"Predicted Activation Times (Sample {i})")
#     plt.xticks(index)
#     plt.grid(True, linestyle='--', alpha=0.5)
#     plt.tight_layout()
#     plt.show()

# print("✅ Real Data Visualization Complete!")









import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# === 📌 读取电极坐标 ===
points = pd.read_csv('triElectrodeFlat_Points.csv', header=None).values
electrode_x, electrode_y = points[:, 0], points[:, 1]
coords = np.stack([electrode_x, electrode_y], axis=1)

x_min, x_max, y_min, y_max = -10, 10, -10, 10

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
    plt.imshow(predicted_maps[i], cmap='jet', extent=[x_min, x_max, y_min, y_max], origin='lower')
    plt.colorbar(label="Activation Time")
    plt.scatter(electrode_x, electrode_y, c='black', s=50, edgecolors='k', label="Electrodes")
    for idx, (x, y) in enumerate(zip(electrode_x, electrode_y), start=1):
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
    pred_dirs = compute_direction_vectors_numpy(times_pred, coords)

    # 归一化方向向量以便可视化（单位向量）
    pred_dirs_norm = pred_dirs / (np.linalg.norm(pred_dirs, axis=1, keepdims=True) + 1e-6)

    plt.figure(figsize=(6, 6))
    plt.scatter(electrode_x, electrode_y, c='black', s=50)
    for idx, (x, y) in enumerate(zip(electrode_x, electrode_y), start=1):
        plt.text(x + 0.5, y, str(idx), color='black', fontsize=9)

    plt.quiver(electrode_x, electrode_y,
               pred_dirs_norm[:, 0], pred_dirs_norm[:, 1],
               color='red', angles='xy', scale_units='xy', scale=0.5, label='Predicted')

    plt.title(f"Direction Vectors (Predicted) - Sample {i}")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.gca().set_aspect('equal')
    plt.legend()
    plt.tight_layout()
    plt.show()

print("✅ Real Data Visualization Complete!")


















