# # -*- coding: utf-8 -*-
# """
# Visualization of AFNetCNN Predictions on Synthesis Data
# Includes comparison with ground truth activation maps, labels, and activation times.
# Supports both softmax probabilities or integer labels.
# """

# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd

# # === 📌 读取电极坐标 ===
# points = pd.read_csv('triElectrodeFlat_Points.csv', header=None).values
# electrode_x, electrode_y = points[:, 0], points[:, 1]
# x_min, x_max, y_min, y_max = -10, 10, -10, 10

# # === 📌 加载数据 ===
# predicted_maps = np.load("predicted_activation_maps.npy")
# true_maps = np.load("true_activation_maps.npy")
# predicted_labels = np.load("predicted_labels.npy")
# true_labels = np.load("true_labels.npy")
# predicted_times = np.load("predicted_activation_times.npy")        # 🔹 加载预测激活时间
# true_times = np.load("true_activation_times.npy")                  # 🔹 加载真实激活时间
# test_electrograms = np.load("test_electrograms.npy")

# print("✅ 合成数据加载成功！")

# # === 📌 类别映射 ===
# class_names = ["Focal", "Rotor", "Planar"]

# # 自动兼容标签类型
# if predicted_labels.ndim == 1:
#     predicted_classes = predicted_labels.astype(int)
#     pred_probs_all = None
# else:
#     predicted_classes = np.argmax(predicted_labels, axis=1)
#     pred_probs_all = predicted_labels

# if true_labels.ndim == 1:
#     true_classes = true_labels.astype(int)
# else:
#     true_classes = np.argmax(true_labels, axis=1)

# # === 📌 可视化多个样本 ===
# num_visualizations = 100
# for i in range(min(num_visualizations, len(predicted_maps))):
#     pred_class = predicted_classes[i]
#     true_class = true_classes[i]
#     class_pred_name = class_names[pred_class]
#     class_true_name = class_names[true_class]

#     print(f"\n🔍 Sample {i} | Predicted: {class_pred_name} | True: {class_true_name}", end='')

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

#     # === 🔥 激活图对比：真实 vs 预测 ===
#     fig, axs = plt.subplots(1, 2, figsize=(12, 6))
#     axs[0].imshow(true_maps[i], cmap='jet', extent=[x_min, x_max, y_min, y_max], origin='lower')
#     axs[0].set_title(f"True Activation Map - Sample {i} ({class_true_name})")
#     axs[1].imshow(predicted_maps[i], cmap='jet', extent=[x_min, x_max, y_min, y_max], origin='lower')
#     axs[1].set_title(f"Predicted Activation Map - Sample {i} ({class_pred_name})")

#     for ax in axs:
#         ax.scatter(electrode_x, electrode_y, c='black', s=50, edgecolors='k', label="Electrodes")
#         for idx, (x, y) in enumerate(zip(electrode_x, electrode_y), start=1):
#             ax.text(x + 0.5, y, str(idx), color='black', fontsize=9, ha='left', va='center')
#         ax.set_xlabel("X (mm)")
#         ax.set_ylabel("Y (mm)")
#         ax.legend(loc='upper right')
#         ax.grid(False)

#     fig.colorbar(axs[0].images[0], ax=axs, orientation='vertical', fraction=0.02, pad=0.04, label='Activation Time')
#     plt.tight_layout()
#     plt.show()

#     # === 🕒 激活时间对比柱状图 ===
#     times_true = true_times[i]     # shape: (19,)
#     times_pred = predicted_times[i]

#     plt.figure(figsize=(10, 4))
#     index = np.arange(1, 20)
#     bar_width = 0.35
#     plt.bar(index - bar_width / 2, times_true, bar_width, label='True')
#     plt.bar(index + bar_width / 2, times_pred, bar_width, label='Predicted')
#     plt.xlabel("Electrode Channel")
#     plt.ylabel("Activation Time")
#     plt.title(f"Activation Times (Sample {i})")
#     plt.xticks(index)
#     plt.legend()
#     plt.grid(True, linestyle='--', alpha=0.5)
#     plt.tight_layout()
#     plt.show()

# print("✅ Synthesis Visualization Complete!")











# """Planar-Only"""

# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd

# # === 📌 读取电极坐标 ===
# points = pd.read_csv('triElectrodeFlat_Points.csv', header=None).values
# electrode_x, electrode_y = points[:, 0], points[:, 1]
# x_min, x_max, y_min, y_max = -10, 10, -10, 10

# # === 📌 加载数据 ===
# predicted_maps = np.load("predicted_activation_maps_synthesis.npy")
# true_maps = np.load("true_activation_maps_synthesis.npy")
# predicted_times = np.load("predicted_activation_times_synthesis.npy")        
# true_times = np.load("true_activation_times_synthesis.npy")                  
# test_electrograms = np.load("test_electrograms_synthesis.npy")

# print("✅ 合成数据加载成功！")

# # === 📌 可视化多个样本 ===
# num_visualizations = 50
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

#     # === 🔥 激活图对比：真实 vs 预测 ===
#     fig, axs = plt.subplots(1, 2, figsize=(12, 6))
#     axs[0].imshow(true_maps[i], cmap='jet', extent=[x_min, x_max, y_min, y_max], origin='lower')
#     axs[0].set_title(f"True Activation Map - Sample {i}")
#     axs[1].imshow(predicted_maps[i], cmap='jet', extent=[x_min, x_max, y_min, y_max], origin='lower')
#     axs[1].set_title(f"Predicted Activation Map - Sample {i}")

#     for ax in axs:
#         ax.scatter(electrode_x, electrode_y, c='black', s=50, edgecolors='k', label="Electrodes")
#         for idx, (x, y) in enumerate(zip(electrode_x, electrode_y), start=1):
#             ax.text(x + 0.5, y, str(idx), color='black', fontsize=9, ha='left', va='center')
#         ax.set_xlabel("X (mm)")
#         ax.set_ylabel("Y (mm)")
#         ax.legend(loc='upper right')
#         ax.grid(False)

#     fig.colorbar(axs[0].images[0], ax=axs, orientation='vertical', fraction=0.02, pad=0.04, label='Activation Time')
#     plt.tight_layout()
#     plt.show()

#     # === 🕒 激活时间对比柱状图 ===
#     times_true = true_times[i]
#     times_pred = predicted_times[i]

#     plt.figure(figsize=(10, 4))
#     index = np.arange(1, 20)
#     bar_width = 0.35
#     plt.bar(index - bar_width / 2, times_true, bar_width, label='True')
#     plt.bar(index + bar_width / 2, times_pred, bar_width, label='Predicted')
#     plt.xlabel("Electrode Channel")
#     plt.ylabel("Activation Time")
#     plt.title(f"Activation Times (Sample {i})")
#     plt.xticks(index)
#     plt.legend()
#     plt.grid(True, linestyle='--', alpha=0.5)
#     plt.tight_layout()
#     plt.show()

# print("✅ Synthesis Visualization Complete!")









import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# === 📌 读取电极坐标 ===
points = pd.read_csv('triElectrodeFlat_Points.csv', header=None).values
electrode_x, electrode_y = points[:, 0], points[:, 1]
coords = np.stack([electrode_x, electrode_y], axis=1)

x_min, x_max, y_min, y_max = -10, 10, -10, 10

# === 📌 加载预测数据 ===
predicted_maps = np.load("predicted_activation_maps_synthesis.npy")
true_maps = np.load("true_activation_maps_synthesis.npy")
predicted_times = np.load("predicted_activation_times_synthesis.npy")        
true_times = np.load("true_activation_times_synthesis.npy")                  
test_electrograms = np.load("test_electrograms_synthesis.npy")

print("✅ 合成数据加载成功！")

# === ✅ NumPy 实现方向向量计算函数 ===
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
num_visualizations = 50
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

    # === 🔥 激活图对比 ===
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    axs[0].imshow(true_maps[i], cmap='jet', extent=[x_min, x_max, y_min, y_max], origin='lower')
    axs[0].set_title(f"True Activation Map - Sample {i}")
    axs[1].imshow(predicted_maps[i], cmap='jet', extent=[x_min, x_max, y_min, y_max], origin='lower')
    axs[1].set_title(f"Predicted Activation Map - Sample {i}")

    for ax in axs:
        ax.scatter(electrode_x, electrode_y, c='black', s=50, edgecolors='k', label="Electrodes")
        for idx, (x, y) in enumerate(zip(electrode_x, electrode_y), start=1):
            ax.text(x + 0.5, y, str(idx), color='black', fontsize=9, ha='left', va='center')
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")
        ax.legend(loc='upper right')
        ax.grid(False)

    fig.colorbar(axs[0].images[0], ax=axs, orientation='vertical', fraction=0.02, pad=0.04, label='Activation Time')
    plt.tight_layout()
    plt.show()

    # === 🕒 激活时间对比柱状图 ===
    times_true = true_times[i]
    times_pred = predicted_times[i]

    plt.figure(figsize=(10, 4))
    index = np.arange(1, 20)
    bar_width = 0.35
    plt.bar(index - bar_width / 2, times_true, bar_width, label='True')
    plt.bar(index + bar_width / 2, times_pred, bar_width, label='Predicted')
    plt.xlabel("Electrode Channel")
    plt.ylabel("Activation Time")
    plt.title(f"Activation Times (Sample {i})")
    plt.xticks(index)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

    # === 🨝 方向向量可视化 ===
    pred_dirs = compute_direction_vectors_numpy(predicted_times[i], coords)
    true_dirs = compute_direction_vectors_numpy(true_times[i], coords)

    plt.figure(figsize=(6, 6))
    plt.scatter(electrode_x, electrode_y, c='black', s=50)
    for idx, (x, y) in enumerate(zip(electrode_x, electrode_y), start=1):
        plt.text(x + 0.5, y, str(idx), color='black', fontsize=9)


    # Normalize to unit vectors
    true_dirs_norm = true_dirs / (np.linalg.norm(true_dirs, axis=1, keepdims=True) + 1e-6)
    pred_dirs_norm = pred_dirs / (np.linalg.norm(pred_dirs, axis=1, keepdims=True) + 1e-6)
    
    # 然后用 normalized vectors 可视化
    plt.quiver(electrode_x, electrode_y, true_dirs_norm[:, 0], true_dirs_norm[:, 1],
                color='green', angles='xy', scale_units='xy', scale=0.5, label='True')
    plt.quiver(electrode_x, electrode_y, pred_dirs_norm[:, 0], pred_dirs_norm[:, 1],
                color='red', angles='xy', scale_units='xy', scale=0.5, label='Predicted')
    


    plt.title(f"Direction Vectors (Sample {i})")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

print("✅ Directional Visualization Complete!")















