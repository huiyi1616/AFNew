


# import torch
# import torch.nn as nn
# import scipy.io
# import numpy as np
# import os
# from tqdm import tqdm
# from scipy.ndimage import gaussian_filter
# from baseline_CNN_train_synthesis_data import AFNetResNet

# def min_max_normalize(data, new_min=-1, new_max=1):
#     old_min, old_max = np.min(data), np.max(data)
#     return (data - old_min) / (old_max - old_min + 1e-8) * (new_max - new_min) + new_min

# # === 📌 设置路径 ===
# model_path = "AFNetResNet.pth"
# test_data_path = r"C:\AF\NickMatlab\Library\RetroMapping_GIT\DataProcessing\DataProcessing\AT_data\20250206\Processed_MAT_Files\Peak_Windows_MAT\test_data"

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = AFNetResNet().to(device)
# model.load_state_dict(torch.load(model_path, map_location=device))
# model.eval()

# all_test_files = [f for f in os.listdir(test_data_path) if f.endswith('.mat')]

# # === 📌 初始化结果列表 ===
# predicted_maps = []
# predicted_labels = []
# predicted_times = []
# test_electrograms = []
# true_labels = []

# # === 📌 分类正确统计
# correct = 0
# total = 0

# for test_file in tqdm(all_test_files, desc="Processing Test Files"):
#     print(f"\nProcessing: {test_file}")
#     mat_data = scipy.io.loadmat(os.path.join(test_data_path, test_file))
#     real_waveform = mat_data['peak_window_signals']

#     num_samples = 250
#     if real_waveform.shape[1] == 19 and real_waveform.shape[0] == 161:
#         real_waveform = real_waveform.T
#     else: 
#         real_waveform = real_waveform
#     real_waveform = min_max_normalize(real_waveform, new_min=-1, new_max=1)

#     baseline_wandering = np.random.randn(19, num_samples) * 0.2
#     baseline_wandering = gaussian_filter(baseline_wandering, sigma=[0, 15])

#     start_idx = np.random.randint(0, num_samples - 161)
#     smooth_window = np.hanning(161)
#     smoothed_signal = real_waveform * smooth_window

#     baseline_wandering[:, start_idx:start_idx + 161] += smoothed_signal
#     real_waveform = baseline_wandering  # (19, 400)


#     real_waveform = torch.tensor(real_waveform, dtype=torch.float32).unsqueeze(0).to(device)  # (1, 19, 400)

#     with torch.no_grad():
#         pred_class, pred_activation_map, pred_activation_time = model(real_waveform)

#     probs = torch.softmax(pred_class, dim=1).cpu().numpy().squeeze()  # (3,)
#     pred_label = np.argmax(probs)  # scalar

#     predicted_maps.append(pred_activation_map.cpu().numpy().squeeze())
#     predicted_labels.append(probs)
#     predicted_times.append(pred_activation_time.cpu().numpy().squeeze())
#     test_electrograms.append(real_waveform.cpu().numpy().squeeze())
#     true_labels.append([0, 0, 1])  # Planar → class 2

#     # === 📊 分类正确性统计
#     if pred_label == 2:
#         correct += 1
#     total += 1

# # === 📌 保存结果 ===
# np.save("test_electrograms.npy", np.array(test_electrograms))
# np.save("predicted_labels.npy", np.array(predicted_labels))
# np.save("true_labels.npy", np.array(true_labels))
# np.save("predicted_activation_times.npy", np.array(predicted_times))
# np.save("predicted_activation_maps.npy", np.array(predicted_maps))

# # === ✅ 打印准确率 ===
# accuracy = correct / total if total > 0 else 0.0
# print(f"\n✅ Classification Accuracy on Test Data: {accuracy:.4f} ({correct}/{total})")
# print("✅ 所有测试结果已保存！")














# """Planar-Only"""

# import torch
# import torch.nn as nn
# import scipy.io
# import numpy as np
# import os
# from tqdm import tqdm
# from scipy.ndimage import gaussian_filter
# from baseline_CNN_train_synthesis_data_planar import AFNetResNet

# def min_max_normalize(data, new_min=-1, new_max=1):
#     old_min, old_max = np.min(data), np.max(data)
#     return (data - old_min) / (old_max - old_min + 1e-8) * (new_max - new_min) + new_min

# # === 📌 设置路径 ===
# #model_path = "AFNetResNet_planar_only.pth"
# model_path = "best_AFNetResNet_planar_only.pth"
# test_data_path = r"C:\AF\NickMatlab\Library\RetroMapping_GIT\DataProcessing\DataProcessing\AT_data\20250206\Processed_MAT_Files\Peak_Windows_MAT\test_data"

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = AFNetResNet().to(device)
# model.load_state_dict(torch.load(model_path, map_location=device))
# model.eval()

# all_test_files = [f for f in os.listdir(test_data_path) if f.endswith('.mat')]

# # === 📌 初始化结果列表 ===
# predicted_maps = []
# predicted_times = []
# test_electrograms = []

# for test_file in tqdm(all_test_files, desc="Processing Test Files"):
#     print(f"\nProcessing: {test_file}")
#     mat_data = scipy.io.loadmat(os.path.join(test_data_path, test_file))
#     real_waveform = mat_data['peak_window_signals']

#     num_samples = 250
#     if real_waveform.shape[1] == 19 and real_waveform.shape[0] == 161:
#         real_waveform = real_waveform.T
#     real_waveform = min_max_normalize(real_waveform, new_min=-1, new_max=1)

#     baseline_wandering = np.random.randn(19, num_samples) * 0.2
#     baseline_wandering = gaussian_filter(baseline_wandering, sigma=[0, 15])

#     start_idx = np.random.randint(0, num_samples - 161)
#     smooth_window = np.hanning(161)
#     smoothed_signal = real_waveform * smooth_window

#     baseline_wandering[:, start_idx:start_idx + 161] += smoothed_signal
#     processed_waveform = baseline_wandering  # (19, 250)

#     input_tensor = torch.tensor(processed_waveform, dtype=torch.float32).unsqueeze(0).to(device)  # (1, 19, 250)

#     with torch.no_grad():
#         pred_activation_map, pred_activation_time = model(input_tensor)

#     predicted_maps.append(pred_activation_map.cpu().numpy().squeeze())
#     predicted_times.append(pred_activation_time.cpu().numpy().squeeze())
#     test_electrograms.append(processed_waveform)

# # === 📌 保存结果 ===
# np.save("test_electrograms_real.npy", np.array(test_electrograms))
# np.save("predicted_activation_times_real.npy", np.array(predicted_times))
# np.save("predicted_activation_maps_real.npy", np.array(predicted_maps))

# print("\n✅ 所有测试结果已保存！")




"""Planar-Only | Real Test Data Prediction"""
import torch
import torch.nn as nn
import scipy.io
import numpy as np
import os
from tqdm import tqdm
from scipy.ndimage import gaussian_filter
from baseline_CNN_train_synthesis_data_planar import AFNetResNet

# === ✅ Normalization Utility ===
def min_max_normalize(data, new_min=-1, new_max=1):
    old_min, old_max = np.min(data), np.max(data)
    return (data - old_min) / (old_max - old_min + 1e-8) * (new_max - new_min) + new_min

# === 📌 路径设置 ===
model_path = "best_AFNetResNet_planar_only.pth"
test_data_path = r"C:\AF\NickMatlab\Library\RetroMapping_GIT\DataProcessing\DataProcessing\AT_data\20250206\Processed_MAT_Files\Peak_Windows_MAT\test_data"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AFNetResNet().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# === ✅ 文件读取 ===
all_test_files = [f for f in os.listdir(test_data_path) if f.endswith('.mat')]
print(f"📂 Found {len(all_test_files)} test samples")

# === 📌 初始化结果列表 ===
predicted_maps = []
predicted_times = []
test_electrograms = []

# === 📌 主处理流程 ===
for test_file in tqdm(all_test_files, desc="🔍 Predicting"):
    try:
        mat_data = scipy.io.loadmat(os.path.join(test_data_path, test_file))
        real_waveform = mat_data['peak_window_signals']  # shape: (250, 19) or (19, 250) or (161, 19)

        # === ✅ 保证 waveform 为 (19, T)
        if real_waveform.shape[0] == 19:
            real_waveform = real_waveform
        elif real_waveform.shape[1] == 19:
            real_waveform = real_waveform.T
        else:
            raise ValueError(f"Unexpected shape {real_waveform.shape} in {test_file}")

        num_channels, signal_len = real_waveform.shape
        real_waveform = min_max_normalize(real_waveform, new_min=-1, new_max=1)

        # === ✅ 增加 baseline wandering 模拟
        baseline = np.random.randn(19, 250) * 0.2
        baseline = gaussian_filter(baseline, sigma=[0, 15])

        # === ✅ 将 real_waveform 嵌入 baseline（使用 Hanning 窗加权）
        if signal_len >= 161:
            smooth_window = np.hanning(161)
            smoothed_signal = real_waveform[:, :161] * smooth_window
            start_idx = np.random.randint(0, 250 - 161)
            baseline[:, start_idx:start_idx + 161] += smoothed_signal
        else:
            # 若太短，直接填充
            baseline[:, :signal_len] += real_waveform

        processed_waveform = baseline[:, :250]  # 保证大小 (19, 250)

        # === ✅ 模型预测 ===
        input_tensor = torch.tensor(processed_waveform, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            pred_activation_map, pred_activation_time = model(input_tensor)

        # === ✅ 存储结果（注意 shape 保持一致性）
        predicted_maps.append(pred_activation_map.cpu().numpy().squeeze())
        predicted_times.append(pred_activation_time.cpu().numpy().squeeze())
        test_electrograms.append(processed_waveform)

    except Exception as e:
        print(f"❌ Error processing {test_file}: {e}")
        continue

# === ✅ 保存结果 ===
np.save("test_electrograms_real.npy", np.array(test_electrograms))
np.save("predicted_activation_times_real.npy", np.array(predicted_times))
np.save("predicted_activation_maps_real.npy", np.array(predicted_maps))

print("\n✅ 所有测试结果已保存为 .npy 文件！")


















