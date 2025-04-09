

"""Planar-Only | Real Test Data Prediction"""
import torch
import torch.nn as nn
import scipy.io
import numpy as np
import os
from tqdm import tqdm
from scipy.ndimage import gaussian_filter
from train_baseline import AFNetResNet
from utils.signal_processing import min_max_normalize
from config.constants import (
    REAL_DATA_FOLDER, ELECTRODE_X, ELECTRODE_Y, BASE_DIR,
    NUM_SAMPLES, GRID_RESOLUTION, X_MIN, X_MAX, Y_MIN, Y_MAX, POINTS
)
from config.train_config import DEVICE


model_path = os.path.join(BASE_DIR, "models", "AFNetResNet_planar_only.pth")
test_data_path = REAL_DATA_FOLDER

device = DEVICE
model = AFNetResNet().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# === ✅ 文件读取 ===
all_test_files = [f for f in os.listdir(test_data_path) if f.endswith('.mat')]
print(f"📂 Found {len(all_test_files)} test samples")


predicted_maps = []
predicted_times = []
test_electrograms = []


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
result_folder = os.path.join(BASE_DIR, "results", "results_for_visualization")
 
np.save(os.path.join(result_folder, "test_electrograms_real.npy"), np.array(test_electrograms))
np.save(os.path.join(result_folder, "predicted_activation_times_real.npy"), np.array(predicted_times))
np.save(os.path.join(result_folder, "predicted_activation_maps_real.npy"), np.array(predicted_maps))

print("\n✅ 所有测试结果已保存为 .npy 文件！")


















