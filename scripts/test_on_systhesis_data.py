

"""Planar-Only Directional Evaluation"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import os
import pandas as pd
import torch.nn.functional as F

# 👇 导入 Dataset 和 Model（注意路径是否正确）
from train_baseline import ElectrogramDataset, AFNetResNet, fast_compute_direction_vectors, directional_vector_loss
from config.constants import (
    SYNTHESIS_SAVE_PATH, SYNTHESIS_DATA_FOLDER, ELECTRODE_X, ELECTRODE_Y,
    NUM_SAMPLES, GRID_RESOLUTION, X_MIN, X_MAX, Y_MIN, Y_MAX, POINTS, BASE_DIR
)

points_tensor = torch.tensor(POINTS, dtype=torch.float32)



# === 📌 模型测试函数 ===
def test_model(model_path, data_dir, batch_size=32, device=None):
    all_electrograms = []
    all_activation_times_true = []
    all_activation_times_pred = []
    all_activation_maps_true = []
    all_activation_maps_pred = []

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"📦 Loading model on {device}...")

    model = AFNetResNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("✅ Model loaded.")

    dataset = ElectrogramDataset(data_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    criterion_map = nn.MSELoss()
    criterion_time = nn.MSELoss()

    total_loss = 0
    activation_time_mae_total = 0
    activation_map_mae_total = 0
    directional_loss_total = 0
    n_batches = 0

    with torch.no_grad():
        for electrograms, activation_maps, activation_times_target in tqdm(dataloader, desc="Testing"):
            electrograms = electrograms.to(device)
            activation_maps = activation_maps.to(device)
            activation_times_target = activation_times_target.to(device)

            activation_map_pred, activation_time_pred = model(electrograms)

            loss_map = criterion_map(activation_map_pred, activation_maps)
            loss_time = criterion_time(activation_time_pred, activation_times_target)

            pred_dirs = fast_compute_direction_vectors(activation_time_pred, points_tensor.to(device))
            true_dirs = fast_compute_direction_vectors(activation_times_target, points_tensor.to(device))
            loss_dir_vec = directional_vector_loss(pred_dirs, true_dirs)

            loss = 0.3 * loss_map + 0.2 * loss_time + 0.5 * loss_dir_vec
            total_loss += loss.item()

            activation_time_mae_total += torch.mean(torch.abs(activation_time_pred - activation_times_target)).item()
            activation_map_mae_total += torch.mean(torch.abs(activation_map_pred - activation_maps)).item()
            directional_loss_total += loss_dir_vec.item()
            n_batches += 1

            all_electrograms.append(electrograms.cpu().numpy())
            all_activation_times_true.append(activation_times_target.cpu().numpy())
            all_activation_times_pred.append(activation_time_pred.cpu().numpy())
            all_activation_maps_true.append(activation_maps.cpu().numpy())
            all_activation_maps_pred.append(activation_map_pred.cpu().numpy())

    activation_time_mae_avg = activation_time_mae_total / n_batches
    activation_map_mae_avg = activation_map_mae_total / n_batches
    directional_loss_avg = directional_loss_total / n_batches
    activation_time_score = 1 - activation_time_mae_avg
    activation_map_score = 1 - activation_map_mae_avg

    print("\n📊 Test Results on Validation Set:")
    print(f"   Activation Time MAE     : {activation_time_mae_avg:.4f}")
    print(f"   Activation Time Score   : {activation_time_score:.4f}")
    print(f"   Activation Map MAE      : {activation_map_mae_avg:.4f}")
    print(f"   Activation Map Score    : {activation_map_score:.4f}")
    print(f"   Direction Vector Loss   : {directional_loss_avg:.4f}")

    return {
        "electrograms": np.concatenate(all_electrograms, axis=0),
        "activation_times_true": np.concatenate(all_activation_times_true, axis=0),
        "activation_times_pred": np.concatenate(all_activation_times_pred, axis=0),
        "activation_maps_true": np.concatenate(all_activation_maps_true, axis=0),
        "activation_maps_pred": np.concatenate(all_activation_maps_pred, axis=0)
    }


if __name__ == "__main__":
    model_path = os.path.join(BASE_DIR, "models", "AFNetResNet_planar_only.pth")
    validation_data_dir = os.path.join(SYNTHESIS_SAVE_PATH, "validation")
    results = test_model(model_path, validation_data_dir)

    result_folder = os.path.join(BASE_DIR, "results", "results_for_visualization")
    
    np.save(os.path.join(result_folder, "test_electrograms_synthesis.npy"), results["electrograms"])
    np.save(os.path.join(result_folder, "predicted_activation_times_synthesis.npy"), results["activation_times_pred"])
    np.save(os.path.join(result_folder, "true_activation_times_synthesis.npy"), results["activation_times_true"])
    np.save(os.path.join(result_folder, "predicted_activation_maps_synthesis.npy"), results["activation_maps_pred"])
    np.save(os.path.join(result_folder, "true_activation_maps_synthesis.npy"), results["activation_maps_true"])

    print("✅ 所有预测结果已保存为 .npy 文件！")











































