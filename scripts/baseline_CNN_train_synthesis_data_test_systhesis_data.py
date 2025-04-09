# # test_model.py

# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader
# from tqdm import tqdm
# import numpy as np
# import os

# # 👇 导入之前写的 Dataset 和 Model
# from baseline_CNN_train_synthesis_data import ElectrogramDataset, AFNetResNet

# def test_model(model_path, data_dir, batch_size=32, device=None):
#     all_electrograms = []
#     all_labels_true = []
#     all_labels_pred = []
#     all_activation_times_true = []
#     all_activation_times_pred = []
#     all_activation_maps_true = []
#     all_activation_maps_pred = []

#     if device is None:
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"📦 Loading model on {device}...")

#     # ✅ Load model
#     model = AFNetResNet().to(device)
#     model.load_state_dict(torch.load(model_path, map_location=device))
#     model.eval()
#     print("✅ Model loaded.")

#     # ✅ Load data
#     dataset = ElectrogramDataset(data_dir)
#     dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

#     criterion_cls = nn.CrossEntropyLoss()
#     criterion_map = nn.MSELoss()
#     criterion_time = nn.MSELoss()

#     total_loss = 0
#     correct = 0
#     total = 0
#     activation_time_mae_total = 0
#     activation_map_mae_total = 0
#     n_batches = 0

#     with torch.no_grad():
#         for electrograms, activation_maps, labels, activation_times_target in tqdm(dataloader, desc="Testing"):
#             electrograms = electrograms.to(device)
#             activation_maps = activation_maps.to(device)
#             labels = labels.to(device)
#             activation_times_target = activation_times_target.to(device)

#             class_out, activation_map_pred, activation_time_pred = model(electrograms)

#             loss_cls = criterion_cls(class_out, labels)
#             loss_map = criterion_map(activation_map_pred, activation_maps)
#             loss_time = criterion_time(activation_time_pred, activation_times_target)
#             loss = 0.1 * loss_cls + 0.4 * loss_map + 0.5 * loss_time

#             total_loss += loss.item()
#             preds = class_out.argmax(dim=1)
#             correct += (preds == labels).sum().item()
#             total += labels.size(0)

#             # 🔹 统计 MAE
#             activation_time_mae_batch = torch.mean(torch.abs(activation_time_pred - activation_times_target)).item()
#             activation_map_mae_batch = torch.mean(torch.abs(activation_map_pred - activation_maps)).item()
#             activation_time_mae_total += activation_time_mae_batch
#             activation_map_mae_total += activation_map_mae_batch
#             n_batches += 1

#             # 🔹 收集预测结果
#             all_electrograms.append(electrograms.cpu().numpy())
#             all_labels_true.append(labels.cpu().numpy())
#             all_labels_pred.append(preds.cpu().numpy())
#             all_activation_times_true.append(activation_times_target.cpu().numpy())
#             all_activation_times_pred.append(activation_time_pred.cpu().numpy())
#             all_activation_maps_true.append(activation_maps.cpu().numpy())
#             all_activation_maps_pred.append(activation_map_pred.cpu().numpy())

#     avg_loss = total_loss / len(dataloader)
#     acc = correct / total
#     activation_time_mae_avg = activation_time_mae_total / n_batches
#     activation_map_mae_avg = activation_map_mae_total / n_batches
#     activation_time_score = 1 - activation_time_mae_avg
#     activation_map_score = 1 - activation_map_mae_avg

#     print("\n📊 Test Results on Validation Set:")
#     print(f"   Classification Accuracy : {acc:.4f}")
#     print(f"   Activation Time MAE     : {activation_time_mae_avg:.4f}")
#     print(f"   Activation Time Score   : {activation_time_score:.4f}")
#     print(f"   Activation Map MAE      : {activation_map_mae_avg:.4f}")
#     print(f"   Activation Map Score    : {activation_map_score:.4f}")

#     # ✅ 返回所有预测结果（拼接为 numpy）
#     return {
#         "electrograms": np.concatenate(all_electrograms, axis=0),
#         "labels_true": np.concatenate(all_labels_true, axis=0),
#         "labels_pred": np.concatenate(all_labels_pred, axis=0),
#         "activation_times_true": np.concatenate(all_activation_times_true, axis=0),
#         "activation_times_pred": np.concatenate(all_activation_times_pred, axis=0),
#         "activation_maps_true": np.concatenate(all_activation_maps_true, axis=0),
#         "activation_maps_pred": np.concatenate(all_activation_maps_pred, axis=0)
#     }


# # ✅ Main 调用 & 保存文件
# if __name__ == "__main__":
#     model_path = "best_AFNetResNet.pth"
#     validation_data_dir = r"C:\AF\NickMatlab\Library\RetroMapping_GIT\DataProcessing\DataProcessing\AT_data\20250206\Processed_MAT_Files\dataloader_data\sythesis_data\validation"

#     results = test_model(model_path, validation_data_dir)

#     np.save("test_electrograms.npy", results["electrograms"])
#     np.save("predicted_labels.npy", results["labels_pred"])
#     np.save("true_labels.npy", results["labels_true"])
#     np.save("predicted_activation_times.npy", results["activation_times_pred"])
#     np.save("true_activation_times.npy", results["activation_times_true"])
#     np.save("predicted_activation_maps.npy", results["activation_maps_pred"])
#     np.save("true_activation_maps.npy", results["activation_maps_true"])

#     print("✅ 所有预测结果已保存为 .npy 文件！")




# """Planar-Only"""

# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader
# from tqdm import tqdm
# import numpy as np
# import os

# # 👇 导入 Dataset 和 Model（注意路径是否正确）
# from baseline_CNN_train_synthesis_data_planar import ElectrogramDataset, AFNetResNet

# def test_model(model_path, data_dir, batch_size=32, device=None):
#     all_electrograms = []
#     all_activation_times_true = []
#     all_activation_times_pred = []
#     all_activation_maps_true = []
#     all_activation_maps_pred = []

#     if device is None:
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"📦 Loading model on {device}...")

#     # ✅ 加载模型
#     model = AFNetResNet().to(device)
#     model.load_state_dict(torch.load(model_path, map_location=device))
#     model.eval()
#     print("✅ Model loaded.")

#     # ✅ 加载数据
#     dataset = ElectrogramDataset(data_dir)
#     dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

#     criterion_map = nn.MSELoss()
#     criterion_time = nn.MSELoss()

#     total_loss = 0
#     activation_time_mae_total = 0
#     activation_map_mae_total = 0
#     n_batches = 0

#     with torch.no_grad():
#         for electrograms, activation_maps, activation_times_target in tqdm(dataloader, desc="Testing"):
#             electrograms = electrograms.to(device)
#             activation_maps = activation_maps.to(device)
#             activation_times_target = activation_times_target.to(device)

#             activation_map_pred, activation_time_pred = model(electrograms)

#             loss_map = criterion_map(activation_map_pred, activation_maps)
#             loss_time = criterion_time(activation_time_pred, activation_times_target)
#             loss = 0.5 * loss_map + 0.3 * loss_time + 0.2 * 0  # 不计算方向一致性 loss

#             total_loss += loss.item()

#             # 🔹 统计 MAE
#             activation_time_mae_total += torch.mean(torch.abs(activation_time_pred - activation_times_target)).item()
#             activation_map_mae_total += torch.mean(torch.abs(activation_map_pred - activation_maps)).item()
#             n_batches += 1

#             # 🔹 收集预测结果
#             all_electrograms.append(electrograms.cpu().numpy())
#             all_activation_times_true.append(activation_times_target.cpu().numpy())
#             all_activation_times_pred.append(activation_time_pred.cpu().numpy())
#             all_activation_maps_true.append(activation_maps.cpu().numpy())
#             all_activation_maps_pred.append(activation_map_pred.cpu().numpy())

#     # ✅ 汇总结果
#     activation_time_mae_avg = activation_time_mae_total / n_batches
#     activation_map_mae_avg = activation_map_mae_total / n_batches
#     activation_time_score = 1 - activation_time_mae_avg
#     activation_map_score = 1 - activation_map_mae_avg

#     print("\n📊 Test Results on Validation Set:")
#     print(f"   Activation Time MAE     : {activation_time_mae_avg:.4f}")
#     print(f"   Activation Time Score   : {activation_time_score:.4f}")
#     print(f"   Activation Map MAE      : {activation_map_mae_avg:.4f}")
#     print(f"   Activation Map Score    : {activation_map_score:.4f}")

#     return {
#         "electrograms": np.concatenate(all_electrograms, axis=0),
#         "activation_times_true": np.concatenate(all_activation_times_true, axis=0),
#         "activation_times_pred": np.concatenate(all_activation_times_pred, axis=0),
#         "activation_maps_true": np.concatenate(all_activation_maps_true, axis=0),
#         "activation_maps_pred": np.concatenate(all_activation_maps_pred, axis=0)
#     }

# # ✅ Main 测试调用 & 保存预测结果
# if __name__ == "__main__":
#     model_path = "best_AFNetResNet_planar_only.pth"
#     validation_data_dir = r"C:\AF\NickMatlab\Library\RetroMapping_GIT\DataProcessing\DataProcessing\AT_data\20250206\Processed_MAT_Files\dataloader_data\sythesis_data\validation"

#     results = test_model(model_path, validation_data_dir)

#     np.save("test_electrograms_synthesis.npy", results["electrograms"])
#     np.save("predicted_activation_times_synthesis.npy", results["activation_times_pred"])
#     np.save("true_activation_times_synthesis.npy", results["activation_times_true"])
#     np.save("predicted_activation_maps_synthesis.npy", results["activation_maps_pred"])
#     np.save("true_activation_maps_synthesis.npy", results["activation_maps_true"])

#     print("✅ 所有预测结果已保存为 .npy 文件！")








# test_model.py
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
from baseline_CNN_train_synthesis_data_planar import ElectrogramDataset, AFNetResNet, fast_compute_direction_vectors, directional_vector_loss

# === 📌 electrode 坐标加载 ===
points = pd.read_csv('triElectrodeFlat_Points.csv', header=None).values
points_tensor = torch.tensor(points, dtype=torch.float32)



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

# === 📌 主程序调用 ===
if __name__ == "__main__":
    model_path = "best_AFNetResNet_planar_only.pth"
    validation_data_dir = r"C:\AF\NickMatlab\Library\RetroMapping_GIT\DataProcessing\DataProcessing\AT_data\20250206\Processed_MAT_Files\dataloader_data\sythesis_data\validation"

    results = test_model(model_path, validation_data_dir)

    np.save("test_electrograms_synthesis.npy", results["electrograms"])
    np.save("predicted_activation_times_synthesis.npy", results["activation_times_pred"])
    np.save("true_activation_times_synthesis.npy", results["activation_times_true"])
    np.save("predicted_activation_maps_synthesis.npy", results["activation_maps_pred"])
    np.save("true_activation_maps_synthesis.npy", results["activation_maps_true"])

    print("✅ 所有预测结果已保存为 .npy 文件！")











































