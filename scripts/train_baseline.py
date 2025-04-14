# -*- coding: utf-8 -*-
"""
Created on Tue Apr  8 15:01:36 2025

@author: hw1616
"""

# -*- coding: utf-8 -*-
"""
Training script for AF Electrogram-to-Activation Map & Class Prediction
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import h5py
import glob
import os
import torchvision.models as models
from tqdm import tqdm
import torch.nn.functional as F
import pandas as pd
from config.constants import (
    SYNTHESIS_SAVE_PATH, SYNTHESIS_DATA_FOLDER, REAL_DATA_FOLDER, ELECTRODE_X, ELECTRODE_Y,
    NUM_SAMPLES, GRID_RESOLUTION, X_MIN, X_MAX, Y_MIN, Y_MAX, POINTS, BASE_DIR
)
from config.train_config import BATCH_SIZE, EPOCHS, LEARNING_RATE, DEVICE
import csv
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

points_tensor = torch.tensor(POINTS, dtype=torch.float32)  # 暂时不放到 device 上

def fast_compute_direction_vectors(activation_times, coords, neighbor_radius=5.0):
    """
    向量化方向向量计算，避免三重 for 循环
    输入：
    - activation_times: (B, N)
    - coords: (N, 2)
    返回：
    - direction_vectors: (B, N, 2)
    """
    B, N = activation_times.shape
    device = activation_times.device
    coords = coords.to(device)  # (N, 2)

    # 计算邻接矩阵 (N, N)
    diffs = coords.unsqueeze(0) - coords.unsqueeze(1)  # (N, N, 2)
    dist_matrix = torch.norm(diffs, dim=-1)  # (N, N)
    adj = ((dist_matrix < neighbor_radius) & (dist_matrix > 0)).float()  # 去掉自己

    dx = diffs / (dist_matrix.unsqueeze(-1) + 1e-8)  # 单位向量 (N, N, 2)

    at_i = activation_times.unsqueeze(2)  # (B, N, 1)
    at_j = activation_times.unsqueeze(1)  # (B, 1, N)
    #dt = at_i - at_j  # 这样是反过来
    dt = at_j - at_i  # 这样是从early place propagate to late place

    weighted_dx = dt.unsqueeze(-1) * dx.unsqueeze(0)  # (B, N, N, 2)

    # 只保留邻居（乘以邻接矩阵）
    weighted_dx = weighted_dx * adj.unsqueeze(0).unsqueeze(-1)  # (B, N, N, 2)

    sum_vec = weighted_dx.sum(dim=2)  # (B, N, 2)
    num_neighbors = adj.sum(dim=1).clamp(min=1.0)  # (N,)

    direction_vectors = sum_vec / num_neighbors.unsqueeze(0).unsqueeze(-1)  # (B, N, 2)
    return direction_vectors


def monotonicity_loss(activation_time, coords, radius=5.0):
    B, N = activation_time.shape
    diffs = coords.unsqueeze(0) - coords.unsqueeze(1)
    dists = torch.norm(diffs, dim=-1)
    mask = ((dists < radius) & (dists > 0)).float()  # 只对邻居生效
    diff_time = activation_time.unsqueeze(2) - activation_time.unsqueeze(1)
    penalty = F.relu(-diff_time)  # 若违反单调性，则惩罚
    return torch.sum(penalty * mask.unsqueeze(0)) / (torch.sum(mask) + 1e-8)


# === ✅ Directional Loss Function ===
def directional_vector_loss(pred_vectors, true_vectors):
    pred_unit = F.normalize(pred_vectors, dim=-1)
    true_unit = F.normalize(true_vectors, dim=-1)
    cosine_sim = torch.sum(pred_unit * true_unit, dim=-1)  # (B, 19)
    return 1 - torch.mean(cosine_sim ** 2)  # 越接近1表示越方向一致
    ## 用余弦相似度的平方loss来强化一致性
    

# **📌 数据集定义**
class ElectrogramDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir):
        self.files = glob.glob(os.path.join(data_dir, "*.h5"))
        self.label_map = {"focal": 0, "rotor": 1, "planar": 2}  # 分类标签

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        filename = self.files[idx]
        with h5py.File(filename, 'r') as hf:
            electrograms = torch.tensor(hf['electrograms'][:], dtype=torch.float32)  # (19, 400)
            activation_map = torch.tensor(hf['activation_map'][:], dtype=torch.float32)  # (100, 100)
            activation_times = torch.tensor(hf['activation_times'][:], dtype=torch.float32)  # <== 加这行
        return electrograms, activation_map, activation_times





# **📌 ResNet-based Model**
class AFNetResNet(nn.Module):
    def __init__(self):
        super(AFNetResNet, self).__init__()

        # **ResNet18 作为 Feature Extractor**
        resnet = models.resnet18(pretrained=False)
        resnet.conv1 = nn.Conv2d(19, 64, kernel_size=7, stride=2, padding=3, bias=False)  # 适配 19 通道
        self.resnet = nn.Sequential(*list(resnet.children())[:-1])  # 去掉最后的 FC 层

        self.shared_fc = nn.Linear(512, 1024)  # ResNet18 输出 512 维，扩展到 1024 维
        self.shared_relu = nn.ReLU()
        self.activation_time_fc = nn.Linear(1024, 19)

        # **Activation Map 预测任务**
        self.activation_fc = nn.Linear(1024, 100 * 100)  

    def forward(self, x):
        x = x.unsqueeze(-1)  # **变为 `[batch, 19, 400, 1]`**
        x = x.repeat(1, 1, 1, 7)  # **扩展到 `[batch, 19, 400, 7]` 适应 ResNet**
        
        x = self.resnet(x)  # **ResNet 提取特征**
        x = x.view(x.size(0), -1)  # 展平
        shared_features = self.shared_relu(self.shared_fc(x))  # 共享特征层
        activation_times = self.activation_time_fc(shared_features)  # (batch, 19)

        activation_pred = self.activation_fc(shared_features).view(-1, 100, 100)  # **Activation Map 任务**

        return activation_pred, activation_times


# **📌 训练函数**
def train(model, dataloader, criterion_map, criterion_time, optimizer, device):
    model.train()
    total_loss = 0

    activation_time_mae_total = 0
    activation_map_mae_total = 0
    directional_loss_total = 0 
    
    n_batches = 0

    for electrograms, activation_maps, activation_times_target in tqdm(dataloader, desc="Training"):
        electrograms, activation_maps, activation_times_target = (
            electrograms.to(device), activation_maps.to(device), activation_times_target.to(device)
        )

        optimizer.zero_grad()
        activation_map_pred, activation_time_pred = model(electrograms)


        loss_map = criterion_map(activation_map_pred, activation_maps)
        loss_time = criterion_time(activation_time_pred, activation_times_target)

        # # 在每个 batch 中，加入：
        pred_dirs = fast_compute_direction_vectors(activation_time_pred, points_tensor.to(device))
        true_dirs = fast_compute_direction_vectors(activation_times_target, points_tensor.to(device))
        loss_dir_vec = directional_vector_loss(pred_dirs, true_dirs)

        loss = loss_dir_vec


        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        directional_loss_total += loss_dir_vec.item()

        # 🔹 统计 MAE 和 Score
        activation_time_mae_batch = torch.mean(torch.abs(activation_time_pred - activation_times_target)).item()
        activation_map_mae_batch = torch.mean(torch.abs(activation_map_pred - activation_maps)).item()

        activation_time_mae_total += activation_time_mae_batch
        activation_map_mae_total += activation_map_mae_batch
        n_batches += 1

    avg_loss = total_loss / len(dataloader)
    activation_time_mae_avg = activation_time_mae_total / n_batches
    activation_map_mae_avg = activation_map_mae_total / n_batches
    activation_time_score = 1 - activation_time_mae_avg
    activation_map_score = 1 - activation_map_mae_avg
    direction_vector_score = 1 - (directional_loss_total / n_batches)


    print("🧪 Training Results:")
    print(f"   Activation Time Score   : {activation_time_score:.4f}")
    print(f"   Activation Map Score    : {activation_map_score:.4f}")
    print(f"   Direction Vector Score  : {direction_vector_score:.4f}")


    return avg_loss, activation_time_score, activation_map_score, direction_vector_score



# **📌 评估函数**
def evaluate(model, dataloader, criterion_map, criterion_time, device):
    model.eval()
    total_loss = 0

    
    activation_time_mae_total = 0
    activation_map_mae_total = 0
    directional_loss_total = 0 
    n_batches = 0

    with torch.no_grad():
        for electrograms, activation_maps, activation_times_target in tqdm(dataloader, desc="Evaluating"):
            electrograms, activation_maps, activation_times_target = (
                electrograms.to(device), activation_maps.to(device), activation_times_target.to(device)
            )

            activation_map_pred, activation_time_pred = model(electrograms)

            loss_map = criterion_map(activation_map_pred, activation_maps)
            loss_time = criterion_time(activation_time_pred, activation_times_target)
            
            pred_dirs = fast_compute_direction_vectors(activation_time_pred, points_tensor.to(device))
            true_dirs = fast_compute_direction_vectors(activation_times_target, points_tensor.to(device))
            loss_dir_vec = directional_vector_loss(pred_dirs, true_dirs)
            loss_mono = monotonicity_loss(activation_time_pred, points_tensor.to(device))

            loss = loss_dir_vec + loss_mono
            
            
            total_loss += loss.item()
            directional_loss_total += loss_dir_vec.item()  # ✅ 累加方向损失
            # 🔹 统计 MAE
            activation_time_mae_batch = torch.mean(torch.abs(activation_time_pred - activation_times_target)).item()
            activation_map_mae_batch = torch.mean(torch.abs(activation_map_pred - activation_maps)).item()
            
            activation_time_mae_total += activation_time_mae_batch
            activation_map_mae_total += activation_map_mae_batch
            
            n_batches += 1

    avg_loss = total_loss / len(dataloader)
    activation_time_mae_avg = activation_time_mae_total / n_batches
    activation_map_mae_avg = activation_map_mae_total / n_batches
    directional_loss_avg = directional_loss_total / n_batches  # ✅ 平均方向损失
    activation_time_score = 1 - activation_time_mae_avg
    activation_map_score = 1 - activation_map_mae_avg
    direction_vector_score = 1 - directional_loss_avg # 越高越好
    
    
    print("📊 Validation Results:")
    print(f"   Activation Time MAE     : {activation_time_mae_avg:.4f}")
    print(f"   Activation Time Score   : {activation_time_score:.4f}")
    print(f"   Activation Map MAE      : {activation_map_mae_avg:.4f}")
    print(f"   Activation Map Score    : {activation_map_score:.4f}")
    print(f"   Direction Vector Score  : {direction_vector_score:.4f}") 

    return avg_loss, activation_time_score, activation_map_score, direction_vector_score




# **📌 主训练过程**
def main():
    train_data_dir = os.path.join(SYNTHESIS_SAVE_PATH, 'train')
    validation_data_dir = os.path.join(SYNTHESIS_SAVE_PATH, 'validation')

    batch_size = BATCH_SIZE
    epochs = EPOCHS
    learning_rate = LEARNING_RATE
    device = DEVICE
    print(f"Using device: {device}")

    train_dataset = ElectrogramDataset(train_data_dir)
    validation_dataset = ElectrogramDataset(validation_data_dir)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    model = AFNetResNet().to(device)
    
    
    # ✅ 初始化 TensorBoard SummaryWriter
    tb_writer  = SummaryWriter(log_dir=os.path.join(BASE_DIR, "logs", "runs_AFNet", "1"))
    # 初始化 CSV 文件并写入表头
    best_val_direction_vector_score = -1
    log_filename = os.path.join(BASE_DIR, "results", f"training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    with open(log_filename, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Epoch', 'TrainLoss', 'ValLoss', 'TimeScore', 'MapScore'])


    criterion_map = nn.MSELoss()  # Activation Map 任务
    criterion_time = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        train_loss, train_time_score, train_map_score, train_direction_vector_score = train(model, train_loader, criterion_map, criterion_time, optimizer, device)
        val_loss, val_time_score, val_map_score, val_direction_vector_score = evaluate(model, validation_loader, criterion_map, criterion_time, device)
    
        # ✅ 打印 epoch 信息
        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Train Direction Score: {train_direction_vector_score:.4f} | " 
              f"Val Loss: {val_loss:.4f} | "
              f"Time Score: {val_time_score:.4f}, Map Score: {val_map_score:.4f} | "
              f"Validation Direction Score: {val_direction_vector_score:.4f} | " 
              )

        # ✅ 保存最佳模型（基于 activation time score）
        if val_direction_vector_score < best_val_direction_vector_score:
            best_val_direction_vector_score = val_direction_vector_score
            torch.save(model.state_dict(), os.path.join(BASE_DIR, "models", "best_AFNetResNet_planar_only.pth"))
            print("✅ Best model updated and saved!")
        # ✅ 写入 CSV 日志
        with open(log_filename, 'a', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow([epoch+1, train_loss, val_loss, val_time_score, val_map_score])
        
        # Tensorboard 展示 logging
        tb_writer.add_scalar("Score/activation_time_train", train_time_score, epoch)
        tb_writer.add_scalar("Score/activation_map_train", train_map_score, epoch)

        tb_writer.add_scalar("Loss/train", train_loss, epoch)
        tb_writer.add_scalar("Loss/val", val_loss, epoch)
        tb_writer.add_scalar("Loss/direction_train", train_direction_vector_score, epoch)
        tb_writer.add_scalar("Loss/direction_val", val_direction_vector_score, epoch)

        
        tb_writer.add_scalar("Score/activation_time_validation", val_time_score, epoch)
        tb_writer.add_scalar("Score/activation_map_validation", val_map_score, epoch)
                
    
    torch.save(model.state_dict(), os.path.join(BASE_DIR, "models", "AFNetResNet_planar_only.pth"))
    print("✅ Model saved!")
    tb_writer.close()


if __name__ == "__main__":
    main()
