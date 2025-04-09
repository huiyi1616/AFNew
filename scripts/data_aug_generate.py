# -*- coding: utf-8 -*-
"""
生成合成 Electrogram 数据并存入 Train/Validation
"""
import numpy as np
import pandas as pd
import h5py
import os
from tqdm import tqdm  # 进度条
from data_augmentation import generate_electrogram
from config.constants import (
    SYNTHESIS_SAVE_PATH, SYNTHESIS_DATA_FOLDER, ELECTRODE_X, ELECTRODE_Y,
    NUM_SAMPLES, GRID_RESOLUTION, X_MIN, X_MAX, Y_MIN, Y_MAX
)


train_save_path = os.path.join(SYNTHESIS_SAVE_PATH, "train")
validation_save_path = os.path.join(SYNTHESIS_SAVE_PATH, "validation")
# **📌 创建 Train / Validation 文件夹**
os.makedirs(train_save_path, exist_ok=True)
os.makedirs(validation_save_path, exist_ok=True)


# **📌 设定要生成的数据量**
num_train_data = 8000  # 训练集
num_validation_data = 2000  # 验证集

data_distribution = {"focal": 0.00, "rotor": 0.00, "planar": 1.0}  # 增加 Planar 数据的比例


def generate_and_save_data(num_data, real_data_folder, save_path, dataset_type="train"):
    """
    生成数据并保存
    - num_data: 生成数据数量
    - real_data_folder: 真实数据存放路径
    - save_path: 存储合成数据路径
    - dataset_type: "train" or "validation"
    """
    for i in tqdm(range(num_data), desc=f"Generating {dataset_type} Dataset"):
        # **📌 随机选择 Activation 类型**
        activation_type = np.random.choice(
            ["focal", "rotor", "planar"],
            p=[data_distribution["focal"], data_distribution["rotor"], data_distribution["planar"]]
        )
        activation_time, electrograms, x_source, y_source, activation_times_19 = generate_electrogram(
            mode = activation_type,
            num_samples=NUM_SAMPLES,
            grid_resolution=GRID_RESOLUTION,
            x_min=X_MIN,
            y_min=Y_MIN,
            x_max=X_MAX,
            y_max=Y_MAX,
            electrode_x=ELECTRODE_X,
            electrode_y=ELECTRODE_Y,
            dataFolder=real_data_folder,
            status = dataset_type
        )

        # **📌 存储数据**
        filename = os.path.join(save_path, f"sample_{i:05d}.h5")
        with h5py.File(filename, "w") as hf:
            hf.create_dataset("electrograms", data=electrograms, compression="gzip")  # 19 x 400
            hf.create_dataset("activation_map", data=activation_time, compression="gzip")  # 100 x 100
            hf.create_dataset("activation_type", data=np.string_(activation_type))  # "focal", "rotor", "planar"
            hf.create_dataset("source_position", data=[x_source, y_source], compression="gzip")  # (x, y)
            hf.create_dataset("activation_times", data=activation_times_19, compression="gzip")  # 新增！19 个通道的激活时间

            
    print(f"✅ {dataset_type} 数据生成完成！")


if __name__ == "__main__":
    # **📌 生成 Train 数据**
    generate_and_save_data(num_train_data, SYNTHESIS_DATA_FOLDER, train_save_path, dataset_type="train")
    
    # **📌 生成 Validation 数据**
    generate_and_save_data(num_validation_data, SYNTHESIS_DATA_FOLDER, validation_save_path, dataset_type="validation")











