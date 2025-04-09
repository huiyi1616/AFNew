# -*- coding: utf-8 -*-
"""
生成合成 Electrogram 数据并存入 Train/Validation
"""
import numpy as np
import pandas as pd
import h5py
import os
from tqdm import tqdm  # 进度条
from real_data_augmentation_synthesis2 import generate_electrogram


# **📌 设定真实数据存储路径**
real_data_folder = r"C:\AF\NickMatlab\Library\RetroMapping_GIT\DataProcessing\DataProcessing\AT_data\20250206\Processed_MAT_Files\Peak_Windows_MAT\train_data_for_synthesis"

# **📌 设定合成数据存储路径**
base_save_path = r"C:\AF\NickMatlab\Library\RetroMapping_GIT\DataProcessing\DataProcessing\AT_data\20250206\Processed_MAT_Files\dataloader_data\sythesis_data"
train_save_path = os.path.join(base_save_path, "train")
validation_save_path = os.path.join(base_save_path, "validation")

# **📌 创建 Train / Validation 文件夹**
os.makedirs(train_save_path, exist_ok=True)
os.makedirs(validation_save_path, exist_ok=True)

# **📌 设定基础参数**
NUM_SAMPLES = 250
GRID_RESOLUTION = 100
X_MIN, X_MAX = -10, 10
Y_MIN, Y_MAX = -10, 10

# **📌 读取电极坐标**
points = pd.read_csv("triElectrodeFlat_Points.csv", header=None).values
ELECTRODE_X, ELECTRODE_Y = points[:, 0], points[:, 1]

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


# **📌 生成 Train 数据**
generate_and_save_data(num_train_data, real_data_folder, train_save_path, dataset_type="train")

# **📌 生成 Validation 数据**
generate_and_save_data(num_validation_data, real_data_folder, validation_save_path, dataset_type="validation")











