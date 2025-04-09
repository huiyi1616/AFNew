# -*- coding: utf-8 -*-
"""
Created on Wed Apr  9 15:41:42 2025

@author: hw1616
"""

# config/constants.py

import os
import pandas as pd
import numpy as np

# 获取项目根目录
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 数据路径
SYNTHESIS_DATA_FOLDER = os.path.join(BASE_DIR, "data", "Peak_Windows_MAT", "train_data_for_synthesis")
REAL_DATA_FOLDER = os.path.join(BASE_DIR, "data", "Peak_Windows_MAT", "test_data")
SYNTHESIS_SAVE_PATH = os.path.join(BASE_DIR, "data", "dataloader_data", "synthesis_data")
TRI_ELECTRODE_POINTS = os.path.join(BASE_DIR, "data", "triElectrodeFlat_Points.csv")

# 读取电极位置
POINTS = pd.read_csv(TRI_ELECTRODE_POINTS, header=None).values
ELECTRODE_X, ELECTRODE_Y = POINTS[:, 0], POINTS[:, 1]

# 网格 & 采样参数
NUM_SAMPLES = 250
GRID_RESOLUTION = 100
X_MIN, X_MAX = -10, 10
Y_MIN, Y_MAX = -10, 10
COORDS = np.stack([ELECTRODE_X, ELECTRODE_Y], axis=1)