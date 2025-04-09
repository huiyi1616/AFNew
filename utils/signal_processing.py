# -*- coding: utf-8 -*-
"""
Created on Wed Apr  9 15:11:55 2025

@author: hw1616
"""
import numpy as np
from scipy.signal import butter, filtfilt
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

# **📌 归一化函数**
def min_max_normalize(data, new_min=-1, new_max=1):
    old_min, old_max = np.min(data), np.max(data)
    return (data - old_min) / (old_max - old_min + 1e-8) * (new_max - new_min) + new_min

# **📌 带通滤波器**
def bandpass_filter(data, lowcut=1, highcut=50, fs=980, order=3):
    nyq = 0.5 * fs
    low, high = lowcut / nyq, highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

def compute_activation_phase(activation_times, frequency=5):
    """ 计算 activation time 对应的相位 """
    return 2 * np.pi * frequency * activation_times  # 线性转换

def match_phase_to_activation(peak_phases, activation_phases):
    """ 使用匈牙利算法（Hungarian Algorithm）找到 Peak Phase 到 Activation Phase 的最佳匹配 """
    # 计算 phase 之间的欧几里得距离矩阵
    distance_matrix = cdist(peak_phases.reshape(-1, 1), activation_phases.reshape(-1, 1), metric='euclidean')
    # 使用匈牙利算法（最小成本匹配）
    row_indices, col_indices = linear_sum_assignment(distance_matrix)
    return col_indices  # 返回匹配后的通道索引

