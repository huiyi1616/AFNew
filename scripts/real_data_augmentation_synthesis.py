

import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import scipy.io
import os
from scipy.ndimage import gaussian_filter
from scipy.signal import hilbert, butter, filtfilt
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

# 参数设计
NUM_SAMPLES = 250
GRID_RESOLUTION = 100
X_MIN, X_MAX = -10, 10  
Y_MIN, Y_MAX = -10, 10
points = pd.read_csv('triElectrodeFlat_Points.csv', header=None).values
ELECTRODE_X, ELECTRODE_Y = points[:, 0], points[:, 1]

    
   


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




# """ 📌 生成 Focal Activation Map & Electrograms """
# def generate_focal_electrogram(num_samples=NUM_SAMPLES, grid_resolution=GRID_RESOLUTION, 
#                                 x_min=X_MIN, y_min=Y_MIN, x_max=X_MAX, y_max=Y_MAX, 
#                                 electrode_x=ELECTRODE_X, electrode_y=ELECTRODE_Y):
#     """第一步： 通过真实数据real_waveform做unwrap希尔伯特变换来得到phase"""
#     # **使用真实波形 morphology**
#     dataFolder = r'C:\AF\NickMatlab\Library\RetroMapping_GIT\DataProcessing\DataProcessing\AT_data\20250206\Processed_MAT_Files\Peak_Windows_MAT\train_data_for_synthesis\validation'
     
#     all_files = [f for f in os.listdir(dataFolder) if f.endswith('.mat')]
#     np.random.seed(None) # 随机选取
#     selected_file = np.random.choice(all_files)  
#     mat_data = scipy.io.loadmat(os.path.join(dataFolder, selected_file))
#     real_waveform = mat_data['peak_window_signals']
    
#     if real_waveform.shape[1] == 19:
#         real_waveform = real_waveform.T  
    
#     num_channels, real_waveform_length = real_waveform.shape
    
#     # 计算real_waveform每个peak的phase
#     phase_map = np.zeros((num_channels, real_waveform_length))
#     peak_phases = np.zeros(num_channels)
#     peak_indices = np.zeros(num_channels, dtype=int)
    
#     for i in range(num_channels):
#         filtered_signal = bandpass_filter(real_waveform[i])  # **滤波**
#         analytic_signal = hilbert(filtered_signal)  # **计算 Hilbert 变换**
#         phase = np.angle(analytic_signal)  # **瞬时相位**
#         phase_unwrapped = np.unwrap(phase)  # **展开相位，避免跳跃**
#         phase_map[i] = phase_unwrapped
#         peak_index = np.argmax(np.abs(filtered_signal))  # **找到 peak**
#         peak_phases[i] = phase_map[i, peak_index]  # **存储 peak 对应的相位**
#         peak_indices[i] = peak_index  # **存储 peak index**
        
    
#     """第二步： 先做出 Focal activation map ground truth"""
#     xx, yy = np.meshgrid(np.linspace(x_min, x_max, grid_resolution),
#                          np.linspace(y_min, y_max, grid_resolution))
    
#     # **设定 pacing site**
#     x_focal, y_focal = np.random.uniform(-5, 5), np.random.uniform(-5, 5) 
#     x_source = x_focal
#     y_source = y_focal
#     # **确保传播速度每次变化**
#     np.random.seed(None) # 确保激活时间的一致性
#     random_field = np.random.randn(grid_resolution, grid_resolution)
#     wave_speed_base = 50  # 基础传播速度 (mm/s)
#     wave_speed_variation = np.random.uniform(20, 60) # 速度变化范围！！可调整参数 20-60
#     sigma_value = np.random.uniform(5, 8) # 平滑程度！！可调整参数 5-8
#     wave_speed = wave_speed_base + wave_speed_variation * gaussian_filter(random_field, sigma=sigma_value)
#     wave_speed = np.clip(wave_speed, 10, 110)  
    
    
#     # **计算 Activation Time**
#     distance_to_focal = np.sqrt((xx - x_focal) ** 2 + (yy - y_focal) ** 2)
#     activation_time = distance_to_focal / wave_speed  
#     # **归一化 Activation Time 到 `[0, 1]` 以确保映射范围**
#     activation_time = min_max_normalize(activation_time, new_min=0, new_max=1)
    
#     # **读取电极点坐标**
#     points = pd.read_csv('triElectrodeFlat_Points.csv', header=None).values  
#     electrode_x, electrode_y = points[:, 0], points[:, 1]
    
#     # **获取 19 个电极点的 activation time**
#     electrode_activation_times = np.zeros(19)
#     for i in range(19):
#         xi = int((electrode_x[i] - x_min) / (x_max - x_min) * grid_resolution)
#         yi = int((electrode_y[i] - y_min) / (y_max - y_min) * grid_resolution)
#         electrode_activation_times[i] = activation_time[yi, xi]  
    
#     # **重新匹配 Activation Time 与 信号**
#     window_length = np.random.randint(40, 50)  # 让 window length 变化
#     activation_time_relative_index = np.round(electrode_activation_times * window_length).astype(int)
#     activation_window_position = np.random.randint(window_length, num_samples - window_length - 10)
#     activation_time_index = activation_window_position + activation_time_relative_index

#     """第三步： 得到electrogram"""
#     activation_phases = compute_activation_phase(electrode_activation_times)
#     # 归一化 peak_phases 到 [0, 2π]
#     peak_phases_normalized = (peak_phases - np.min(peak_phases)) / (np.max(peak_phases) - np.min(peak_phases)) * 2 * np.pi
#     # 归一化 activation_phases 到 [0, 2π]
#     activation_phases_normalized = (activation_phases - np.min(activation_phases)) / (np.max(activation_phases) - np.min(activation_phases)) * 2 * np.pi
#     sorted_indices = match_phase_to_activation(peak_phases_normalized, activation_phases_normalized)
    
    
#     normalized_waveform = min_max_normalize(real_waveform, new_min=-1, new_max=1)
#     sorted_real_waveform = normalized_waveform[sorted_indices]  # 重新排列 19 个通道
    
#     electrograms = np.zeros((19, num_samples))
#     # ** 生成 502 采样点的低频 Baseline Wandering 作为基底 **
#     low_freq_noise = np.random.randn(19, num_samples) * 0.2  
#     low_freq_noise = gaussian_filter(low_freq_noise, sigma=[0, 15]) 
    
#     for i in range(19):
#         normalized_signal = sorted_real_waveform[i]
#         peak_index = np.argmax(np.abs(normalized_signal))
#         shift_amount = activation_time_index[i] - peak_index
#         shift_amount = np.clip(shift_amount, 0, num_samples - real_waveform_length)
        
#         shifted_signal = np.zeros(num_samples)
#         start_idx = shift_amount
#         end_idx = min(num_samples, start_idx + real_waveform_length)
        
#         # **平滑处理 (可选)**
#         smooth_window = np.hanning(real_waveform_length)  # **Hanning 平滑窗口**
#         smoothed_signal = normalized_signal[:real_waveform_length] * smooth_window  # **平滑信号**
        
#         shifted_signal[start_idx:end_idx] = smoothed_signal[:end_idx - start_idx]
#         electrograms[i] = low_freq_noise[i] + shifted_signal
    


#     return activation_time, electrograms, x_source, y_source, electrode_activation_times








# """ 📌 生成 rotor Activation Map & Electrograms """
# def generate_rotor_electrogram(num_samples=NUM_SAMPLES, grid_resolution=GRID_RESOLUTION, 
#                                 x_min=X_MIN, y_min=Y_MIN, x_max=X_MAX, y_max=Y_MAX, 
#                                 electrode_x=ELECTRODE_X, electrode_y=ELECTRODE_Y):
#     """第一步： 通过真实数据real_waveform做unwrap希尔伯特变换来得到phase"""
#     # **使用真实波形 morphology**
#     dataFolder = r'C:\AF\NickMatlab\Library\RetroMapping_GIT\DataProcessing\DataProcessing\AT_data\20250206\Processed_MAT_Files\Peak_Windows_MAT\train_data_for_synthesis\train'
#     all_files = [f for f in os.listdir(dataFolder) if f.endswith('.mat')]
#     np.random.seed(None) # 随机选取
#     selected_file = np.random.choice(all_files)  
#     mat_data = scipy.io.loadmat(os.path.join(dataFolder, selected_file))
#     real_waveform = mat_data['peak_window_signals']
    
#     if real_waveform.shape[1] == 19:
#         real_waveform = real_waveform.T  
    
#     num_channels, real_waveform_length = real_waveform.shape
    
#     # 计算real_waveform每个peak的phase
#     phase_map = np.zeros((num_channels, real_waveform_length))
#     peak_phases = np.zeros(num_channels)
#     peak_indices = np.zeros(num_channels, dtype=int)
    
#     for i in range(num_channels):
#         filtered_signal = bandpass_filter(real_waveform[i])  # **滤波**
#         analytic_signal = hilbert(filtered_signal)  # **计算 Hilbert 变换**
#         phase = np.angle(analytic_signal)  # **瞬时相位**
#         phase_unwrapped = np.unwrap(phase)  # **展开相位，避免跳跃**
#         phase_map[i] = phase_unwrapped
#         peak_index = np.argmax(np.abs(filtered_signal))  # **找到 peak**
#         peak_phases[i] = phase_map[i, peak_index]  # **存储 peak 对应的相位**
#         peak_indices[i] = peak_index  # **存储 peak index**
        
 
#     """Option 2 第二步： 生成符合 Rotor 传播的 Activation Map"""
#     xx, yy = np.meshgrid(np.linspace(x_min, x_max, grid_resolution),
#                          np.linspace(y_min, y_max, grid_resolution))
#     # **🌟 随机设定 Rotor 旋转中心**
#     x_rotor, y_rotor = np.random.uniform(-5, 5), np.random.uniform(-5, 5)
#     x_source = x_rotor
#     y_source = y_rotor
#     # **🌟 计算每个点的角度 (theta)**
#     theta = np.arctan2(yy - y_rotor, xx - x_rotor)
#     # **🌟 旋转方向随机**
#     rotation_direction = np.random.choice([-1, 1])  
#     # **🌟 额外引入一个 `initial_phase` 让旋转起点随机**
#     initial_phase = np.random.uniform(0, 2 * np.pi)  # 0° ~ 360° 之间随机
#     theta = rotation_direction * (theta + initial_phase)  # **调整初始相位并随机旋转方向**
#     # **🌟 生成高斯随机场的传播速度**
#     np.random.seed(None)  # 每次变化
#     random_field = np.random.randn(grid_resolution, grid_resolution)
#     wave_speed_base = 50  
#     wave_speed_variation = np.random.uniform(20, 60) # 速度变化范围！！可调整参数 20-60
#     sigma_value = np.random.uniform(5, 8) # 平滑程度！！可调整参数 5-8
#     wave_speed = wave_speed_base + wave_speed_variation * gaussian_filter(random_field, sigma=sigma_value)
#     wave_speed = np.clip(wave_speed, 10, 110)
#     distance_to_rotor = np.sqrt((xx - x_rotor) ** 2 + (yy - y_rotor) ** 2)
#     activation_time = (theta % (2 * np.pi)) / (2 * np.pi) + distance_to_rotor / wave_speed
#     activation_time = min_max_normalize(activation_time, new_min=0, new_max=1)
    
#     points = pd.read_csv('triElectrodeFlat_Points.csv', header=None).values
#     electrode_x, electrode_y = points[:, 0], points[:, 1]
#     electrode_activation_times = np.zeros(19)
#     for i in range(19):
#         xi = int((electrode_x[i] - x_min) / (x_max - x_min) * grid_resolution)
#         yi = int((electrode_y[i] - y_min) / (y_max - y_min) * grid_resolution)
#         electrode_activation_times[i] = activation_time[yi, xi]
    
#     window_length = np.random.randint(40, 50)  # 让 window length 变化
#     activation_time_relative_index = np.round(electrode_activation_times * window_length).astype(int)
#     activation_window_position = np.random.randint(window_length, num_samples - window_length - 10)
#     activation_time_index = activation_window_position + activation_time_relative_index
    


#     """第三步： 得到electrogram"""
#     activation_phases = compute_activation_phase(electrode_activation_times)
#     # 归一化 peak_phases 到 [0, 2π]
#     peak_phases_normalized = (peak_phases - np.min(peak_phases)) / (np.max(peak_phases) - np.min(peak_phases)) * 2 * np.pi
#     # 归一化 activation_phases 到 [0, 2π]
#     activation_phases_normalized = (activation_phases - np.min(activation_phases)) / (np.max(activation_phases) - np.min(activation_phases)) * 2 * np.pi
#     sorted_indices = match_phase_to_activation(peak_phases_normalized, activation_phases_normalized)
    
    
#     normalized_waveform = min_max_normalize(real_waveform, new_min=-1, new_max=1)
#     sorted_real_waveform = normalized_waveform[sorted_indices]  # 重新排列 19 个通道
    
#     electrograms = np.zeros((19, num_samples))
#     # ** 生成 502 采样点的低频 Baseline Wandering 作为基底 **
#     low_freq_noise = np.random.randn(19, num_samples) * 0.2  
#     low_freq_noise = gaussian_filter(low_freq_noise, sigma=[0, 15]) 
    
#     for i in range(19):
#         normalized_signal = sorted_real_waveform[i]
#         peak_index = np.argmax(np.abs(normalized_signal))
#         shift_amount = activation_time_index[i] - peak_index
#         shift_amount = np.clip(shift_amount, 0, num_samples - real_waveform_length)
        
#         shifted_signal = np.zeros(num_samples)
#         start_idx = shift_amount
#         end_idx = min(num_samples, start_idx + real_waveform_length)
        
#         # **平滑处理 (可选)**
#         smooth_window = np.hanning(real_waveform_length)  # **Hanning 平滑窗口**
#         smoothed_signal = normalized_signal[:real_waveform_length] * smooth_window  # **平滑信号**
        
#         shifted_signal[start_idx:end_idx] = smoothed_signal[:end_idx - start_idx]
#         electrograms[i] = low_freq_noise[i] + shifted_signal
    


#     return activation_time, electrograms, x_source, y_source, electrode_activation_times





# """ 📌 生成 Planar Activation Map & Electrograms """
# def generate_planar_electrogram(num_samples=NUM_SAMPLES, grid_resolution=GRID_RESOLUTION, 
#                                 x_min=X_MIN, y_min=Y_MIN, x_max=X_MAX, y_max=Y_MAX, 
#                                 electrode_x=ELECTRODE_X, electrode_y=ELECTRODE_Y):
#     """第一步： 通过真实数据real_waveform做unwrap希尔伯特变换来得到phase"""
#     # **使用真实波形 morphology**
#     dataFolder = r'C:\AF\NickMatlab\Library\RetroMapping_GIT\DataProcessing\DataProcessing\AT_data\20250206\Processed_MAT_Files\Peak_Windows_MAT\train_data_for_synthesis\train'
#     all_files = [f for f in os.listdir(dataFolder) if f.endswith('.mat')]
#     np.random.seed(None) # 随机选取
#     selected_file = np.random.choice(all_files)  
#     mat_data = scipy.io.loadmat(os.path.join(dataFolder, selected_file))
#     real_waveform = mat_data['peak_window_signals']
    
#     if real_waveform.shape[1] == 19:
#         real_waveform = real_waveform.T  
    
#     num_channels, real_waveform_length = real_waveform.shape
    
#     # 计算real_waveform每个peak的phase
#     phase_map = np.zeros((num_channels, real_waveform_length))
#     peak_phases = np.zeros(num_channels)
#     peak_indices = np.zeros(num_channels, dtype=int)
    
#     for i in range(num_channels):
#         filtered_signal = bandpass_filter(real_waveform[i])  # **滤波**
#         analytic_signal = hilbert(filtered_signal)  # **计算 Hilbert 变换**
#         phase = np.angle(analytic_signal)  # **瞬时相位**
#         phase_unwrapped = np.unwrap(phase)  # **展开相位，避免跳跃**
#         phase_map[i] = phase_unwrapped
#         peak_index = np.argmax(np.abs(filtered_signal))  # **找到 peak**
#         peak_phases[i] = phase_map[i, peak_index]  # **存储 peak 对应的相位**
#         peak_indices[i] = peak_index  # **存储 peak index**
        
 
#     """Optional 3 第二步： 生成 Planar Activation Map（从外部区域传播）"""  
#     xx, yy = np.meshgrid(np.linspace(x_min, x_max, grid_resolution),
#                          np.linspace(y_min, y_max, grid_resolution))
#     # **✅ 确定 Planar Wave 的来源点**
#     # 让 source 出现在边界之外（外围区域）
#     source_angle = np.random.uniform(0, 2 * np.pi)  # 让 source 角度在 0° 到 360° 之间随机
#     source_distance = np.random.uniform(10, 25)  # source 位置在 grid 之外
#     x_source = source_distance * np.cos(source_angle)  # 计算 source x 坐标
#     y_source = source_distance * np.sin(source_angle)  # 计算 source y 坐标
#     # **✅ 计算传播方向**
#     propagation_vector = np.array([-np.cos(source_angle), -np.sin(source_angle)])  # 传播方向指向网格中心
#     # **✅ 计算 Activation Time（线性传播）**
#     activation_time = (xx - x_source) * propagation_vector[0] + (yy - y_source) * propagation_vector[1]
#     # **✅ 生成高斯随机场的传播速度**
#     np.random.seed(None)  
#     random_field = np.random.randn(grid_resolution, grid_resolution)
#     wave_speed_base = 50  
#     wave_speed_variation = np.random.uniform(20, 60) # 速度变化范围！！可调整参数 20-60
#     sigma_value = np.random.uniform(5, 8) # 平滑程度！！可调整参数 5-8
#     wave_speed = wave_speed_base + wave_speed_variation * gaussian_filter(random_field, sigma=sigma_value)
#     wave_speed = np.clip(wave_speed, 10, 110)
#     # **✅ 计算 Final Activation Time**
#     activation_time = activation_time / wave_speed  
#     activation_time = min_max_normalize(activation_time, new_min=0, new_max=1)
#     # **✅ 获取 19 个电极点的 Activation Time**
#     electrode_activation_times = np.zeros(19)
#     for i in range(19):
#         xi = int((electrode_x[i] - x_min) / (x_max - x_min) * grid_resolution)
#         yi = int((electrode_y[i] - y_min) / (y_max - y_min) * grid_resolution)
#         electrode_activation_times[i] = activation_time[yi, xi]
    
#     window_length = np.random.randint(40, 50)  # 让 window length 变化
#     activation_time_relative_index = np.round(electrode_activation_times * window_length).astype(int)
#     activation_window_position = np.random.randint(window_length, num_samples - window_length - 10)
#     activation_time_index = activation_window_position + activation_time_relative_index
    

#     """第三步： 得到electrogram"""
#     activation_phases = compute_activation_phase(electrode_activation_times)
#     # 归一化 peak_phases 到 [0, 2π]
#     peak_phases_normalized = (peak_phases - np.min(peak_phases)) / (np.max(peak_phases) - np.min(peak_phases)) * 2 * np.pi
#     # 归一化 activation_phases 到 [0, 2π]
#     activation_phases_normalized = (activation_phases - np.min(activation_phases)) / (np.max(activation_phases) - np.min(activation_phases)) * 2 * np.pi
#     sorted_indices = match_phase_to_activation(peak_phases_normalized, activation_phases_normalized)
#     normalized_waveform = min_max_normalize(real_waveform, new_min=-1, new_max=1)
#     sorted_real_waveform = normalized_waveform[sorted_indices]  # 重新排列 19 个通道
    
#     electrograms = np.zeros((19, num_samples))
#     # ** 生成 502 采样点的低频 Baseline Wandering 作为基底 **
#     low_freq_noise = np.random.randn(19, num_samples) * 0.2  
#     low_freq_noise = gaussian_filter(low_freq_noise, sigma=[0, 15]) 
    
#     for i in range(19):
#         normalized_signal = sorted_real_waveform[i]
#         peak_index = np.argmax(np.abs(normalized_signal))
#         shift_amount = activation_time_index[i] - peak_index
#         shift_amount = np.clip(shift_amount, 0, num_samples - real_waveform_length)
        
#         shifted_signal = np.zeros(num_samples)
#         start_idx = shift_amount
#         end_idx = min(num_samples, start_idx + real_waveform_length)
        
#         # **平滑处理 (可选)**
#         smooth_window = np.hanning(real_waveform_length)  # **Hanning 平滑窗口**
#         smoothed_signal = normalized_signal[:real_waveform_length] * smooth_window  # **平滑信号**
        
#         shifted_signal[start_idx:end_idx] = smoothed_signal[:end_idx - start_idx]
#         electrograms[i] = low_freq_noise[i] + shifted_signal
    


#     return activation_time, electrograms, x_source, y_source, electrode_activation_times









DATAFOLDER = r'C:\AF\NickMatlab\Library\RetroMapping_GIT\DataProcessing\DataProcessing\AT_data\20250206\Processed_MAT_Files\Peak_Windows_MAT\train_data_for_synthesis'
""" 📌 生成 Planar Activation Map & Electrograms """
def generate_electrogram(mode="focal", num_samples=NUM_SAMPLES, grid_resolution=GRID_RESOLUTION,
                         x_min=X_MIN, y_min=Y_MIN, x_max=X_MAX, y_max=Y_MAX,
                         electrode_x=ELECTRODE_X, electrode_y=ELECTRODE_Y, dataFolder=DATAFOLDER, status = 'train'):
    """
    统一生成 electrogram 的函数，mode 可选：'focal', 'rotor', 'planar'
    """
    # 💡 Part 1: waveform + phase 处理（完全相同）
    file_dataFolder = os.path.join(DATAFOLDER, status)
    all_files = [f for f in os.listdir(file_dataFolder) if f.endswith('.mat')]
    np.random.seed(None) # 随机选取
    selected_file = np.random.choice(all_files)
    mat_data = scipy.io.loadmat(os.path.join(file_dataFolder, selected_file))
    real_waveform = mat_data['peak_window_signals']
    if real_waveform.shape[1] == 19:
        real_waveform = real_waveform.T
    num_channels, real_waveform_length = real_waveform.shape
    peak_phases = np.zeros(num_channels)
    for i in range(num_channels):
        filtered_signal = bandpass_filter(real_waveform[i])
        analytic_signal = hilbert(filtered_signal)
        phase_unwrapped = np.unwrap(np.angle(analytic_signal))
        peak_index = np.argmax(np.abs(filtered_signal))
        peak_phases[i] = phase_unwrapped[peak_index]


    # 💡 Part 2: 根据 mode 生成不同的 activation map
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, grid_resolution),
                         np.linspace(y_min, y_max, grid_resolution))
    np.random.seed(None)
    random_field = np.random.randn(grid_resolution, grid_resolution)
    wave_speed_base = 50
    wave_speed_variation = np.random.uniform(20, 60)
    sigma_value = np.random.uniform(5, 8)
    wave_speed = wave_speed_base + wave_speed_variation * gaussian_filter(random_field, sigma=sigma_value)
    wave_speed = np.clip(wave_speed, 10, 110) 

    if mode == "focal":
        x_source, y_source = np.random.uniform(-5, 5), np.random.uniform(-5, 5)
        distance = np.sqrt((xx - x_source) ** 2 + (yy - y_source) ** 2)
        activation_time = distance / wave_speed

    elif mode == "rotor":
        x_source, y_source = np.random.uniform(-5, 5), np.random.uniform(-5, 5)
        theta = np.arctan2(yy - y_source, xx - x_source)
        direction = np.random.choice([-1, 1])
        theta = direction * (theta + np.random.uniform(0, 2 * np.pi))
        distance = np.sqrt((xx - x_source) ** 2 + (yy - y_source) ** 2)
        activation_time = (theta % (2 * np.pi)) / (2 * np.pi) + distance / wave_speed

    elif mode == "planar":
        angle = np.random.uniform(0, 2 * np.pi)
        dist = np.random.uniform(10, 25)
        x_source = dist * np.cos(angle)
        y_source = dist * np.sin(angle)
        direction = np.array([-np.cos(angle), -np.sin(angle)])
        activation_time = (xx - x_source) * direction[0] + (yy - y_source) * direction[1]
        activation_time = activation_time / wave_speed

    else:
        raise ValueError("Mode must be 'focal', 'rotor', or 'planar'.")
    activation_time = min_max_normalize(activation_time, new_min=0, new_max=1)


    # 💡 Part 3: 获取 electrode activation time 和生成信号（完全相同）
    electrode_activation_times = np.zeros(19)
    for i in range(19):
        xi = int((electrode_x[i] - x_min) / (x_max - x_min) * grid_resolution)
        yi = int((electrode_y[i] - y_min) / (y_max - y_min) * grid_resolution)
        electrode_activation_times[i] = activation_time[yi, xi]

    activation_phases = compute_activation_phase(electrode_activation_times)
    peak_phases = (peak_phases - np.min(peak_phases)) / (np.max(peak_phases) - np.min(peak_phases)) * 2 * np.pi
    activation_phases = (activation_phases - np.min(activation_phases)) / (np.max(activation_phases) - np.min(activation_phases)) * 2 * np.pi
    sorted_indices = match_phase_to_activation(peak_phases, activation_phases)
    sorted_real_waveform = min_max_normalize(real_waveform, -1, 1)[sorted_indices]

    window_length = np.random.randint(40, 50)
    activation_time_relative_index = np.round(electrode_activation_times * window_length).astype(int)
    activation_window_position = np.random.randint(window_length, num_samples - window_length - 10)
    activation_time_index = activation_window_position + activation_time_relative_index

    electrograms = np.zeros((19, num_samples))
    low_freq_noise = np.random.randn(19, num_samples) * 0.2  
    low_freq_noise = gaussian_filter(low_freq_noise, sigma=[0, 15]) 

    for i in range(19):
        normalized_signal = sorted_real_waveform[i]
        peak_index = np.argmax(np.abs(normalized_signal))
        shift_amount = activation_time_index[i] - peak_index
        shift_amount = np.clip(shift_amount, 0, num_samples - real_waveform_length)

        shifted_signal = np.zeros(num_samples)
        start_idx = shift_amount
        end_idx = min(num_samples, start_idx + real_waveform_length)

        smooth_window = np.hanning(real_waveform_length)
        smoothed_signal = normalized_signal[:real_waveform_length] * smooth_window
        shifted_signal[start_idx:end_idx] = smoothed_signal[:end_idx - start_idx]

        electrograms[i] = low_freq_noise[i] + shifted_signal

    return activation_time, electrograms, x_source, y_source, electrode_activation_times






"""运行测试可视化结果"""
if __name__ == "__main__":
    
    activation_time, electrograms, x_source, y_source, activation_times_19 = generate_electrogram(mode = 'focal', num_samples=NUM_SAMPLES, grid_resolution=GRID_RESOLUTION,
                                                                                    x_min=X_MIN, y_min=Y_MIN, x_max=X_MAX, y_max=Y_MAX,
                                                                                    electrode_x=ELECTRODE_X, electrode_y=ELECTRODE_Y, dataFolder = DATAFOLDER, status ='train')
    
    activation_time, electrograms, x_source, y_source, activation_times_19 = generate_electrogram(mode = 'rotor', num_samples=NUM_SAMPLES, grid_resolution=GRID_RESOLUTION,
                                                                                    x_min=X_MIN, y_min=Y_MIN, x_max=X_MAX, y_max=Y_MAX,
                                                                                    electrode_x=ELECTRODE_X, electrode_y=ELECTRODE_Y, dataFolder = DATAFOLDER, status ='train')
    
    activation_time, electrograms, x_source, y_source, activation_times_19 = generate_electrogram(mode = 'planar', num_samples=NUM_SAMPLES, grid_resolution=GRID_RESOLUTION,
                                                                                    x_min=X_MIN, y_min=Y_MIN, x_max=X_MAX, y_max=Y_MAX,
                                                                                    electrode_x=ELECTRODE_X, electrode_y=ELECTRODE_Y, dataFolder = DATAFOLDER, status ='train')
    
    # **📌 绘制 Electrogram 信号**
    time = np.linspace(0, NUM_SAMPLES / 980, NUM_SAMPLES)
    plt.figure(figsize=(10, 5))
    for i in range(19):
        plt.plot(time, electrograms[i] + i * 2, label=f'Electrode {i+1}')
    plt.xlabel('Time (s)')
    plt.ylabel('Electrogram Amplitude')
    plt.title('Synthesized Electrograms')
    plt.legend()
    plt.show()
    
    # **📌 绘制 Activation Map**
    plt.figure(figsize=(6, 6))
    plt.imshow(activation_time, cmap='jet', vmin=np.min(activation_time), vmax=np.max(activation_time), 
               extent=[X_MIN, X_MAX, Y_MIN, Y_MAX], origin='lower')  
    plt.colorbar(label='Activation Time (s)')
    plt.scatter(x_source, y_source, c='red', marker='x', s=100, label='Source Site')
    plt.scatter(ELECTRODE_X, ELECTRODE_Y, c='black', edgecolors='k', label="Electrodes", s=50)  
    for idx, (x, y) in enumerate(zip(ELECTRODE_X, ELECTRODE_Y), start=1):
        plt.text(x + 1, y, str(idx), color='black', fontsize=10, weight='bold', ha='left', va='center')
    plt.legend()
    plt.title('Generated Activation Map with Gaussian Random Field')
    plt.xlabel('X Coordinate (mm)')
    plt.ylabel('Y Coordinate (mm)')
    plt.show()



























