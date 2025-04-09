import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import scipy.io
import os
from scipy.ndimage import gaussian_filter
from scipy.signal import hilbert
from utils.signal_processing import (min_max_normalize, bandpass_filter,
                                     compute_activation_phase, match_phase_to_activation)


from config.constants import (
    NUM_SAMPLES, GRID_RESOLUTION, X_MIN, X_MAX, Y_MIN, Y_MAX,
    ELECTRODE_X, ELECTRODE_Y, SYNTHESIS_DATA_FOLDER
)

""" 📌 生成 Planar Activation Map & Electrograms """
def generate_electrogram(mode="focal", num_samples=NUM_SAMPLES, grid_resolution=GRID_RESOLUTION,
                         x_min=X_MIN, y_min=Y_MIN, x_max=X_MAX, y_max=Y_MAX,
                         electrode_x=ELECTRODE_X, electrode_y=ELECTRODE_Y, dataFolder=SYNTHESIS_DATA_FOLDER, status = 'train'):
    """
    统一生成 electrogram 的函数，mode 可选：'focal', 'rotor', 'planar'
    """
    # 💡 Part 1: waveform + phase 处理（完全相同）
    file_dataFolder = os.path.join(dataFolder, status)
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
                                                                                    electrode_x=ELECTRODE_X, electrode_y=ELECTRODE_Y, dataFolder = SYNTHESIS_DATA_FOLDER, status ='train')
    
    activation_time, electrograms, x_source, y_source, activation_times_19 = generate_electrogram(mode = 'rotor', num_samples=NUM_SAMPLES, grid_resolution=GRID_RESOLUTION,
                                                                                    x_min=X_MIN, y_min=Y_MIN, x_max=X_MAX, y_max=Y_MAX,
                                                                                    electrode_x=ELECTRODE_X, electrode_y=ELECTRODE_Y, dataFolder = SYNTHESIS_DATA_FOLDER, status ='train')
    
    activation_time, electrograms, x_source, y_source, activation_times_19 = generate_electrogram(mode = 'planar', num_samples=NUM_SAMPLES, grid_resolution=GRID_RESOLUTION,
                                                                                    x_min=X_MIN, y_min=Y_MIN, x_max=X_MAX, y_max=Y_MAX,
                                                                                    electrode_x=ELECTRODE_X, electrode_y=ELECTRODE_Y, dataFolder = SYNTHESIS_DATA_FOLDER, status ='train')
    
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



























