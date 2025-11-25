# -*- coding: utf-8 -*-
"""
快速傅里叶变换（FFT）函数模块
用于对振动信号进行时域和频域分析
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

plt.rcParams['font.sans-serif'] = ['SimHei']      # 设置中文黑体
plt.rcParams['axes.unicode_minus'] = False        # 解决负号显示为方块


def compute_fft(signal_data, fs):
    """
    计算信号的快速傅里叶变换（FFT）
    
    参数:
        signal_data: 输入信号数组 (1D numpy array)
        fs: 采样频率 (Hz)
    
    返回:
        f: 频率数组 (Hz)
        magnitude: 归一化后的幅度谱
        phase: 相位谱
    """
    n = len(signal_data)
    fft_result = np.fft.fft(signal_data)
    
    f = np.fft.fftfreq(n, 1/fs)
    f = f[:n//2]
    
    # 归一化：除以采样点数，使幅值更接近原始信号的幅度
    magnitude = np.abs(fft_result[:n//2]) / n
    phase = np.angle(fft_result[:n//2])
    
    # 去掉直流分量（f=0的分量，索引0）
    f = f[1:]
    magnitude = magnitude[1:]
    phase = phase[1:]
    
    return f, magnitude, phase


def plot_time_and_frequency_domain(signal_data, fs,
                                   time_title='时域图',
                                   freq_title='频域图',
                                   max_display_time=None,
                                   max_frequency=None,
                                   yscale='linear',
                                   figsize=(12, 8),
                                   block=True):
    """
    在一个窗口中同时绘制时域图和频域图（两个子图）

    参数:
        signal_data: 输入信号数组 (1D numpy array)
        fs: 采样频率 (Hz)
        time_title: 时域图标题
        freq_title: 频域图标题
        max_display_time: 时域图最大显示时间（秒），None表示显示全部
        max_frequency: 频域图最大显示频率（Hz），None表示显示到奈奎斯特频率
        yscale: 频域图Y轴刻度类型，'linear'或'log'
        figsize: 图形大小，默认(12, 8)
        block: 是否阻塞显示，默认True
    """
    # 创建时间数组
    time = np.arange(len(signal_data)) / fs
    
    # 如果指定了最大显示时间，只显示部分数据
    if max_display_time is not None:
        max_samples = int(max_display_time * fs)
        if max_samples < len(signal_data):
            signal_data_display = signal_data[:max_samples]
            time_display = time[:max_samples]
        else:
            signal_data_display = signal_data
            time_display = time
    else:
        signal_data_display = signal_data
        time_display = time
    
    # 计算FFT
    f, magnitude, _ = compute_fft(signal_data, fs)
    
    # 如果指定了最大频率，只显示到该频率
    if max_frequency is None:
        max_frequency = fs / 2  # 奈奎斯特频率
    
    # 过滤频率范围
    mask = f <= max_frequency
    f_display = f[mask]
    magnitude_display = magnitude[mask]
    
    # 创建图形和子图
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
    
    # 绘制时域图（上子图）
    ax1.plot(time_display, signal_data_display, linewidth=0.5, color='blue')
    ax1.set_title(time_title)
    ax1.set_xlabel('时间 (秒)')
    ax1.set_ylabel('幅度')
    ax1.grid(True, alpha=0.3)
    
    # 绘制频域图（下子图）
    ax2.plot(f_display, magnitude_display, linewidth=0.8, color='red')
    ax2.set_title(freq_title)
    ax2.set_xlabel('频率 (Hz)')
    ax2.set_ylabel('幅度')
    ax2.set_xlim([0, max_frequency])
    ax2.set_yscale(yscale)
    ax2.xaxis.set_major_locator(MaxNLocator(nbins=20))# 修改 nbins 参数，设置横坐标刻度更细（增加刻度数量）
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show(block=block)

    return fig

