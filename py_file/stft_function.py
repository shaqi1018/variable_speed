# -*- coding: utf-8 -*-
"""
短时傅里叶变换（STFT）函数模块
用于对去除直流分量后的振动信号进行时频分析
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

plt.rcParams['font.sans-serif'] = ['SimHei']      # 设置中文黑体
plt.rcParams['axes.unicode_minus'] = False        # 解决负号显示为方块


def remove_dc_component(signal_data):
    """
    去除信号的直流分量（均值）
    
    参数:
        signal_data: 输入信号数组 (1D numpy array)
    
    返回:
        去除直流分量后的信号
    """
    return signal_data - np.mean(signal_data)


def compute_stft(signal_data, fs, nperseg=1024, noverlap=None, nfft=None, window='hann'):
    """
    计算信号的短时傅里叶变换（STFT）
    
    参数:
        signal_data: 输入信号数组 (1D numpy array)
        fs: 采样频率 (Hz)
        nperseg: 每个段的长度（样本数），默认1024
        noverlap: 段之间的重叠样本数，默认为nperseg的75%
        nfft: FFT的长度，默认等于nperseg
        window: 窗函数类型，默认'hann'（汉宁窗）
    
    返回:
        f: 频率数组 (Hz)
        t: 时间数组 (秒)
        Zxx: STFT结果的复数矩阵
    """
    # 去除直流分量
    signal_no_dc = remove_dc_component(signal_data)
    
    # 设置默认重叠
    if noverlap is None:
        noverlap = nperseg * 3 // 4  # 75%重叠
    
    # 计算STFT
    f, t, Zxx = signal.stft(signal_no_dc, fs=fs, window=window, 
                            nperseg=nperseg, noverlap=noverlap, nfft=nfft)
    
    return f, t, Zxx


def plot_stft_spectrogram(signal_data, fs,
                          title='时频图 (STFT)',
                          nperseg=1024,
                          noverlap=None,
                          nfft=None,
                          window='hann',
                          max_frequency=None,
                          vmin=None,
                          vmax=None,
                          cmap='jet',
                          figsize=(12, 8),
                          block=True):
    """
    绘制STFT时频图（频谱图）

    参数:
        signal_data: 输入信号数组 (1D numpy array)
        fs: 采样频率 (Hz)
        title: 图形标题
        nperseg: 每个段的长度（样本数），默认1024
        noverlap: 段之间的重叠样本数，默认为nperseg的75%
        nfft: FFT的长度，默认等于nperseg
        window: 窗函数类型，默认'hann'（汉宁窗）
        max_frequency: 最大显示频率（Hz），None表示显示到奈奎斯特频率
        vmin: 颜色映射的最小值（dB），None表示自动
        vmax: 颜色映射的最大值（dB），None表示自动
        cmap: 颜色映射，默认'jet'
        figsize: 图形大小，默认(12, 8)
        block: 是否阻塞显示，默认True

    返回:
        fig: matplotlib图形对象
    """
    # 计算STFT
    f, t, Zxx = compute_stft(signal_data, fs, nperseg=nperseg,
                            noverlap=noverlap, nfft=nfft, window=window)

    # 计算幅度谱（转换为dB）
    magnitude = np.abs(Zxx)
    magnitude_db = 20 * np.log10(magnitude + 1e-10)  # 加小值避免log(0)

    # 设置最大显示频率
    if max_frequency is None:
        max_frequency = fs / 2  # 奈奎斯特频率

    # 创建新的图形窗口
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)

    # 绘制时频图
    im = ax.pcolormesh(t, f, magnitude_db, shading='gouraud',
                       cmap=cmap, vmin=vmin, vmax=vmax)

    # 设置标题和标签
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('时间 (秒)', fontsize=12)
    ax.set_ylabel('频率 (Hz)', fontsize=12)
    ax.set_ylim([0, max_frequency])

    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('幅度 (dB)', fontsize=12)

    # 添加网格
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

    plt.tight_layout()
    plt.show(block=block)

    return fig


def plot_stft_3d(signal_data, fs,
                 title='时频图 3D (STFT)',
                 nperseg=1024,
                 noverlap=None,
                 nfft=None,
                 window='hann',
                 max_frequency=None,
                 cmap='jet',
                 figsize=(14, 10)):
    """
    绘制STFT时频图的3D视图
    
    参数:
        signal_data: 输入信号数组 (1D numpy array)
        fs: 采样频率 (Hz)
        title: 图形标题
        nperseg: 每个段的长度（样本数），默认1024
        noverlap: 段之间的重叠样本数，默认为nperseg的75%
        nfft: FFT的长度，默认等于nperseg
        window: 窗函数类型，默认'hann'（汉宁窗）
        max_frequency: 最大显示频率（Hz），None表示显示到奈奎斯特频率
        cmap: 颜色映射，默认'jet'
        figsize: 图形大小，默认(14, 10)
    
    返回:
        fig: matplotlib图形对象
    """
    from mpl_toolkits.mplot3d import Axes3D
    
    # 计算STFT
    f, t, Zxx = compute_stft(signal_data, fs, nperseg=nperseg,
                            noverlap=noverlap, nfft=nfft, window=window)
    
    # 计算幅度谱
    magnitude = np.abs(Zxx)
    
    # 设置最大显示频率
    if max_frequency is None:
        max_frequency = fs / 2
    
    # 过滤频率范围
    freq_mask = f <= max_frequency
    f_display = f[freq_mask]
    magnitude_display = magnitude[freq_mask, :]
    
    # 创建网格
    T, F = np.meshgrid(t, f_display)
    
    # 创建3D图形
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制3D表面
    surf = ax.plot_surface(T, F, magnitude_display, cmap=cmap, 
                          linewidth=0, antialiased=True)
    
    # 设置标题和标签
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('时间 (秒)', fontsize=12)
    ax.set_ylabel('频率 (Hz)', fontsize=12)
    ax.set_zlabel('幅度', fontsize=12)
    
    # 添加颜色条
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    
    plt.tight_layout()
    plt.show()
    
    return fig

