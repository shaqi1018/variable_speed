# -*- coding: utf-8 -*-
"""
测试脚本 - 基于熵的LPS算法可视化
功能：
1. 打开文件选择对话框，选择.mat文件
2. 应用LPS算法处理数据
3. 在一个窗口中显示频谱图和时频图
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # 设置后端
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog
import scipy.io as scio

# 添加上级目录到路径，以便导入算法模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from algorithms.lps import entropy_based_lps, convert_if_to_rpm
from algorithms.time_freq import compute_stft, compute_fft, remove_dc_component
from algorithms.if_tools import smooth_if_curve
from algorithms.pipeline import downsample_signal
from data.data_utils import load_channel1_data


# ==============================
# 配置参数
# ==============================
# 采样频率配置
FS_ORIGINAL = 200000  # 原始采样频率 200 kHz
DOWNSAMPLE_FACTOR = 10  # 降采样因子
FS = FS_ORIGINAL // DOWNSAMPLE_FACTOR  # 降采样后频率 20 kHz

# LPS算法参数
LPS_CONFIG = {
    'nperseg': 32768,          # STFT窗口长度
    'noverlap': 31130,         # STFT重叠长度 (75%重叠)
    'min_freq': 5.0,          # 最小搜索频率 Hz
    'max_freq': 50.0,         # 最大搜索频率 Hz
    'c1': 0.9,                # 自适应搜索下限系数
    'c2': 1.1,                # 自适应搜索上限系数
    'adaptive_range': True,   # 使用自适应搜索范围
    'lambda_smooth': 0.1,     # 平滑约束系数
    'use_interpolation': False,# 使用抛物线插值
    'bidirectional': True,    # 使用双向追踪
}

# 后处理平滑参数
SMOOTH_CONFIG = {
    'enabled': True,          # 是否启用平滑
    'window_size': 15,        # 滤波器窗口大小（必须为奇数）
    'poly_order': 3,          # 多项式阶数（保留形状特征）
}


def select_mat_file():
    """
    打开文件选择对话框，选择.mat文件
    
    返回:
        file_path: 选中的文件路径，如果取消返回None
    """
    # 创建Tk根窗口（隐藏）
    root = Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    
    # 设置初始目录
    initial_dir = r"D:\Variable_speed\University_of_Ottawa\Original_data\UO_Bearing"
    
    # 打开文件选择对话框
    file_path = filedialog.askopenfilename(
        title="请选择 .mat 文件",
        initialdir=initial_dir,
        filetypes=[("MAT文件", "*.mat"), ("所有文件", "*.*")]
    )
    
    root.destroy()
    
    if file_path:
        print(f"\n已选择文件: {file_path}")
        return file_path
    else:
        print("\n未选择文件，程序退出")
        return None


def plot_results(signal, fs, freq_spectrum, power_spectrum, 
                 t_if, if_estimated, if_smoothed, f_stft, Zxx, file_name):
    """
    在一个窗口中显示频谱图和时频图
    
    参数:
        signal: 原始信号
        fs: 采样频率
        freq_spectrum: 频谱的频率轴
        power_spectrum: 功率谱
        t_if: IF时间轴
        if_estimated: 提取的瞬时频率（原始）
        if_smoothed: 平滑后的瞬时频率
        f_stft: STFT频率轴
        Zxx: STFT时频矩阵
        file_name: 文件名（用于标题）
    """
    import matplotlib
    # 在创建图形之前设置字体，彻底解决负号问题
    matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
    matplotlib.rcParams['axes.unicode_minus'] = False
    matplotlib.rcParams['font.family'] = 'sans-serif'
    
    # 创建图形窗口 (1行3列布局)
    fig = plt.figure(figsize=(16, 5))
    fig.suptitle(f'信号分析结果 - {file_name}', fontsize=16, fontweight='bold')
    
    # 计算时频图数据
    magnitude = np.abs(Zxx)
    magnitude_db = 20 * np.log10(magnitude + 1e-10)
    
    # 子图1: 功率谱
    ax1 = plt.subplot(1, 3, 1)
    ax1.semilogy(freq_spectrum, power_spectrum, 'r-', linewidth=1)
    ax1.set_xlabel('频率 (Hz)', fontsize=12)
    ax1.set_ylabel('功率 (对数尺度)', fontsize=12)
    ax1.set_title('功率谱', fontsize=13, fontweight='bold')
    ax1.set_xlim([0, 100])  # 只显示0-100Hz
    ax1.grid(True, alpha=0.3)
    
    # 子图2: 时频图 (STFT)
    ax2 = plt.subplot(1, 3, 2)
    im = ax2.pcolormesh(t_if, f_stft, magnitude_db, 
                        shading='gouraud', cmap='jet', vmin=-80, vmax=10)
    ax2.set_xlabel('时间 (s)', fontsize=12)
    ax2.set_ylabel('频率 (Hz)', fontsize=12)
    ax2.set_title('短时傅里叶变换 (STFT)', fontsize=13, fontweight='bold')
    ax2.set_ylim([0, 40])  # 只显示0-40Hz
    ax2.set_yticks(np.arange(0, 41, 5))  # 5分度
    
    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax2)
    cbar.set_label('幅值 (dB)', fontsize=10)
    
    # 子图3: 时频图 + IF脊线叠加（原始 vs 平滑）
    ax3 = plt.subplot(1, 3, 3)
    im2 = ax3.pcolormesh(t_if, f_stft, magnitude_db, 
                         shading='gouraud', cmap='jet', vmin=-80, vmax=10)
    ax3.plot(t_if, if_estimated, 'w--', linewidth=1.5, alpha=0.6, label='原始IF')
    ax3.plot(t_if, if_smoothed, 'lime', linewidth=2.5, label='平滑IF (Savitzky-Golay)')
    ax3.set_xlabel('时间 (s)', fontsize=12)
    ax3.set_ylabel('频率 (Hz)', fontsize=12)
    ax3.set_title('时频图 + 瞬时频率脊线', fontsize=13, fontweight='bold')
    ax3.set_ylim([0, 40])
    ax3.set_yticks(np.arange(0, 41, 5))  # 5分度
    ax3.legend(fontsize=10, loc='upper right')
    
    # 添加颜色条
    cbar2 = plt.colorbar(im2, ax=ax3)
    cbar2.set_label('幅值 (dB)', fontsize=10)
    
    # 调整子图布局
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    
    # 显示图形
    plt.show()


def main():
    """主函数"""
    print("=" * 60)
    print("基于熵的LPS算法测试脚本")
    print("=" * 60)
    
    # 1. 选择文件
    file_path = select_mat_file()
    if file_path is None:
        return
    
    file_name = os.path.basename(file_path)
    
    # 2. 加载数据
    print("\n正在加载数据...")
    try:
        signal_original = load_channel1_data(file_path)
        print(f"原始信号长度: {len(signal_original)} 点")
    except Exception as e:
        print(f"加载数据失败: {str(e)}")
        return
    
    # 3. 降采样
    print(f"\n正在降采样 (因子={DOWNSAMPLE_FACTOR})...")
    signal = downsample_signal(signal_original, DOWNSAMPLE_FACTOR)
    print(f"降采样后信号长度: {len(signal)} 点")
    print(f"降采样后采样频率: {FS} Hz")
    
    # 4. 计算功率谱 (复用time_freq模块的compute_fft)
    print("\n正在计算功率谱...")
    freq_spectrum, magnitude, _ = compute_fft(signal, FS)
    power_spectrum = magnitude ** 2  # 幅度谱的平方即功率谱
    
    # 5. 应用LPS算法提取瞬时频率
    print("\n正在应用基于熵的LPS算法...")
    print("-" * 60)
    try:
        t_if, if_estimated, f_stft, Zxx = entropy_based_lps(
            signal, FS, 
            **LPS_CONFIG,
            verbose=True
        )
        print("-" * 60)
        print(f"IF提取完成! 共 {len(if_estimated)} 个时间点")
    except Exception as e:
        print(f"LPS算法执行失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return
    
    # 6. 应用后处理平滑
    if SMOOTH_CONFIG['enabled']:
        print(f"\n正在应用Savitzky-Golay平滑 (窗口={SMOOTH_CONFIG['window_size']}, 阶数={SMOOTH_CONFIG['poly_order']})...")
        if_smoothed = smooth_if_curve(
            if_estimated, 
            window_size=SMOOTH_CONFIG['window_size'],
            poly_order=SMOOTH_CONFIG['poly_order']
        )
    else:
        if_smoothed = if_estimated
    
    # 7. 绘制结果
    print("\n正在绘制结果...")
    plot_results(signal, FS, freq_spectrum, power_spectrum, 
                 t_if, if_estimated, if_smoothed, f_stft, Zxx, file_name)
    
    print("\n处理完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
