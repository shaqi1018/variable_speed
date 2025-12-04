# -*- coding: utf-8 -*-
"""
主函数：使用LPS算法提取轴承振动信号的瞬时频率（IF），并进行平滑插值
"""

import os
import matplotlib.pyplot as plt
from load_data import load_channel1_data
from fft_function import plot_time_and_frequency_domain
from LPS import entropy_based_lps, plot_if_extraction_result, convert_if_to_rpm
from if_data import smooth_and_interpolate_if, plot_if_interpolation


# ================================
# 配置参数
# ================================
DATA_BASE_PATH = r"D:\Variable_speed\University_of_Ottawa\Original_data\UO_Bearing"
FS = 200000  # 采样频率 200kHz
N_SAMPLES = 2000000  # 原始振动信号点数

# 选择文件夹索引（0-4）
# 0: 健康轴承  1: 内圈故障  2: 外圈故障  3: 滚珠故障  4: 组合故障
SELECTED_FOLDER_INDEX = 2
EXAMPLE_FILE = "O-C-1.mat"

# LPS算法参数
LPS_CONFIG = {
    'nperseg': 131072,
    'noverlap': 124518,
    'min_freq': 5,
    'max_freq': 35,
    'c1': 0.9,
    'c2': 1.1,
    'adaptive_range': True,
    'use_interpolation': True,
    'bidirectional': True,
    'lambda_smooth': 2.61,
}

# IF平滑插值参数
IF_SMOOTH_CONFIG = {
    'n_samples': N_SAMPLES,
    'pad_size': 20,
    'smooth_factor': 0.5,
    't_start': 0,
    't_end': 10,
}


# ================================
# 主函数
# ================================
if __name__ == "__main__":
    # 构建文件路径
    subdirs = [d for d in os.listdir(DATA_BASE_PATH)
               if os.path.isdir(os.path.join(DATA_BASE_PATH, d))]

    print("可用数据文件夹:")
    for i, subdir in enumerate(subdirs):
        marker = " <-- 当前" if i == SELECTED_FOLDER_INDEX else ""
        print(f"  [{i}] {subdir}{marker}")

    file_path = os.path.join(DATA_BASE_PATH, subdirs[SELECTED_FOLDER_INDEX], EXAMPLE_FILE)

    if not os.path.exists(file_path):
        print(f"\n文件不存在: {file_path}")
        exit(1)

    # 加载数据
    print(f"\n加载文件: {EXAMPLE_FILE}")
    signal_data = load_channel1_data(file_path)
    print(f"信号长度: {len(signal_data)} 样本 ({len(signal_data)/FS:.2f} 秒)")

    # ================================
    # 窗口1: 振动信号 + FFT频谱图
    # ================================
    plot_time_and_frequency_domain(
        signal_data, FS,
        time_title=f'时域图 - {EXAMPLE_FILE}',
        freq_title=f'频域图 - {EXAMPLE_FILE}',
        block=False
    )

    # ================================
    # 窗口2: LPS算法IF提取结果
    # ================================
    print("\n运行LPS算法提取瞬时频率...")
    t_if, if_estimated, f, Zxx = entropy_based_lps(
        signal_data, FS,
        **LPS_CONFIG,
        verbose=True
    )

    rpm = convert_if_to_rpm(if_estimated)
    print(f"转速范围: {rpm.min():.1f} - {rpm.max():.1f} RPM")

    plot_if_extraction_result(
        t_if, if_estimated, f, Zxx,
        max_display_freq=40,
        title=f'IF提取结果 - {EXAMPLE_FILE}',
        block=False
    )

    # ================================
    # 窗口3: IF平滑插值结果
    # ================================
    print("\n对IF进行平滑和插值...")
    t_full, if_interp, trend_interp = smooth_and_interpolate_if(
        t_if, if_estimated, **IF_SMOOTH_CONFIG
    )
    print(f"IF插值完成: {len(if_estimated)} 点 → {len(if_interp)} 点")

    plot_if_interpolation(
        t_full, if_interp, trend_interp,
        title=f'IF平滑插值 - {EXAMPLE_FILE}',
        block=False
    )

    # 显示所有窗口
    print("\n三个窗口已显示，关闭所有窗口以退出...")
    plt.show()
