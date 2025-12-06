# -*- coding: utf-8 -*-
"""
主函数 - 程序唯一入口
使用LPS算法提取轴承振动信号的瞬时频率（IF），并进行平滑插值
"""

from data import OttawaDataset


# ================================
# 配置参数
# ================================
DATA_BASE_PATH = r"D:\Variable_speed\University_of_Ottawa\Original_data\UO_Bearing"
FS = 200000  # 采样频率 200kHz
N_SAMPLES = 2000000  # 原始振动信号点数

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

# 切片参数
DOWNSAMPLE_FACTOR = 10  # 降采样因子（每10个点取1个）
WINDOW_SIZE = 3072      # 每个样本的长度
HOP_SIZE = 3072         # 步长（无重叠）


# ================================
# 主函数
# ================================
if __name__ == "__main__":
    # 初始化数据集
    dataset = OttawaDataset(DATA_BASE_PATH, fs=FS)

    print("可用数据文件夹:")
    for i, subdir in enumerate(dataset.subdirs):
        print(f"  [{i}] {subdir}")

    # 构建数据集：遍历0-3文件夹，提取IF、降采样、切片
    print("\n开始构建数据集...")
    dataset.build_dataset(
        LPS_CONFIG, IF_SMOOTH_CONFIG,
        window_size=WINDOW_SIZE,
        hop_size=HOP_SIZE,
        downsample_factor=DOWNSAMPLE_FACTOR,
        verbose=True
    )

    # 数据已切片存储到:
    # - dataset.all_data: (n_samples, 3072) 切片后的信号
    # - dataset.all_if: (n_samples,) 每个切片的IF均值
    # - dataset.all_labels: (n_samples,) 标签

    print("\n处理完成！")
