# -*- coding: utf-8 -*-
"""
Ottawa轴承数据集类
负责数据加载、遍历、打标签（不负责计算数学公式）
"""

import os
from .data_utils import load_channel1_data
from algorithms.pipeline import process_single_file


class OttawaDataset:
    """
    Ottawa大学轴承数据集

    文件夹结构:
        0: 健康轴承 (Healthy)
        1: 内圈故障 (Inner race fault)
        2: 外圈故障 (Outer race fault)
        3: 滚珠故障 (Ball fault)
        4: 组合故障 (Combination fault)
    """

    FAULT_TYPES = {
        0: 'Healthy',
        1: 'Inner',
        2: 'Outer',
        3: 'Ball',
        4: 'Combination'
    }

    def __init__(self, base_path, fs=200000, signal_length=2000000):
        """
        初始化数据集

        参数:
            base_path: 数据集根目录
            fs: 采样频率 (Hz)
            signal_length: 每个文件的信号长度
        """
        self.base_path = base_path
        self.fs = fs
        self.signal_length = signal_length
        self.subdirs = self._get_subdirs()

        # 数据容器（build_dataset后填充）
        self.all_data = []
        self.all_if = []
        self.all_labels = []

    def _get_subdirs(self):
        """获取子文件夹列表"""
        subdirs = [d for d in os.listdir(self.base_path)
                   if os.path.isdir(os.path.join(self.base_path, d))]
        subdirs.sort()
        return subdirs

    def get_file_path(self, folder_idx, filename):
        """获取文件完整路径"""
        return os.path.join(self.base_path, self.subdirs[folder_idx], filename)

    def load_signal(self, folder_idx, filename):
        """加载单个信号文件"""
        file_path = self.get_file_path(folder_idx, filename)
        return load_channel1_data(file_path)

    def get_label(self, folder_idx):
        """获取故障类型标签"""
        return folder_idx

    def get_label_name(self, folder_idx):
        """获取故障类型名称"""
        return self.FAULT_TYPES.get(folder_idx, 'Unknown')

    def list_files(self, folder_idx):
        """列出指定文件夹中的所有.mat文件"""
        folder_path = os.path.join(self.base_path, self.subdirs[folder_idx])
        return [f for f in os.listdir(folder_path) if f.endswith('.mat')]

    def build_dataset(self, lps_config, if_smooth_config,
                      window_size=3072, hop_size=None,
                      downsample_factor=10,
                      folder_indices=None, verbose=True):
        """
        遍历所有文件，提取IF并同步切片存储

        参数:
            lps_config: LPS算法参数字典
            if_smooth_config: IF平滑插值参数字典
            window_size: 切片窗口长度，默认3072
            hop_size: 切片步长，默认为window_size（无重叠）
            downsample_factor: 降采样因子，默认10（每10个点取1个）
            folder_indices: 要处理的文件夹索引列表，默认[0,1,2,3]
            verbose: 是否打印进度信息
        """
        import numpy as np

        # 默认步长 = 窗口长度（无重叠）
        if hop_size is None:
            hop_size = window_size

        # 默认只处理0-3文件夹
        if folder_indices is None:
            folder_indices = [0, 1, 2, 3]

        # 清空容器
        self.all_data = []
        self.all_if = []
        self.all_labels = []
        file_count = 0

        # 外层循环：遍历指定的故障类型
        for folder_idx in folder_indices:
            folder_name = self.subdirs[folder_idx]
            label = self.get_label(folder_idx)
            label_name = self.get_label_name(folder_idx)

            if verbose:
                print(f"\n[{folder_idx}] {label_name}: {folder_name}")

            # 内层循环：遍历.mat文件
            mat_files = self.list_files(folder_idx)

            for filename in mat_files:
                file_count += 1
                if verbose:
                    print(f"  处理: {filename}", end=" ... ")

                # 加载原始信号
                signal_data = self.load_signal(folder_idx, filename)

                # 调用核心处理函数（计算数学公式在algorithms中）
                result = process_single_file(
                    signal_data, self.fs, lps_config, if_smooth_config,
                    verbose=False
                )
                if_interp = result['if_interp']  # 拉伸后IF (2000000,)

                # 降采样：每隔 downsample_factor 个点取1个点
                signal_data = signal_data[::downsample_factor]   # 2000000 -> 200000
                if_interp = if_interp[::downsample_factor]       # 2000000 -> 200000

                # 计算切片数量: (总点数 - 窗口长度) // 步长 + 1
                n_slices = (len(signal_data) - window_size) // hop_size + 1

                # 切片循环
                for i in range(n_slices):
                    start = i * hop_size
                    end = start + window_size
                    signal_slice = signal_data[start:end]
                    if_slice = if_interp[start:end]
                    if_mean = np.mean(if_slice)

                    self.all_data.append(signal_slice)
                    self.all_if.append(if_mean)
                    self.all_labels.append(label)

                if verbose:
                    print(f"切片: {n_slices} 片, IF均值: {np.mean(if_interp):.2f} Hz")

        # 转换为numpy数组，并为Conv1d添加通道维度
        self.all_data = np.array(self.all_data)[:, np.newaxis, :]
        self.all_if = np.array(self.all_if)
        self.all_labels = np.array(self.all_labels)

        if verbose:
            print(f"\n{'='*50}")
            print(f"数据集构建完成!")
            print(f"  文件总数: {file_count}")
            print(f"  降采样: 每{downsample_factor}取1 ({self.signal_length} -> {self.signal_length // downsample_factor})")
            print(f"  窗口长度: {window_size}, 步长: {hop_size}")
            print(f"  总切片数: {len(self.all_data)}")
            print(f"  all_data shape: {self.all_data.shape}")
            print(f"  all_if shape: {self.all_if.shape}")
            print(f"  all_labels shape: {self.all_labels.shape}")
            print(f"  IF范围: 【{self.all_if.min():.2f} - {self.all_if.max():.2f}】Hz")
            print(f"  标签分布: {self._count_labels()}")

    def _count_labels(self):
        """统计各标签数量"""
        from collections import Counter
        counts = Counter(self.all_labels)
        return {self.get_label_name(k): v for k, v in sorted(counts.items())}

