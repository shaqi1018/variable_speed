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
        self.all_data = []      # 原始振动信号
        self.all_if = []        # 拉伸后的IF曲线
        self.all_labels = []    # 标签
        self.all_filenames = [] # 文件名列表

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

    def build_dataset(self, lps_config, if_smooth_config, verbose=True):
        """
        遍历所有文件，提取IF并收集数据

        参数:
            lps_config: LPS算法参数字典
            if_smooth_config: IF平滑插值参数字典
            verbose: 是否打印进度信息
        """
        # 清空容器
        self.all_data = []
        self.all_if = []
        self.all_labels = []
        self.all_filenames = []

        # 外层循环：遍历故障类型（5个文件夹）
        for folder_idx in range(len(self.subdirs)):
            folder_name = self.subdirs[folder_idx]
            label = self.get_label(folder_idx)
            label_name = self.get_label_name(folder_idx)

            if verbose:
                print(f"\n[{folder_idx}] {label_name}: {folder_name}")

            # 内层循环：遍历.mat文件
            mat_files = self.list_files(folder_idx)

            for filename in mat_files:
                if verbose:
                    print(f"  处理: {filename}", end=" ... ")

                # 加载原始信号
                signal_data = self.load_signal(folder_idx, filename)

                # 调用核心处理函数（计算数学公式在algorithms中）
                result = process_single_file(
                    signal_data, self.fs, lps_config, if_smooth_config,
                    verbose=False
                )

                # 收集数据
                self.all_data.append(signal_data)           # 原始信号 (2000000,)
                self.all_if.append(result['if_interp'])     # 拉伸后IF (2000000,)
                self.all_labels.append(label)               # 标签
                self.all_filenames.append(filename)         # 文件名

                if verbose:
                    print(f"完成 (IF均值: {result['if_interp'].mean():.2f} Hz)")

        if verbose:
            print(f"\n{'='*50}")
            print(f"数据集构建完成!")
            print(f"  文件总数: {len(self.all_data)}")
            print(f"  每条信号: {self.signal_length} 点")
            print(f"  每条IF: {len(self.all_if[0])} 点")
            print(f"  标签分布: {self._count_labels()}")

            # 打印O-B-1.mat的转速Hz范围
            target_file = "O-B-1.mat"
            if target_file in self.all_filenames:
                idx = self.all_filenames.index(target_file)
                sample_if = self.all_if[idx]
                min_hz = sample_if.min()
                max_hz = sample_if.max()
                print(f"  {target_file} 转速范围: 【{min_hz:.2f} - {max_hz:.2f}】Hz")

    def _count_labels(self):
        """统计各标签数量"""
        from collections import Counter
        counts = Counter(self.all_labels)
        return {self.get_label_name(k): v for k, v in sorted(counts.items())}

