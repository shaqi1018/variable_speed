# -*- coding: utf-8 -*-
"""
LWPI网络架构
Learnable Wavelet-driven Physically Interpretable Network

网络结构:
    1. LearnableWaveletLayer: 可学习小波滤波层 (论文核心创新)
    2. Convolutional Backbone: 卷积特征提取
    3. Global Average Pooling: 全局平均池化
    4. Classifier: 全连接分类器
"""

import torch
import torch.nn as nn
from .layers import LearnableWaveletLayer


class LWPINet(nn.Module):
    """
    LWPI网络 (Learnable Wavelet-driven Physically Interpretable Network)

    参数:
        num_classes: 故障类别数 (默认4: 健康/内圈/外圈/滚珠)
        num_filters: 小波层滤波器数量 (默认32)
        seq_len: 输入序列长度 (默认3072)
        fs: 采样频率 Hz (默认20000, 降采样后)
    """

    def __init__(self, num_classes=4, num_filters=32, seq_len=3072, fs=20000.0):
        super(LWPINet, self).__init__()

        self.num_classes = num_classes
        self.num_filters = num_filters
        self.seq_len = seq_len
        self.fs = fs

        # ====================================================
        # 第一部分: 可学习小波层 (论文核心创新)
        # ====================================================
        # 输入: (Batch, 1, 3072) + IF值
        # 输出: (Batch, num_filters, 3072)
        self.wavelet_layer = LearnableWaveletLayer(
            num_filters=num_filters,
            seq_len=seq_len,
            fs=fs
        )

        # ====================================================
        # 第二部分: 卷积特征提取骨干 (Backbone)
        # ====================================================
        self.features = nn.Sequential(
            # --- Block 1 ---
            # 输入: (B, 32, 3072) -> 输出: (B, 64, 1536)
            nn.Conv1d(in_channels=num_filters, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),  # 3072 -> 1536

            # --- Block 2 ---
            # 输入: (B, 64, 1536) -> 输出: (B, 128, 768)
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),  # 1536 -> 768

            # --- Block 3 ---
            # 输入: (B, 128, 768) -> 输出: (B, 256, 768)
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            # 不再池化，交给 GAP 处理
        )

        # ====================================================
        # 第三部分: 全局平均池化 (Global Average Pooling)
        # ====================================================
        # 输入: (B, 256, 768) -> 输出: (B, 256, 1)
        self.gap = nn.AdaptiveAvgPool1d(1)

        # ====================================================
        # 第四部分: 分类器 (Classifier)
        # ====================================================
        # 输入: (B, 256) -> 输出: (B, num_classes)
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x, if_val):
        """
        前向传播

        参数:
            x: 原始振动信号 (Batch, 1, seq_len)
            if_val: 瞬时频率 Hz (Batch,)

        返回:
            out: 分类logits (Batch, num_classes)
        """
        # 1. 可学习小波层 (需要 IF 信息)
        # (B, 1, 3072) -> (B, 32, 3072)
        x = self.wavelet_layer(x, if_val)

        # 2. 卷积骨干提取特征
        # (B, 32, 3072) -> (B, 256, 768)
        x = self.features(x)

        # 3. 全局平均池化
        # (B, 256, 768) -> (B, 256, 1)
        x = self.gap(x)

        # 4. 展平
        # (B, 256, 1) -> (B, 256)
        x = x.view(x.size(0), -1)

        # 5. 分类输出
        # (B, 256) -> (B, num_classes)
        out = self.classifier(x)

        return out

    def get_wavelet_params(self):
        """获取小波层的可学习参数 (用于可视化/分析)"""
        return {
            'kappa1': self.wavelet_layer.kappa1.data.cpu().numpy(),
            'kappa0': self.wavelet_layer.kappa0.data.cpu().numpy(),
            'c': self.wavelet_layer.c.data.cpu().numpy(),
        }

