# -*- coding: utf-8 -*-
"""
多尺度小波残差网络 (Multi-Scale Learnable Wavelet Residual Network)
创新点：多尺度小波头部 + 卷积骨干 + 分类器
"""

import torch
import torch.nn as nn
from .ms_layers import MultiScaleWaveletBlock


class MSLWRNet(nn.Module):
    """
    多尺度可学习小波残差网络 (MS-LWR-Net)
    
    网络结构:
    1. 多尺度小波头部 (MultiScaleWaveletBlock) - 核心创新
    2. 卷积骨干网络 (Backbone) - 特征提取
    3. 分类器 (Classifier) - 故障分类
    
    参数:
        num_classes: 分类数 (默认5: 健康/内圈/外圈/滚珠/组合故障)
        num_filters: 头部输出通道数 (默认64)
        seq_len: 输入序列长度 (默认3072)
        fs: 采样频率 (默认20000 Hz)
    """
    
    def __init__(self, num_classes=5, num_filters=64, seq_len=3072, fs=20000.0):
        super(MSLWRNet, self).__init__()
        
        self.seq_len = seq_len
        self.fs = fs
        
        # ========================================
        # 1. 创新点核心：多尺度残差块
        # ========================================
        # 输入单通道(1)，输出丰富特征(64通道)
        self.head = MultiScaleWaveletBlock(
            in_channels=1, 
            out_channels=num_filters, 
            seq_len=seq_len,
            fs=fs
        )
        
        # ========================================
        # 2. 骨干网络 (Backbone)
        # ========================================
        # 类似 ResNet 或论文原版卷积，但通道数要对齐头部输出
        self.backbone = nn.Sequential(
            # Layer 1: 64 -> 128 (降采样)
            nn.Conv1d(num_filters, 128, kernel_size=7, padding=3, stride=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            
            # Layer 2: 128 -> 256 (降采样)
            nn.Conv1d(128, 256, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            
            # 强力池化: 无论多长都压成 6 (原论文设定)
            nn.AdaptiveAvgPool1d(6)
        )
        
        # ========================================
        # 3. 分类器
        # ========================================
        # 输入维度: 256(通道) * 6(长度) = 1536
        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(
            nn.Linear(256 * 6, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x, if_val):
        """
        前向传播
        
        参数:
            x: 输入振动信号 (Batch, 1, seq_len)
            if_val: 瞬时频率 (Batch,)
        
        返回:
            out: 分类logits (Batch, num_classes)
        """
        # 头部处理 (带IF物理信息)
        x = self.head(x, if_val)
        
        # 骨干特征提取
        x = self.backbone(x)
        
        # 分类
        x = self.flatten(x)
        out = self.classifier(x)
        
        return out
