# -*- coding: utf-8 -*-
"""
多尺度小波层模块 (Multi-Scale Wavelet Layers)
创新点：双路径高低频分离 + 残差连接
"""

import torch
import torch.nn as nn
from .layers import LearnableWaveletLayer  # 复用基础组件


class MultiScaleWaveletBlock(nn.Module):
    """
    多尺度小波残差块 (Multi-Scale Wavelet Residual Block)
    
    创新点:
    - 双路径设计：高频共振路径 + 低频特征路径
    - 物理初始化：根据轴承故障频率特征初始化参数
    - 残差连接：保留原始信号信息
    
    参数:
        in_channels: 输入通道数 (原始信号为1)
        out_channels: 输出通道数 (必须为偶数，高低频各占一半)
        seq_len: 序列长度
        fs: 采样频率
    """
    
    def __init__(self, in_channels, out_channels, seq_len=3072, fs=20000.0):
        super(MultiScaleWaveletBlock, self).__init__()
        
        # 输出通道必须为偶数，高低频各占一半
        assert out_channels % 2 == 0, "out_channels must be even"
        branch_channels = out_channels // 2
        
        # === 路径 1: 高频共振捕手 (38x) ===
        # 物理意义：捕捉轴承BPFO、BPFI等高阶故障频率
        self.branch_high = LearnableWaveletLayer(
            num_filters=branch_channels, 
            seq_len=seq_len,
            fs=fs
        )
        # 物理初始化: 盯着高频
        self.branch_high.kappa1.data.fill_(38.0)  # 38倍转速
        self.branch_high.c.data.fill_(15.0)       # 宽带宽，捕捉多个谐波
        # 为了增加多样性，kappa0 依然保持线性分布
        nn.init.uniform_(self.branch_high.kappa0, 0.1, 4.0)

        # === 路径 2: 低频特征捕手 (5x) ===
        # 物理意义：捕捉转速基频及低阶调制
        self.branch_low = LearnableWaveletLayer(
            num_filters=branch_channels, 
            seq_len=seq_len,
            fs=fs
        )
        # 物理初始化: 盯着低频
        self.branch_low.kappa1.data.fill_(5.0)    # 5倍转速
        self.branch_low.c.data.fill_(3.0)         # 窄带宽，精确定位
        nn.init.uniform_(self.branch_low.kappa0, 0.1, 2.0)  # 低频偏移小一点

        # === 路径 3: 融合层 ===
        # 将两个支路(32+32) 融合，并保持特征深度
        self.fusion = nn.Sequential(
            nn.Conv1d(branch_channels * 2, out_channels, kernel_size=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU()
        )

        # === 路径 4: 残差连接 (Shortcut) ===
        # 如果输入(1通道)和输出(64通道)不一样，需要卷积调整
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x, if_val):
        """
        前向传播
        
        参数:
            x: 输入信号 (Batch, in_channels, seq_len)
            if_val: 瞬时频率 (Batch,)
        
        返回:
            out: 多尺度特征 (Batch, out_channels, seq_len)
        """
        # 1. 高频路径
        out_h = self.branch_high(x, if_val)
        
        # 2. 低频路径
        out_l = self.branch_low(x, if_val)
        
        # 3. 拼接高低频特征
        out_cat = torch.cat([out_h, out_l], dim=1)  # shape: (B, out_channels, L)
        
        # 4. 融合
        out_fused = self.fusion(out_cat)
        
        # 5. 残差连接
        identity = self.shortcut(x)
        
        # 物理意义：原始信号信息 + 增强的故障特征
        return torch.relu(out_fused + identity)
