# -*- coding: utf-8 -*-
"""
神经网络层模块
包含LearnableWaveletLayer等自定义层
"""

import torch
import torch.nn as nn
import math


class LearnableWaveletLayer(nn.Module):
    """
    可学习小波层 (Learnable Wavelet Layer)

    基于论文公式，根据输入的瞬时频率(IF)动态构造Morlet小波滤波器。

    核心公式:
        fc = kappa1 * IF + kappa0   (中心频率)
        beta = c * IF                (带宽)
        H(f) = exp(-2*ln2 * (f - fc)^2 / beta^2)  (频域滤波器)

    参数:
        num_filters: 滤波器数量（输出通道数）
        seq_len: 输入序列长度
        fs: 采样频率 (Hz)
    """

    def __init__(self, num_filters=32, seq_len=3072, fs=20000.0):
        super(LearnableWaveletLayer, self).__init__()
        self.num_filters = num_filters
        self.seq_len = seq_len
        self.fs = fs

        # 可学习参数 (论文中的 kappa1, kappa0, c)
        # kappa1: 阶次系数，决定中心频率是转速的多少倍
        # kappa0: 频率偏移量
        # c: 带宽系数
        self.kappa1 = nn.Parameter(torch.rand(num_filters) * 10.0)  # 初始化: 0~10 倍频
        self.kappa0 = nn.Parameter(torch.randn(num_filters))        # 初始化: 标准正态
        self.c = nn.Parameter(torch.rand(num_filters) * 5.0 + 1.0)  # 初始化: 1~6

    def forward(self, x, if_val):
        """
        前向传播

        参数:
            x: 原始振动信号 (Batch, 1, seq_len)
            if_val: 瞬时频率 Hz (Batch,)

        返回:
            out_time: 滤波后信号 (Batch, num_filters, seq_len)
        """
        batch_size = x.shape[0]
        n = self.seq_len

        # ==========================================
        # 1. 转换到频域 (FFT)
        # ==========================================
        # 使用 rfft (实数傅里叶变换)，得到 n//2+1 个频率点
        x_fft = torch.fft.rfft(x, n=n, dim=-1)  # (Batch, 1, n//2+1)

        # 生成频率轴 f (0 ~ fs/2)
        freqs = torch.fft.rfftfreq(n, d=1/self.fs).to(x.device)  # (n//2+1,)

        # ==========================================
        # 2. 根据 IF 动态计算滤波器参数
        # ==========================================
        # 扩展维度以便广播: (Batch, Filters, 1)
        if_val = if_val.view(batch_size, 1, 1)
        k1 = self.kappa1.view(1, self.num_filters, 1)
        k0 = self.kappa0.view(1, self.num_filters, 1)
        ci = self.c.view(1, self.num_filters, 1)

        # 论文公式: fc = kappa1 * IF + kappa0
        f_center = k1 * if_val + k0

        # 论文公式: beta = c * IF
        beta = ci * if_val
        beta = torch.clamp(beta, min=1e-5)  # 防止除零

        # ==========================================
        # 3. 构造 Morlet 滤波器 (频域高斯带通)
        # ==========================================
        # 扩展频率轴: (1, 1, n//2+1)
        f = freqs.view(1, 1, -1)

        # 论文公式: H(f) = exp(-2*ln2 * (f - fc)^2 / beta^2)
        exponent = -2 * math.log(2) * ((f - f_center) ** 2) / (beta ** 2)
        filters = torch.exp(exponent)  # (Batch, Filters, n//2+1)

        # ==========================================
        # 4. 频域滤波 (Element-wise Multiply)
        # ==========================================
        # x_fft: (Batch, 1, n//2+1) 广播到 (Batch, Filters, n//2+1)
        out_fft = x_fft * filters

        # ==========================================
        # 5. 转换回时域 (IFFT)
        # ==========================================
        out_time = torch.fft.irfft(out_fft, n=n, dim=-1)  # (Batch, Filters, seq_len)

        return out_time
