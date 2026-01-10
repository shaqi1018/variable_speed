# -*- coding: utf-8 -*-
"""
阶次分析模块 (Computed Order Tracking)

核心逻辑:
1. 转速变相位 (积分): IF (Hz) -> 累积相位 (弧度)
2. 角度重采样 (插值): 等时间间隔 -> 等角度间隔
3. 变换求谱 (FFT): 角域信号 FFT -> 阶次谱
"""

import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import get_window


def compute_order_spectrum(time_signal, if_curve, fs=200000, target_length=1537):
    """
    计算阶次谱 (Computed Order Tracking)
    
    将变速工况下的时域振动信号转换为阶次谱, 消除转速波动的影响.
    
    参数:
        time_signal: 时域振动信号, 形状 (N,)
        if_curve: 瞬时频率曲线 (Hz), 形状 (N,), 与 time_signal 等长
        fs: 采样频率 (Hz), 默认 200000
        target_length: 输出阶次谱长度, 默认 1537 (便于神经网络输入)
    
    返回:
        order_spec: 归一化阶次谱, 形状 (target_length,), 值域 [0, 1]
    """
    # 0. 基础校验: 确保信号与IF曲线等长
    if len(time_signal) != len(if_curve):
        min_len = min(len(time_signal), len(if_curve))
        time_signal = time_signal[:min_len]
        if_curve = if_curve[:min_len]

    # 1. 转速变相位 (积分)
    # 物理公式: phase(t) = 2*pi * integral(f(t)) dt
    dt = 1.0 / fs
    phase = np.cumsum(if_curve) * dt * 2 * np.pi
    
    # 2. 角度重采样 (插值)
    # 建立等角度网格, 点数与原信号一致以保证分辨率
    num_samples = len(time_signal)
    uniform_angle_grid = np.linspace(phase[0], phase[-1], num_samples)
    
    # 三次样条插值: 时域信号 -> 角域信号
    interpolator = interp1d(phase, time_signal, kind='cubic', fill_value='extrapolate')
    angle_domain_signal = interpolator(uniform_angle_grid)
    
    # 3. 加窗 (减少频谱泄漏)
    window = get_window('hann', len(angle_domain_signal))
    angle_domain_signal = angle_domain_signal * window
    
    # 4. FFT 变换得到阶次谱
    fft_result = np.fft.rfft(angle_domain_signal)
    order_amp = np.abs(fft_result)
    
    # 5. 尺寸对齐 (插值到目标长度)
    if target_length is not None and len(order_amp) != target_length:
        x_old = np.linspace(0, 1, len(order_amp))
        x_new = np.linspace(0, 1, target_length)
        order_amp = interp1d(x_old, order_amp, kind='linear')(x_new)
        
    # 6. 归一化到 [0, 1]
    order_spec = order_amp / (np.max(order_amp) + 1e-8)
    
    return order_spec
