# -*- coding: utf-8 -*-
"""
IF处理工具模块
包含IF插值、拉伸、平滑、RPM转换等功能
"""

import numpy as np
from scipy.interpolate import CubicSpline, UnivariateSpline


def smooth_and_interpolate_if(t_if, if_estimated, n_samples=2000000,
                               pad_size=20, smooth_factor=0.5,
                               t_start=0, t_end=10):
    """
    对IF脊线进行平滑和插值，拉伸到与原始振动信号等长
    
    参数:
        t_if: IF脊线的时间轴 (ndarray)
        if_estimated: IF脊线数据 (ndarray)
        n_samples: 目标采样点数，与原始振动信号对齐 (default: 2000000)
        pad_size: 首尾填充点数 (default: 20)
        smooth_factor: UnivariateSpline平滑因子 (default: 0.5)
        t_start: 目标时间轴起点 (default: 0)
        t_end: 目标时间轴终点 (default: 10)
    
    返回:
        t_full: 插值后的时间轴 (n_samples,)
        if_interp: 插值后的IF脊线 (n_samples,)
        trend_interp: 插值后的趋势线 (n_samples,)
    """
    # 填充技巧 + 单变量样条平滑
    if_padded = np.concatenate([
        np.full(pad_size, if_estimated[0]),
        if_estimated,
        np.full(pad_size, if_estimated[-1])
    ])
    
    dt = t_if[1] - t_if[0]
    t_padded = np.concatenate([
        np.linspace(t_if[0] - pad_size * dt, t_if[0] - dt, pad_size),
        t_if,
        np.linspace(t_if[-1] + dt, t_if[-1] + pad_size * dt, pad_size)
    ])
    
    # UnivariateSpline平滑
    spline = UnivariateSpline(t_padded, if_padded, s=len(if_padded) * smooth_factor)
    if_smoothed_padded = spline(t_padded)
    
    # 切除填充部分
    trend = if_smoothed_padded[pad_size:-pad_size]
    
    # Cubic Spline插值 - 拉伸到目标长度
    t_full = np.linspace(t_start, t_end, n_samples)
    
    cs_if = CubicSpline(t_if, if_estimated)
    cs_trend = CubicSpline(t_if, trend)
    
    if_interp = cs_if(t_full)
    trend_interp = cs_trend(t_full)
    
    return t_full, if_interp, trend_interp
