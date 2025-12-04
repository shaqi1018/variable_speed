# -*- coding: utf-8 -*-
"""
IF脊线平滑与插值模块
- 填充技巧 + UnivariateSpline平滑
- Cubic Spline插值拉伸到原始信号长度
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline, UnivariateSpline

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


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
    # ================================
    # 填充技巧 + 单变量样条平滑
    # ================================
    # 1. 填充：左边补pad_size个第一个点，右边补pad_size个最后一个点
    if_padded = np.concatenate([
        np.full(pad_size, if_estimated[0]),
        if_estimated,
        np.full(pad_size, if_estimated[-1])
    ])

    # 创建填充后的时间轴
    dt = t_if[1] - t_if[0]
    t_padded = np.concatenate([
        np.linspace(t_if[0] - pad_size * dt, t_if[0] - dt, pad_size),
        t_if,
        np.linspace(t_if[-1] + dt, t_if[-1] + pad_size * dt, pad_size)
    ])

    # 2. UnivariateSpline平滑
    spline = UnivariateSpline(t_padded, if_padded, s=len(if_padded) * smooth_factor)
    if_smoothed_padded = spline(t_padded)

    # 3. 切除填充部分
    trend = if_smoothed_padded[pad_size:-pad_size]

    # ================================
    # Cubic Spline插值 - 拉伸到目标长度
    # ================================
    t_full = np.linspace(t_start, t_end, n_samples)

    cs_if = CubicSpline(t_if, if_estimated)
    cs_trend = CubicSpline(t_if, trend)

    if_interp = cs_if(t_full)
    trend_interp = cs_trend(t_full)

    return t_full, if_interp, trend_interp


def plot_if_interpolation(t_full, if_interp, trend_interp, title='IF插值结果',
                          figsize=(14, 5), block=True):
    """
    可视化插值后的IF脊线和趋势线

    参数:
        t_full: 插值后的时间轴
        if_interp: 插值后的IF脊线
        trend_interp: 插值后的趋势线
        title: 图标题
        figsize: 图尺寸
        block: 是否阻塞显示
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    ax.plot(t_full, if_interp, 'b-', linewidth=0.5, alpha=0.7, label='IF插值')
    ax.plot(t_full, trend_interp, 'r-', linewidth=1.5, label='趋势线')
    ax.set_xlabel('时间 (s)')
    ax.set_ylabel('IF (Hz)')
    ax.set_title(f'{title} ({len(if_interp)}点)')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([t_full[0], t_full[-1]])
    ax.legend(loc='upper right')

    stats_text = f'插值点数: {len(if_interp)}\nIF均值: {if_interp.mean():.2f} Hz'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            verticalalignment='top', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    plt.show(block=block)
    return fig

