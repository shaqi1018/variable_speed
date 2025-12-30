# -*- coding: utf-8 -*-
"""
IF提取算法验证指标模块
用于定量评估瞬时频率(IF)提取质量

指标列表:
1. 脊线能量占比 (Ridge Energy Ratio, RER)
2. 平滑度指数 (Smoothness Index, SI)
3. 频率跳变率 (Frequency Jump Rate, FJR)
"""

import numpy as np


def ridge_energy_ratio(stft_matrix, if_curve, frequencies, freq_tolerance=None):
    """
    脊线能量占比 (Ridge Energy Ratio, RER)
    
    定义: 提取的IF脊线上的能量占时频图总能量的比例
    
    公式: RER = Σ|S(f_IF(t), t)|² / Σ|S(f, t)|²
    
    意义:
    - RER越高，说明提取的IF脊线经过了能量集中的区域
    - 理想情况下，RER应接近信号主成分的能量占比
    
    参数:
        stft_matrix: STFT时频矩阵 (ndarray), shape = (n_freq, n_time)
        if_curve: 提取的IF曲线 (ndarray), shape = (n_time,)
        frequencies: 频率轴 (ndarray), shape = (n_freq,)
        freq_tolerance: 频率容差 (float)，在脊线附近多少Hz范围内累计能量
                       默认为频率分辨率的2倍
    
    返回:
        rer: 脊线能量占比 (float), 范围 [0, 1]
    """
    n_freq, n_time = stft_matrix.shape
    power_matrix = np.abs(stft_matrix) ** 2
    
    # 总能量
    total_energy = np.sum(power_matrix)
    if total_energy < 1e-12:
        return 0.0
    
    # 频率分辨率
    df = frequencies[1] - frequencies[0] if len(frequencies) > 1 else 1.0
    if freq_tolerance is None:
        freq_tolerance = 2 * df
    
    # 脊线能量
    ridge_energy = 0.0
    for t_idx in range(n_time):
        if np.isnan(if_curve[t_idx]):
            continue
        
        # 找到IF对应的频率索引范围
        f_if = if_curve[t_idx]
        freq_mask = np.abs(frequencies - f_if) <= freq_tolerance
        ridge_energy += np.sum(power_matrix[freq_mask, t_idx])
    
    rer = ridge_energy / total_energy
    return rer


def smoothness_index(if_curve, fs=None, dt=None):
    """
    平滑度指数 (Smoothness Index, SI)
    
    定义: IF曲线的一阶差分的标准差的倒数（归一化）
    
    公式: SI = 1 / (1 + σ(Δf))
    其中: Δf = f(t+1) - f(t)
    
    意义:
    - SI越高，说明IF曲线越平滑，变化越缓慢
    - SI = 1 表示完全平滑（无变化）
    - SI → 0 表示剧烈波动
    
    参数:
        if_curve: 提取的IF曲线 (ndarray), shape = (n_time,)
        fs: 采样率 (float)，用于计算每秒的频率变化率
        dt: 时间步长 (float)，与fs二选一
    
    返回:
        si: 平滑度指数 (float), 范围 (0, 1]
    """
    # 去除NaN值
    valid_if = if_curve[~np.isnan(if_curve)]
    if len(valid_if) < 2:
        return 1.0
    
    # 一阶差分
    diff = np.diff(valid_if)
    
    # 如果提供了时间信息，转换为变化率
    if dt is not None:
        diff = diff / dt
    elif fs is not None:
        diff = diff * fs  # 每秒的频率变化
    
    # 计算标准差
    std_diff = np.std(diff)
    
    # 归一化到 (0, 1]
    si = 1.0 / (1.0 + std_diff)
    
    return si


def frequency_jump_rate(if_curve, threshold_ratio=0.1, relative=True):
    """
    频率跳变率 (Frequency Jump Rate, FJR)
    
    定义: IF曲线中发生"突变"的时刻占比
    
    公式: FJR = N_jump / N_total
    其中: 跳变判定 |Δf| > threshold
    
    意义:
    - FJR越低，说明IF曲线越稳定，没有异常跳变
    - FJR = 0 表示完全无跳变
    - FJR = 1 表示每个时刻都在跳变（极差的结果）
    
    参数:
        if_curve: 提取的IF曲线 (ndarray), shape = (n_time,)
        threshold_ratio: 跳变阈值比例 (float)
                        - relative=True时: 相对于上一时刻IF的比例，默认10%
                        - relative=False时: 绝对频率阈值 (Hz)
        relative: 是否使用相对阈值 (bool)
    
    返回:
        fjr: 频率跳变率 (float), 范围 [0, 1]
        jump_indices: 发生跳变的时刻索引 (ndarray)
    """
    # 去除NaN值，保留索引映射
    valid_mask = ~np.isnan(if_curve)
    valid_indices = np.where(valid_mask)[0]
    valid_if = if_curve[valid_mask]
    
    if len(valid_if) < 2:
        return 0.0, np.array([])
    
    # 一阶差分
    diff = np.diff(valid_if)
    
    # 判定跳变
    if relative:
        # 相对阈值: |Δf| > threshold_ratio * f_prev
        prev_if = valid_if[:-1]
        threshold = threshold_ratio * np.abs(prev_if)
        jump_mask = np.abs(diff) > threshold
    else:
        # 绝对阈值
        jump_mask = np.abs(diff) > threshold_ratio
    
    # 统计跳变
    n_jumps = np.sum(jump_mask)
    n_total = len(diff)
    fjr = n_jumps / n_total
    
    # 跳变时刻的原始索引
    jump_local_indices = np.where(jump_mask)[0]
    jump_indices = valid_indices[jump_local_indices + 1]  # +1 因为跳变发生在后一个时刻
    
    return fjr, jump_indices


def evaluate_if_extraction(stft_matrix, if_curve, frequencies, time_axis=None,
                           freq_tolerance=None, jump_threshold=0.1):
    """
    综合评估IF提取质量
    
    参数:
        stft_matrix: STFT时频矩阵 (ndarray), shape = (n_freq, n_time)
        if_curve: 提取的IF曲线 (ndarray), shape = (n_time,)
        frequencies: 频率轴 (ndarray)
        time_axis: 时间轴 (ndarray)，可选
        freq_tolerance: RER的频率容差 (float)
        jump_threshold: FJR的跳变阈值比例 (float)
    
    返回:
        metrics: 包含所有指标的字典
    """
    # 计算时间步长
    dt = None
    if time_axis is not None and len(time_axis) > 1:
        dt = time_axis[1] - time_axis[0]
    
    # 计算各指标
    rer = ridge_energy_ratio(stft_matrix, if_curve, frequencies, freq_tolerance)
    si = smoothness_index(if_curve, dt=dt)
    fjr, jump_indices = frequency_jump_rate(if_curve, threshold_ratio=jump_threshold)
    
    metrics = {
        'RER': rer,           # 脊线能量占比
        'SI': si,             # 平滑度指数
        'FJR': fjr,           # 频率跳变率
        'n_jumps': len(jump_indices),  # 跳变次数
        'jump_indices': jump_indices,   # 跳变位置
    }
    
    return metrics


def print_metrics(metrics, title="IF提取质量评估"):
    """
    格式化打印评估指标
    
    参数:
        metrics: evaluate_if_extraction返回的指标字典
        title: 标题
    """
    print(f"\n{'='*50}")
    print(f"{title:^50}")
    print(f"{'='*50}")
    print(f"  脊线能量占比 (RER): {metrics['RER']:.4f}  (越高越好)")
    print(f"  平滑度指数   (SI):  {metrics['SI']:.4f}  (越高越好)")
    print(f"  频率跳变率   (FJR): {metrics['FJR']:.4f}  (越低越好)")
    print(f"  跳变次数:           {metrics['n_jumps']}")
    print(f"{'='*50}\n")


# ============================================================
# 测试代码
# ============================================================
if __name__ == "__main__":
    # 生成测试数据
    np.random.seed(42)
    
    # 模拟STFT矩阵
    n_freq, n_time = 100, 200
    frequencies = np.linspace(0, 50, n_freq)
    time_axis = np.linspace(0, 10, n_time)
    
    # 真实IF: 线性调频
    true_if = 10 + 2 * time_axis  # 10-30 Hz
    
    # 生成时频矩阵（在真实IF附近有能量）
    stft_matrix = np.random.randn(n_freq, n_time) * 0.1
    for t_idx, f_if in enumerate(true_if):
        f_idx = np.argmin(np.abs(frequencies - f_if))
        stft_matrix[max(0, f_idx-2):min(n_freq, f_idx+3), t_idx] += 5
    
    # 模拟提取的IF（加入噪声和跳变）
    extracted_if = true_if + np.random.randn(n_time) * 0.5
    # 人为添加几个跳变
    extracted_if[50] += 5
    extracted_if[100] -= 4
    extracted_if[150] += 6
    
    # 评估
    metrics = evaluate_if_extraction(
        stft_matrix, extracted_if, frequencies, time_axis,
        freq_tolerance=1.0, jump_threshold=0.1
    )
    
    print_metrics(metrics, "测试数据评估结果")
    print(f"跳变位置索引: {metrics['jump_indices']}")
