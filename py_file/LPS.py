# -*- coding: utf-8 -*-
"""
基于熵的LPS算法 - 瞬时频率(IF)提取
Entropy-based Local Peak Search Algorithm with Adaptive IF Search Range
"""

import numpy as np
import matplotlib.pyplot as plt
from stft_function import compute_stft

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ============================================================================
# 核心计算函数
# ============================================================================

def compute_probability_distribution(power_spectrum, freq_indices):
    """计算概率分布 d_k(μ_ki) = S_k(μ_ki) / Σ S_k(μ_kj)"""
    power = power_spectrum[freq_indices]
    total_power = np.sum(power)
    if total_power > 1e-10:
        return power / total_power
    return np.ones(len(freq_indices)) / len(freq_indices)


def compute_entropy(prob_distribution):
    """计算熵 H = -Σ d_k · log(d_k)"""
    prob = np.clip(prob_distribution, 1e-10, 1.0)
    return -np.sum(prob * np.log(prob))


def find_initial_if(power_spectrum, frequencies, min_freq, max_freq):
    """在搜索范围内找到功率最大的频率作为IF起点"""
    freq_mask = (frequencies >= min_freq) & (frequencies <= max_freq)
    search_indices = np.where(freq_mask)[0]

    if len(search_indices) == 0:
        raise ValueError(f"搜索范围 [{min_freq}, {max_freq}] Hz 内没有有效频率点")

    power_in_range = power_spectrum[search_indices]
    best_idx = search_indices[np.argmax(power_in_range)]
    return frequencies[best_idx], best_idx


def parabolic_interpolation(power_spectrum, peak_idx, frequencies):
    """抛物线插值：获得亚像素级频率精度"""
    if peak_idx <= 0 or peak_idx >= len(power_spectrum) - 1:
        return frequencies[peak_idx]

    y_left = power_spectrum[peak_idx - 1]
    y_center = power_spectrum[peak_idx]
    y_right = power_spectrum[peak_idx + 1]

    denominator = 2 * (2 * y_center - y_left - y_right)
    if abs(denominator) < 1e-10:
        return frequencies[peak_idx]

    delta = np.clip((y_left - y_right) / denominator, -0.5, 0.5)
    freq_resolution = frequencies[1] - frequencies[0]
    return frequencies[peak_idx] + delta * freq_resolution


def compute_penalty_function(power_spectrum, freq_indices, frequencies,
                             prev_if=None, lambda_smooth=0.0, use_interpolation=True,
                             entropy_weight=0.5):
    """
    计算惩罚函数并返回最优频率

    惩罚函数 (论文公式8): χ = -|TFD|² + entropy_weight * H
    目标: 最小化χ = 最大化功率 + 惩罚高熵(噪声)

    参数:
        entropy_weight: 熵的惩罚权重，越大越倾向于选择低熵(能量集中)的频率
    """
    power_in_range = power_spectrum[freq_indices]
    max_power = np.max(power_in_range)
    normalized_power = power_in_range / max_power if max_power > 1e-10 else np.zeros(len(freq_indices))

    # 计算每个候选频率点的局部熵
    local_entropies = np.zeros(len(freq_indices))
    window_size = max(3, len(freq_indices) // 10)  # 局部窗口大小

    for i, idx in enumerate(freq_indices):
        # 取该频率点附近的局部窗口计算熵
        left = max(0, idx - window_size)
        right = min(len(power_spectrum), idx + window_size + 1)
        local_indices = np.arange(left, right)
        local_prob = compute_probability_distribution(power_spectrum, local_indices)
        local_entropies[i] = compute_entropy(local_prob)

    # 归一化局部熵到[0,1]
    max_entropy = np.max(local_entropies)
    min_entropy = np.min(local_entropies)
    if max_entropy - min_entropy > 1e-10:
        normalized_entropy = (local_entropies - min_entropy) / (max_entropy - min_entropy)
    else:
        normalized_entropy = np.zeros(len(freq_indices))

    # 惩罚函数: χ = -功率 + entropy_weight * 熵 + lambda_smooth * 频率偏离
    penalty = -normalized_power + entropy_weight * normalized_entropy

    if lambda_smooth > 0 and prev_if is not None:
        freq_deviation = np.abs(frequencies[freq_indices] - prev_if) / prev_if
        penalty += lambda_smooth * freq_deviation

    local_idx = np.argmin(penalty)
    best_idx = freq_indices[local_idx]
    best_freq = parabolic_interpolation(power_spectrum, best_idx, frequencies) if use_interpolation else frequencies[best_idx]

    # 返回全局熵用于调试
    global_prob = compute_probability_distribution(power_spectrum, freq_indices)
    global_entropy = compute_entropy(global_prob)

    return best_freq, best_idx, global_entropy



# ============================================================================
# 双向追踪内部函数
# ============================================================================

def _track_if_single_direction(power, f, start_idx, end_idx, step,
                               min_freq, max_freq, c1, c2,
                               adaptive_range, lambda_smooth, use_interpolation):
    """单方向追踪IF脊"""
    num_time_points = power.shape[1]
    if_path = np.zeros(num_time_points)

    initial_if, _ = find_initial_if(power[:, start_idx], f, min_freq, max_freq)
    if_path[start_idx] = initial_if

    for k in range(start_idx + step, end_idx + step, step):
        prev_if = if_path[k - step]

        if adaptive_range:
            search_min = max(c1 * prev_if, min_freq)
            search_max = min(c2 * prev_if, max_freq)
        else:
            search_min, search_max = min_freq, max_freq

        search_indices = np.where((f >= search_min) & (f <= search_max))[0]
        if len(search_indices) == 0:
            if_path[k] = prev_if
            continue

        best_freq, _, _ = compute_penalty_function(
            power[:, k], search_indices, f,
            prev_if=prev_if, lambda_smooth=lambda_smooth,
            use_interpolation=use_interpolation
        )
        if_path[k] = best_freq

    return if_path


def _compute_path_smoothness(if_path):
    """计算路径平滑度（一阶差分标准差）"""
    return np.std(np.diff(if_path))


def _fuse_bidirectional_paths(if_forward, if_backward, verbose=False):
    """融合正向和反向追踪路径"""
    smoothness_fwd = _compute_path_smoothness(if_forward)
    smoothness_bwd = _compute_path_smoothness(if_backward)
    mean_diff = np.mean(np.abs(if_forward - if_backward))

    if verbose:
        print(f"  正向平滑度: {smoothness_fwd:.4f}, 反向平滑度: {smoothness_bwd:.4f}")
        print(f"  路径平均差异: {mean_diff:.4f} Hz")

    if mean_diff < 0.5:
        if_fused = (if_forward + if_backward) / 2
        fusion_method = "平均融合"
    elif smoothness_fwd < smoothness_bwd * 0.8:
        if_fused = if_forward.copy()
        fusion_method = "选择正向"
    elif smoothness_bwd < smoothness_fwd * 0.8:
        if_fused = if_backward.copy()
        fusion_method = "选择反向"
    else:
        if_fused = np.minimum(if_forward, if_backward)
        fusion_method = "逐点选择低频"

    if verbose:
        print(f"  融合策略: {fusion_method}")

    return if_fused


# ============================================================================
# 主函数
# ============================================================================

def entropy_based_lps(signal_data, fs, nperseg=131072, noverlap=None,
                      min_freq=5, max_freq=50, c1=0.9, c2=1.1,
                      adaptive_range=True, lambda_smooth=0.0,
                      use_interpolation=True, bidirectional=True, verbose=True):
    """
    基于熵的LPS算法主函数

    参数:
        signal_data: 输入振动信号
        fs: 采样频率 (Hz)
        nperseg: STFT窗口长度
        noverlap: STFT重叠长度，默认75%
        min_freq/max_freq: 频率搜索范围 (Hz)
        c1/c2: 自适应搜索范围系数
        adaptive_range: 是否使用自适应搜索范围
        lambda_smooth: 平滑约束系数
        use_interpolation: 是否使用抛物线插值
        bidirectional: 是否使用双向追踪
        verbose: 是否打印进度信息

    返回:
        t: 时间数组, if_estimated: IF数组, f: 频率数组, Zxx: STFT结果
    """
    if verbose:
        print("=" * 60)
        print("基于熵的LPS算法 - IF脊追踪")
        print("=" * 60)
        print(f"参数: fs={fs}Hz, nperseg={nperseg}, freq=[{min_freq},{max_freq}]Hz")
        print(f"      adaptive={adaptive_range}, c1={c1}, c2={c2}, bidirectional={bidirectional}")

    # 步骤1: 计算STFT
    if verbose:
        print("\n[1/4] 计算STFT...")
    f, t, Zxx = compute_stft(signal_data, fs, nperseg=nperseg, noverlap=noverlap)
    power = np.abs(Zxx) ** 2
    num_time_points = len(t)

    if verbose:
        print(f"  时间点: {num_time_points}, 频率分辨率: {f[1]-f[0]:.4f}Hz, 时间分辨率: {t[1]-t[0]:.4f}s")

    # 步骤2: 正向追踪
    if verbose:
        print("\n[2/4] 正向追踪...")
    if_forward = _track_if_single_direction(
        power, f, 0, num_time_points - 1, +1,
        min_freq, max_freq, c1, c2, adaptive_range, lambda_smooth, use_interpolation
    )
    if verbose:
        print(f"  {if_forward[0]:.2f}Hz → {if_forward[-1]:.2f}Hz")

    if not bidirectional:
        if_estimated = if_forward
    else:
        # 步骤3: 反向追踪
        if verbose:
            print("\n[3/4] 反向追踪...")
        if_backward = _track_if_single_direction(
            power, f, num_time_points - 1, 0, -1,
            min_freq, max_freq, c1, c2, adaptive_range, lambda_smooth, use_interpolation
        )
        if verbose:
            print(f"  {if_backward[-1]:.2f}Hz → {if_backward[0]:.2f}Hz")

        # 步骤4: 融合
        if verbose:
            print("\n[4/4] 双向融合...")
        if_estimated = _fuse_bidirectional_paths(if_forward, if_backward, verbose=verbose)

    if verbose:
        print(f"\n结果: 平均IF={np.mean(if_estimated):.2f}Hz, "
              f"范围=[{np.min(if_estimated):.2f}, {np.max(if_estimated):.2f}]Hz, "
              f"RPM={np.mean(if_estimated)*60:.1f}")

    return t, if_estimated, f, Zxx



# ============================================================================
# 可视化函数
# ============================================================================

def plot_if_extraction_result(t, if_estimated, f, Zxx, max_display_freq=50,
                              title='IF提取结果', figsize=(14, 10),
                              edge_trim_ratio=0.05, block=True):
    """绘制IF提取结果：时频图+IF曲线"""
    fig, axes = plt.subplots(2, 1, figsize=figsize)

    # 上图：时频图 + IF曲线
    magnitude_db = 20 * np.log10(np.abs(Zxx) + 1e-10)
    freq_mask = f <= max_display_freq

    im = axes[0].pcolormesh(t, f[freq_mask], magnitude_db[freq_mask, :],
                            shading='gouraud', cmap='jet')
    axes[0].plot(t, if_estimated, 'w--', linewidth=2.5, label='提取的IF', alpha=0.9)
    axes[0].plot(t, if_estimated, 'k-', linewidth=1, alpha=0.5)
    axes[0].set_xlabel('时间 (秒)')
    axes[0].set_ylabel('频率 (Hz)')
    axes[0].set_title(f'{title} - 时频图与IF脊')
    axes[0].legend(loc='upper right')
    axes[0].set_ylim([0, max_display_freq])
    plt.colorbar(im, ax=axes[0], label='幅度 (dB)')

    # 下图：IF曲线（带趋势线）
    axes[1].plot(t, if_estimated, 'b-', linewidth=1.5, label='瞬时频率 (IF)')

    # 趋势线（镜像填充处理边缘效应）
    if len(if_estimated) > 10:
        window_size = max(5, len(if_estimated) // 20) | 1  # 确保奇数
        pad_size = window_size // 2
        if_padded = np.concatenate([
            if_estimated[pad_size:0:-1],
            if_estimated,
            if_estimated[-2:-pad_size-2:-1]
        ])
        kernel = np.ones(window_size) / window_size
        if_trend = np.convolve(if_padded, kernel, mode='same')[pad_size:-pad_size]
        axes[1].plot(t, if_trend, 'r-', linewidth=2, alpha=0.7, label='趋势线')

    axes[1].set_xlabel('时间 (秒)')
    axes[1].set_ylabel('瞬时频率 (Hz)')
    axes[1].set_title('提取的瞬时频率 (IF)')
    axes[1].grid(True, alpha=0.3, linestyle='--')
    axes[1].legend(loc='upper right')

    # 统计信息
    n = len(if_estimated)
    trim_n = max(1, int(n * edge_trim_ratio))
    if_trimmed = if_estimated[trim_n:-trim_n] if n > 2*trim_n else if_estimated
    stats_text = (f'平均IF: {np.mean(if_trimmed):.2f} Hz\n'
                  f'范围: [{np.min(if_trimmed):.2f}, {np.max(if_trimmed):.2f}] Hz\n'
                  f'RPM: {np.mean(if_trimmed)*60:.1f}')
    axes[1].text(0.02, 0.98, stats_text, transform=axes[1].transAxes,
                 verticalalignment='top', fontsize=10,
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    plt.show(block=block)
    return fig


def convert_if_to_rpm(if_array):
    """将瞬时频率(Hz)转换为转速(RPM)"""
    return if_array * 60
