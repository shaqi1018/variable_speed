# -*- coding: utf-8 -*-
"""
可视化模块
包含绘图相关函数
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def plot_time_and_frequency_domain(signal_data, fs,
                                   time_title='时域图',
                                   freq_title='频域图',
                                   max_display_time=None,
                                   max_frequency=None,
                                   figsize=(12, 8),
                                   block=True):
    """绘制时域图和频域图"""
    from algorithms.time_freq import compute_fft
    
    time = np.arange(len(signal_data)) / fs
    
    if max_display_time is not None:
        max_samples = int(max_display_time * fs)
        if max_samples < len(signal_data):
            signal_data_display = signal_data[:max_samples]
            time_display = time[:max_samples]
        else:
            signal_data_display = signal_data
            time_display = time
    else:
        signal_data_display = signal_data
        time_display = time
    
    f, magnitude, _ = compute_fft(signal_data, fs)
    
    if max_frequency is None:
        max_frequency = fs / 2
    
    mask = f <= max_frequency
    f_display = f[mask]
    magnitude_display = magnitude[mask]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
    
    ax1.plot(time_display, signal_data_display, linewidth=0.5, color='blue')
    ax1.set_title(time_title)
    ax1.set_xlabel('时间 (秒)')
    ax1.set_ylabel('幅度')
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(f_display, magnitude_display, linewidth=0.8, color='red')
    ax2.set_title(freq_title)
    ax2.set_xlabel('频率 (Hz)')
    ax2.set_ylabel('幅度')
    ax2.set_xlim([0, max_frequency])
    ax2.xaxis.set_major_locator(MaxNLocator(nbins=20))
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show(block=block)
    return fig


def plot_if_extraction_result(t, if_estimated, f, Zxx, max_display_freq=50,
                              title='IF提取结果', figsize=(14, 10), block=True):
    """绘制IF提取结果：时频图+IF曲线"""
    fig, axes = plt.subplots(2, 1, figsize=figsize)
    
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
    
    axes[1].plot(t, if_estimated, 'b-', linewidth=1.5, label='瞬时频率 (IF)')
    axes[1].set_xlabel('时间 (秒)')
    axes[1].set_ylabel('瞬时频率 (Hz)')
    axes[1].set_title('提取的瞬时频率 (IF)')
    axes[1].grid(True, alpha=0.3, linestyle='--')
    axes[1].legend(loc='upper right')
    
    stats_text = (f'平均IF: {np.mean(if_estimated):.2f} Hz\n'
                  f'范围: [{np.min(if_estimated):.2f}, {np.max(if_estimated):.2f}] Hz\n'
                  f'RPM: {np.mean(if_estimated)*60:.1f}')
    axes[1].text(0.02, 0.98, stats_text, transform=axes[1].transAxes,
                 verticalalignment='top', fontsize=10,
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.show(block=block)
    return fig


def plot_if_interpolation(t_full, if_interp, trend_interp, title='IF插值结果',
                          figsize=(14, 5), block=True):
    """可视化插值后的IF脊线和趋势线"""
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


def plot_training_history(train_losses, val_losses, train_accs, val_accs, save_dir='results'):
    """
    绘制训练过程曲线并保存
    
    参数:
        train_losses: 训练集损失列表
        val_losses: 验证集损失列表
        train_accs: 训练集准确率列表
        val_accs: 验证集准确率列表
        save_dir: 保存目录
    """
    os.makedirs(save_dir, exist_ok=True)
    
    epochs = range(1, len(train_losses) + 1)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss 曲线
    axes[0].plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
    axes[0].plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epochs', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy 曲线
    axes[1].plot(epochs, train_accs, 'b-', label='Train Acc', linewidth=2)
    axes[1].plot(epochs, val_accs, 'r-', label='Val Acc', linewidth=2)
    axes[1].set_xlabel('Epochs', fontsize=12)
    axes[1].set_ylabel('Accuracy (%)', fontsize=12)
    axes[1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'training_history.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"训练历史曲线已保存至: {save_path}")
    plt.close()


def plot_confusion_matrix(y_true, y_pred, class_names, save_dir='results'):
    """
    绘制混淆矩阵并保存
    
    参数:
        y_true: 真实标签
        y_pred: 预测标签
        class_names: 类别名称列表
        save_dir: 保存目录
    """
    os.makedirs(save_dir, exist_ok=True)
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, 
                yticklabels=class_names,
                cbar_kws={'label': 'Count'},
                annot_kws={'fontsize': 12})
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'confusion_matrix.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"混淆矩阵已保存至: {save_path}")
    plt.close()
    
    return cm


def generate_classification_report(y_true, y_pred, class_names, save_dir='results'):
    """
    生成分类报告并保存
    
    参数:
        y_true: 真实标签
        y_pred: 预测标签
        class_names: 类别名称列表
        save_dir: 保存目录
    
    返回:
        report_str: 分类报告字符串
    """
    os.makedirs(save_dir, exist_ok=True)
    
    report_str = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    
    # 保存为文本文件
    save_path = os.path.join(save_dir, 'classification_report.txt')
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("Classification Report\n")
        f.write("=" * 60 + "\n\n")
        f.write(report_str)
        f.write("\n" + "=" * 60 + "\n")
    
    print(f"分类报告已保存至: {save_path}")
    print("\n分类报告:")
    print(report_str)
    
    return report_str
