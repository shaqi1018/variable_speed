# -*- coding: utf-8 -*-
"""
TO-XNet Training Script
训练脚本

Usage:
    python train_toxnet.py --data_path "path/to/data" --epochs 100
"""

import os
# 解决 OpenMP 运行时冲突问题
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import sys
import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# Add parent directory to path
_current_dir = os.path.dirname(os.path.abspath(__file__))
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)

from models.toxnet import TOXNet, TOXNetLite
from data.dual_stream_api import create_dataset, get_default_config


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """Train one epoch"""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (time_signal, order_spec, labels) in enumerate(dataloader):
        time_signal = time_signal.to(device)
        order_spec = order_spec.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(time_signal, order_spec)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        if (batch_idx + 1) % 10 == 0:
            print(f"  Batch [{batch_idx+1}/{len(dataloader)}] Loss: {loss.item():.4f}")

    return total_loss / len(dataloader), 100.0 * correct / total


def evaluate(model, dataloader, criterion, device):
    """Evaluate model"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for time_signal, order_spec, labels in dataloader:
            time_signal = time_signal.to(device)
            order_spec = order_spec.to(device)
            labels = labels.to(device)

            outputs = model(time_signal, order_spec)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return total_loss / len(dataloader), 100.0 * correct / total, all_preds, all_labels


def main(args):
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Get default config
    config = get_default_config()

    # Create dataset
    print("\n" + "="*50)
    print("Loading and processing dataset...")
    print("="*50)

    dataset = create_dataset(
        base_path=args.data_path,
        lps_config=config['lps_config'],
        if_smooth_config=config['if_smooth_config'],
        window_size=args.window_size,
        downsample_factor=args.downsample_factor,
        folder_indices=[0, 1, 2, 3, 4],  # 包含全部5个类别
        verbose=True
    )

    # Split dataset
    total_size = len(dataset)
    train_size = int(total_size * 0.8)
    val_size = total_size - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    print(f"\nDataset split: Train={train_size}, Val={val_size}")

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                               shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                             shuffle=False, num_workers=0, pin_memory=True)

    # Create model
    print("\n" + "="*50)
    print("Creating TO-XNet model...")
    print("="*50)

    if args.lite:
        model = TOXNetLite(num_classes=args.num_classes).to(device)
        print("Using TOXNetLite (ablation version)")
    else:
        model = TOXNet(
            num_classes=args.num_classes,
            dim=64,
            num_heads=4,
            use_multiscale=True,
            use_pos_encoding=True,
            use_bidirectional=True
        ).to(device)
        print("Using TOXNet (full version)")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training loop
    print("\n" + "="*50)
    print("Starting training...")
    print("="*50)

    best_val_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch [{epoch}/{args.epochs}]")
        print("-" * 30)

        start_time = time.time()

        # Train
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)

        # Update scheduler
        scheduler.step()

        epoch_time = time.time() - start_time

        # Log
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        print(f"Time: {epoch_time:.1f}s | LR: {scheduler.get_last_lr()[0]:.6f}")

        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = os.path.join(args.save_dir, 'best_toxnet.pth')
            os.makedirs(args.save_dir, exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'history': history
            }, save_path)
            print(f"*** Best model saved (Acc: {val_acc:.2f}%) ***")

    print("\n" + "="*50)
    print(f"Training complete! Best Val Acc: {best_val_acc:.2f}%")
    print("="*50)

    # ==================== 训练结束后生成图表 ====================
    print("\n生成训练结果图表...")
    
    # 类别名称
    class_names = ['Healthy', 'Inner', 'Outer', 'Ball', 'Combination']
    
    # 最终评估获取预测结果
    _, _, all_preds, all_labels = evaluate(model, val_loader, criterion, device)
    
    # 创建图表保存目录
    fig_dir = os.path.join(args.save_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # ========== 图1: 损失曲线 ==========
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    epochs_range = range(1, len(history['train_loss']) + 1)
    ax1.plot(epochs_range, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs_range, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    fig1.tight_layout()
    fig1.savefig(os.path.join(fig_dir, 'loss_curve.png'), dpi=150)
    print(f"  保存: {os.path.join(fig_dir, 'loss_curve.png')}")
    
    # ========== 图2: 准确率曲线 ==========
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.plot(epochs_range, history['train_acc'], 'b-', label='Train Acc', linewidth=2)
    ax2.plot(epochs_range, history['val_acc'], 'r-', label='Val Acc', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Training and Validation Accuracy', fontsize=14)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 105])
    fig2.tight_layout()
    fig2.savefig(os.path.join(fig_dir, 'accuracy_curve.png'), dpi=150)
    print(f"  保存: {os.path.join(fig_dir, 'accuracy_curve.png')}")
    
    # ========== 图3: 混淆矩阵 ==========
    cm = confusion_matrix(all_labels, all_preds)
    fig3, ax3 = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names[:args.num_classes], 
                yticklabels=class_names[:args.num_classes],
                ax=ax3, annot_kws={'size': 14})
    ax3.set_xlabel('Predicted Label', fontsize=12)
    ax3.set_ylabel('True Label', fontsize=12)
    ax3.set_title('Confusion Matrix', fontsize=14)
    fig3.tight_layout()
    fig3.savefig(os.path.join(fig_dir, 'confusion_matrix.png'), dpi=150)
    print(f"  保存: {os.path.join(fig_dir, 'confusion_matrix.png')}")
    
    # ========== 图4: 归一化混淆矩阵 ==========
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    fig4, ax4 = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues',
                xticklabels=class_names[:args.num_classes],
                yticklabels=class_names[:args.num_classes],
                ax=ax4, annot_kws={'size': 12})
    ax4.set_xlabel('Predicted Label', fontsize=12)
    ax4.set_ylabel('True Label', fontsize=12)
    ax4.set_title('Normalized Confusion Matrix', fontsize=14)
    fig4.tight_layout()
    fig4.savefig(os.path.join(fig_dir, 'confusion_matrix_normalized.png'), dpi=150)
    print(f"  保存: {os.path.join(fig_dir, 'confusion_matrix_normalized.png')}")
    
    # ========== 图5: 综合图 (2x2布局) ==========
    fig5, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 子图1: 损失曲线
    axes[0, 0].plot(epochs_range, history['train_loss'], 'b-', label='Train', linewidth=2)
    axes[0, 0].plot(epochs_range, history['val_loss'], 'r-', label='Val', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Loss Curve')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 子图2: 准确率曲线
    axes[0, 1].plot(epochs_range, history['train_acc'], 'b-', label='Train', linewidth=2)
    axes[0, 1].plot(epochs_range, history['val_acc'], 'r-', label='Val', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].set_title('Accuracy Curve')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 子图3: 混淆矩阵
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names[:args.num_classes],
                yticklabels=class_names[:args.num_classes],
                ax=axes[1, 0], annot_kws={'size': 10})
    axes[1, 0].set_xlabel('Predicted')
    axes[1, 0].set_ylabel('True')
    axes[1, 0].set_title('Confusion Matrix')
    
    # 子图4: 每类准确率柱状图
    per_class_acc = cm.diagonal() / cm.sum(axis=1) * 100
    colors = plt.cm.Blues(np.linspace(0.4, 0.8, len(per_class_acc)))
    bars = axes[1, 1].bar(class_names[:args.num_classes], per_class_acc, color=colors, edgecolor='black')
    axes[1, 1].set_xlabel('Class')
    axes[1, 1].set_ylabel('Accuracy (%)')
    axes[1, 1].set_title('Per-Class Accuracy')
    axes[1, 1].set_ylim([0, 105])
    # 添加数值标签
    for bar, acc in zip(bars, per_class_acc):
        axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                        f'{acc:.1f}%', ha='center', va='bottom', fontsize=10)
    
    fig5.suptitle(f'TO-XNet Training Results (Best Val Acc: {best_val_acc:.2f}%)', fontsize=16)
    fig5.tight_layout()
    fig5.savefig(os.path.join(fig_dir, 'training_summary.png'), dpi=150)
    print(f"  保存: {os.path.join(fig_dir, 'training_summary.png')}")
    
    # ========== 保存分类报告 ==========
    report = classification_report(all_labels, all_preds, 
                                   target_names=class_names[:args.num_classes],
                                   digits=4)
    report_path = os.path.join(fig_dir, 'classification_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("TO-XNet Classification Report\n")
        f.write("="*60 + "\n\n")
        f.write(f"Best Validation Accuracy: {best_val_acc:.2f}%\n\n")
        f.write(report)
    print(f"  保存: {report_path}")
    
    # 打印分类报告
    print("\n" + "="*50)
    print("Classification Report:")
    print("="*50)
    print(report)
    
    # 显示所有图表
    plt.show()

    return history


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train TO-XNet')
    parser.add_argument('--data_path', type=str, 
                        default=r'D:/Variable_speed/University_of_Ottawa/Original_data/UO_Bearing',
                        help='Path to dataset')
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='Save directory')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--num_classes', type=int, default=5, help='Number of classes')
    parser.add_argument('--window_size', type=int, default=3072, help='Window size')
    parser.add_argument('--downsample_factor', type=int, default=10, help='Downsample factor')
    parser.add_argument('--lite', action='store_true', help='Use TOXNetLite')

    args = parser.parse_args()
    main(args)

