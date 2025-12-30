# -*- coding: utf-8 -*-
"""
主函数 - 程序唯一入口
LWPI网络训练与评估
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # 解决OpenMP库冲突

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from data import OttawaDataset
from models import LWPINet, MSLWRNet
from train import train_one_epoch, evaluate
from utils.visualization import plot_training_history, plot_confusion_matrix, generate_classification_report


# ================================
# 数据预处理参数
# ================================
DATA_BASE_PATH = r"D:\Variable_speed\University_of_Ottawa\Original_data\UO_Bearing"
FS = 200000  # 原始采样频率 200kHz
N_SAMPLES = 2000000  # 原始振动信号点数

# ================================
# LPS算法策略配置
# ================================
# 可选策略:
#   'baseline'  - 香农熵 (论文原版，稳定可靠)
#   'renyi_v1'  - 瑞利熵 (创新策略，对峰值更敏感)
LPS_STRATEGY = 'renyi_v1'  # 修改此处切换IF提取策略

# 策略参数
LPS_STRATEGY_PARAMS = {
    # ---- baseline (香农熵) 参数 ----
    # 'entropy_weight': 0.5,  # 熵项权重
    
    # ---- renyi_v1 (瑞利熵) 参数 ----
    'alpha': 3,    # 瑞利熵阶数 (α>1对峰值敏感, 常用3或5)
    'w1': 1.0,     # 能量权重 (功率越大越好)
    'w2': 0.7,     # 熵权重 (峰值越尖锐越好)
}

# LPS算法参数
LPS_CONFIG = {
    'nperseg': 131072,
    'noverlap': 124518,
    'min_freq': 5,
    'max_freq': 35,
    'c1': 0.9,
    'c2': 1.1,
    'adaptive_range': True,
    'use_interpolation': True,
    'bidirectional': True,
    'lambda_smooth': 2.61,
    # 策略注入
    'strategy': LPS_STRATEGY,
    'strategy_params': LPS_STRATEGY_PARAMS,
}

# IF平滑插值参数
IF_SMOOTH_CONFIG = {
    'n_samples': N_SAMPLES,
    'pad_size': 20,
    'smooth_factor': 0.5,
    't_start': 0,
    't_end': 10,
}

# 切片参数
DOWNSAMPLE_FACTOR = 10  # 降采样因子（每10个点取1个）
WINDOW_SIZE = 3072      # 每个样本的长度
HOP_SIZE = 3072         # 步长（无重叠）

# ================================
# 训练超参数
# ================================
BATCH_SIZE = 64
LEARNING_RATE = 0.001
NUM_EPOCHS = 100
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_DIR = "../results"  # 结果保存目录

# ================================
# 模型选择配置
# ================================
# 可选模型:
#   'lwpi'    - 原版LWPI网络 (Baseline)
#   'mslwr'   - 多尺度小波残差网络 (创新模型)
MODEL_TYPE = 'mslwr'  # 修改此处切换模型


# ================================
# 主函数
# ================================
def main():
    # 创建结果保存目录
    os.makedirs(SAVE_DIR, exist_ok=True)
    print(f"使用设备: {DEVICE}")
    print(f"结果保存目录: {SAVE_DIR}")

    # ========== 1. 准备数据 ==========
    dataset = OttawaDataset(DATA_BASE_PATH, fs=FS)

    print("可用数据文件夹:")
    for i, subdir in enumerate(dataset.subdirs):
        print(f"  [{i}] {subdir}")

    # 构建数据集：遍历0-4文件夹，提取IF、降采样、切片
    print("\n开始构建数据集...")
    dataset.build_dataset(
        LPS_CONFIG, IF_SMOOTH_CONFIG,
        window_size=WINDOW_SIZE,
        hop_size=HOP_SIZE,
        downsample_factor=DOWNSAMPLE_FACTOR,
        folder_indices=[0, 1, 2, 3, 4],  # 包含组合故障类
        verbose=True
    )

    # 划分训练集/测试集 (7:3)
    train_size = int(0.7 * len(dataset))
    test_size = len(dataset) - train_size
    train_set, test_set = random_split(dataset, [train_size, test_size])

    # 创建 DataLoader
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

    print(f"\n训练集: {len(train_set)} | 测试集: {len(test_set)}")

    # ========== 2. 初始化模型 ==========
    print(f"\n使用模型: {MODEL_TYPE}")
    
    if MODEL_TYPE == 'lwpi':
        # 原版 LWPI 网络 (Baseline)
        model = LWPINet(
            num_classes=5,  # 5类: 健康/内圈/外圈/滚珠/组合故障
            num_filters=32,
            seq_len=WINDOW_SIZE,
            fs=FS // DOWNSAMPLE_FACTOR  # 降采样后的采样率 20kHz
        ).to(DEVICE)
    elif MODEL_TYPE == 'mslwr':
        # 多尺度小波残差网络 (创新模型)
        model = MSLWRNet(
            num_classes=5,
            num_filters=64,  # 双路径，高低频各32
            seq_len=WINDOW_SIZE,
            fs=FS // DOWNSAMPLE_FACTOR
        ).to(DEVICE)
    else:
        raise ValueError(f"未知模型类型: {MODEL_TYPE}, 可选: 'lwpi', 'mslwr'")

    # ========== 3. 定义 Loss 和 Optimizer ==========
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 学习率衰减 (每30个epoch衰减为原来的0.1)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    # ========== 4. 训练循环 ==========
    best_acc = 0.0
    
    # 用于记录训练过程
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []

    for epoch in range(NUM_EPOCHS):
        # 训练一轮
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, DEVICE
        )

        # 验证一轮
        val_loss, val_acc = evaluate(
            model, test_loader, criterion, DEVICE
        )

        # 更新学习率
        scheduler.step()
        
        # 记录训练数据
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        # 打印进度
        print(f"Epoch [{epoch+1:3d}/{NUM_EPOCHS}] "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            model_path = os.path.join(SAVE_DIR, "best_lwpi_model.pth")
            torch.save(model.state_dict(), model_path)
            print(f"  --> 保存最佳模型 (Acc: {best_acc:.2f}%)")

    print(f"\n训练结束！最佳验证准确率: {best_acc:.2f}%")

    # ========== 5. 绘制训练曲线 ==========
    print("\n生成训练结果可视化...")
    plot_training_history(train_losses, val_losses, train_accs, val_accs, SAVE_DIR)

    # ========== 6. 测试最佳模型并生成混淆矩阵 ==========
    print("\n使用最佳模型进行测试...")
    model_path = os.path.join(SAVE_DIR, "best_lwpi_model.pth")
    model.load_state_dict(torch.load(model_path))
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, if_vals, labels in test_loader:
            inputs = inputs.to(DEVICE)
            if_vals = if_vals.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(inputs, if_vals)
            _, predicted = outputs.max(1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 获取类别名称
    unique_labels = sorted(set(all_labels))
    class_names = [dataset.FAULT_TYPES[i] for i in unique_labels]

    # 绘制混淆矩阵
    plot_confusion_matrix(all_labels, all_preds, class_names, SAVE_DIR)

    # 生成分类报告
    generate_classification_report(all_labels, all_preds, class_names, SAVE_DIR)

    print(f"\n{'='*60}")
    print(f"所有结果已保存至 {SAVE_DIR} 文件夹")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
