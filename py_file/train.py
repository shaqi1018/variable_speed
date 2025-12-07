# -*- coding: utf-8 -*-
"""
训练模块
包含单个Epoch训练和验证的逻辑
"""

import torch


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    训练一个 Epoch

    参数:
        model: LWPI网络模型
        dataloader: 训练数据加载器
        criterion: 损失函数 (CrossEntropyLoss)
        optimizer: 优化器 (Adam/SGD)
        device: 设备 (cuda/cpu)

    返回:
        epoch_loss: 平均损失
        epoch_acc: 准确率 (%)
    """
    model.train()  # 开启训练模式 (BN更新, Dropout启用)

    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, if_vals, labels) in enumerate(dataloader):
        # 1. 数据搬运到设备
        inputs = inputs.to(device)      # (B, 1, 3072)
        if_vals = if_vals.to(device)    # (B,)
        labels = labels.to(device)      # (B,) long类型

        # 2. 梯度清零
        optimizer.zero_grad()

        # 3. 前向传播 (LWPI需要同时传入信号和IF)
        outputs = model(inputs, if_vals)  # (B, num_classes)

        # 4. 计算损失
        loss = criterion(outputs, labels)

        # 5. 反向传播
        loss.backward()

        # 6. 参数更新
        optimizer.step()

        # 7. 统计
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    # 计算平均损失和准确率
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100.0 * correct / total

    return epoch_loss, epoch_acc


def evaluate(model, dataloader, criterion, device):
    """
    验证/测试模型

    参数:
        model: LWPI网络模型
        dataloader: 验证/测试数据加载器
        criterion: 损失函数
        device: 设备

    返回:
        val_loss: 平均损失
        val_acc: 准确率 (%)
    """
    model.eval()  # 开启评估模式 (BN冻结, Dropout禁用)

    running_loss = 0.0
    correct = 0
    total = 0

    # 不计算梯度，节省显存
    with torch.no_grad():
        for inputs, if_vals, labels in dataloader:
            # 数据搬运
            inputs = inputs.to(device)
            if_vals = if_vals.to(device)
            labels = labels.to(device)

            # 前向传播
            outputs = model(inputs, if_vals)
            loss = criterion(outputs, labels)

            # 统计
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    val_loss = running_loss / len(dataloader)
    val_acc = 100.0 * correct / total

    return val_loss, val_acc