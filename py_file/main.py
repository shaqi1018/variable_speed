# -*- coding: utf-8 -*-
"""
主函数：调用FFT函数处理UO_Bearing数据集中的Channel_1振动数据
"""

import os
import numpy as np
from fft_function import plot_time_and_frequency_domain
from load_data import load_channel1_data


# ================================
# 配置参数
# ================================
# 数据文件夹路径
DATA_BASE_PATH = r"D:\Variable_speed\University_of_Ottawa\Original_data\UO_Bearing"

# 采样频率（University of Ottawa数据集为200kHz）
FS = 200000

# 选择要处理的文件夹索引（0-4，对应5个文件夹）
# 0: 1 Data collected from a healthy bearing
# 1: 2 Data collected from a bearing with inner race fault
# 2: 3 Data collected from a bearing with outer race fault
# 3: 4 Data collected from a bearing with ball fault
# 4: 5 Data collected from a bearing with a combination of faults
SELECTED_FOLDER_INDEX = 1  # 修改这个值来选择要处理的文件夹（0-4）

EXAMPLE_FILE = "I-A-1.mat"  # 修改这个值来选择要处理的文件名


# ================================
# 辅助函数
# ================================
def process_single_file(mat_path, plot=True):
    """
    处理单个.mat文件的Channel_1数据
    
    参数:
        mat_path: .mat文件路径
        plot: 是否绘制时域图和频域图
    
    返回:
        channel1_data: Channel_1的振动数据
    """
    print("\n" + "=" * 60)
    print(f"处理文件: {os.path.basename(mat_path)}")
    print("=" * 60)
    
    channel1_data = load_channel1_data(mat_path)
    
    print(f"\n信号信息:")
    print(f"  信号长度: {len(channel1_data)} 样本 ({len(channel1_data)/FS:.2f} 秒)")
    print(f"  采样频率: {FS} Hz")
    print(f"  奈奎斯特频率: {FS/2} Hz")
    
    if plot:
        time_title = f'时域图 - {os.path.basename(mat_path)} (Channel_1)'
        freq_title = f'频域图 - {os.path.basename(mat_path)} (Channel_1)'
        
        print("\n显示时域图和频域图...")
        plot_time_and_frequency_domain(channel1_data, FS, 
                                      time_title=time_title,
                                      freq_title=freq_title,
                                      max_display_time=None,
                                      max_frequency=None,
                                      yscale='linear')
    
    return channel1_data


def process_directory(directory_path, plot=True):
    """
    处理目录下所有.mat文件的Channel_1数据
    
    参数:
        directory_path: 包含.mat文件的目录路径
        plot: 是否绘制时域图和频域图
    """
    # 查找所有.mat文件
    mat_files = [f for f in os.listdir(directory_path) if f.endswith('.mat')]
    
    if len(mat_files) == 0:
        print(f"在目录 {directory_path} 中未找到.mat文件")
        return
    
    print(f"找到 {len(mat_files)} 个.mat文件")
    
    # 处理每个文件
    results = {}
    for mat_file in mat_files:
        mat_path = os.path.join(directory_path, mat_file)
        try:
            channel1_data = process_single_file(mat_path, plot=plot)
            results[mat_file] = channel1_data
        except Exception as e:
            print(f"处理文件 {mat_file} 时出错: {str(e)}")
            continue
    
    print(f"\n成功处理 {len(results)} 个文件")
    return results


# ================================
# 主函数
# ================================
def main():
    """
    主函数：处理UO_Bearing数据集中的Channel_1数据
    """
    if os.path.exists(DATA_BASE_PATH):
        subdirs = [d for d in os.listdir(DATA_BASE_PATH) 
                  if os.path.isdir(os.path.join(DATA_BASE_PATH, d))]
        
        print(f"找到 {len(subdirs)} 个子目录:")
        for i, subdir in enumerate(subdirs):
            marker = " <-- 当前选择" if i == SELECTED_FOLDER_INDEX else ""
            print(f"  [{i}] {subdir}{marker}")
        
        # 检查索引是否有效
        if len(subdirs) == 0:
            print("未找到任何子目录")
            return
        
        if SELECTED_FOLDER_INDEX < 0 or SELECTED_FOLDER_INDEX >= len(subdirs):
            print(f"\n错误: SELECTED_FOLDER_INDEX ({SELECTED_FOLDER_INDEX}) 超出范围")
            print(f"有效范围: 0 到 {len(subdirs)-1}")
            return
        
        # 处理选定的目录
        target_dir = os.path.join(DATA_BASE_PATH, subdirs[SELECTED_FOLDER_INDEX])
        print(f"\n处理目录: {subdirs[SELECTED_FOLDER_INDEX]}")
        process_directory(target_dir, plot=True)
    else:
        print(f"数据路径不存在: {DATA_BASE_PATH}")
        print("请修改 DATA_BASE_PATH 变量为正确的路径")


if __name__ == "__main__":
    if os.path.exists(DATA_BASE_PATH):
        subdirs = [d for d in os.listdir(DATA_BASE_PATH) 
                  if os.path.isdir(os.path.join(DATA_BASE_PATH, d))]
        
        if len(subdirs) > 0 and SELECTED_FOLDER_INDEX < len(subdirs):
            example_file = os.path.join(DATA_BASE_PATH, subdirs[SELECTED_FOLDER_INDEX], EXAMPLE_FILE)
            
            if os.path.exists(example_file):
                print("=" * 60)
                print("示例：处理单个文件")
                print("=" * 60)
                process_single_file(example_file, plot=True)
            else:
                print(f"示例文件不存在: {example_file}")
                print(f"请检查 SELECTED_FOLDER_INDEX ({SELECTED_FOLDER_INDEX}) 和 EXAMPLE_FILE ({EXAMPLE_FILE})")
        else:
            print(f"文件夹索引 {SELECTED_FOLDER_INDEX} 无效")
            print(f"有效范围: 0 到 {len(subdirs)-1}")
    else:
        print(f"数据路径不存在: {DATA_BASE_PATH}")
