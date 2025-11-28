#!/usr/bin/env python3
"""
视频帧抽取工具
从指定视频中抽取特定时间点的帧并保存为图片
"""

import cv2
import os
import sys
from pathlib import Path
from typing import List, Tuple


def extract_frames_at_times(video_path: str, output_dir: str, time_points: List[Tuple[int, int]]) -> List[str]:
    """
    从视频中抽取指定时间点的帧
    
    Args:
        video_path: 视频文件路径
        output_dir: 输出目录路径
        time_points: 时间点列表，格式为[(分钟, 秒), ...]
    
    Returns:
        成功保存的图片文件路径列表
    """
    # 确保输出目录存在
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频文件: {video_path}")
    
    # 获取视频帧率
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        raise ValueError("无法获取视频帧率")
    
    saved_files = []
    video_name = Path(video_path).stem
    
    try:
        for minutes, seconds in time_points:
            # 计算总秒数和对应的帧数
            total_seconds = minutes * 60 + seconds
            frame_number = int(total_seconds * fps)
            
            # 跳转到指定帧
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            
            # 读取帧
            ret, frame = cap.read()
            if not ret:
                print(f"警告: 无法读取第 {minutes}分{seconds}秒 的帧")
                continue
            
            # 生成输出文件名
            timestamp = f"{minutes:02d}_{seconds:02d}"
            filename = f"{video_name}_{timestamp}.jpg"
            output_file = output_path / filename
            
            # 保存图片
            success = cv2.imwrite(str(output_file), frame)
            if success:
                saved_files.append(str(output_file))
                print(f"已保存: {output_file}")
            else:
                print(f"警告: 无法保存文件 {output_file}")
    
    finally:
        cap.release()
    
    return saved_files


def main():
    """主函数"""
    # 直接在代码中指定参数
    video_path = "/home/darwin/projects/Lab/有尘_II020618回风顺槽回风绕道掘进面T1朝向迎头_20250820190630-20250820191650_1.mp4"  # 视频文件路径
    output_dir = "output/extracted_frames"  # 输出目录
    
    # 指定时间点: 40秒、1分08秒、1分47秒
    time_points = [(0, 40), (1, 8), (1, 47)]
    
    # 检查视频文件是否存在
    if not os.path.exists(video_path):
        print(f"错误: 视频文件不存在: {video_path}")
        sys.exit(1)
    
    try:
        # 抽取帧
        saved_files = extract_frames_at_times(video_path, output_dir, time_points)
        
        if saved_files:
            print(f"\n成功抽取 {len(saved_files)} 帧:")
            for file in saved_files:
                print(f"  - {file}")
        else:
            print("\n没有成功保存任何帧")
            
    except Exception as e:
        print(f"错误: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
