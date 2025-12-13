# Run command: python -m hls.HAND.scripts.replay_speed /home/dw/INFI_hand/hls/HAND/orcahand_v1_right
import time
import yaml
import argparse
import sys
import os
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from hls.HAND.contrl.core3 import OrcaHand


def wait_for_arrival(hand, target_angles, threshold=2, timeout=5.0):
    """
    智能等待函数：不断检查当前位置是否接近目标位置
    threshold: 允许的误差范围 (弧度/度)
    timeout: 超时强制跳过，防止死循环
    """
    start_wait = time.time()
    while time.time() - start_wait < timeout:
        # 获取当前所有关节位置
        current_angles = hand.get_joint_pos(as_list=True)

        # 计算最大误差
        diffs = [abs(c - t) for c, t in zip(current_angles, target_angles)]
        max_diff = max(diffs)

        # 如果所有关节误差都小于阈值，说明到了
        if max_diff < threshold:
            return True

        time.sleep(0.05)# 设置短暂延时，避免频繁访问

    print("Warning: 动作超时，强制进入下一步")
    return False


def main():
    parser = argparse.ArgumentParser(description='Replay with direct speed control')
    parser.add_argument('model_path', type=str, help='Path to model')
    args = parser.parse_args()

    # --- 文件读取逻辑 ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    filename_input = input("请输入速度模式录制的文件名: ").strip()

    if os.path.exists(filename_input):
        file_path = filename_input
    elif os.path.exists(os.path.join(script_dir, filename_input)):
        file_path = os.path.join(script_dir, filename_input)
    else:
        test = os.path.join(script_dir, filename_input + ".yaml")
        file_path = test if os.path.exists(test) else None

    if not file_path:
        print("找不到文件")
        return

    with open(file_path, "r") as file:
        data = yaml.safe_load(file)
    points = data.get("waypoints", [])

    if not points:
        print("无数据")
        return

    # --- 初始化 ---
    hand = OrcaHand(args.model_path)
    if not hand.connect()[0]:  #连接硬件
        print("连接失败")
        return
    hand.enable_torque()  #启用扭矩
    print("\n>>> 开始流畅播放 (硬件插值模式)")

    #----开始复位到起点------#
    start_pose = points[0]["angles"]
    # 复位时用较慢速度安全归位
    hand.set_joint_pos(start_pose, num_steps=1, speed=100,acceleration=50)
    time.sleep(2.0)

    try:
        while True:
            for i in range(len(points)):
                # 获取当前帧数据
                point = points[i]
                target_angles = point["angles"]

                # 获取该帧设定的速度 (如果没有则默认200)
                speed_val = point.get("speed", 200)

                print(f"执行步骤 {i + 1}/{len(points)} | 速度: {speed_val}")

                # --- 核心修改 ---
                # num_steps=1: 关键！禁止电脑插值，只发一次指令
                # speed=speed_val: 把速度直接传给电机
                hand.set_joint_pos(target_angles, num_steps=1, speed=speed_val,acceleration=0)

                # --- 等待电机跑完 ---
                # 因为我们只发了一次指令，程序瞬间就执行完了
                # 所以必须在这里卡住，直到手爪物理上真的跑到了位置
                # 下一个动作才会开始

                # 跳过第一个点的等待（因为前面已经复位过了）
                if i > 0:
                    wait_for_arrival(hand, target_angles)
                else:
                    time.sleep(0.5)

            print("--- 循环 ---\n")
            time.sleep(0.5)

    except KeyboardInterrupt:
        print("停止")
    finally:
        hand.disable_torque()


if __name__ == "__main__":
    main()