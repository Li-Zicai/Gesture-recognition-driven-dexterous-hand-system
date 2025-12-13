# Run command: python -m hls.HAND.scripts.replay_smooth
import time
import yaml
import argparse
import sys
import os
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from hls.HAND.contrl.core3 import OrcaHand

# --- 核心参数调整 ---

# 过渡比例 (0.1 ~ 0.5)
# 0.3 表示：当跑完 70% 的路程（还剩 30%）时，就立刻执行下一步
# 值越大，动作越连贯（圆滑），但越容易“抄近道”导致动作不到位
# 值越小，动作越精准，但顿挫感越强
TRANSITION_RATIO = 0.3

# 最小过渡角度 (度)
# 如果动作很小，至少要保留这个角度的误差就开始切换，防止卡顿
MIN_TRANSITION_ANGLE = 10.0

# 严格停止阈值 (仅用于最后一个点)
FINAL_STOP_THRESHOLD = 5.0


def get_max_distance(current_angles, target_angles):
    """计算所有关节中，距离目标最远的那一个关节的距离"""
    diffs = []
    for c, t in zip(current_angles, target_angles):
        if c is not None and t is not None:
            diffs.append(abs(c - t))
    return max(diffs) if diffs else 0


def main():
    parser = argparse.ArgumentParser(description='Replay with Smooth Transition')
    parser.add_argument('model_path', type=str, help='Path to model')
    args = parser.parse_args()

    # --- 文件读取 ---
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
    if not points: return

    hand = OrcaHand(args.model_path)
    if not hand.connect()[0]: return
    hand.enable_torque()

    print(f"\n>>> 开始丝滑播放 (过渡比例: {TRANSITION_RATIO * 100}%)")
    print(">>> 注意：中间动作将不再完全停稳，直接平滑过渡")

    # 1. 复位
    start_pose = points[0]["angles"]
    hand.set_joint_pos(start_pose, num_steps=1, speed=100, acceleration=50)
    time.sleep(2.0)

    try:
        while True:
            print("\n--- 循环开始 ---")

            for i in range(len(points)):
                point = points[i]
                target_angles = point["angles"]
                speed_val = point.get("speed", 400)

                # 判断是否是这一组动作的最后一步
                is_last_point = (i == len(points) - 1)

                # 1. 获取起始位置，计算这一步的总行程 (为了计算何时切换)
                start_angles = hand.get_joint_pos(as_list=True)
                total_distance = get_max_distance(start_angles, target_angles)

                # 2. 计算触发切换的阈值
                if is_last_point:
                    # 最后一步：必须严格停稳
                    pass_threshold = FINAL_STOP_THRESHOLD
                    print(f"步骤 {i + 1} (终点) | 速度:{speed_val} | 等待停稳...")
                else:
                    # 中间步骤：计算动态阈值
                    # 阈值 = 总距离 * 30%，或者是最小10度，取大者
                    # 意思就是：只要进入这个范围内，马上发下一条指令
                    pass_threshold = max(total_distance * TRANSITION_RATIO, MIN_TRANSITION_ANGLE)
                    print(f"步骤 {i + 1} | 速度:{speed_val} | 距离:{total_distance:.1f}° | 剩余 {pass_threshold:.1f}° 时切换")

                # 3. 发送指令 (极速加速度)
                hand.set_joint_pos(target_angles, num_steps=1, speed=speed_val, acceleration=0)

                # 4. 智能等待 (Transition Logic)
                start_wait = time.time()
                while True:
                    # 实时获取误差
                    current_angles = hand.get_joint_pos(as_list=True)
                    current_error = get_max_distance(current_angles, target_angles)

                    # 满足切换条件？
                    if current_error < pass_threshold:
                        # 到了切换点，直接 break，去执行下一次循环(发送下一个指令)
                        break

                    # 超时保护 (防止卡死)
                    if time.time() - start_wait > 3.0:
                        break

                    # 极速检查，减少延迟
                    time.sleep(0.01)

            # 循环之间的停顿 (可选)
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\n停止")
    finally:
        hand.disable_torque()


if __name__ == "__main__":
    main()