# Run command: python -m hls.HAND.scripts.replay_fast_v3 /home/dw/INFI_hand/hls/HAND/orcahand_v1_right
import time
import yaml
import argparse
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from hls.HAND.contrl.core3 import OrcaHand


def main():
    parser = argparse.ArgumentParser(description='Replay with smart speed control')
    # 将默认 step_time 稍微调大到 0.02s，减少通信拥堵，人眼看不出区别
    parser.add_argument('--step_time', type=float, default=0.02,
                        help='Control loop timestep (default: 0.02)')
    parser.add_argument('model_path', type=str,
                        help='Path to the orcahand model folder')
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    filename_input = input("请输入录制的文件名: ").strip()

    # --- 路径查找逻辑 ---
    if os.path.exists(filename_input):
        file_path = filename_input
    elif os.path.exists(os.path.join(script_dir, filename_input)):
        file_path = os.path.join(script_dir, filename_input)
    else:
        test_path = os.path.join(script_dir, filename_input + ".yaml")
        if os.path.exists(test_path):
            file_path = test_path
        else:
            print(f"错误: 找不到文件 {filename_input}")
            return

    try:
        with open(file_path, "r") as file:
            replay_data = yaml.safe_load(file)
    except Exception as e:
        print(f"读取出错: {e}")
        return

    points = replay_data.get("waypoints", [])
    if not points:
        print("未找到 waypoints 数据。")
        return

    # 初始化
    hand = OrcaHand(args.model_path)
    status = hand.connect()
    if not status[0]:
        print(f"连接失败: {status[1]}")
        return

    hand.enable_torque()

    # 设置一个较高的速度和加速度参数，解除电机内部的限制
    # 注意：这些参数可以在 set_joint_pos 中传递
    HIGH_SPEED = 1000  # 给一个较大的值
    HIGH_ACCEL = 0  # 0 通常代表最大加速度（无限制）或根据具体电机协议设定为254

    print(">>> 正在复位到起始点...")
    start_pose = points[0]["angles"]
    hand.set_joint_pos(start_pose, num_steps=100, step_size=0.01)
    time.sleep(1.0)

    try:
        while True:
            print(">>> 开始新循环")

            for i in range(len(points) - 1):
                curr_point = points[i]
                next_point = points[i + 1]

                start_angles = curr_point["angles"]
                end_angles = next_point["angles"]
                target_duration = max(next_point.get("duration", 0.04), 0.005)

                start_time = time.time()

                # --- 核心改进逻辑 ---

                # 情况A：极速动作 (时间小于 0.25秒)
                # 这种情况下，插值没有意义，还会导致通信堵塞。
                # 直接发送终点命令，让电机全速执行。
                if target_duration < 0.25:
                    # speed=0 或 HIGH_SPEED 让电机全速
                    hand.set_joint_pos(end_angles, speed=HIGH_SPEED, acceleration=HIGH_ACCEL)

                    # 等待剩余时间，确保节奏对齐
                    elapsed = time.time() - start_time
                    remaining = target_duration - elapsed
                    if remaining > 0:
                        time.sleep(remaining)

                # 情况B：平滑慢动作 (时间较长)
                # 使用插值来保证动作细腻
                else:
                    n_steps = int(target_duration / args.step_time)
                    if n_steps < 1: n_steps = 1

                    for step in range(1, n_steps + 1):
                        t = step / n_steps
                        pose = [(1 - t) * s + t * e for s, e in zip(start_angles, end_angles)]

                        # 关键修改：在这里也传入高速参数
                        # 这样电机就会紧跟我们的插值点，不会因为内部PID限速而拖后腿
                        hand.set_joint_pos(pose, speed=HIGH_SPEED, acceleration=HIGH_ACCEL)

                        # 时间控制
                        loop_elapsed = time.time() - start_time
                        expected = step * (target_duration / n_steps)
                        sleep_needed = expected - loop_elapsed

                        if sleep_needed > 0:
                            time.sleep(sleep_needed)

    except KeyboardInterrupt:
        print("\n中断。")

    finally:
        hand.disable_torque()
        print("结束。")


if __name__ == "__main__":
    main()