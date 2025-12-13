# Run command: python -m hls.HAND.scripts.replay_variable_v2 /home/dw/INFI_hand/hls/HAND/orcahand_v1_right
import time
import yaml
import argparse
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from hls.HAND.contrl.core3 import OrcaHand


def main():
    parser = argparse.ArgumentParser(description='Replay hand movements with variable speed')
    # step_time 是发送指令的频率（插值密度），0.01s 比较平滑
    parser.add_argument('--step_time', type=float, default=0.01,
                        help='Control loop timestep (default: 0.01)')
    parser.add_argument('model_path', type=str,
                        help='Path to the orcahand model folder')
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))

    # 获取文件名
    filename_input = input("请输入变速录制的文件名: ").strip()

    # 智能查找路径
    if os.path.exists(filename_input):
        file_path = filename_input
    elif os.path.exists(os.path.join(script_dir, filename_input)):
        file_path = os.path.join(script_dir, filename_input)
    else:
        # 尝试自动补全 .yaml
        test_path = os.path.join(script_dir, filename_input + ".yaml")
        if os.path.exists(test_path):
            file_path = test_path
        else:
            print(f"错误: 找不到文件 {filename_input}")
            return

    print(f"正在读取: {file_path}")
    try:
        with open(file_path, "r") as file:
            replay_data = yaml.safe_load(file)
    except Exception as e:
        print(f"读取文件出错: {e}")
        return

    # 提取数据
    points = replay_data.get("waypoints", [])
    if not points:
        print("错误：文件中没有找到 'waypoints' 数据。")
        return

    # 自动识别数据格式
    # 格式1：新版变速格式 (list of dicts) -> [{"angles": [...], "duration": 0.04}, ...]
    # 格式2：旧版格式 (list of lists) -> [[...], [...]]
    if isinstance(points[0], dict) and "angles" in points[0]:
        print(f"--> 检测到【变速轨迹】数据，共 {len(points)} 帧。")
        mode = "variable"
    else:
        print("--> 检测到【旧版固定速度】数据，将按固定速度播放。")
        mode = "fixed"
        # 转换旧格式以兼容下方逻辑
        points = [{"angles": p, "duration": 0.05} for p in points]

    hand = OrcaHand(args.model_path)
    status = hand.connect()
    if not status[0]:
        print(f"连接手爪失败: {status[1]}")
        return

    hand.enable_torque()
    print("力矩已开启。")

    # --- 1. 复位到起始点 ---
    print("正在缓慢复位到起始位置...")
    start_pose = points[0]["angles"]
    # 这里的 num_steps=100 是为了让手爪慢慢走到起点，防止突然跳变
    hand.set_joint_pos(start_pose, num_steps=100, step_size=0.01)
    time.sleep(1.0)

    # --- 2. 循环播放 ---
    try:
        while True:
            print(f">>> 开始播放循环 (模式: {'变速' if mode == 'variable' else '固定'})")

            for i in range(len(points) - 1):
                curr_point = points[i]
                next_point = points[i + 1]

                start_angles = curr_point["angles"]
                end_angles = next_point["angles"]

                # 获取这一段路程应有的持续时间
                # 如果是旧版数据或者录制时卡顿了，给一个最小保护值 0.005s
                duration = max(next_point.get("duration", 0.04), 0.005)

                # 计算插值步数：总时间 / 每一步的时间
                # 例如：这一段要动0.04秒，每步0.01秒 -> 分4步走完
                n_steps = int(duration / args.step_time)
                if n_steps < 1: n_steps = 1

                segment_start_time = time.time()

                # 插值执行
                for step in range(1, n_steps + 1):
                    t = step / n_steps

                    # 线性插值
                    pose = [(1 - t) * s + t * e for s, e in zip(start_angles, end_angles)]
                    hand.set_joint_pos(pose)

                    # 严格的时间控制
                    elapsed = time.time() - segment_start_time
                    expected = step * (duration / n_steps)
                    sleep_needed = expected - elapsed

                    if sleep_needed > 0:
                        time.sleep(sleep_needed)

    except KeyboardInterrupt:
        print("\n播放被用户中断。")

    finally:
        hand.disable_torque()
        print("力矩已关闭，程序结束。")


if __name__ == "__main__":
    main()