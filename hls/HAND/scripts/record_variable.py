# Run command: python -m hls.HAND.scripts.record_manual_step
import time
import yaml
import sys
import os

# 路径设置
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from contrl.core3 import OrcaHand


# --- YAML 格式美化技巧 (保持方括号风格) ---
class FlowList(list): pass


def flow_list_representer(dumper, data):
    return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=True)


yaml.add_representer(FlowList, flow_list_representer)


# ---------------------------------------

def main():
    filename = input("请输入保存文件名 (默认: manual_step.yaml): ") or "manual_step.yaml"
    if not filename.endswith(".yaml"):
        filename += ".yaml"

    # 初始化
    hand = OrcaHand('D:\\桌面\\研究生\\INFI_hand\\hls\\HAND\\orcahand_v1_right')
    status = hand.connect()
    if not status[0]:
        print("连接手爪失败。")
        return

    hand.init_joints()
    time.sleep(1)
    hand.disable_torque()  # 关力矩，方便你摆弄手型

    print("\n" + "=" * 50)
    print("【手动分步录制模式】操作说明：")
    print("1. 用手摆好想要的姿势。")
    print("2. 按 [Enter] 键捕获当前姿势。")
    print("3. 输入这一步需要的时间 (秒)，然后按 [Enter]。")
    print("4. 重复上述步骤。")
    print("5. 结束时，在“按 Enter 捕获”的提示下输入 'q' 并回车。")
    print("=" * 50 + "\n")

    waypoints = []
    step_count = 1

    try:
        while True:
            # --- 步骤1：等待用户摆好姿势 ---
            user_input = input(f"[{step_count}] 摆好姿势后按 Enter (输入 q 结束录制): ").strip()

            # 检查是否退出
            if user_input.lower() == 'q':
                break

            # --- 步骤2：获取关节数据 ---
            # 获取数据并保留4位小数
            current_angles = hand.get_joint_pos(as_list=True)
            formatted_angles = [float(f"{x:.4f}") for x in current_angles]
            print(f"   -> 姿势已捕获。")

            # --- 步骤3：询问时间 ---
            valid_duration = False
            duration_val = 1.0  # 默认值

            while not valid_duration:
                # 第一帧通常是起始点，时间其实不重要，给个提示
                if step_count == 1:
                    prompt = "   -> 请输入到达此起始姿势的准备时间 (默认 2.0秒): "
                    default_time = 2.0
                else:
                    prompt = f"   -> 请输入从 [步骤{step_count - 1}] 变到 [步骤{step_count}] 的时间 (秒): "
                    default_time = 1.0

                time_input = input(prompt).strip()

                if time_input == "":
                    duration_val = default_time
                    valid_duration = True
                    print(f"      使用默认时间: {duration_val}s")
                else:
                    try:
                        duration_val = float(time_input)
                        if duration_val <= 0:
                            print("      时间必须大于0，请重新输入。")
                        else:
                            valid_duration = True
                    except ValueError:
                        print("      输入格式错误，请输入数字（例如 0.5 或 1.5）。")

            # --- 步骤4：存入列表 ---
            record_point = {
                "angles": FlowList(formatted_angles),  # 使用 FlowList 强制变成 [ ... ]
                "duration": duration_val
            }
            waypoints.append(record_point)
            print(f"   ✓ 步骤 {step_count} 已保存。\n")
            step_count += 1

    except KeyboardInterrupt:
        print("\n检测到 Ctrl+C，停止录制。")

    # --- 保存文件 ---
    if not waypoints:
        print("未记录任何数据，程序退出。")
        return

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    final_filename = filename.replace(".yaml", f"_{timestamp}.yaml")

    print(f"\n正在保存 {len(waypoints)} 组数据...")
    with open(final_filename, "w") as file:
        # sort_keys=False 保证 angles 在前，duration 在后，好看一点
        yaml.dump({"waypoints": waypoints}, file, sort_keys=False)

    print(f"录制完成！文件已保存至: {final_filename}")
    print("你可以使用 replay_variable_v2.py 来播放此文件。")


if __name__ == "__main__":
    main()