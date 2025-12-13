# Run command: python -m hls.HAND.scripts.record_speed
import time
import yaml
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from contrl.core3 import OrcaHand


# --- YAML 格式美化 (方括号风格) ---
class FlowList(list): pass


def flow_list_representer(dumper, data):
    return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=True)


yaml.add_representer(FlowList, flow_list_representer)


# -------------------------------

def main():
    filename = input("请输入保存文件名 (默认: speed_motion.yaml): ") or "speed_motion.yaml"
    if not filename.endswith(".yaml"):
        filename += ".yaml"

    hand = OrcaHand('D:\桌面\Sustech\INFI_hand\hls\HAND\orcahand_v1_right')
    status = hand.connect()
    if not status[0]:
        print("连接失败")
        return

    hand.init_joints()
    time.sleep(1)
    hand.disable_torque()

    print("\n" + "=" * 50)
    print("【速度控制模式录制】")
    print("1. 摆好姿势 -> 按 Enter")
    print("2. 输入前往该姿势的速度值 (范围 1-1000)")
    print("   (参考: 50=很慢, 200=中速, 800=很快, 0=极速)")
    print("3. 输入 'q' 结束")
    print("=" * 50 + "\n")

    waypoints = []
    step_count = 1

    try:
        while True:
            # 1. 捕获姿势
            user_input = input(f"[{step_count}] 摆好姿势按 Enter (q 结束): ").strip()
            if user_input.lower() == 'q':
                break

            current_angles = hand.get_joint_pos(as_list=True)
            formatted_angles = [float(f"{x:.4f}") for x in current_angles]
            print(f"   -> 姿势已记录")

            # 2. 输入速度
            valid_speed = False
            speed_val = 200  # 默认中速

            while not valid_speed:
                if step_count == 1:
                    prompt = "   -> 起始复位速度 (默认 100): "
                    default_speed = 100
                else:
                    prompt = f"   -> 下一步的运行速度 (1-1000, 默认300): "
                    default_speed = 300

                sp_input = input(prompt).strip()

                if sp_input == "":
                    speed_val = default_speed
                    valid_speed = True
                else:
                    try:
                        speed_val = int(sp_input)
                        if speed_val < 0 or speed_val > 1000:
                            print("      范围错误，请输入 0-1000 之间的整数")
                        else:
                            valid_speed = True
                    except ValueError:
                        print("      请输入整数")

            # 3. 保存 (Duration 字段被改为 speed)
            record_point = {
                "angles": FlowList(formatted_angles),
                "speed": speed_val  # <--- 这里改成了 speed
            }
            waypoints.append(record_point)
            print(f"   ✓ 步骤 {step_count} 保存 (速度: {speed_val})\n")
            step_count += 1

    except KeyboardInterrupt:
        print("\n中断")

    if not waypoints:
        return

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    final_filename = filename.replace(".yaml", f"_{timestamp}.yaml")

    with open(final_filename, "w") as file:
        # 保存时，angles用方括号，speed在后面
        yaml.dump({"waypoints": waypoints}, file, sort_keys=False)

    print(f"文件已保存: {final_filename}")


if __name__ == "__main__":
    main()