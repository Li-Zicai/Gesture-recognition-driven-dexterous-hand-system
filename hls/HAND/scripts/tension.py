import argparse
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from contrl.core3 import OrcaHand

def main():
    parser = argparse.ArgumentParser(
        description="Enable torque and hold tension on the ORCA Hand. "
                    "Specify the path to the orcahand model folder."
    )
    parser.add_argument(
        "model_path",
        type=str,
        help="Path to the orcahand model folder (e.g., /path/to/orcahand_v1_left)"
    )
    args = parser.parse_args()

    hand = OrcaHand(args.model_path)
    status = hand.connect()
    print(status)
    if not status[0]:
        print("Failed to connect to the hand.")
        exit(1)

    hand.enable_torque()
    print("Torque enabled. Press Ctrl+C to exit.")

    try:
        while True:
            pass
    except KeyboardInterrupt:
        print("\nExiting. Disabling torque.")
        hand.disable_torque()

if __name__ == "__main__":
    main()

# 程序执行流程：

#    解析命令行参数获取模型路径
#    创建 OrcaHand 对象并尝试连接
#    检查连接状态，失败则退出
#    启用扭矩控制，机械臂进入工作状态
#    进入无限循环等待用户操作
#    用户按 Ctrl+C 时捕获中断信号
#    关闭扭矩控制，安全释放机械臂

# 这个程序是控制 ORCA 机械臂的基础框架