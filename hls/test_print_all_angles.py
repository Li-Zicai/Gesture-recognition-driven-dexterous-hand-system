#!/usr/bin/env python
import time
from scservo_sdk import PortHandler, hls, COMM_SUCCESS

PORT = "COM8"  # 端口路径
BAUDRATE = 1000000  # 波特率
MOTOR_IDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]  # 所有电机ID
JOINT_NAMES = [  # 与motor_ids对应的关节名称（来自config.yaml的joint_ids）
    "thumb_mcp", "thumb_abd", "thumb_pip", "thumb_dip",
    "index_abd", "index_mcp", "index_pip", "middle_abd",
    "middle_mcp", "middle_pip", "ring_abd", "ring_mcp",
    "ring_pip", "pinky_abd", "pinky_mcp", "pinky_pip", "wrist"
]


def main():
    # 初始化端口处理器
    port_handler = PortHandler(PORT)

    # 初始化HLS协议包处理器
    packet_handler = hls(port_handler)

    # Open port
    if port_handler.openPort():
        print("Succeeded to open the port")
    else:
        print("Failed to open the port")
        quit()

    # Set port baudrate 1000000
    if port_handler.setBaudRate(1000000):
        print("Succeeded to change the baudrate")
    else:
        print("Failed to change the baudrate")
        quit()

    try:
        print("开始读取所有关节的位置和速度（按Ctrl+C停止）...")
        print("-" * 80)
        while True:
            # 遍历所有电机ID，读取并打印信息
            for motor_id, joint_name in zip(MOTOR_IDS, JOINT_NAMES):
                # 调用ReadPosSpeed获取位置、速度及通信状态
                pos, speed, comm_result, error = packet_handler.ReadPosSpeed(motor_id)

                # 处理通信结果
                if comm_result != COMM_SUCCESS:
                    print(f"关节 {joint_name} (ID:{motor_id}) 通信失败: {packet_handler.getTxRxResult(comm_result)}")
                elif error != 0:
                    print(f"关节 {joint_name} (ID:{motor_id}) 错误: {packet_handler.getRxPacketError(error)}")
                else:
                    print(f"关节 {joint_name} (ID:{motor_id}): 位置={pos} 速度={speed}")

            print("-" * 80)
            time.sleep(0.5)  # 每0.5秒刷新一次

    except KeyboardInterrupt:
        print("\n用户终止程序")

    finally:
        # 关闭端口
        port_handler.closePort()
        print("端口已关闭")


if __name__ == "__main__":
    main()