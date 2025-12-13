# collect_gesture_data.py - 收集手势训练数据
import asyncio
import json
import websockets
import time
from collections import defaultdict

async def collect_data():
    """从WebSocket收集手势数据"""
    uri = "ws://localhost:8765"
    training_data = defaultdict(lambda: {"sequences": []})

    current_gesture = None
    current_sequence = []

    print("Gesture data collection started")
    print("Commands:")
    print("  start <gesture_name> - Start collecting data for a gesture")
    print("  stop - Stop collecting current gesture")
    print("  save - Save collected data")
    print("  quit - Exit")

    async with websockets.connect(uri) as ws:
        async def send_command():
            while True:
                cmd = input("Command: ").strip().split()
                if not cmd:
                    continue

                if cmd[0] == "start" and len(cmd) > 1:
                    nonlocal current_gesture, current_sequence
                    current_gesture = cmd[1]
                    current_sequence = []
                    print(f"Started collecting data for gesture: {current_gesture}")

                elif cmd[0] == "stop":
                    if current_gesture and current_sequence:
                        training_data[current_gesture]["sequences"].append(current_sequence)
                        print(f"Saved sequence for {current_gesture} ({len(current_sequence)} frames)")
                        current_sequence = []
                    current_gesture = None

                elif cmd[0] == "save":
                    with open('gesture_training_data.json', 'w') as f:
                        json.dump(dict(training_data), f, indent=2)
                    print("Training data saved")

                elif cmd[0] == "quit":
                    break

                await asyncio.sleep(0.1)

        # 启动命令输入任务
        command_task = asyncio.create_task(send_command())

        try:
            async for msg in ws:
                data = json.loads(msg)
                if data.get("type") == "joints" and current_gesture:
                    joints = data["joints"]
                    current_sequence.append(joints)
                    print(f"Frame {len(current_sequence)} collected for {current_gesture}")

        except KeyboardInterrupt:
            pass
        finally:
            command_task.cancel()

if __name__ == "__main__":
    asyncio.run(collect_data())