# collect_gesture_data.py - 收集手势训练数据
import asyncio
import json
import websockets
import time
import os
from collections import defaultdict
from datetime import datetime

class GestureDataCollector:
    """手势数据收集器"""
    
    def __init__(self, ws_uri: str = "ws://localhost:8765", output_file: str = "gesture_training_data.json"):
        self.ws_uri = ws_uri
        self.output_file = output_file
        self.training_data = defaultdict(lambda: {"sequences": []})
        self.current_gesture = None
        self.current_sequence = []
        self.frame_count = 0
        self.gesture_count = defaultdict(int)
        self.auto_stop_event = None  # 用于自动停止
        
        # 加载现有数据
        self._load_existing_data()
    
    def _load_existing_data(self):
        """加载现有的训练数据"""
        if os.path.exists(self.output_file):
            try:
                with open(self.output_file, 'r') as f:
                    self.training_data = defaultdict(lambda: {"sequences": []}, json.load(f))
                    # 统计现有手势数量
                    for gesture, data in self.training_data.items():
                        self.gesture_count[gesture] = len(data["sequences"])
                    print(f"✓ Loaded existing training data from {self.output_file}")
                    print(f"  Current gestures: {dict(self.gesture_count)}\n")
            except Exception as e:
                print(f"⚠ Failed to load existing data: {e}\n")
    
    def _print_menu(self):
        """打印菜单和统计信息"""
        print("\n" + "="*60)
        print("🎯 手势数据收集工具")
        print("="*60)
        print("\n📊 当前收集进度:")
        if self.gesture_count:
            for gesture, count in sorted(self.gesture_count.items()):
                print(f"  - {gesture}: {count} 个序列")
        else:
            print("  (还未收集任何数据)")
        
        print("\n📝 可用命令:")
        print("  start <gesture_name>           - 开始收集某个手势的数据")
        print("  start <gesture_name> <seconds> - 自动收集 N 秒后停止 (推荐: 5-10秒)")
        print("  quick <gesture1> <gesture2>... - 快速模式：连续收集多个手势（每个10秒）")
        print("  stop                           - 停止当前手势的收集")
        print("  [按空格]                       - 快速停止（收集中时按空格立即停止）")
        print("  list                           - 列出所有已收集的手势")
        print("  clear <gesture_name>           - 清除某个手势的所有数据")
        print("  save                           - 保存所有收集的数据")
        print("  info                           - 显示当前统计信息")
        print("  help                           - 显示帮助信息")
        print("  quit                           - 退出程序\n")
    
    async def run(self):
        """运行数据收集器"""
        print(f"✓ 已连接到 {self.ws_uri}")
        print("\n" + "="*60)
        print("🎯 手势数据收集工具 - 简易模式")
        print("="*60)
        print("\n📝 使用方法:")
        print("  1. 输入动作名称（中英文都可以），然后按回车")
        print("  2. 脚本会自动收集 500 帧")
        print("  3. 达到 500 帧后，继续输入下一个动作名称")
        print("  4. 输入 'q' 并回车完成收集\n")
        
        try:
            async with websockets.connect(self.ws_uri) as ws:
                # 接收WebSocket消息的任务
                async def receive_messages():
                    try:
                        async for msg in ws:
                            try:
                                data = json.loads(msg)
                                # 监听服务器广播的原始前端数据
                                if data.get("type") == "raw_joints" and self.current_gesture:
                                    joints = data["joints"]
                                    self.current_sequence.append(joints)
                                    self.frame_count += 1
                            except json.JSONDecodeError:
                                pass
                    except asyncio.CancelledError:
                        pass
                
                # 启动消息接收任务
                receive_task = asyncio.create_task(receive_messages())
                
                # 主循环：等待用户输入动作名
                loop = asyncio.get_event_loop()
                
                while True:
                    # 等待用户输入
                    gesture_name = await loop.run_in_executor(None, input, "输入动作名称 (q=完成): ")
                    gesture_name = gesture_name.strip()
                    
                    if gesture_name.lower() == 'q':
                        print("\n✅ 数据收集完成!")
                        break
                    
                    if not gesture_name:
                        print("⚠️  请输入有效的动作名称\n")
                        continue
                    
                    # 开始收集该动作
                    self.current_gesture = gesture_name
                    self.current_sequence = []
                    self.frame_count = 0
                    
                    print(f"\n✓ 开始收集: {gesture_name}")
                    print(f"  ⏱️  收集 500 帧中... ", end="", flush=True)
                    
                    # 持续收集直到达到500帧
                    last_frame_count = 0
                    while self.frame_count < 500 and self.current_gesture:
                        # 每0.1秒检查一次帧数变化
                        await asyncio.sleep(0.1)
                        if self.frame_count > last_frame_count:
                            remaining = 500 - self.frame_count
                            print(f"\r  ⏱️  收集 500 帧中... [{self.frame_count}/500] ", end="", flush=True)
                            last_frame_count = self.frame_count
                    
                    # 停止收集
                    if self.current_sequence:
                        self.training_data[gesture_name]["sequences"].append(self.current_sequence)
                        self.gesture_count[gesture_name] += 1
                        print(f"\n✓ 已保存 {gesture_name} 的一个序列 ({len(self.current_sequence)} 帧)\n")
                    else:
                        print(f"\n⚠️  未收集到任何数据\n")
                    
                    self.current_gesture = None
                    self.current_sequence = []
                
                # 取消接收任务
                receive_task.cancel()
                
                # 保存数据
                if self.gesture_count:
                    self._save_data()
                
        except ConnectionRefusedError:
            print("❌ 错误：无法连接到服务器")
            print("   请确保后端服务器已启动：python -m hls.HAND.scripts.hand_netserver")
        except Exception as e:
            print(f"❌ 连接错误: {e}")
    
    async def _auto_stop_timer(self, duration: float):
        """自动停止计时器"""
        try:
            for i in range(int(duration * 2)):  # 每0.5秒检查一次
                await asyncio.sleep(0.5)
                if not self.current_gesture:
                    break
            
            # 时间到，自动停止
            if self.current_gesture:
                self._stop_collecting()
        except asyncio.CancelledError:
            pass
    
    async def _auto_stop_timer(self, duration: float):
        """自动停止计时器"""
        try:
            for i in range(int(duration * 2)):  # 每0.5秒检查一次
                await asyncio.sleep(0.5)
                if not self.current_gesture:
                    break
            
            # 时间到，自动停止
            if self.current_gesture:
                self._stop_collecting()
        except asyncio.CancelledError:
            pass
    
    def _stop_collecting(self):
        """停止收集"""
        if self.current_gesture:
            if self.current_sequence:
                self.training_data[self.current_gesture]["sequences"].append(self.current_sequence)
                self.gesture_count[self.current_gesture] += 1
                print(f"\n✓ 已保存 {self.current_gesture} 的一个序列 ({len(self.current_sequence)} 帧)")
            else:
                print(f"\n⚠ 没有收集到任何帧数据")
            self.current_sequence = []
            self.current_gesture = None
            self.frame_count = 0
    
    async def _handle_commands(self):
        """处理用户命令（已弃用，改用run()中的主循环）"""
        pass
    
    def _save_data(self):
        """保存数据到JSON文件"""
        try:
            # 将defaultdict转换为普通dict
            data_to_save = {k: v for k, v in self.training_data.items()}
            
            with open(self.output_file, 'w', encoding='utf-8') as f:
                json.dump(data_to_save, f, indent=2, ensure_ascii=False)
            
            total_sequences = sum(len(v["sequences"]) for v in data_to_save.values())
            print(f"\n✓ 数据已保存到 {self.output_file}")
            print(f"  - 手势类别: {len(data_to_save)}")
            print(f"  - 总序列数: {total_sequences}")
            print(f"  - 时间戳: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        except Exception as e:
            print(f"❌ 保存失败: {e}")

async def main():
    """主函数"""
    collector = GestureDataCollector()
    await collector.run()

if __name__ == "__main__":
    asyncio.run(main())