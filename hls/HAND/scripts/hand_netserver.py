import asyncio
import json
import websockets
import sys
import os
from typing import List

# 新增手势匹配导入
from .gesture_matcher import GestureMatcher, MLGestureMatcher

# ---- 配置 ----
HOST = "0.0.0.0"
PORT = 8765
QUEUE_MAXSIZE = 10
# 旧的全局 min/max
HARDWARE_MIN = 0.0
HARDWARE_MAX = 180.0

# 前端输出角度范围（默认 0-180）
INPUT_MIN = 0.0
INPUT_MAX = 180.0

# 每个关节对应的硬件最小/最大角度（顺序应与重排序后的一致）
MIN_ANGLES = [
    -61.6006, -43.2354, 128.002, 97.6487, -50.1023, 101.9852, 148.8604, 58.3419,
    81.6574, 68.1355, -61.6529, 71.6728, 108.0935, -86.5317, 98.0125, 62.3053,
    -0.1958
]  #原本应该是中指的abd关节的位置现在是拇指mcp关节

MAX_ANGLES = [
    -42.9028, -31.7866, 187.1923, 149.0736, -62.4206, 136.5112, 194.9323, 80.1749,
    120.1984, 113.2749, -46.4382, 110.2107, 160.5983, -99.9564, 134.6903, 117.5737,
    9.7608
]  #中指abd，食指abd，食指pip，食指mcp，拇指abd，拇指dip，拇指pip，拇指mcp，
  # 中指pip，中指mcp，小指abd，小指mcp，小指pip，无名指abd，无名指mcp，无名指pip

# 每个关节的前端输入最小角度（长度17，与关节顺序一致）
JOINT_INPUT_MIN = [0.0, 0.0, 15.0, 0.0, 30.0, 20.0, 20.0, 0.0,
                   0.0, 0.0,0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                   0.0]  #17个关节的输入最小值
# 每个关节的前端输入最大角度（长度17，与关节顺序一致）
JOINT_INPUT_MAX = [80.0, 90.0, 50.0,75.0,45.0,80.0,80.0,50.0,
                   75.0,70.0,90.0,75.0,75.0,90.0,75.0,75.0,
                   90.0,]  # 不同关节的输入最大值不同

# 输出节流/变化阈值（度）——当任一关节变化超过此值时输出 angles 
DELTA_THRESHOLD = 15.0




###-----1.
# ---- 手势匹配配置 ----
ENABLE_GESTURE_MATCHING = True  # 是否启用手势匹配
GESTURE_HISTORY_SIZE = 10  # 历史帧数（用于ML模型）
USE_ML_MODEL = False  # 是否使用ML模型（需要模型文件）
ML_MODEL_PATH = "gesture_model.h5"  # ML模型路径
TEMPLATE_FILE = "gesture_templates.json"  # 模板文件路径
###-------





# ---- 命令队列和后台 worker ----
cmd_queue: asyncio.Queue = asyncio.Queue(maxsize=QUEUE_MAXSIZE)





###-------2.
# ---- 手势匹配全局变量 ----
gesture_matcher = None
joint_history = []  # 存储最近的关节角度历史
###-----------




def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def map_angle(angle: float, idx: int) -> float:
    """
    为每个关节单独实现映射逻辑
    按关节索引将前端角度映射为硬件角度。

    映射规则（按需求实现）:
    - 前端 JOINT_INPUT_MIN[idx]  -> 硬件 MIN_ANGLES[idx]
    - 前端 MAX_ANGLES[idx] -> 硬件 MAX_ANGLES[idx]
    - 前端小于 0° 的值 clamp 到 MIN
    - 前端大于 MAX_ANGLES[idx] 的值 clamp 到 MAX（可选策略：也可以选择把 90-180 线性映射回 MIN，这里采用截断为 MAX）

    注意：请确认 `MIN_ANGLES`/`MAX_ANGLES` 的顺序与 `joints_reordered` 保持一致。
    """
    try:
        # 读取当前关节的硬件范围
        hardware_min = float(MIN_ANGLES[idx])
        hardware_max = float(MAX_ANGLES[idx])

        # 读取当前关节的前端输入范围
        input_min = float(JOINT_INPUT_MIN[idx])
        input_max = float(JOINT_INPUT_MAX[idx])

    except Exception:
        # 回退到全局范围
        return clamp(angle, HARDWARE_MIN, HARDWARE_MAX)

    # 0 -> min, 90 -> max
    if angle <= input_min:
        return hardware_min
    if angle >= 90.0:
        return hardware_max

    # 1. 先将前端角度截断到当前关节的输入范围内
    clamped_angle = clamp(angle, input_min, input_max)

    if input_max == input_min:
        # 避免除零，直接返回硬件最小值
        return hardware_min
    # ratio = float(angle) / 90.0
    ratio = (clamped_angle - input_min) / (input_max - input_min)
    mapped = hardware_min + ratio * (hardware_max - hardware_min)
    return mapped

# 这是一个同步函数，调用现有的驱动 API
def send_to_hardware_sync(joints: List[float]):
    # 延迟导入并维护一个 OrcaHand 单例，以避免在模块导入时立即连接硬件
    global _ORCA_HAND_INSTANCE
    try:
        if '_ORCA_HAND_INSTANCE' not in globals() or _ORCA_HAND_INSTANCE is None:
            # model path 设为与脚本目录相邻的 orcahand_v1_right 文件夹
            script_dir = os.path.dirname(os.path.dirname(__file__))
            model_path = os.path.join(script_dir, "orcahand_v1_right")
            try:
                from hls.HAND.contrl.core3 import OrcaHand
            except Exception:
                # 兼容直接包内导入路径
                from HAND.contrl.core3 import OrcaHand

            _ORCA_HAND_INSTANCE = OrcaHand(model_path)

        hand = _ORCA_HAND_INSTANCE

        # 尝试连接（如果尚未连接）
        if not hand.is_connected():
            ok, msg = hand.connect()
            if not ok:
                return False
            # 开启 torque（与 replay 脚本一致）
            try:
                hand.enable_torque()
            except Exception:
                pass

        # 将接收到的角度（度）直接发送给驱动
        # 驱动内部使用角度单位（degree）作为 joint_pos 输入（与 replay_smooth 保持一致）
        try:
            # 直接设置为一步到位，速度与加速度使用合理默认值
            hand.set_joint_pos(joints, num_steps=1, speed=100, acceleration=100)
            return True
        except Exception:
            return False
    except Exception:
        return False

async def hardware_worker():
    loop = asyncio.get_running_loop()
    while True:
        joints = await cmd_queue.get()
        try:
            await loop.run_in_executor(None, send_to_hardware_sync, joints)
        except Exception:
            # 不在控制台打印额外信息，保持输出简洁
            pass
        finally:
            cmd_queue.task_done()


# 维护上一次输出用于变化检测
LAST_SENT = None


async def handler(ws, path=None):  # path 可选，兼容旧/新版本
    try:
        async for msg in ws:
            try:
                data = json.loads(msg)
            except Exception:
                await ws.send(json.dumps({"type": "error", "msg": "invalid json"}))
                continue

            if data.get("type") != "joints":
                await ws.send(json.dumps({"type": "error", "msg": "unknown type"}))
                continue

            joints_in = data.get("joints", [])
            if not isinstance(joints_in, list) or len(joints_in) != 17:
                await ws.send(json.dumps({"type": "error", "msg": "need 17 joints"}))
                continue

            # 假定前端顺序为 types.ts 的 TARGET_JOINTS_CONFIG 顺序：
            # [wrist, thumb_abd, thumb_mcp, thumb_pip, thumb_dip, index_abd, index_mcp, index_pip,
            #  middle_abd, middle_mcp, middle_pip, ring_abd, ring_mcp, ring_pip, pinky_abd, pinky_mcp, pinky_pip]
            # 目标硬件顺序（你要求）:
            # 中指abd，食指abd，食指pip，食指mcp，拇指abd，拇指dip，拇指pip，拇指mcp，
            # 中指pip，中指mcp，小指abd，小指mcp，小指pip，无名指abd，无名指mcp，无名指pip
            # 手腕wrist
            reorder_idx = [8, 5, 7, 6, 1, 4, 3, 2, 10,9, 14, 15, 16, 11, 12, 13, 0]
            try:
                joints_reordered = [float(joints_in[i]) for i in reorder_idx]
            except Exception:
                await ws.send(json.dumps({"type": "error", "msg": "reorder_failed"}))
                continue

            # 映射到硬件范围（逐关节映射）
            joints_mapped = [map_angle(x, i) for i, x in enumerate(joints_reordered)]




            ###----3.
            # ---- 手势匹配和校正 ----
            if ENABLE_GESTURE_MATCHING and gesture_matcher:
                # 更新历史
                global joint_history
                joint_history.append(joints_mapped.copy())
                if len(joint_history) > GESTURE_HISTORY_SIZE:
                    joint_history.pop(0)

                # 匹配手势
                gesture_name, confidence, corrected_joints = gesture_matcher.match_gesture(
                    joints_mapped, joint_history if USE_ML_MODEL else None
                )

                # 如果置信度足够高，使用校正后的关节角度
                if confidence > 0.7:  # 可调整阈值
                    joints_mapped = corrected_joints
                    print(f"Gesture: {gesture_name} (conf: {confidence:.2f})")

            ###-----------
            
            
            
            
            # 变化检测
            global LAST_SENT
            changed = False
            if LAST_SENT is None:
                changed = True
            else:
                for a, b in zip(joints_mapped, LAST_SENT):
                    if abs(a - b) > DELTA_THRESHOLD:
                        changed = True
                        break

            if changed:
                s = ",".join(str(x) for x in joints_mapped)
                # 精简输出，严格格式：angles:[v1,v2,...,v17]
                print(f"angles:[{s}]")
                LAST_SENT = joints_mapped

            # 入队给后台 worker 处理（非阻塞）
            try:
                cmd_queue.put_nowait(joints_mapped)
                await ws.send(json.dumps({"type": "ack", "ok": True, "queued": True}))
            except asyncio.QueueFull:
                await ws.send(json.dumps({"type": "ack", "ok": False, "reason": "queue_full"}))
    finally:
        return


async def main():



    ###-----4.
    # 初始化手势匹配器
    global gesture_matcher
    if ENABLE_GESTURE_MATCHING:
        script_dir = os.path.dirname(__file__)
        template_path = os.path.join(script_dir, TEMPLATE_FILE)
        if USE_ML_MODEL:
            model_path = os.path.join(script_dir, ML_MODEL_PATH)
            gesture_matcher = MLGestureMatcher(model_path, template_path)
        else:
            gesture_matcher = GestureMatcher(template_path)
        print("Gesture matching enabled")
    ###----------





    # 启动后台 worker
    asyncio.create_task(hardware_worker())
    server = await websockets.serve(handler, HOST, PORT)
    # 仅输出启动信息，其余运行时仅输出 angles 行
    print(f"Hand server listening on ws://{HOST}:{PORT}")
    await server.wait_closed()


if __name__ == "__main__":
    asyncio.run(main())