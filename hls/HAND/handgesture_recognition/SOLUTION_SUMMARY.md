# 手势识别系统完整解决方案

## 问题总结

用户提出了两个关键问题：

### 问题 1: 应用脚本与模型不匹配
- **症状**: 有训练好的 `gesture_model.pth` 和 `label_encoder.npy`，但 `hand_netserver.py` 和 `gesture_matcher.py` 的代码与这些文件格式不兼容
- **根本原因**: 代码期望 `.h5` 格式的模型，但使用的是 `.pth` 格式

### 问题 2: 缺少手势到硬件动作的映射
- **症状**: 系统能识别用户做的手势（如"one sign"），但不知道硬件应该如何执行这个手势
- **根本原因**: 没有将识别到的手势名称映射到具体的硬件关节位置

---

## 解决方案

### 步骤 1: 更新 `gesture_matcher.py` (✅ 已完成)

**改动内容**:
1. **支持 `.pth` 格式模型**:
   ```python
   checkpoint = torch.load(model_path, map_location='cpu')
   ```

2. **加载 `label_encoder.npy`**:
   - 支持两种格式：字典 `{"one_sign": 0, "V_sign": 1, ...}` 和数组 `["one_sign", "V_sign", ...]`
   - 自动转换为统一的字典格式

3. **正确的索引到手势名称的映射**:
   ```python
   # 模型输出: 索引 0, 1, 2
   # label_encoder 转换: {"one_sign": 0, "V_sign": 1, "OK_sign": 2}
   # 结果: "one_sign"
   ```

### 步骤 2: 创建 `gesture_actions.json` (✅ 已完成)

**新建文件**: `gesture_actions.json`

**文件结构**:
```json
{
  "one_sign": {
    "joints": [-50, -35, 30, 50, ..., 0],  // 17个关节的目标位置
    "name": "一个手指竖起",
    "description": "食指竖起，其他手指弯曲"
  },
  "V_sign": { ... },
  "OK_sign": { ... }
}
```

**关键特点**:
- 为每个手势定义 17 个关节的具体位置
- 这些位置在硬件的 MIN_ANGLES 和 MAX_ANGLES 范围内
- 系统识别到该手势时，会自动使用这些位置控制硬件

### 步骤 3: 更新 `hand_netserver.py` (✅ 已完成)

**配置参数**:
```python
# 启用手势识别
ENABLE_GESTURE_MATCHING = True

# 使用 PyTorch 模型
USE_ML_MODEL = True
ML_MODEL_PATH = "gesture_model.pth"           # ✓ 现在正确指向 .pth
LABEL_ENCODER_PATH = "label_encoder.npy"      # ✓ 新增

# 手势动作映射
GESTURE_ACTIONS_FILE = "gesture_actions.json"  # ✓ 新增
GESTURE_CONFIDENCE_THRESHOLD = 0.7             # ✓ 新增
```

**核心逻辑** (handler函数中):
```python
# 1. 识别手势
gesture_name, confidence, _ = gesture_matcher.match_gesture(...)

# 2. 如果置信度足够高且手势有对应动作
if confidence > GESTURE_CONFIDENCE_THRESHOLD:
    if gesture_name in gesture_actions:
        # 3. 使用手势对应的硬件动作
        joints_mapped = gesture_actions[gesture_name]
        print(f"🎯 Gesture detected: {gesture_name} -> Executing action")
```

---

## 三层架构

```
┌──────────────────────────────────────────────────┐
│ 层1: 前端识别                                    │
│ 输入: 10帧关节序列 (手机/传感器捕捉的动作)      │
│ 输出: (手势名称, 置信度)                         │
│ 文件: gesture_model.pth + label_encoder.npy    │
└──────────────────────────────────────────────────┘
                    ↓
┌──────────────────────────────────────────────────┐
│ 层2: 决策判断                                    │
│ 规则: if 置信度 > 70%:                           │
│     在 gesture_actions.json 中查找该手势       │
└──────────────────────────────────────────────────┘
                    ↓
┌──────────────────────────────────────────────────┐
│ 层3: 硬件执行                                    │
│ 输入: 17个关节位置                              │
│ 输出: 灵巧手执行对应的手势动作                  │
│ 文件: gesture_actions.json                      │
└──────────────────────────────────────────────────┘
```

---

## 工作流示例

### 场景: 用户做"one sign"手势

```
1. 前端传输关节角度序列:
   Frame 1: [30°, 45°, 28°, 40°, ...]
   Frame 2: [31°, 46°, 29°, 41°, ...]
   ...
   Frame 10: [32°, 47°, 30°, 42°, ...]
   ↓

2. 手势识别 (gesture_matcher.py):
   LSTM模型处理 10帧 × 17关节
   输出: [0.02, 0.01, 0.97]  (概率分布)
   索引 2 对应 confidence=0.97
   ↓

3. 标签解码 (label_encoder):
   索引 2 → "one_sign"
   ↓

4. 置信度检查:
   0.97 > 0.7? YES ✓
   ↓

5. 动作查找 (gesture_actions.json):
   gesture_actions["one_sign"] = [-50, -35, 30, 50, ..., 0]
   ↓

6. 硬件执行 (OrcaHand):
   设置 17 个关节到指定位置
   → 灵巧手做出"one sign"手势
   ↓

7. 日志输出:
   angles:[-50,-35,30,50,-45,90,90,30,30,60,-55,60,30,-75,70,30,0]
   🎯 Gesture detected: one_sign (conf: 97.00%) -> Executing action
```

---

## 文件对应关系

| 文件 | 作用 | 来源 |
|------|------|------|
| `gesture_training_data.json` | 手势训练数据 (50帧序列) | collect_gesture_data.py 收集 |
| `gesture_model.pth` | 训练好的 LSTM 模型 | train_gesture_model.py 训练 |
| `label_encoder.npy` | 标签编码器 (索引→手势名) | train_gesture_model.py 生成 |
| `gesture_actions.json` | 手势→硬件关节位置映射 | **用户定义** |
| `gesture_matcher.py` | 识别逻辑 | **已更新** |
| `hand_netserver.py` | 主服务器程序 | **已更新** |

---

## 关键映射表

### 映射 1: 模型输出索引 → 手势名称
```python
label_encoder = {
    "one_sign": 0,     # 模型输出[1,0,0] → 索引0
    "V_sign": 1,       # 模型输出[0,1,0] → 索引1
    "OK_sign": 2       # 模型输出[0,0,1] → 索引2
}
```

### 映射 2: 手势名称 → 硬件关节位置
```python
gesture_actions = {
    "one_sign": [
        -50,   # 关节 0: 中指abd
        -35,   # 关节 1: 食指abd
        30,    # 关节 2: 食指pip
        # ... (共17个)
    ],
    "V_sign": [...],
    "OK_sign": [...]
}
```

### 映射 3: 前端角度 → 硬件角度 (已有)
```python
JOINT_INPUT_MIN/MAX → MIN_ANGLES/MAX_ANGLES
```

---

## 验证系统工作

### 快速检查清单

- [ ] `gesture_model.pth` 存在 (227KB)
- [ ] `label_encoder.npy` 存在 (282B)
- [ ] `gesture_actions.json` 存在且包含 one_sign/V_sign/OK_sign
- [ ] `hand_netserver.py` 的配置:
  - [ ] `ENABLE_GESTURE_MATCHING = True`
  - [ ] `USE_ML_MODEL = True`
  - [ ] `GESTURE_CONFIDENCE_THRESHOLD = 0.7`

### 启动服务器
```bash
python hand_netserver.py
```

### 期望日志输出
```
✓ Loaded gesture actions: ['one_sign', 'V_sign', 'OK_sign', 'neutral']
✓ ML gesture matching enabled
Hand server listening on ws://0.0.0.0:8765
```

### 识别日志
```
angles:[-50,-35,30,50,-45,90,90,30,30,60,-55,60,30,-75,70,30,0]
🎯 Gesture detected: one_sign (conf: 92.00%) -> Executing action
```

---

## 常见问题解答

### Q1: 如何调整手势的硬件执行位置？
**A**: 编辑 `gesture_actions.json` 的 `joints` 数组，确保每个值在对应关节的 MIN/MAX 范围内

### Q2: 识别总是出错怎么办？
**A**: 
- 检查 `label_encoder.npy` 是否正确加载
- 增加训练数据并重新训练模型
- 降低 `GESTURE_CONFIDENCE_THRESHOLD` (如改为 0.5)

### Q3: 想添加第4个手势？
**A**:
1. 收集新手势的训练数据
2. 重新运行 `train_gesture_model.py`
3. 在 `gesture_actions.json` 中添加新手势的关节位置

### Q4: 硬件位置超出范围？
**A**: 检查 MIN_ANGLES 和 MAX_ANGLES，确保 gesture_actions.json 中的值在范围内

---

## 完整流程图

```
数据收集阶段:
  collect_gesture_data.py
  → 前端关节角 → gesture_training_data.json (50帧序列)

训练阶段:
  train_gesture_model.py
  → gesture_training_data.json → gesture_model.pth + label_encoder.npy

配置阶段 (新):
  手动编写 gesture_actions.json
  → 为每个手势定义 17 个硬件关节位置

运行阶段:
  hand_netserver.py (已更新的版本)
  → 接收前端角度
  → gesture_matcher.py 识别手势
  → gesture_actions.json 查找硬件动作
  → OrcaHand 驱动执行动作
```

---

## 总结

通过本次更新，系统现在可以：

✅ **识别手势**  
- 使用训练好的 PyTorch LSTM 模型
- 准确读取 `.pth` 格式的模型文件
- 正确解析 `label_encoder.npy` 的标签映射

✅ **执行对应动作**  
- 根据识别的手势名称查找硬件动作
- 自动控制灵巧手的 17 个关节
- 用户可灵活修改 `gesture_actions.json` 调整动作

✅ **完整的数据流**  
- 前端 (0-180°) → 映射 → 识别 → 动作查找 → 硬件执行 (OrcaHand)

**系统已完全集成，可以开始使用！**
