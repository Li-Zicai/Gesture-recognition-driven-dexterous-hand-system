# 快速启动指南

## 前置条件

确保以下文件存在于此目录：
- ✅ `gesture_model.pth` (模型文件)
- ✅ `label_encoder.npy` (标签编码器)
- ✅ `gesture_actions.json` (硬件动作映射)
- ✅ `gesture_templates.json` (备用模板)

## 启动步骤

### 1. 验证系统 (可选)
```bash
python verify_system.py
```
输出应该显示所有文件都已加载。

### 2. 启动手势识别服务器
```bash
python hand_netserver.py
```

期望输出：
```
✓ Loaded gesture actions: ['one_sign', 'V_sign', 'OK_sign', 'neutral']
✓ ML gesture matching enabled
Hand server listening on ws://0.0.0.0:8765
```

### 3. 前端连接
前端通过 WebSocket 连接到 `ws://localhost:8765` 或 `ws://目标IP:8765`

### 4. 观察识别效果
做出手势时，服务器会输出：
```
angles:[v1,v2,v3,...]
✓ Gesture: one_sign (conf: 0.92)

angles:[w1,w2,w3,...]
🎯 Gesture detected: one_sign (conf: 92.00%) -> Executing action
```

---

## 调整手势动作

编辑 `gesture_actions.json` 中的 `joints` 值：

```json
{
  "one_sign": {
    "joints": [
      -50,    // ← 修改这些数字
      -35,
      30,
      ...
    ]
  }
}
```

**范围限制**:
- 关节 0: [-61.6, -42.9]
- 关节 1: [-43.2, -31.8]
- 关节 2: [128.0, 187.2]
- ... (参考 hand_netserver.py 中的 MIN_ANGLES 和 MAX_ANGLES)

修改后无需重启，直接重启服务器即可生效。

---

## 故障排查

### 症状: 没有看到手势识别的日志

**检查**:
1. `ENABLE_GESTURE_MATCHING = True` 是否启用？
2. `gesture_model.pth` 和 `label_encoder.npy` 是否存在？
3. PyTorch 是否正确安装？

**解决方案**:
```bash
# 测试 PyTorch
python -c "import torch; print(torch.__version__)"

# 测试模型加载
python -c "import torch; ckpt = torch.load('gesture_model.pth'); print(ckpt.keys())"

# 测试标签编码器
python -c "import numpy as np; le = np.load('label_encoder.npy', allow_pickle=True); print(type(le), le.shape if hasattr(le, 'shape') else le)"
```

### 症状: 识别置信度总是很低

**原因**:
- 训练数据不足或不多样
- 用户做手势的方式与训练数据不同

**解决方案**:
1. 收集更多训练数据：`python collect_gesture_data.py`
2. 重新训练模型：`python train_gesture_model.py`
3. 或降低阈值：在 hand_netserver.py 中改为 `GESTURE_CONFIDENCE_THRESHOLD = 0.5`

### 症状: 硬件关节报错"超出范围"

**原因**:
`gesture_actions.json` 中的值超出了该关节的 MIN/MAX 范围

**解决方案**:
```python
# 在 hand_netserver.py 中查看范围
MIN_ANGLES[i]  # 最小值
MAX_ANGLES[i]  # 最大值

# 确保 gesture_actions.json 中的值在这个范围内
```

---

## 文件结构

```
handgesture_recognition/
├── hand_netserver.py              # 主服务器（已更新）
├── gesture_matcher.py             # 识别逻辑（已更新）
├── gesture_model.pth              # 训练的模型 ✓
├── label_encoder.npy              # 标签编码器 ✓
├── gesture_actions.json           # 硬件动作映射 ✓（新建）
├── gesture_templates.json         # 备用模板
├── gesture_training_data.json     # 训练数据（用于重新训练）
├── train_gesture_model.py         # 训练脚本
├── collect_gesture_data.py        # 数据收集脚本
├── verify_system.py               # 系统验证脚本（新建）
├── GESTURE_README.md              # 详细说明（已更新）
├── QUICK_REFERENCE.md             # 快速参考（新建）
├── SOLUTION_SUMMARY.md            # 解决方案总结（新建）
└── START_GUIDE.md                 # 本文件
```

---

## 关键参数对照表

```python
# hand_netserver.py 中的配置

# 识别相关
ENABLE_GESTURE_MATCHING = True              # 启用/禁用手势识别
USE_ML_MODEL = True                         # 使用 ML 模型（vs 模板匹配）
GESTURE_CONFIDENCE_THRESHOLD = 0.7          # 置信度阈值（0-1）
GESTURE_HISTORY_SIZE = 10                   # ML 输入的帧数

# 文件路径
ML_MODEL_PATH = "gesture_model.pth"         # 模型文件
LABEL_ENCODER_PATH = "label_encoder.npy"    # 标签编码器
GESTURE_ACTIONS_FILE = "gesture_actions.json"  # 硬件动作映射
TEMPLATE_FILE = "gesture_templates.json"    # 备用模板
```

---

## 常用命令速查

```bash
# 启动服务器
python hand_netserver.py

# 收集新的手势数据
python collect_gesture_data.py

# 重新训练模型（使用已收集的数据）
python train_gesture_model.py

# 验证系统（检查所有文件和配置）
python verify_system.py

# 查看标签编码器内容
python -c "import numpy as np; le = np.load('label_encoder.npy', allow_pickle=True); print('Loaded:', le)"

# 查看模型结构
python -c "import torch; ckpt = torch.load('gesture_model.pth'); print('Model info:', {k:v for k,v in ckpt.items() if k != 'model_state_dict'})"
```

---

## 提示和建议

💡 **要快速测试而不依赖 ML 模型**:
```python
USE_ML_MODEL = False
GESTURE_CONFIDENCE_THRESHOLD = 0.6
# 使用 gesture_templates.json 的模板匹配
```

💡 **要完全禁用手势识别，只接收前端数据**:
```python
ENABLE_GESTURE_MATCHING = False
```

💡 **调整识别灵敏度**:
- 提高 `GESTURE_CONFIDENCE_THRESHOLD` (如 0.9) → 更严格
- 降低 `GESTURE_CONFIDENCE_THRESHOLD` (如 0.5) → 更敏感

💡 **改进识别准确率**:
1. 收集更多样本 (不同角度、速度)
2. 增加训练轮数
3. 调整 LSTM 参数 (hidden_size, num_layers)

---

## 完整工作流

```
启动服务器
    ↓
前端发送关节角
    ↓
关节映射 (0-180° → 硬件范围)
    ↓
手势识别 (ML 模型)
    ↓
置信度检查 (> 70%)
    ↓
查找硬件动作 (gesture_actions.json)
    ↓
驱动硬件 (OrcaHand)
    ↓
灵巧手执行对应手势
```

**现在系统已完全集成，可以开始使用了！** 🎉
