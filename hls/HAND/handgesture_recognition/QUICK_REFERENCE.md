# 快速参考卡 - 手势识别系统

## 系统三层结构

```
┌─────────────────────────────────────────────────┐
│  第1层: 识别层 (gesture_matcher.py)            │
│  输入: 10帧关节序列                             │
│  输出: (手势名称, 置信度 0-1)                   │
│  例: ("one_sign", 0.92)                        │
└─────────────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────────────┐
│  第2层: 决策层 (hand_netserver.py)              │
│  if 置信度 > 70%:                               │
│      查找 gesture_actions[手势名称]             │
└─────────────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────────────┐
│  第3层: 执行层 (OrcaHand硬件驱动)               │
│  设置 17 个关节位置                             │
│  灵巧手执行对应的手势动作                       │
└─────────────────────────────────────────────────┘
```

## 关键映射关系

### 1. 手势识别映射
```
输入序列 (10帧 × 17关节)
    ↓ [LSTM处理]
预测索引 (0, 1, 2, ...)
    ↓ [label_encoder.npy查表]
手势名称 ("one_sign", "V_sign", "OK_sign")
```

### 2. 手势到硬件动作映射
```
gesture_actions.json:
{
  "one_sign": [关节1, 关节2, ..., 关节17],
  "V_sign":   [关节1, 关节2, ..., 关节17],
  "OK_sign":  [关节1, 关节2, ..., 关节17]
}
```

### 3. 关节索引顺序 (0-16)
```
0: 中指abd       7: 拇指mcp       14: 无名指mcp
1: 食指abd       8: 中指pip       15: 无名指pip
2: 食指pip       9: 中指mcp       16: 手腕
3: 食指mcp      10: 小指abd
4: 拇指abd      11: 小指mcp
5: 拇指dip      12: 小指pip
6: 拇指pip      13: 无名指abd
```

## 配置速查表

```python
# hand_netserver.py 中的三个关键开关

1. ENABLE_GESTURE_MATCHING = True/False
   ├─ True: 启用手势识别
   └─ False: 关闭手势识别（只接收前端角度）

2. USE_ML_MODEL = True/False
   ├─ True: 使用 gesture_model.pth (需要 label_encoder.npy)
   └─ False: 仅使用 gesture_templates.json

3. GESTURE_CONFIDENCE_THRESHOLD = 0.0-1.0
   ├─ 0.9: 只有 90%+ 置信度时执行
   ├─ 0.7: 默认，70%+ 时执行
   └─ 0.5: 低标准，50%+ 时执行
```

## 启动检查清单

- [ ] `gesture_model.pth` 存在
- [ ] `label_encoder.npy` 存在
- [ ] `gesture_actions.json` 存在且包含所有手势
- [ ] `gesture_actions.json` 中的关节值在 MIN/MAX 范围内
- [ ] `hand_netserver.py` 配置正确：
  - [ ] `ENABLE_GESTURE_MATCHING = True`
  - [ ] `USE_ML_MODEL = True`
  - [ ] 路径正确

## 常用命令

```bash
# 数据收集 (50帧自动采集)
python collect_gesture_data.py

# 模型训练 (自动保存 .pth 和 .npy)
python train_gesture_model.py

# 启动服务器 (同时启用手势识别)
python hand_netserver.py

# 检查 label_encoder 内容
python -c "import numpy as np; le = np.load('label_encoder.npy', allow_pickle=True).item(); print(le)"

# 检查模型结构
python -c "import torch; ckpt = torch.load('gesture_model.pth'); print(ckpt.keys())"
```

## 调整手势执行位置

### 方法 1: 编辑 gesture_actions.json (推荐)
```json
{
  "one_sign": {
    "joints": [
      -50,    // ← 中指abd: 改这个数字调整食指位置
      ...
    ]
  }
}
```
- 修改后无需重新训练
- 重启服务器立即生效

### 方法 2: 找到对应的关节范围
```
MIN_ANGLES[i] ≤ 新值 ≤ MAX_ANGLES[i]

例如，关节0（中指abd）:
MIN_ANGLES[0] = -61.6
MAX_ANGLES[0] = -42.9
可选范围: [-61.6, -42.9] 之间的任何值
```

## 日志解读

```
✓ Loaded gesture actions: ['one_sign', 'V_sign', 'OK_sign', 'neutral']
  → 动作文件成功加载

✓ ML gesture matching enabled
  → 使用了 PyTorch 模型（未看到这行说明用的是模板匹配）

angles:[-50.123,-35.456,30.789,...]
  → 当前硬件目标关节位置

✓ Gesture: one_sign (conf: 0.85)
  → 识别成功但置信度不足阈值（85%，可能阈值设为 90%）

🎯 Gesture detected: one_sign (conf: 85.00%) -> Executing action
  → 识别成功且执行了对应的硬件动作！
```

## 问题诊断

```
问题 1: 没看到 "✓ Loaded gesture actions"
→ 检查 gesture_actions.json 是否存在
→ 检查文件路径是否正确
→ 检查 JSON 格式是否有效

问题 2: 识别不了某个手势
→ 识别置信度低 (<70%)
  • 增加该手势的训练数据
  • 重新训练模型
→ 没有识别到任何手势
  • 检查 ML 模型是否正确加载
  • 确认 USE_ML_MODEL = True

问题 3: 硬件动作错误或超范围
→ 检查 gesture_actions.json 的关节值
→ 确保值在 MIN_ANGLES[i] 到 MAX_ANGLES[i] 之间
→ 打印 MIN/MAX 范围进行对比

问题 4: 想快速测试
→ 暂时关闭手势识别:
   ENABLE_GESTURE_MATCHING = False
→ 或仅用模板匹配:
   USE_ML_MODEL = False
```

## 核心数据结构

### gesture_training_data.json (训练数据)
```json
{
  "one_sign": {
    "sequences": [
      [[j1, j2, ..., j17], [j1, j2, ..., j17], ...],  // 50帧
      [[j1, j2, ..., j17], [j1, j2, ..., j17], ...],  // 50帧
      ...
    ]
  }
}
```

### gesture_model.pth (模型文件)
```
{
  'model_state_dict': {...},      // 神经网络权重
  'input_size': 17,                // 关节数
  'hidden_size': 64,               // LSTM隐层大小
  'num_layers': 2,                 // LSTM层数
  'num_classes': 3                 // 手势类别数 (one_sign, V_sign, OK_sign)
}
```

### label_encoder.npy (标签映射)
```python
{
  'one_sign': 0,
  'V_sign': 1,
  'OK_sign': 2
}
```

### gesture_actions.json (硬件动作)
```json
{
  "one_sign": {
    "joints": [v0, v1, ..., v16],  // 17个关节位置
    "description": "..."
  }
}
```

## 性能优化建议

1. **降低 GESTURE_HISTORY_SIZE** (10 → 5)
   - 更快响应，但准确率可能下降

2. **提高 GESTURE_CONFIDENCE_THRESHOLD** (0.7 → 0.9)
   - 减少误触发

3. **使用 CPU 模式** (默认)
   - 如果 GPU 不可用，自动降级到 CPU

4. **缓存模型**
   - 第一次加载后，模型会保留在内存中
