# 手势识别系统集成指南

## 系统架构

```
前端 (0-180°关节角) 
    ↓
hand_netserver.py (WebSocket服务器)
    ↓ 
[关节映射] → 硬件范围 (-90°~180°)
    ↓
[手势识别] → ML模型 (gesture_model.pth)
    ↓
[手势到动作映射] → gesture_actions.json
    ↓
[硬件控制] → OrcaHand驱动程序
    ↓
灵巧手执行动作
```

## 关键文件说明

### 1. **gesture_model.pth** - 训练的神经网络模型
- **格式**: PyTorch checkpoint (.pth)
- **架构**: 2层LSTM (64隐藏单元) + 全连接层
- **输入**: 10帧序列 × 17个关节
- **输出**: 3个手势类别的概率分布
- **由**: `train_gesture_model.py` 从 `gesture_training_data.json` 生成

### 2. **label_encoder.npy** - 标签编码器
- **用途**: 将模型输出的索引 (0, 1, 2) 映射到手势名称
- **内容**: `{"one_sign": 0, "V_sign": 1, "OK_sign": 2}`
- **由**: `train_gesture_model.py` 生成

### 3. **gesture_actions.json** - 关键：手势到硬件动作的映射
```json
{
  "one_sign": {
    "joints": [-50, -35, 30, 50, ..., 0],  // 17个关节位置
    "description": "食指竖起，其他手指弯曲",
    "name": "一个手指竖起"
  },
  "V_sign": { ... },
  "OK_sign": { ... }
}
```

**关节顺序** (17个):
```
中指abd, 食指abd, 食指pip, 食指mcp, 拇指abd, 拇指dip, 拇指pip, 拇指mcp,
中指pip, 中指mcp, 小指abd, 小指mcp, 小指pip, 无名指abd, 无名指mcp, 无名指pip, 手腕
```

**角度范围**: 对应各关节的 MIN_ANGLES 到 MAX_ANGLES

## 问题解决

### 问题 1: 模型与应用脚本不匹配
**症状**: 模型文件是 `.pth` 格式但代码期望 `.h5`

**解决方案** (已实现):
- ✅ 更新了 `gesture_matcher.py` 的 `MLGestureMatcher` 类以支持 `.pth` 格式
- ✅ 添加了 `label_encoder.npy` 的加载和使用
- ✅ 更新了 `hand_netserver.py` 中的模型配置

### 问题 2: 缺少手势到硬件动作的映射
**症状**: 系统能识别手势，但不知道硬件应该怎样做这个手势

**解决方案** (已实现):
- ✅ 创建了 `gesture_actions.json` 文件
- ✅ 为每个手势定义17个关节位置
- ✅ `hand_netserver.py` 现在会：
  ```python
  if gesture_name in gesture_actions:
      joints_mapped = gesture_actions[gesture_name]  # 使用手势对应的硬件动作
      print(f"🎯 Gesture detected: {gesture_name} -> Executing action")
  ```

## 完整工作流

### 第1步: 数据收集 (使用 50 帧固定长度)
```bash
python collect_gesture_data.py
```
- 连接到 `hand_netserver.py`
- 输入手势名称: `one_sign`, `V_sign`, `OK_sign`
- 自动收集 50 帧
- 保存到 `gesture_training_data.json`

### 第2步: 训练模型
```bash
python train_gesture_model.py
```
- 加载 `gesture_training_data.json`
- 自动将 50 帧序列分割为 10 帧输入窗口
- 训练 LSTM 模型
- **输出**:
  - `gesture_model.pth` ← 模型文件
  - `label_encoder.npy` ← 标签编码器

### 第3步: 定义手势硬件动作
编辑 `gesture_actions.json`:
```json
{
  "one_sign": {
    "joints": [
      -50,  // 中指abd
      -35,  // 食指abd
      30,   // 食指pip
      50,   // 食指mcp
      -45,  // 拇指abd
      90,   // 拇指dip
      90,   // 拇指pip
      30,   // 拇指mcp
      30,   // 中指pip
      60,   // 中指mcp
      -55,  // 小指abd
      60,   // 小指mcp
      30,   // 小指pip
      -75,  // 无名指abd
      70,   // 无名指mcp
      30,   // 无名指pip
      0     // 手腕
    ]
  }
}
```

### 第4步: 启动服务器
```bash
python hand_netserver.py
```

服务器将：
1. ✅ 接收前端的 0-180° 关节角
2. ✅ 映射到硬件范围
3. ✅ 识别当前手势（使用 ML 模型）
4. ✅ **如果置信度 > 70%，自动执行 `gesture_actions.json` 中的硬件动作**
5. ✅ 驱动 OrcaHand 灵巧手

## 配置参数

在 `hand_netserver.py` 中修改:

```python
# 启用手势识别
ENABLE_GESTURE_MATCHING = True

# 使用 ML 模型（True）或仅模板匹配（False）
USE_ML_MODEL = True

# 模型文件路径
ML_MODEL_PATH = "gesture_model.pth"           # .pth 格式
LABEL_ENCODER_PATH = "label_encoder.npy"      # 标签编码器

# ML 输入的历史帧数
GESTURE_HISTORY_SIZE = 10

# 置信度阈值（低于此值不执行手势动作）
GESTURE_CONFIDENCE_THRESHOLD = 0.7  # 70% 以上才执行

# 手势动作文件
GESTURE_ACTIONS_FILE = "gesture_actions.json"
```

## 运行示例

### 场景: 用户做"one sign"手势

```
1. 前端检测关节角: [30°, 45°, 28°, 40°, ...]
   ↓
2. 映射到硬件范围: [-50°, -35°, 30°, 50°, ...]
   ↓
3. 历史 10 帧 → LSTM 模型
   ↓
4. 输出: "one_sign" (置信度: 85%)
   ↓
5. 置信度 85% > 70% 阈值 ✓
   ↓
6. 从 gesture_actions.json 获取 one_sign 的关节位置
   ↓
7. 发送到 OrcaHand: [-50, -35, 30, 50, -45, 90, 90, 30, 30, 60, -55, 60, 30, -75, 70, 30, 0]
   ↓
8. 硬件执行: 食指竖起，其他手指弯曲
```

### 服务器日志输出

```
Hand server listening on ws://0.0.0.0:8765
✓ Loaded gesture actions: ['one_sign', 'V_sign', 'OK_sign', 'neutral']
✓ ML gesture matching enabled
angles:[-50.123,-35.456,30.789,...]
✓ Gesture: one_sign (conf: 0.85)
angles:[-50,-35,30,50,-45,90,90,30,30,60,-55,60,30,-75,70,30,0]
🎯 Gesture detected: one_sign (conf: 85.00%) -> Executing action
```

## 自定义手势动作

### 调整关节位置

1. 查看硬件范围:
```python
# 在 hand_netserver.py 中
MIN_ANGLES = [-61.6006, -43.2354, 128.002, 97.6487, ...]
MAX_ANGLES = [-42.9028, -31.7866, 187.1923, 149.0736, ...]
```

2. 编辑 `gesture_actions.json` 中的 `joints` 数组，确保每个值在对应的 MIN/MAX 范围内

3. 示例调整:
```json
{
  "one_sign": {
    "joints": [
      -50,   // 中指abd: 在 [-61.6, -42.9] 范围内 ✓
      -35,   // 食指abd: 在 [-43.2, -31.8] 范围内 ✓
      ...
    ]
  }
}
```

### 测试新的关节位置

- 修改 `gesture_actions.json` 后，无需重新训练
- 直接重启服务器，下次识别该手势时会使用新的关节位置
- 观察硬件反应，再微调参数

## 调试建议

1. **检查模型加载**:
```
服务器启动输出:
✓ Loaded gesture actions: ['one_sign', 'V_sign', 'OK_sign']
✓ ML gesture matching enabled
```

2. **检查识别置信度**:
```
✓ Gesture: one_sign (conf: 0.92)     // 92% 置信度
🎯 Gesture detected: one_sign (conf: 92.00%) -> Executing action
```

3. **关闭手势识别快速调试**:
```python
ENABLE_GESTURE_MATCHING = False  # 只接收前端角度，不执行手势识别
```

4. **仅使用模板匹配**:
```python
USE_ML_MODEL = False  # 使用 gesture_templates.json，更快更简单
GESTURE_CONFIDENCE_THRESHOLD = 0.6  # 调整阈值
```

## 常见问题

**Q: 识别错误或漂移怎么办？**
A: 
- 重新收集该手势的更多训练数据 (建议 10+ 个序列)
- 确保在不同角度、速度下都有样本
- 重新训练模型

**Q: 硬件关节值超出范围报错？**
A:
- 检查 `gesture_actions.json` 中的值是否在各关节的 MIN/MAX 范围内
- 参考上面的范围表进行调整

**Q: 置信度总是很低？**
A:
- 增加训练数据
- 降低 `GESTURE_CONFIDENCE_THRESHOLD` (如改为 0.5)
- 检查数据收集时是否正确执行了该手势

**Q: 想快速测试而不训练模型？**
A:
- 设置 `USE_ML_MODEL = False`
- 编辑 `gesture_templates.json` 定义手势范围
- 系统会使用简单的范围匹配

**Q: 想添加第4个手势？**
A:
1. 收集新手势的训练数据 (至少 5 个 50 帧序列)
2. 重新运行 `train_gesture_model.py`
3. 在 `gesture_actions.json` 中定义新手势的 17 个关节位置
4. 重启服务器，系统会自动识别新手势
}
```

## 推荐的ML方法
**PyTorch LSTM网络** 是最佳选择，因为：
- **适合时间序列**: 关节角度是连续的时序数据
- **捕捉动态**: 能学习手势的时间变化模式
- **准确性高**: 比传统方法更能处理复杂手势
- **灵活性**: PyTorch提供了丰富的定制选项

## PyTorch模型架构
- **输入**: (batch_size, sequence_length=10, input_size=17)
- **LSTM层**: 2层，隐藏单元64
- **Dropout**: 0.2防止过拟合
- **全连接层**: 16单元 + 输出层
- **激活函数**: ReLU + Softmax

## 训练参数
- **优化器**: Adam (lr=0.001)
- **损失函数**: CrossEntropyLoss
- **Batch size**: 32
- **Epochs**: 50
- **验证**: 20%数据用于验证

## 性能考虑
- 模板匹配：低延迟，适合实时应用
- PyTorch模型：稍高延迟，但准确性更高
- 可以根据需要切换两种方法

## 扩展建议
- 添加更多手势模板
- 实现动态模板更新
- 集成更先进的模型（如Transformer）