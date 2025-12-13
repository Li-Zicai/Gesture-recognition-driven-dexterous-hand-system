# 手势识别集成说明

## 概述
INFI_HAND 项目现在支持在后端进行手势上下文感知匹配，以提升关节角度识别的准确性。

## 功能特性
- **模板匹配**: 使用预定义的手势模板进行简单匹配
- **PyTorch ML**: 可选使用PyTorch LSTM神经网络进行复杂手势识别
- **实时校正**: 根据识别到的手势校正关节角度

## 文件结构
- `gesture_matcher.py`: 手势匹配核心模块
- `gesture_templates.json`: 手势模板定义
- `train_gesture_model.py`: PyTorch模型训练脚本
- `collect_gesture_data.py`: 数据收集脚本

## 配置选项
在 `hand_netserver.py` 中：
- `ENABLE_GESTURE_MATCHING`: 启用/禁用手势匹配
- `USE_ML_MODEL`: 使用PyTorch模型还是模板匹配
- `GESTURE_HISTORY_SIZE`: 历史帧数（用于ML）

## 使用方法

### 1. 模板匹配（推荐开始）
1. 编辑 `gesture_templates.json` 定义手势
2. 运行服务器：`python -m hls.HAND.scripts.hand_netserver`
3. 服务器会自动应用匹配和校正

### 2. PyTorch机器学习方法
1. 收集训练数据：
   ```bash
   python collect_gesture_data.py
   ```
   然后按提示收集不同手势的数据

2. 训练模型：
   ```bash
   python train_gesture_model.py
   ```

3. 在 `hand_netserver.py` 中设置 `USE_ML_MODEL = True`

## 手势模板格式
```json
{
  "gesture_name": {
    "ranges": [[min1, max1], [min2, max2], ...],
    "weights": [w1, w2, ...]
  }
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