# gesture_matcher.py - 手势匹配模块
import numpy as np
from typing import List, Dict, Tuple, Optional
import json
import os

# PyTorch导入（可选，用于ML模型）
try:
    import torch
    import torch.nn as nn
    PYTORCH_AVAILABLE = True

    class GestureLSTM(nn.Module):
        """PyTorch LSTM模型用于手势识别（与训练脚本保持一致）"""

        def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.2):
            super(GestureLSTM, self).__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers

            self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                               batch_first=True, dropout=dropout if num_layers > 1 else 0)
            self.dropout = nn.Dropout(dropout)
            self.fc1 = nn.Linear(hidden_size, 16)
            self.fc2 = nn.Linear(16, num_classes)
            self.relu = nn.ReLU()

        def forward(self, x):
            # 初始化隐藏状态
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

            # 前向传播LSTM
            out, _ = self.lstm(x, (h0, c0))

            # 取最后一个时间步的输出
            out = out[:, -1, :]

            # 全连接层
            out = self.dropout(out)
            out = self.relu(self.fc1(out))
            out = self.fc2(out)
            return out

except ImportError:
    PYTORCH_AVAILABLE = False
    GestureLSTM = None

class GestureMatcher:
    """简单的手势匹配器，使用模板匹配和可选的ML模型"""

    def __init__(self, template_file: str = None):
        self.templates = self._load_templates(template_file)
        self.ml_model = None  # 如果使用ML，加载模型

    def _load_templates(self, template_file: str) -> Dict[str, Dict]:
        """加载手势模板（JSON格式）"""
        if not template_file:
            # 默认模板示例
            return {
                "fist": {
                    "ranges": [[0, 10], [0, 10], [0, 10], [0, 10], [0, 10], [0, 10], [0, 10], [0, 10],
                              [0, 10], [0, 10], [0, 10], [0, 10], [0, 10], [0, 10], [0, 10], [0, 10], [0, 10]],
                    "weights": [1.0] * 17  # 每个关节的权重
                },
                "open_hand": {
                    "ranges": [[70, 90], [70, 90], [70, 90], [70, 90], [70, 90], [70, 90], [70, 90], [70, 90],
                              [70, 90], [70, 90], [70, 90], [70, 90], [70, 90], [70, 90], [70, 90], [70, 90], [70, 90]],
                    "weights": [1.0] * 17
                },
                # 添加更多手势...
            }

        if os.path.exists(template_file):
            with open(template_file, 'r') as f:
                return json.load(f)
        return {}

    def match_gesture(self, joints: List[float]) -> Tuple[str, float, List[float]]:
        """
        匹配手势并返回校正后的关节角度
        返回: (手势名称, 置信度, 校正后的关节角度)
        """
        best_gesture = "unknown"
        best_score = 0.0
        corrected_joints = joints.copy()

        for gesture_name, template in self.templates.items():
            score = self._calculate_match_score(joints, template)
            if score > best_score:
                best_score = score
                best_gesture = gesture_name
                corrected_joints = self._correct_joints(joints, template)

        return best_gesture, best_score, corrected_joints

    def _calculate_match_score(self, joints: List[float], template: Dict) -> float:
        """计算匹配分数（0-1）"""
        ranges = template["ranges"]
        weights = template.get("weights", [1.0] * len(joints))

        total_score = 0.0
        total_weight = 0.0

        for i, (joint, joint_range, weight) in enumerate(zip(joints, ranges, weights)):
            min_val, max_val = joint_range
            if min_val <= joint <= max_val:
                score = 1.0
            else:
                # 距离范围的惩罚
                distance = min(abs(joint - min_val), abs(joint - max_val))
                score = max(0.0, 1.0 - distance / 20.0)  # 20度容差

            total_score += score * weight
            total_weight += weight

        return total_score / total_weight if total_weight > 0 else 0.0

    def _correct_joints(self, joints: List[float], template: Dict) -> List[float]:
        """根据模板校正关节角度"""
        ranges = template["ranges"]
        corrected = []

        for joint, joint_range in zip(joints, ranges):
            min_val, max_val = joint_range
            # 简单校正：拉到范围内
            if joint < min_val:
                corrected.append(min_val)
            elif joint > max_val:
                corrected.append(max_val)
            else:
                corrected.append(joint)

        return corrected

# 如果使用ML模型的示例（需要安装PyTorch）
class MLGestureMatcher(GestureMatcher):
    """使用PyTorch神经网络的手势匹配器"""

    def __init__(self, model_path: str, template_file: str = None):
        super().__init__(template_file)
        self._load_ml_model(model_path)

    def _load_ml_model(self, model_path: str):
        """加载PyTorch ML模型"""
        if not PYTORCH_AVAILABLE or GestureLSTM is None:
            print("PyTorch not installed, ML matching disabled")
            self.ml_model = None
            return

        try:
            # 加载完整的检查点
            checkpoint = torch.load(model_path, map_location='cpu')

            # 从检查点创建模型
            model = GestureLSTM(
                input_size=checkpoint['input_size'],
                hidden_size=checkpoint['hidden_size'],
                num_layers=checkpoint['num_layers'],
                num_classes=checkpoint['num_classes']
            )

            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()  # 设置为评估模式
            self.ml_model = model
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.ml_model.to(self.device)
        except Exception as e:
            print(f"Failed to load PyTorch model: {e}")
            self.ml_model = None

    def match_gesture(self, joints: List[float], history: List[List[float]] = None) -> Tuple[str, float, List[float]]:
        """使用PyTorch ML模型匹配手势"""
        if self.ml_model and history and len(history) >= 10:  # 需要时间序列
            if not PYTORCH_AVAILABLE:
                return super().match_gesture(joints)

            try:
                # 准备输入数据
                sequence = np.array(history[-10:])  # 最后10帧
                sequence = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)  # 添加batch维度

                # 预测
                with torch.no_grad():
                    outputs = self.ml_model(sequence)
                    probabilities = torch.softmax(outputs, dim=1)
                    confidence, predicted_idx = torch.max(probabilities, 1)

                gesture_idx = predicted_idx.item()
                confidence_score = confidence.item()

                gesture_names = list(self.templates.keys())
                gesture_name = gesture_names[gesture_idx] if gesture_idx < len(gesture_names) else "unknown"

                # 使用模板校正
                template = self.templates.get(gesture_name, {})
                corrected_joints = self._correct_joints(joints, template) if template else joints

                return gesture_name, float(confidence_score), corrected_joints
            except Exception as e:
                print(f"ML prediction failed: {e}")
                # 回退到模板匹配
                return super().match_gesture(joints)
        else:
            # 回退到模板匹配
            return super().match_gesture(joints)