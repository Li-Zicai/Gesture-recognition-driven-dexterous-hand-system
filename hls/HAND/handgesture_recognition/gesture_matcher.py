# gesture_matcher.py - 完整升级版
import numpy as np
import json
import os
import joblib  # [新增]
from typing import List, Tuple, Dict

try:
    import torch
    import torch.nn as nn

    PYTORCH_AVAILABLE = True


    # 定义必须与训练时完全一致的模型结构
    class GestureLSTM(nn.Module):
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
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
            out, _ = self.lstm(x, (h0, c0))
            out = out[:, -1, :]
            out = self.dropout(out)
            out = self.relu(self.fc1(out))
            out = self.fc2(out)
            return out

except ImportError:
    PYTORCH_AVAILABLE = False
    GestureLSTM = None


class GestureMatcher:
    def __init__(self, template_file: str = None):
        self.templates = self._load_templates(template_file)
        self.ml_model = None

    def _load_templates(self, template_file: str) -> Dict:
        if template_file and os.path.exists(template_file):
            with open(template_file, 'r') as f:
                return json.load(f)
        return {}  # 返回空字典或你的默认模板

    def match_gesture(self, joints: List[float]) -> Tuple[str, float, List[float]]:
        # 默认回退逻辑：只做简单的模板匹配
        return "unknown", 0.0, joints

    def _correct_joints(self, joints, template):
        # 简单的范围截断逻辑
        ranges = template.get("ranges", [])
        corrected = []
        for i, val in enumerate(joints):
            if i < len(ranges):
                min_v, max_v = ranges[i]
                corrected.append(max(min_v, min(max_v, val)))
            else:
                corrected.append(val)
        return corrected


class MLGestureMatcher(GestureMatcher):
    def __init__(self, model_path: str, template_file: str = None):
        super().__init__(template_file)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.scaler = None
        self.target_len = 60  # 默认值，会从 checkpoint 更新
        self.classes = []
        self._load_ml_model(model_path)

    def _load_ml_model(self, model_path: str):
        if not PYTORCH_AVAILABLE:
            print("PyTorch not available.")
            return

        try:
            print(f"Loading model from {model_path}...")
            checkpoint = torch.load(model_path, map_location=self.device)

            # 1. 加载模型结构参数
            self.target_len = checkpoint.get('target_len', 60)
            self.classes = checkpoint.get('classes', [])

            self.ml_model = GestureLSTM(
                input_size=checkpoint['input_size'],
                hidden_size=checkpoint['hidden_size'],
                num_layers=checkpoint['num_layers'],
                num_classes=checkpoint['num_classes']
            )
            self.ml_model.load_state_dict(checkpoint['model_state_dict'])
            self.ml_model.to(self.device)
            self.ml_model.eval()

            # 2. 加载 Scaler (在模型同目录下找 scaler.pkl)
            scaler_path = os.path.join(os.path.dirname(model_path), 'scaler.pkl')
            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)
                print("✓ Scaler loaded.")
            else:
                print("⚠ Warning: Scaler not found! Prediction will be inaccurate.")

        except Exception as e:
            print(f"Error loading model: {e}")
            self.ml_model = None

    def match_gesture(self, joints: List[float], history: List[List[float]] = None) -> Tuple[str, float, List[float]]:
        # 如果没有历史数据或模型未加载，回退到基类逻辑
        if not self.ml_model or not history:
            return super().match_gesture(joints)

        try:
            # 1. 准备数据：取最近的 history
            # 如果 history 长度不够 target_len，需要 Pad
            sequence = np.array(history, dtype=np.float32)
            curr_len = sequence.shape[0]

            if curr_len < self.target_len:
                # 补零
                pad_width = ((0, self.target_len - curr_len), (0, 0))
                sequence = np.pad(sequence, pad_width, mode='constant')
            else:
                # 截取最后 target_len 帧
                sequence = sequence[-self.target_len:, :]

            # 2. 归一化 (关键步骤)
            if self.scaler:
                # scaler 需要 2D 输入 (TimeSteps, Features)
                sequence = self.scaler.transform(sequence)

            # 3. 转 Tensor 并预测
            input_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)  # (1, Seq, Feat)

            with torch.no_grad():
                outputs = self.ml_model(input_tensor)
                probs = torch.softmax(outputs, dim=1)
                confidence, predicted_idx = torch.max(probs, 1)

            idx = predicted_idx.item()
            conf = confidence.item()

            gesture_name = self.classes[idx] if idx < len(self.classes) else "unknown"

            # 4. 获取对应的模板进行关节修正
            template = self.templates.get(gesture_name, {})
            corrected_joints = self._correct_joints(joints, template) if template else joints

            return gesture_name, conf, corrected_joints

        except Exception as e:
            print(f"Prediction Error: {e}")
            return super().match_gesture(joints)