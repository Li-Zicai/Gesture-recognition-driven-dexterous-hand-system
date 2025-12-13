# train_gesture_model.py - 使用PyTorch训练手势识别模型
import numpy as np
import json
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

class GestureLSTM(nn.Module):
    """PyTorch LSTM模型用于手势识别"""

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

def load_training_data(data_file: str):
    """加载训练数据"""
    if not os.path.exists(data_file):
        print(f"Training data file {data_file} not found")
        return None, None

    with open(data_file, 'r') as f:
        data = json.load(f)

    sequences = []
    labels = []

    for gesture_name, gesture_data in data.items():
        for sequence in gesture_data['sequences']:
            sequences.append(sequence)
            labels.append(gesture_name)

    return np.array(sequences), np.array(labels)

def train_model():
    """训练PyTorch模型"""
    # 检查CUDA可用性
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 加载数据
    sequences, labels = load_training_data('gesture_training_data.json')
    if sequences is None:
        print("No training data available")
        return

    print(f"Loaded {len(sequences)} sequences with {len(set(labels))} gesture classes")

    # 编码标签
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        sequences, encoded_labels, test_size=0.2, random_state=42
    )

    # 转换为PyTorch张量
    X_train = torch.FloatTensor(X_train)
    X_test = torch.FloatTensor(X_test)
    y_train = torch.LongTensor(y_train)
    y_test = torch.LongTensor(y_test)

    # 创建数据加载器
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # 模型参数
    input_size = X_train.shape[2]  # 关节数量 (17)
    sequence_length = X_train.shape[1]  # 序列长度
    num_classes = len(label_encoder.classes_)

    # 创建模型
    model = GestureLSTM(
        input_size=input_size,
        hidden_size=64,
        num_layers=2,
        num_classes=num_classes,
        dropout=0.2
    ).to(device)

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练循环
    num_epochs = 50
    best_accuracy = 0.0

    print("Starting training...")
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # 验证
        model.eval()
        val_predictions = []
        val_targets = []

        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)

                val_predictions.extend(predicted.cpu().numpy())
                val_targets.extend(targets.cpu().numpy())

        val_accuracy = accuracy_score(val_targets, val_predictions)

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss/len(train_loader):.4f}, Val Accuracy: {val_accuracy:.4f}')

        # 保存最佳模型
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(model.state_dict(), 'gesture_model_best.pth')
            print(f"Best model saved with accuracy: {best_accuracy:.4f}")

    # 保存最终模型
    torch.save({
        'model_state_dict': model.state_dict(),
        'input_size': input_size,
        'hidden_size': 64,
        'num_layers': 2,
        'num_classes': num_classes,
        'sequence_length': sequence_length
    }, 'gesture_model.pth')

    # 保存标签编码器
    np.save('label_encoder.npy', label_encoder.classes_)

    print(f"Training completed. Best validation accuracy: {best_accuracy:.4f}")
    print("Model saved as 'gesture_model.pth'")
    print("Label encoder saved as 'label_encoder.npy'")

def load_and_test_model():
    """加载并测试模型（可选）"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载检查点
    checkpoint = torch.load('gesture_model.pth')
    model = GestureLSTM(
        input_size=checkpoint['input_size'],
        hidden_size=checkpoint['hidden_size'],
        num_layers=checkpoint['num_layers'],
        num_classes=checkpoint['num_classes']
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 加载标签编码器
    label_encoder = LabelEncoder()
    label_encoder.classes_ = np.load('label_encoder.npy', allow_pickle=True)

    print("Model loaded successfully")
    print(f"Classes: {label_encoder.classes_}")

if __name__ == "__main__":
    train_model()
    # 可选：加载并测试模型
    # load_and_test_model()