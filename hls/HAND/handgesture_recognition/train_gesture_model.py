# train_gesture_model.py - 完整升级版
import numpy as np
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
import joblib  # 用于保存归一化参数 pip install joblib

# --- 配置参数 ---
JSON_PATH = 'gesture_training_data.json'
MODEL_SAVE_PATH = 'gesture_model.pth'
SCALER_SAVE_PATH = 'scaler.pkl'
TARGET_SEQ_LEN = 60  # 【重要】统一序列长度，建议60帧(约2秒)
HIDDEN_SIZE = 64
NUM_LAYERS = 2
BATCH_SIZE = 32
EPOCHS = 100
LR = 0.001


# --- 1. 定义模型 ---
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
        out = out[:, -1, :]  # 取最后一个时间步
        out = self.dropout(out)
        out = self.relu(self.fc1(out))
        out = self.fc2(out)
        return out


# --- 2. 数据增强函数 ---
def augment_data(X, y, noise_level=0.05):
    """
    数据增强：给原始数据添加高斯噪声
    noise_level: 噪声强度，归一化后的数据通常在 -3~3 之间，0.05 是合理的抖动
    """
    print(f"Applying data augmentation (Noise level: {noise_level})...")
    noise = np.random.normal(0, noise_level, X.shape)
    X_aug = X + noise

    # 将原始数据和增强数据合并
    X_combined = np.concatenate((X, X_aug), axis=0)
    y_combined = np.concatenate((y, y), axis=0)

    return X_combined, y_combined


# --- 3. 数据处理核心 ---
def load_and_process_data(json_path, target_len=60):
    print(f"Loading data from {json_path}...")
    with open(json_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    X_list = []
    y_list = []

    for label_name, content in raw_data.items():
        sequences = content.get('sequences', [])
        if not sequences: continue
        print(f"Processing '{label_name}': {len(sequences)} sequences found.")

        for seq in sequences:
            seq_arr = np.array(seq, dtype=np.float32)
            if seq_arr.size == 0: continue

            curr_len = seq_arr.shape[0]

            # Padding / Truncating
            if curr_len < target_len:
                pad_width = ((0, target_len - curr_len), (0, 0))
                seq_processed = np.pad(seq_arr, pad_width, mode='constant', constant_values=0)
            else:
                seq_processed = seq_arr[:target_len, :]  # 截取前60帧

            X_list.append(seq_processed)
            y_list.append(label_name)

    if not X_list:
        raise ValueError("Error: No valid data found.")

    X = np.array(X_list)  # (Samples, TimeSteps, Features)
    y = np.array(y_list)

    # --- 归一化 (Standardization) ---
    # LSTM输入是3维，StandardScaler只能处理2维，需要先拍扁
    num_samples, seq_len, num_features = X.shape
    X_reshaped = X.reshape(-1, num_features)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_reshaped)

    # 保存 Scaler 供推理使用 (这一步至关重要！)
    joblib.dump(scaler, SCALER_SAVE_PATH)
    print(f"Scaler saved to {SCALER_SAVE_PATH}")

    # 变回3维
    X = X_scaled.reshape(num_samples, seq_len, num_features)

    print(f"Data processing complete. Shape: {X.shape}")
    return X, y


# --- 4. 训练流程 ---
def train_model():
    # 1. 加载数据
    try:
        X, y_raw = load_and_process_data(JSON_PATH, target_len=TARGET_SEQ_LEN)
    except Exception as e:
        print(f"Data Load Error: {e}")
        return

    # 2. 标签编码
    label_encoder = LabelEncoder()
    y_enc = label_encoder.fit_transform(y_raw)
    num_classes = len(label_encoder.classes_)
    np.save('classes.npy', label_encoder.classes_)
    print(f"Classes: {label_encoder.classes_}")

    # 3. 数据增强 (在归一化之后进行)
    X, y_enc = augment_data(X, y_enc, noise_level=0.1)
    print(f"Data shape after augmentation: {X.shape}")

    # 4. 划分数据集
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.LongTensor(y_enc)

    X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)

    train_ds = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    # 5. 初始化模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_size = X.shape[2]
    model = GestureLSTM(input_size, HIDDEN_SIZE, NUM_LAYERS, num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # 6. 训练
    loss_history = []
    print(f"Start Training on {device}...")

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            output = model(batch_X)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        loss_history.append(avg_loss)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{EPOCHS}], Loss: {avg_loss:.4f}")

    # 7. 保存
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'input_size': input_size,
        'hidden_size': HIDDEN_SIZE,
        'num_layers': NUM_LAYERS,
        'num_classes': num_classes,
        'target_len': TARGET_SEQ_LEN,
        'classes': label_encoder.classes_.tolist()
    }
    torch.save(checkpoint, MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")

    # 8. 评估
    model.eval()
    with torch.no_grad():
        test_out = model(X_test.to(device))
        _, predicted = torch.max(test_out, 1)
        acc = (predicted == y_test.to(device)).sum().item() / len(y_test)
        print(f"Test Accuracy: {acc * 100:.2f}%")

    plt.plot(loss_history)
    plt.savefig('loss_plot.png')


if __name__ == "__main__":
    train_model()