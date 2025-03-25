import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from torch.utils.data import Dataset, DataLoader
import joblib  # 用于保存和加载 LabelEncoder
from app.model_training.cnn_rnn_spectrogram import log_results

# 自定义数据集
class EmotionDataset(Dataset):
    def __init__(self, grouped_features, labels):
        self.features = grouped_features  # 二维特征，每个样本为一个文件的特征矩阵
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.tensor(self.features[idx], dtype=torch.float32), self.labels[idx]


# 定义 RNN 模型
class EmotionClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(EmotionClassifier, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )

    def forward(self, x, lengths):
        # 使用 pack_padded_sequence 处理变长序列
        packed_input = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, (hn, _) = self.rnn(packed_input)
        out = self.fc(hn[-1])  # 使用最后时间步的隐藏状态进行分类
        return out


# 数据加载函数
def load_and_group_features(data_folder, file_name):
    """
    从指定路径加载CSV文件，并按 file_name 聚合特征数据。
    input:
    data_folder : 包含CSV文件的文件夹路径
    file_name : 要加载的CSV文件名
    Returns:
    grouped_features : 每个文件的特征矩阵列表（list of 2D arrays）
    labels : 每个文件的类别标签列表
    label_encoder : 编码器对象
    """
    csv_path = os.path.join(data_folder, file_name)
    print(f"加载特征文件: {csv_path}")
    df = pd.read_csv(csv_path)

    # 提取特征和标签
    labels = df["category"].values
    file_names = df["file_name"].values

    # 使用 LabelEncoder 编码类别标签
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)

    # 保存 LabelEncoder 到文件
    folder_name = os.path.basename(data_folder.rstrip("/"))  # 获取 input_folder 的最后一个文件夹名
    encoder_path = os.path.join("../models/label_encoder", f"{folder_name}_label_encoder.joblib")
    joblib.dump(label_encoder, encoder_path)
    print(f"LabelEncoder 已保存到: {encoder_path}")

    # 按 file_name 分组，聚合特征
    grouped_features = []
    grouped_labels = []
    for file_name, group in df.groupby("file_name"):
        group_features = group.drop(columns=["file_name", "category"]).values
        grouped_features.append(group_features)
        grouped_labels.append(labels[group.index[0]])  # 同一文件的标签相同

    # 对特征进行标准化
    scaler = StandardScaler()
    grouped_features = [scaler.fit_transform(group) for group in grouped_features]

    return grouped_features, grouped_labels, label_encoder


# 自定义 collate_fn
def collate_fn(batch):
    """
    自定义 collate_fn，用于填充序列并返回长度。
    Args:
        batch: 一个批次的样本 [(features1, label1), (features2, label2), ...]
    Returns:
        padded_features: 填充后的序列张量 (batch_size, max_seq_len, num_features)
        labels: 标签张量 (batch_size,)
        lengths: 每个序列的实际长度 (batch_size,)
    """
    features = [item[0] for item in batch]
    labels = torch.tensor([item[1] for item in batch], dtype=torch.long)

    # 获取每个序列的长度
    lengths = torch.tensor([len(f) for f in features], dtype=torch.long).to(features[0].device)  # 确保 lengths 与 features 在同一设备上

    # 使用 pad_sequence 对特征进行填充
    padded_features = pad_sequence(features, batch_first=True)  # (batch_size, max_seq_len, num_features)

    return padded_features, labels, lengths


# 模型训练函数
def train_model(model, criterion, optimizer, train_loader, val_loader, device, model_output, epochs=1,
                pretrained_model_path=None):
    model.to(device)
    best_val_acc = 0.0

    # 如果没有预训练模型路径，则直接从头训练
    if pretrained_model_path:
        if os.path.exists(pretrained_model_path):
            print(f"加载预训练模型: {pretrained_model_path}")
            model.load_state_dict(torch.load(pretrained_model_path))
        else:
            print(f"预训练模型文件 {pretrained_model_path} 不存在，从头开始训练")

    for epoch in range(epochs):
        model.train()
        train_loss, correct = 0.0, 0
        for features, labels, lengths in train_loader:
            features, labels, lengths = features.to(device), labels.to(device), lengths.to(device)

            optimizer.zero_grad()
            outputs = model(features, lengths.to(features.device))
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()

        train_acc = correct / len(train_loader.dataset)

        # 验证集评估
        model.eval()
        val_loss, val_correct = 0.0, 0
        with torch.no_grad():
            for features, labels, lengths in val_loader:
                features, labels, lengths = features.to(device), labels.to(device), lengths.to(device)
                outputs = model(features, lengths)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                val_correct += (outputs.argmax(1) == labels).sum().item()

        val_acc = val_correct / len(val_loader.dataset)

        print(f"Epoch [{epoch + 1}/{epochs}], Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        # 记录日志
        log_file = "../model_visible/rnn.txt"
        log_results(log_file, train_loss, train_acc, val_loss, val_acc)

        # 保存最优模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            print(f"保存最优模型 (Epoch {epoch + 1}) 到: {model_output}")
            torch.save(model.state_dict(), model_output)


# 主程序
if __name__ == "__main__":
    # 用户提供的路径
    input_folder = "../features/feature_extraction_2/CREMA-D"  # 包含 train.csv, val.csv, test.csv 的文件夹
    model_output = "../models/rnn_2.pth"  # 模型保存路径
    pretrained_model_path = "../models/rnn_2.pth"  # 预训练模型文件

    # 加载和分组数据
    train_features, train_labels, label_encoder = load_and_group_features(input_folder, "train.csv")
    val_features, val_labels, _ = load_and_group_features(input_folder, "val.csv")

    # 创建数据加载器
    train_dataset = EmotionDataset(train_features, train_labels)
    val_dataset = EmotionDataset(val_features, val_labels)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)

    # 模型定义
    input_size = train_features[0].shape[1]  # 每帧的特征数
    hidden_size = 256  # 增大隐藏层大小
    num_classes = len(label_encoder.classes_)
    model = EmotionClassifier(input_size, hidden_size, num_classes)

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # 使用较小的学习率

    # 训练模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_model(model, criterion, optimizer, train_loader, val_loader, device, model_output, epochs=10,
                pretrained_model_path=pretrained_model_path)
