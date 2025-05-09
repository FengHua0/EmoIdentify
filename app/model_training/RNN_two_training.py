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
import sys
import time

def log_results(log_file,train_loss, train_acc, val_loss, val_acc):
    """
    将每个 epoch 的训练结果记录到日志文件
    Args:
        log_file (str): 日志文件路径
        epoch (int): 当前 epoch
        train_loss (float): 训练损失
        train_acc (float): 训练准确率
        val_loss (float): 验证损失
        val_acc (float): 验证准确率
    """
    with open(log_file, "a") as f:
        f.write(f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f} | "
                f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}\n")

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
        self.rnn = nn.LSTM(input_size, hidden_size, batch_first=True, num_layers=2, bidirectional=True)
        self.bn1 = nn.BatchNorm1d(hidden_size * 2)  # 双向LSTM输出是hidden_size*2
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x, lengths):
        packed_input = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, (hn, _) = self.rnn(packed_input)
        
        # 处理双向LSTM的输出
        if self.rnn.bidirectional:
            hn = torch.cat((hn[-2], hn[-1]), dim=1)
        else:
            hn = hn[-1]
            
        hn = self.bn1(hn)
        out = self.fc(hn)
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
    current_dir = os.path.dirname(os.path.abspath(__file__))
    encoder_path = os.path.join(current_dir, encoder_path)
    joblib.dump(label_encoder, encoder_path)
    print(f"LabelEncoder 已保存到: {encoder_path}")

    # 按 file_name 和 category 双重分组，确保文件名和类别一致
    grouped_features = []
    grouped_labels = []
    groups = df.groupby(["file_name", "category"])
    for (file_name, category), group in groups:
        group_features = group.drop(columns=["file_name", "category"]).values
        grouped_features.append(group_features)
        grouped_labels.append(labels[group.index[0]])  # 同一文件的标签相同
    
    print(f"共找到 {len(groups)} 个文件-类别组合")  # 添加打印分组数量
    print(f"实际处理样本数: {len(grouped_features)}")  # 添加打印实际样本数


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
def train_model(model, criterion, optimizer, train_loader, val_loader, device, model_output, log_file, epochs=1,
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
        # 记录开始时间
        epoch_start = time.time()
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
        # 计算epoch耗时
        epoch_time = time.time() - epoch_start
        print(f"Epoch [{epoch + 1}/{epochs}] 耗时: {epoch_time:.2f}秒")

        print(f"Epoch [{epoch + 1}/{epochs}], Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        # 记录日志
        log_results(log_file, train_loss, train_acc, val_loss, val_acc)

        # 获取文件夹名
        last_folder = os.path.basename(os.path.normpath(input_folder))
        model_path = os.path.join(model_output, f"rnn2_epoch_{epoch + 1}.pth")
        print(f"保存模型 (Epoch {epoch + 1}) 到: {model_path}")
        torch.save(model.state_dict(), model_path)


# 主程序
if __name__ == "__main__":
    # 用户提供的路径
    input_folder = "../features/feature_extraction_2/CASIA"  # 包含 train.csv, val.csv, test.csv 的文件夹
    last_folder = os.path.basename(os.path.normpath(input_folder))
    model_output = f"../models/{last_folder}_rnn_2"  # 现在是文件夹路径
    pretrained_model_path = "../models/rnn_2.pth"
    log_file = "result.txt"

    current_dir = os.path.dirname(os.path.abspath(__file__))
    input_folder = os.path.join(current_dir, input_folder)
    model_output = os.path.join(current_dir, model_output)
    os.makedirs(model_output, exist_ok=True)  # 确保保存文件夹存在
    pretrained_model_path = os.path.join(current_dir, pretrained_model_path)
    log_file = os.path.join(model_output, log_file)

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
    train_model(model, criterion, optimizer, train_loader, val_loader, device, model_output, log_file, epochs=50,
                pretrained_model_path=pretrained_model_path)
