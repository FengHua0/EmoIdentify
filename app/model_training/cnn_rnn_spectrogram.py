import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import Dataset, DataLoader
import numpy as np

# 设备检测
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")


# 数据加载函数
def load_datasets(data_folder, img_size=(224, 224), batch_size=64):
    """
    加载数据集并返回 DataLoader

    Args:
        data_folder (str): 数据集所在路径
        img_size (tuple): 图像大小，默认为 (224, 224)
        batch_size (int): 批量大小

    Returns:
        train_loader, val_loader, test_loader (DataLoader): 训练、验证、测试集的数据加载器
        class_indices (dict): 类别索引映射
    """
    print(f"加载数据集: {data_folder}")

    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    train_dataset = datasets.ImageFolder(os.path.join(data_folder, "train"), transform=transform)
    val_dataset = datasets.ImageFolder(os.path.join(data_folder, "val"), transform=transform)
    test_dataset = datasets.ImageFolder(os.path.join(data_folder, "test"), transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"数据加载完成: 训练集 {len(train_dataset)} 样本, 验证集 {len(val_dataset)} 样本, 测试集 {len(test_dataset)} 样本")

    # 获取类别索引并保存
    class_indices = train_dataset.class_to_idx
    index_to_class = {v: k for k, v in class_indices.items()}
    save_path = "../models/label_encoder/CREMA-D_CNN.json"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(index_to_class, f)
    print(f"类别标签映射已保存: {save_path}")

    return train_loader, val_loader, test_loader, class_indices


# 自定义 CNN-RNN 模型
class CNN_RNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN_RNN, self).__init__()

        # CNN 部分
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)  # 额外增加一层
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.5)

        # RNN 部分
        self.lstm = nn.LSTM(input_size=512, hidden_size=256, num_layers=2, batch_first=True, dropout=0.3)

        # 全连接层
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = self.pool(torch.relu(self.conv4(x)))  # 额外增加的 CNN 层

        # 变换为 (batch_size, 时间步, 特征数) 格式供 RNN 使用
        x = x.view(x.size(0), -1, 512)
        x, _ = self.lstm(x)
        x = self.dropout(torch.relu(self.fc1(x[:, -1, :])))
        x = self.fc2(x)
        return x


def log_results(log_file, epoch, train_loss, train_acc, val_loss, val_acc):
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
        f.write(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f} | "
                f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}\n")

# 模型训练函数
def train_model(model, criterion, optimizer, train_loader, val_loader, device, model_output, epochs=20,
                resume_training=False):
    """
    训练 CNN-RNN 模型，并在训练过程中持续输出 batch 级别的损失和准确率。

    Args:
        model (nn.Module): CNN-RNN 模型
        criterion (nn.Module): 损失函数
        optimizer (optim.Optimizer): 优化器
        train_loader (DataLoader): 训练集
        val_loader (DataLoader): 验证集
        device (torch.device): 计算设备
        model_output (str): 模型保存路径
        epochs (int): 训练轮数
        resume_training (bool): 是否加载已有模型继续训练

    Returns:
        None
    """
    model.to(device)
    best_val_acc = 0.0
    best_val_loss = 1000

    # 继续训练
    if resume_training and os.path.exists(model_output):
        print(f"加载已有模型: {model_output}")
        model.load_state_dict(torch.load(model_output, weights_only=True))

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    print("开始训练模型...")

    for epoch in range(epochs):
        model.train()
        total_loss, correct = 0.0, 0

        print(f"\nEpoch {epoch + 1}/{epochs} 开始训练...")
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()

            # 计算当前 batch 训练准确率
            batch_acc = (outputs.argmax(1) == labels).sum().item() / labels.size(0)

            # 实时输出 batch 级别的损失和准确率
            print(f"\rEpoch [{epoch + 1}/{epochs}] | Batch [{batch_idx + 1}/{len(train_loader)}] "
                  f"Loss: {loss.item():.4f} | Batch Acc: {batch_acc:.4f}", end='', flush=True)

        # 计算 epoch 级别的训练损失和准确率
        train_acc = correct / len(train_loader.dataset)
        train_loss = total_loss / len(train_loader)

        # 验证
        model.eval()
        val_loss, val_correct = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                val_correct += (outputs.argmax(1) == labels).sum().item()

        val_acc = val_correct / len(val_loader.dataset)
        val_loss /= len(val_loader)

        # 清除上一行的 `\r` 影响，并固定最终结果
        print(f"\nEpoch [{epoch + 1}/{epochs}] -> "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        # 记录日志
        log_file = "../model_visible/cnn_rnn.txt"
        log_results(log_file, epoch + 1, train_loss, train_acc, val_loss, val_acc)

        # 保存最优模型
        if val_acc > best_val_acc and val_loss < best_val_loss:
            best_val_acc = val_acc
            best_val_loss = val_loss
            print(f"保存最优模型 (Epoch {epoch + 1}) 到: {model_output}")
            torch.save(model.state_dict(), model_output)


# 主程序
if __name__ == "__main__":
    # 配置路径
    data_folder = "../features/mel_spectrogram/CREMA-D"
    model_output = "../models/cnn_rnn_spectrogram_model.pth"

    # 加载数据
    train_loader, val_loader, test_loader, class_indices = load_datasets(data_folder)

    # 初始化模型
    model = CNN_RNN(num_classes=len(class_indices))

    # 损失函数 & 优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # 训练模型
    train_model(model, criterion, optimizer, train_loader, val_loader, device, model_output, epochs=30,
                resume_training=True)
