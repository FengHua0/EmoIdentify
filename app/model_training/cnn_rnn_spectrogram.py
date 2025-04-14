import os
import json
import torch
import sys
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import torch.nn.functional as F

class CustomDataset(Dataset):
    def __init__(self, data_folder, split='train', img_size=(224, 224), transform=None):
        """
        自定义数据集，加载图像并返回图像、情感标签和说话人ID
        """
        self.data_folder = os.path.join(data_folder, split)
        self.transform = transform
        self.samples = []
        self.speaker_ids = {}

        # 遍历目录，加载每个文件并提取标签
        for label, emotion_folder in enumerate(os.listdir(self.data_folder)):
            emotion_path = os.path.join(self.data_folder, emotion_folder)
            if os.path.isdir(emotion_path):
                for file_name in os.listdir(emotion_path):
                    if file_name.endswith(".png"):
                        # 提取说话人ID (假设文件名中的第一部分是说话人ID)
                        speaker_id = file_name.split('_')[0]
                        self.samples.append((os.path.join(emotion_path, file_name), label, speaker_id))
                        if speaker_id not in self.speaker_ids:
                            self.speaker_ids[speaker_id] = len(self.speaker_ids)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, emotion_label, speaker_id = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        speaker_id = self.speaker_ids[speaker_id]  # 转换为数字索引
        return image, emotion_label, speaker_id

# 修改后的数据加载函数
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
        speaker_indices (dict): 说话人索引映射
    """
    print(f"加载数据集: {data_folder}")

    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                           [0.229, 0.224, 0.225])
    ])

    # 使用 CustomDataset 加载数据集
    train_dataset = CustomDataset(data_folder, split="train", img_size=img_size, transform=transform)
    val_dataset = CustomDataset(data_folder, split="val", img_size=img_size, transform=transform)
    test_dataset = CustomDataset(data_folder, split="test", img_size=img_size, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"数据加载完成: 训练集 {len(train_dataset)} 样本, 验证集 {len(val_dataset)} 样本, 测试集 {len(test_dataset)} 样本")

    # 获取类别索引
    class_indices = {i: emotion for i, emotion in enumerate(os.listdir(os.path.join(data_folder, "train")))}
    speaker_indices = train_dataset.speaker_ids

    # 保存类别映射
    class_save_path = "../models/label_encoder/CREMA-D_CNN_class.json"
    os.makedirs(os.path.dirname(class_save_path), exist_ok=True)
    with open(class_save_path, "w") as f:
        json.dump(class_indices, f)
    print(f"类别标签映射已保存: {class_save_path}")

    return train_loader, val_loader, test_loader, class_indices, speaker_indices

# 修改 CNN_RNN 类的 forward 方法以返回特征
class CNN_RNN(nn.Module):
    def __init__(self, num_classes=10, hidden_dim=128, num_layers=1, bidirectional=False):
        """
        Args:
            num_classes (int): 类别数量
            hidden_dim (int): LSTM 隐藏层维度
            num_layers (int): LSTM 层数
            bidirectional (bool): 是否使用双向 LSTM
        """
        super(CNN_RNN, self).__init__()

        # ------------ CNN 特征提取部分 ------------
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),  # [B, 16, 112, 112]

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # [B, 32, 56, 56]

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)  # [B, 64, 28, 28]
        )

        # 输出的 [B, 64, 28, 28] 视作 (B, C, H, W)
        # W = 时间维度，H = 频率维度
        # 在 forward 中，会将 [B, 64, 28, 28] reshape 成 [B, 28, 64*28]，再输入 LSTM

        # ------------ LSTM ------------
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        # LSTM 的输入维度 = 64*28（CNN 输出的通道数*频率维度）
        lstm_input_dim = 64 * 28
        self.lstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional
        )

        # ------------ 注意力机制 ------------
        # 这里使用一个简单的加性注意力
        # 如果是双向 LSTM，需要把 hidden_dim 变为 2 * hidden_dim
        self.att_dim = hidden_dim * (2 if bidirectional else 1)
        self.attention_fc = nn.Linear(self.att_dim, 1)

        # ------------ 分类层 ------------
        self.fc = nn.Linear(self.att_dim, num_classes)

    def attention(self, lstm_output):
        """
        加性注意力（时间维度上的注意力）

        Args:
            lstm_output: [B, T, H]，其中 T 为时间步数, H 为 LSTM 隐藏维度

        Returns:
            context: [B, H] 加权后的上下文向量
            att_weights: [B, T] 注意力权重（每个时间步一个权重）
        """
        # 计算注意力分数: energy = Linear(lstm_output) => [B, T, 1]
        energy = self.attention_fc(lstm_output)  # [B, T, 1]
        energy = energy.squeeze(-1)  # [B, T]

        att_weights = F.softmax(energy, dim=1)  # [B, T]

        # 加权求和得到 context
        att_weights_expanded = att_weights.unsqueeze(-1)  # [B, T, 1]
        context = torch.sum(lstm_output * att_weights_expanded, dim=1)  # [B, H]

        return context, att_weights

    def forward(self, x):
        """
        Args:
            x: [B, 3, 224, 224] 的输入梅尔频谱图（或其他图像）

        Returns:
            out: [B, num_classes] 最终分类输出
            att_weights: [B, T] 时间维度的注意力权重
        """
        # ------------ CNN ------------
        cnn_out = self.cnn(x)  # [B, 64, 28, 28]

        B, C, H, W = cnn_out.shape  # C=64, H=28, W=28
        # 假设 W=时间维度，H=频率维度
        # 将 (C, H) 合并为一个特征维度，然后把 W 当做时间维度
        # => [B, C*H, W]
        cnn_out = cnn_out.view(B, C * H, W)  # [B, 64*28, 28]

        # 交换维度到 [B, T, feature_dim] => T=28, feature_dim=64*28
        cnn_out = cnn_out.permute(0, 2, 1)  # [B, 28, 64*28]

        # ------------ LSTM ------------
        lstm_out, (h_n, c_n) = self.lstm(cnn_out)  # [B, T, hidden_dim] (若双向，则 hidden_dim=2*原设定)

        # ------------ 注意力 ------------
        context, att_weights = self.attention(lstm_out)  # context: [B, att_dim], att_weights: [B, T]

        # ------------ 分类 ------------
        out = self.fc(context)  # [B, num_classes]

        # 返回分类结果和特征向量
        return out, context  # context 是注意力加权后的特征向量

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

# 模型训练函数
def train_model(model, train_loader, val_loader, device, log_file, model_dir, epochs=10, lr=1e-3, resume_training=True):
    """
    训练模型并在验证集上进行评估
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(device)
    best_val_acc = 0.0
    best_val_loss = float("inf")

    model_path = "../models/cnn_rnn_spectrogram_model.pth"

    # 继续训练
    if resume_training and os.path.exists(model_path):
        print(f"加载已有模型: {model_path}")
        model.load_state_dict(torch.load(model_path))

    os.makedirs(model_dir, exist_ok=True)

    print("开始训练模型...")

    for epoch in range(epochs):
        # ------------ 训练 ------------
        model.train()
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0  # 记录已处理的样本数

        for batch_idx, (images, labels, _) in enumerate(train_loader):  # 修改这里，添加第三个返回值
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs, _ = model(images)  # 这里我们只需要分类输出
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            batch_loss = loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            batch_corrects = torch.sum(preds == labels.data)

            # 累计损失和正确样本数
            running_loss += batch_loss
            running_corrects += batch_corrects
            total_samples += images.size(0)

            # 计算当前批次之后的 Epoch 级别的准确率和损失
            train_loss = running_loss / total_samples
            train_acc = running_corrects.double() / total_samples

            print(f"\rEpoch [{epoch + 1}/{epochs}], Batch [{batch_idx + 1}/{len(train_loader)}]: "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}", end="")
            sys.stdout.flush()

        # ------------ 验证 ------------
        model.eval()
        val_running_loss = 0.0
        val_running_corrects = 0

        with torch.no_grad():
            for images, labels, _ in val_loader:  # 同样修改验证集的遍历
                images = images.to(device)
                labels = labels.to(device)
                outputs, _ = model(images)
                loss = criterion(outputs, labels)

                val_running_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                val_running_corrects += torch.sum(preds == labels.data)

        val_loss = val_running_loss / len(val_loader.dataset)
        val_acc = val_running_corrects.double() / len(val_loader.dataset)

        print(f"\nEpoch [{epoch + 1}/{epochs}] Summary: "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # 记录日志
        log_results(log_file, train_loss, train_acc, val_loss, val_acc)

        epoch_model_path = os.path.join(model_dir, f"cnn_rnn_spectrogram_model_{epoch+1}.pth")
        print(f"保存模型 (Epoch {epoch + 1}) 到: {epoch_model_path}")
        torch.save(model.state_dict(), epoch_model_path)

        # # 保存最优模型
        # if val_acc > best_val_acc and val_loss < best_val_loss:
        #     best_val_acc = val_acc
        #     best_val_loss = val_loss
        #     print(f"保存最优模型 (Epoch {epoch + 1}) 到: {model_output}")
        #     torch.save(model.state_dict(), model_output)


# 主程序
if __name__ == "__main__":
    # 设备检测
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    # 配置路径
    data_folder = "../features/mel_spectrogram/CREMA-D"
    model_dir = "../models/cnn_rnn"
    log_file = "../model_visible/cnn_rnn.txt"

    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_folder = os.path.join(current_dir, data_folder)
    model_dir = os.path.join(current_dir, model_dir)
    log_file = os.path.join(current_dir, log_file)

    # 训练参数
    batch_size = 64
    num_classes = 6  # CREMA-D 数据集的情感类别数
    epochs = 50
    lr = 1e-3
    weight_decay = 1e-5  # L2 正则化

    # 加载数据 - 修改为接收5个返回值
    train_loader, val_loader, test_loader, class_indices, speaker_indices = load_datasets(
        data_folder=data_folder,
        img_size=(224, 224),
        batch_size=batch_size
    )

    # 初始化模型
    model = CNN_RNN(num_classes=len(class_indices)).to(device)

    # 损失函数 & 优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)  # 添加 L2 正则化

    # 训练模型
    train_model(model, train_loader, val_loader, device, log_file, model_dir, epochs=epochs, lr=lr, resume_training=True)

