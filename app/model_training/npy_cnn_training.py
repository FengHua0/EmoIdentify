import os
import json
import torch
import sys # 添加 sys 用于 flush 输出
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import time
import torch.nn.functional as F


def load_datasets(data_folder, batch_size=64, target_length=100):
    """
    加载.npy数据集并返回DataLoader
    
    Args:
        data_folder (str): 数据集所在路径
        batch_size (int): 批量大小
        target_length (int): 目标时间维度长度
        
    Returns:
        train_loader, val_loader, test_loader (DataLoader): 训练、验证、测试集的数据加载器
        class_indices (dict): 类别索引映射
        speaker_indices (dict): 说话人索引映射
    """
    print(f"加载数据集: {data_folder}")

    # 使用NPYDataset加载数据集
    train_dataset = NPYDataset(data_folder, split="train", target_length=target_length)
    val_dataset = NPYDataset(data_folder, split="val", target_length=target_length)
    test_dataset = NPYDataset(data_folder, split="test", target_length=target_length)
    
    # 创建DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"数据加载完成: 训练集 {len(train_dataset)} 样本, 验证集 {len(val_dataset)} 样本, 测试集 {len(test_dataset)} 样本")

    # 获取类别索引并转换为您需要的格式
    class_indices = {i: emotion for i, emotion in enumerate(os.listdir(os.path.join(data_folder, "train")))}
    
    # 提取说话人ID（从文件名的开头数字）
    speaker_indices = train_dataset.speaker_ids

    # 保存类别映射
    class_save_path = "../models/label_encoder/CREMA-D_CNN_class.json"
    current_dir = os.path.dirname(os.path.abspath(__file__))
    class_save_path = os.path.join(current_dir, class_save_path)
    os.makedirs(os.path.dirname(class_save_path), exist_ok=True)
    with open(class_save_path, "w") as f:
        json.dump(class_indices, f)

    print(f"类别标签映射已保存: {class_save_path}")

    # 保存说话人标签映射
    speaker_save_path = "../models/label_encoder/CREMA-D_CNN_speaker.json"
    speaker_save_path = os.path.join(current_dir, speaker_save_path)
    os.makedirs(os.path.dirname(speaker_save_path), exist_ok=True)
    with open(speaker_save_path, "w") as f:
        json.dump(speaker_indices, f)

    print(f"说话人标签映射已保存: {speaker_save_path}")

    return train_loader, val_loader, test_loader, class_indices, speaker_indices
class NPYDataset(Dataset):
    def __init__(self, data_folder, split='train', target_length=100):
        """
        自定义数据集，加载.npy文件并返回特征、情感标签和说话人ID

        Args:
            data_folder (str): 数据集路径
            split (str): 数据集分割 ('train', 'val', 'test')
            target_length (int): 目标时间维度长度
        """
        self.data_folder = os.path.join(data_folder, split)
        self.samples = []
        self.speaker_ids = {} # 仍然收集 speaker_ids 以保持一致性，即使训练中不用
        self.target_length = target_length

        # 遍历目录，加载每个文件并提取标签
        for label, emotion_folder in enumerate(os.listdir(self.data_folder)):
            emotion_path = os.path.join(self.data_folder, emotion_folder)
            if os.path.isdir(emotion_path):
                for file_name in os.listdir(emotion_path):
                    if file_name.endswith(".npy"):
                        # 提取说话人ID (假设ID嵌入文件名，形如 1001_DFA_ANG_XX.npy)
                        speaker_id = file_name.split('_')[0]
                        self.samples.append((os.path.join(emotion_path, file_name), label, speaker_id))
                        if speaker_id not in self.speaker_ids:
                            self.speaker_ids[speaker_id] = len(self.speaker_ids)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        npy_path, emotion_label, speaker_id_str = self.samples[idx] # 使用 speaker_id_str 避免覆盖

        # 加载.npy文件
        features = np.load(npy_path)

        # 确保特征维度正确 (n_mels, time)
        if features.ndim == 3 and features.shape[0] == 1:
            features = features.squeeze(0)

        # 统一时间维度长度
        current_length = features.shape[1]
        if current_length > self.target_length:
            start = 0 # 从头截取
            features = features[:, start:start + self.target_length]
        elif current_length < self.target_length:
            pad_size = self.target_length - current_length
            # 使用 constant 模式填充，默认值为 0
            features = np.pad(features, ((0, 0), (0, pad_size)), mode='constant')

        # 转换为PyTorch张量并添加通道维度 (1, n_mels, time)
        features = torch.from_numpy(features).float().unsqueeze(0)

        # 复制单通道为三通道
        features = features.repeat(3, 1, 1)  # [3, n_mels, time]

        # 获取说话人ID的整数索引
        speaker_id_int = self.speaker_ids[speaker_id_str]

        # 返回 features, emotion_label, speaker_id_int 以保持与 contrastive 一致
        return features, emotion_label, speaker_id_int

class CNN_RNN(nn.Module):
    def __init__(self, num_classes=10, n_mels=128, target_length=100, hidden_dim=128, num_layers=1, bidirectional=False):
        super(CNN_RNN, self).__init__()
        self.n_mels = n_mels
        self.target_length = target_length

        # CNN部分，输入为3通道
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),  # H, W -> H/2, W/2

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # H/2, W/2 -> H/4, W/4

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)  # H/4, W/4 -> H/8, W/8
        )

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        # 动态计算LSTM输入维度
        self.lstm_input_dim = self._get_lstm_input_dim()
        print(f"计算得到的 LSTM 输入维度: {self.lstm_input_dim}")

        # LSTM层
        self.lstm = nn.LSTM(
            input_size=self.lstm_input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional
        )

        # 注意力机制
        attention_dim = hidden_dim * (2 if bidirectional else 1)
        self.attention_fc = nn.Linear(attention_dim, 1)

        # 分类层
        self.fc = nn.Linear(attention_dim, num_classes)

    def _get_lstm_input_dim(self):
        # 创建一个符合预期的虚拟输入
        # 注意：这里使用 self.n_mels 和 self.target_length
        dummy_input = torch.randn(1, 3, self.n_mels, self.target_length)
        with torch.no_grad():
            cnn_out = self.cnn(dummy_input)
            # 计算 CNN 输出特征图的高度 H' 和宽度 W' 后的维度
            # CNN 输出形状: [B, C_out, H_out, W_out]
            # LSTM 输入需要 [B, SeqLen, Features]
            # 这里我们将 W_out 视为序列长度 SeqLen
            # Features = C_out * H_out
            return cnn_out.size(1) * cnn_out.size(2) # C_out * H_out

    def attention(self, lstm_output):
        """
        加性注意力机制
        :param lstm_output: [B, T, H] (T=W', H=hidden_dim * num_directions)
        :return: context: [B, H], att_weights: [B, T]
        """
        # lstm_output: [B, W', hidden_dim * num_directions]
        energy = torch.tanh(self.attention_fc(lstm_output)).squeeze(-1)  # [B, W']
        att_weights = F.softmax(energy, dim=1)  # [B, W']
        # context = torch.bmm(att_weights.unsqueeze(1), lstm_output).squeeze(1) # [B, H]
        context = torch.sum(lstm_output * att_weights.unsqueeze(-1), dim=1) # [B, H]
        return context, att_weights

    def forward(self, x):
        """
        :param x: [B, 3, n_mels, target_length]
        :return: 分类输出 (out), 特征向量 (context)
        """
        # x: [B, 3, n_mels, target_length]
        cnn_out = self.cnn(x)  # [B, 64, H', W'] (H'=n_mels/8, W'=target_length/8)
        B, C, H, W = cnn_out.shape

        # Reshape for LSTM: [B, SeqLen, Features]
        # SeqLen = W', Features = C * H'
        cnn_out = cnn_out.view(B, C * H, W).permute(0, 2, 1)  # [B, W', C*H']

        # LSTM 输入: [B, W', C*H']
        lstm_out, _ = self.lstm(cnn_out)  # [B, W', hidden_dim * num_directions]

        # 应用注意力机制
        # context: [B, hidden_dim * num_directions]
        context, _ = self.attention(lstm_out)

        # 分类层
        # 输入 context: [B, hidden_dim * num_directions]
        out = self.fc(context) # [B, num_classes]

        # 返回分类输出和用于对比学习的特征向量
        return out, context

# --- 添加 calculate_accuracy 函数 (如果尚未存在) ---
def calculate_accuracy(outputs, labels):
    _, predicted = torch.max(outputs, 1)
    correct = (predicted == labels).sum().item()
    total = labels.size(0)
    accuracy = correct / total if total > 0 else 0 # 避免除以零
    return accuracy

# --- 添加 log_results 函数 (如果尚未存在或需要更新) ---
def log_results(log_file, train_loss, train_acc, val_loss, val_acc):
    """
    将每个epoch的训练结果记录到日志文件 (与 contrastive 版本对齐)
    """
    with open(log_file, "a") as f:
        f.write(f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f} | "
                f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}\n")


# --- 修改 train_model 函数 ---
def train_model(model, train_loader, val_loader, device, model_output, log_file, # model_output 现在是目录
               epochs=10, lr=1e-3, resume_training=True, pre_model=None):
    # --- 仅使用 CrossEntropyLoss ---
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.to(device)

    # --- 恢复训练逻辑  ---
    start_epoch = 0

    # 恢复训练
    if resume_training and pre_model:
        if os.path.exists(pre_model):
            print(f"加载已有模型: {pre_model}")
            model.load_state_dict(torch.load(pre_model))
        else:
            print(f"预训练模型 {pre_model} 不存在，跳过恢复训练。初始化新的模型。")

    # --- 确保模型输出目录存在 (model_output 是目录) ---
    os.makedirs(model_output, exist_ok=True) # 直接使用 model_output 作为目录

    print("开始训练模型 (仅分类损失)...")

    # --- 训练循环 (与 contrastive 版本对齐) ---
    for epoch in range(start_epoch, epochs): # 从 start_epoch 开始
        epoch_start = time.time()
        model.train()
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0

        # DataLoader 返回 (images, labels, speaker_ids)，但 speaker_ids 在此不用
        for batch_idx, (images, labels, _) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            # speaker_ids = speaker_ids.to(device) # 不需要移到设备

            optimizer.zero_grad()

            # 模型 forward 返回 (outputs, context)
            outputs, context = model(images) # context 在这里不使用

            # --- 仅计算分类损失 ---
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            # 累加损失和准确率指标
            batch_loss = loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            batch_corrects = torch.sum(preds == labels.data)

            running_loss += batch_loss
            running_corrects += batch_corrects
            total_samples += images.size(0)

            # 计算平均值
            train_loss = running_loss / total_samples if total_samples > 0 else 0
            train_acc = running_corrects.double() / total_samples if total_samples > 0 else 0

            # 打印训练进度
            print(f"\rEpoch [{epoch + 1}/{epochs}], Batch [{batch_idx + 1}/{len(train_loader)}]: "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}", end="")
            sys.stdout.flush() # 强制刷新输出缓冲区

        # ------------ 验证 ------------
        model.eval()
        val_running_loss = 0.0
        val_running_corrects = 0
        val_total_samples = 0

        with torch.no_grad():
            # 验证 DataLoader 也返回 speaker_ids，即使不用
            for inputs, labels, _ in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # 模型前向传播
                outputs, context = model(inputs) # context 在这里不使用

                # --- 仅计算分类损失 (用于评估) ---
                loss = criterion(outputs, labels)

                # 累加验证损失
                val_running_loss += loss.item() * inputs.size(0)

                # 计算并累加验证准确率
                _, preds = torch.max(outputs, 1)
                val_running_corrects += torch.sum(preds == labels.data)
                val_total_samples += labels.size(0)

        # 计算平均验证损失和准确率
        val_loss = val_running_loss / val_total_samples if val_total_samples > 0 else 0
        val_acc = val_running_corrects.double() / val_total_samples if val_total_samples > 0 else 0

        epoch_time = time.time() - epoch_start
        print(f" 耗时: {epoch_time:.2f}秒") # 移到打印 Epoch 总结之前

        # 打印 Epoch 总结
        print(f"\nEpoch [{epoch + 1}/{epochs}] Summary: "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # 记录日志
        log_results(log_file, train_loss, train_acc, val_loss, val_acc)

        # --- 模型保存逻辑  ---
        model_path = os.path.join(model_output, f"npy_cnn_model_{epoch + 1}.pth") 

        try:
            # --- 只保存模型 state_dict ---
            torch.save(model.state_dict(), model_path)
            print(f"模型已保存到: {model_path}")
        except Exception as e:
            print(f"保存模型 (Epoch {epoch + 1}) 时出错: {e}")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # --- 配置参数 (与 contrastive 版本对齐) ---
    data_folder = "../features/mel_npy/CREMA-D"
    pre_model = "../models/npy_cnn_model.pth"
    model_output = "../models/npy_cnn"
    log_file = "../model_visible/npy_cnn.txt"

    # --- 解析路径 (与 contrastive 版本对齐) ---
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_folder = os.path.join(current_dir, data_folder)
    if pre_model:
        pre_model = os.path.join(current_dir, pre_model)
    model_output = os.path.join(current_dir, model_output) # 解析目录路径
    log_file = os.path.join(current_dir, log_file)

    # --- 训练超参数  ---
    batch_size = 64
    epochs = 85
    lr = 1e-4
    target_length = 100
    num_workers = 0

    # --- 加载数据 ---
    train_loader, val_loader, test_loader, class_indices, speaker_ids_map = load_datasets(
        data_folder=data_folder,
        batch_size=batch_size,
        target_length=target_length
    )

    # --- 初始化模型 ---
    num_classes = len(class_indices)
    model = CNN_RNN(num_classes=num_classes, n_mels=128, target_length=target_length).to(device)

    # --- 开始训练  ---
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        model_output=model_output, # 传递模型保存目录
        log_file=log_file,
        epochs=epochs,
        lr=lr,
        resume_training=True,
        pre_model=pre_model
    )

    print("训练完成。")

