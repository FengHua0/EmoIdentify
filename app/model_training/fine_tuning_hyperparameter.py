import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import time
import torch.nn.functional as F

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
        self.speaker_ids = {}
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
        npy_path, emotion_label, speaker_id = self.samples[idx]
        
        # 加载.npy文件
        features = np.load(npy_path)
        
        # 确保特征维度正确 (n_mels, time)
        if features.ndim == 3 and features.shape[0] == 1:
            features = features.squeeze(0)
        
        # 统一时间维度长度
        if features.shape[1] > self.target_length:
            features = features[:, :self.target_length]
        elif features.shape[1] < self.target_length:
            pad_size = self.target_length - features.shape[1]
            features = np.pad(features, ((0,0), (0,pad_size)), mode='constant')
        
        # 转换为PyTorch张量并添加通道维度 (1, n_mels, time)
        features = torch.from_numpy(features).float().unsqueeze(0)
        
        # 复制单通道为三通道
        features = features.repeat(3, 1, 1)  # [3, n_mels, time]
        
        speaker_id = self.speaker_ids[speaker_id]
        
        return features, emotion_label, speaker_id

# 对比损失函数 (修改后，专注于同一情感内的说话人对比)
class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.8):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature
        self.eps = 1e-7

    def forward(self, features, labels, speaker_ids):
        # features: [B, D], labels: [B], speaker_ids: [B]
        device = features.device
        features = nn.functional.normalize(features, dim=1)

        loss = 0.0
        valid_count = 0

        for emotion_label in torch.unique(labels):
            idx = torch.where(labels == emotion_label)[0]
            if len(idx) <= 1:
                continue

            feats = features[idx]
            spk_ids = speaker_ids[idx]
            n = len(idx)

            # 计算相似度矩阵
            sim = torch.matmul(feats, feats.T) / self.temperature

            # 掩码
            mask_self = torch.eye(n, dtype=torch.bool, device=device)
            mask_pos = (spk_ids.unsqueeze(0) == spk_ids.unsqueeze(1)) & (~mask_self)  # 正样本掩码
            mask_neg = ~mask_pos & (~mask_self)  # 负样本掩码

            # 对每个anchor，计算正样本对的loss
            for i in range(n):
                pos_idx = mask_pos[i]
                neg_idx = mask_neg[i]
                if pos_idx.sum() == 0:
                    continue  # 没有正样本对，跳过

                # 分子：所有正样本的exp(sim)
                numerator = torch.exp(sim[i][pos_idx]).sum()
                # 分母：所有正样本和负样本的exp(sim)
                denominator = torch.exp(sim[i][neg_idx]).sum() + numerator + self.eps

                loss_i = -torch.log(numerator / denominator)
                loss += loss_i
                valid_count += 1

        if valid_count > 0:
            loss = loss / valid_count
        else:
            loss = torch.tensor(0.0, device=device, requires_grad=True)
        return loss

# 自定义CNN-RNN模型
# 添加单通道转三通道的转换类
class SingleToThreeChannels:
    """将单通道图像复制为三通道"""
    def __call__(self, x):
        return x.repeat(3, 1, 1) if x.size(0) == 1 else x

# 修改CNN_RNN类
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
        dummy_input = torch.randn(1, 3, self.n_mels, self.target_length)
        with torch.no_grad():
            cnn_out = self.cnn(dummy_input)
            return cnn_out.size(1) * cnn_out.size(2)

    def attention(self, lstm_output):
        """
        加性注意力机制
        :param lstm_output: [B, T, H]
        :return: context: [B, H], att_weights: [B, T]
        """
        energy = torch.tanh(self.attention_fc(lstm_output)).squeeze(-1)  # [B, T]
        att_weights = F.softmax(energy, dim=1)  # [B, T]
        context = torch.sum(lstm_output * att_weights.unsqueeze(-1), dim=1)  # [B, H]
        return context, att_weights

    def forward(self, x):
        """
        :param x: [B, 3, n_mels, target_length]
        :return: 分类输出, 特征向量
        """
        cnn_out = self.cnn(x)  # [B, 64, H', W']
        B, C, H, W = cnn_out.shape
        cnn_out = cnn_out.view(B, C * H, W).permute(0, 2, 1)  # [B, W', C*H']

        lstm_out, _ = self.lstm(cnn_out)  # [B, W', hidden_dim * num_directions]

        # 启用注意力机制
        context, _ = self.attention(lstm_out)

        out = self.fc(context)
        return out, context


# 训练函数
def calculate_accuracy(outputs, labels):
    _, predicted = torch.max(outputs, 1)
    correct = (predicted == labels).sum().item()
    total = labels.size(0)
    accuracy = correct / total
    return accuracy

def log_results(log_file, train_loss, train_acc, val_loss, val_acc):
    """
    将每个epoch的训练结果记录到日志文件
    Args:
        log_file (str): 日志文件路径
        train_loss (float): 训练损失
        train_acc (float): 训练准确率
        val_loss (float): 验证损失
        val_acc (float): 验证准确率
    """
    with open(log_file, "a") as f:
        f.write(f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f} | "
                f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}\n")

def train_model(model, train_loader, val_loader, device, model_output, log_file,
               epochs=10, lr=1e-3, resume_training=True, pre_model=None, contrastive_weight=0.5,
               contrastive_temperature=0.07):  # 新增temperature参数
    # 使用修改后的损失函数
    contrastive_criterion = NTXentLoss(temperature=contrastive_temperature)
    classification_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.to(device)

    # 恢复训练
    if resume_training and pre_model:
        if os.path.exists(pre_model):
            print(f"加载已有模型: {pre_model}")
            model.load_state_dict(torch.load(pre_model))
        else:
            print(f"预训练模型 {pre_model} 不存在，跳过恢复训练。初始化新的模型。")

    for epoch in range(epochs):
        # 记录开始时间
        epoch_start = time.time()

        model.train()
        running_loss = 0.0
        running_cls_loss = 0.0 # 可选：记录分类损失
        running_con_loss = 0.0 # 可选：记录对比损失
        correct_train = 0
        total_train = 0

        for batch_idx, (features, labels, speaker_ids) in enumerate(train_loader):
            features, labels, speaker_ids = features.to(device), labels.to(device), speaker_ids.to(device)
            optimizer.zero_grad()
            outputs, features_for_contrastive = model(features)

            # 计算分类损失
            cls_loss = classification_criterion(outputs, labels)
            # 计算对比损失
            con_loss = contrastive_criterion(features_for_contrastive, labels, speaker_ids)

            # 计算总损失，使用权重平衡
            loss = cls_loss + contrastive_weight * con_loss

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * features.size(0)
            running_cls_loss += cls_loss.item() * features.size(0)
            running_con_loss += con_loss.item() * features.size(0)
            correct_train += (outputs.argmax(1) == labels).sum().item()
            total_train += labels.size(0)

        train_loss = running_loss / len(train_loader.dataset)
        train_cls_loss = running_cls_loss / len(train_loader.dataset)
        train_con_loss = running_con_loss / len(train_loader.dataset)
        train_accuracy = correct_train / total_train

        model.eval()
        val_running_loss = 0.0
        val_running_cls_loss = 0.0
        val_running_con_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for features, labels, speaker_ids in val_loader:
                features, labels, speaker_ids = features.to(device), labels.to(device), speaker_ids.to(device)
                outputs, features_for_contrastive = model(features)

                # 计算分类损失
                cls_loss = classification_criterion(outputs, labels)
                # 计算对比损失
                con_loss = contrastive_criterion(features_for_contrastive, labels, speaker_ids)

                # 计算总损失，使用相同的权重
                val_loss_batch = cls_loss + contrastive_weight * con_loss

                val_running_loss += val_loss_batch.item() * features.size(0)
                val_running_cls_loss += cls_loss.item() * features.size(0)
                val_running_con_loss += con_loss.item() * features.size(0)
                correct_val += (outputs.argmax(1) == labels).sum().item()
                total_val += labels.size(0)

        val_loss = val_running_loss / len(val_loader.dataset)
        val_cls_loss = val_running_cls_loss / len(val_loader.dataset)
        val_con_loss = val_running_con_loss / len(val_loader.dataset)
        val_accuracy = correct_val / total_val

        # 计算epoch耗时
        epoch_time = time.time() - epoch_start

        # 打印结果 (可以加入单独的损失项)
        print(f"Epoch [{epoch + 1}/{epochs}]: "
              f"Train Loss: {train_loss:.4f} (Cls: {train_cls_loss:.4f}, Con: {train_con_loss:.4f}), Train Acc: {train_accuracy:.4f} | "
              f"Val Loss: {val_loss:.4f} (Cls: {val_cls_loss:.4f}, Con: {val_con_loss:.4f}), Val Acc: {val_accuracy:.4f}, "
              f"Time: {epoch_time:.2f}s")

        # 日志记录也可以加入单独损失项
        log_results(log_file, train_loss, train_accuracy, val_loss, val_accuracy)
 
        # 确保模型输出目录存在
        os.makedirs(model_output, exist_ok=True)
        model_path = os.path.join(model_output, f"npy_contrastive_model_{epoch + 1}.pth")
        
        try: 
            torch.save(model.state_dict(), model_path)
            print(f"模型已保存到: {model_path}")
        except Exception as e:
            print(f"保存模型时出错: {e}")
            continue

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

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    data_folder = "../features/mel_npy/CREMA-D"
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_folder = os.path.join(current_dir, data_folder)

    batch_size = 64
    epochs = 85
    target_length = 100  # 根据数据分布选择合适的值

    # 超参数搜索空间
    lr_list = [1e-3, 1e-4]
    temperature_list = [0.07, 0.1, 0.3]
    contrastive_weight_list = [0.8, 0.5, 0.2, 0.1]

    # 加载数据
    train_loader, val_loader, test_loader, class_indices, speaker_ids = load_datasets(
        data_folder=data_folder,
        batch_size=batch_size,
        target_length=target_length
    )

    for lr in lr_list:
        for temperature in temperature_list:
            for contrastive_weight in contrastive_weight_list:
                # 构建唯一的模型输出和日志目录
                exp_name = f"lr{lr}_temp{temperature}_cw{contrastive_weight}"
                model_output = os.path.join(current_dir, "../models/npy_contrastive", exp_name)
                log_file = os.path.join(model_output, "train_log.txt")
                os.makedirs(model_output, exist_ok=True)

                print(f"\n==== 开始实验: {exp_name} ====")
                print(f"模型输出目录: {model_output}")
                print(f"日志文件: {log_file}")

                # 初始化模型
                model = CNN_RNN(num_classes=len(class_indices)).to(device)

                # 训练
                train_model(
                    model, train_loader, val_loader, device,
                    model_output, log_file,
                    epochs=epochs, lr=lr, pre_model=None,
                    contrastive_weight=contrastive_weight,
                    # 传递temperature参数
                    contrastive_temperature=temperature
                )

    model = CNN_RNN(num_classes=len(class_indices)).to(device)
    # 将权重传递给训练函数
    train_model(model, train_loader, val_loader, device, model_output, log_file,
                epochs=epochs, lr=lr, pre_model=pre_model, contrastive_weight=contrastive_weight)