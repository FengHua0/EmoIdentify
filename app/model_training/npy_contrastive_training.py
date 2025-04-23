import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
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

# 对比损失函数
class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.5, base_temperature=0.07):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.eps = 1e-8

    def forward(self, features, labels, speaker_ids):
        batch_size = features.shape[0]
        device = features.device

        features = nn.functional.normalize(features, dim=1)
        sim_matrix = torch.matmul(features, features.T) / self.temperature
        
        labels = labels.contiguous().view(-1, 1)
        emotion_mask = torch.eq(labels, labels.T).float().to(device)
        speaker_mask = torch.eq(speaker_ids, speaker_ids.T).float().to(device)
        mask_self = torch.eye(batch_size, dtype=torch.bool).to(device)
        
        loss = 0.0
        num_valid_classes = 0
        
        for emotion_label in torch.unique(labels):
            emotion_indices = (labels == emotion_label).squeeze().nonzero(as_tuple=True)[0]
            if len(emotion_indices) <= 1:
                continue
                
            num_valid_classes += 1
            emotion_features = features[emotion_indices]
            emotion_speaker_ids = speaker_ids[emotion_indices]
            emotion_sim = torch.matmul(emotion_features, emotion_features.T) / self.temperature
            emotion_speaker_mask = torch.eq(emotion_speaker_ids, emotion_speaker_ids.T).float().to(device)
            
            pos_mask = emotion_speaker_mask * (~torch.eye(len(emotion_indices), dtype=torch.bool, device=device)).float()
            neg_mask = (~emotion_speaker_mask.bool()).float() * (~torch.eye(len(emotion_indices), dtype=torch.bool, device=device)).float()
            
            pos_sim = (emotion_sim * pos_mask).sum(dim=1) / (pos_mask.sum(dim=1) + self.eps)
            neg_sim = torch.logsumexp(emotion_sim * neg_mask, dim=1)
            
            class_loss = - (pos_sim - neg_sim).mean()
            loss += class_loss
            
        if num_valid_classes > 0:
            loss = loss / num_valid_classes * self.base_temperature
        else:
            loss = torch.tensor(0.0, device=device)
            
        return loss

# 自定义CNN-RNN模型
# 添加单通道转三通道的转换类
class SingleToThreeChannels:
    """将单通道图像复制为三通道"""
    def __call__(self, x):
        return x.repeat(3, 1, 1) if x.size(0) == 1 else x

# 修改load_datasets函数
def load_datasets(data_folder, batch_size=64, target_length=100):
    """加载.npy数据集并返回DataLoader"""
    print(f"加载数据集: {data_folder}")
    
    transform = transforms.Compose([
        SingleToThreeChannels(),
        transforms.Resize((224, 224)),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
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

# 修改CNN_RNN类
class CNN_RNN(nn.Module):
    def __init__(self, num_classes=10, hidden_dim=128, num_layers=1, bidirectional=False):
        super(CNN_RNN, self).__init__()
        # CNN部分改为3通道输入
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # 动态计算LSTM输入维度
        self.lstm_input_dim = self._get_lstm_input_dim()
        
        self.lstm = nn.LSTM(
            input_size=self.lstm_input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional
        )
        self.fc = nn.Linear(hidden_dim * (2 if bidirectional else 1), num_classes)

    def _get_lstm_input_dim(self):
        # 使用虚拟输入计算CNN输出维度
        # 将虚拟输入的尺寸从 (1, 3, 128, 100) 改为 (1, 3, 224, 224)
        # 以匹配 load_datasets 中 Resize 后的实际尺寸
        dummy_input = torch.randn(1, 3, 224, 224) 
        with torch.no_grad():
            cnn_out = self.cnn(dummy_input)
            # 检查cnn_out的形状，确保计算正确
            # print(f"CNN output shape for dummy input: {cnn_out.shape}")
            return cnn_out.size(1) * cnn_out.size(2)  # C * H

    def attention(self, lstm_output):
        """加性注意力机制"""
        energy = self.attention_fc(lstm_output).squeeze(-1)
        att_weights = F.softmax(energy, dim=1)
        context = torch.sum(lstm_output * att_weights.unsqueeze(-1), dim=1)
        return context, att_weights

    def forward(self, x):
        cnn_out = self.cnn(x)  # [B, 64, H', W'] H'和W'是经过CNN和池化后的维度
        B, C, H, W = cnn_out.shape
        # 调整view和permute以匹配LSTM输入 (B, seq_len, features)
        # 这里假设时间维度是W
        cnn_out = cnn_out.view(B, C * H, W).permute(0, 2, 1)  # [B, W, C*H]
        
        lstm_out, _ = self.lstm(cnn_out)
        
        # 如果使用了注意力机制，取消下面的注释并注释掉 lstm_out.mean(dim=1)
        # context, _ = self.attention(lstm_out) 
        
        # 如果不使用注意力，使用LSTM最后一个时间步的输出或所有时间步输出的平均值
        context = lstm_out.mean(dim=1) # 或者 lstm_out[:, -1, :] 如果是非双向LSTM
        
        out = self.fc(context)
        
        return out, context  # 返回分类结果和特征向量

    # 注意：您在类定义中有两个 _get_lstm_input_dim 和 forward 方法的定义
    # 请确保只保留一个版本的 _get_lstm_input_dim 和 forward 方法
    # 下面是重复的方法定义，我将注释掉它们，请根据您的需要保留正确的版本

    # def _get_lstm_input_dim(self):
    #     # 使用虚拟输入计算CNN输出维度
    #     dummy_input = torch.randn(1, 1, 128, 100)  # 假设输入为(1, 128, 100)
    #     with torch.no_grad():
    #         cnn_out = self.cnn(dummy_input)
    #         return cnn_out.size(1) * cnn_out.size(2)  # C * H

    # def forward(self, x):
    #     cnn_out = self.cnn(x)
    #     B, C, H, W = cnn_out.shape
    #     cnn_out = cnn_out.view(B, C * H, W).permute(0, 2, 1)  # (B, W, C*H)
    #     lstm_out, _ = self.lstm(cnn_out)
    #     context = lstm_out.mean(dim=1)
    #     out = self.fc(context)
    #     return out, context

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
               epochs=10, lr=1e-3, resume_training=True, pre_model=None):
    criterion = NTXentLoss(temperature=0.5)
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
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for batch_idx, (features, labels, speaker_ids) in enumerate(train_loader):
            features, labels, speaker_ids = features.to(device), labels.to(device), speaker_ids.to(device)
            optimizer.zero_grad()
            outputs, features = model(features)
            contrastive_loss = criterion(features, labels, speaker_ids)
            classification_loss = nn.CrossEntropyLoss()(outputs, labels)
            loss = classification_loss + contrastive_loss
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * features.size(0)
            correct_train += (outputs.argmax(1) == labels).sum().item()
            total_train += labels.size(0)

        train_loss = running_loss / len(train_loader.dataset)
        train_accuracy = correct_train / total_train

        model.eval()
        val_running_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for features, labels, speaker_ids in val_loader:
                features, labels, speaker_ids = features.to(device), labels.to(device), speaker_ids.to(device)
                outputs, features = model(features)
                contrastive_loss = criterion(features, labels, speaker_ids)
                classification_loss = nn.CrossEntropyLoss()(outputs, labels)
                val_loss = classification_loss + contrastive_loss
                val_running_loss += val_loss.item() * features.size(0)
                correct_val += (outputs.argmax(1) == labels).sum().item()
                total_val += labels.size(0)

        val_loss = val_running_loss / len(val_loader.dataset)
        val_accuracy = correct_val / total_val

        print(f"Epoch [{epoch + 1}/{epochs}]: Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")
        
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
    pre_model = "../models/npy_contrastive_model.pth"
    model_output = "../models/npy_contrastive"
    log_file = "../model_visible/npy_contrastive.txt"

    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_folder = os.path.join(current_dir, data_folder)
    pre_model = os.path.join(current_dir, pre_model)
    model_output = os.path.join(current_dir, model_output)
    log_file = os.path.join(current_dir, log_file)

    batch_size = 64
    epochs = 70
    lr = 1e-4

    target_length = 100  # 根据数据分布选择合适的值
    
    train_loader, val_loader, test_loader, class_indices, speaker_ids = load_datasets(
        data_folder=data_folder,
        batch_size=batch_size,
        target_length=target_length
    )

    model = CNN_RNN(num_classes=len(class_indices)).to(device)
    train_model(model, train_loader, val_loader, device, model_output, log_file, epochs=epochs, lr=lr, pre_model=pre_model)