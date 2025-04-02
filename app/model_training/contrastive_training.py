import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
from PIL import Image

class CustomDataset(Dataset):
    def __init__(self, data_folder, split='train', img_size=(224, 224), transform=None):
        """
        自定义数据集，加载图像并返回图像、情感标签和说话人ID

        Args:
            data_folder (str): 数据集路径
            split (str): 数据集分割 ('train', 'val', 'test')
            img_size (tuple): 图像尺寸
            transform (callable, optional): 图像预处理转换
        """
        self.data_folder = os.path.join(data_folder, split)
        self.img_size = img_size
        self.transform = transform
        self.samples = []
        self.speaker_ids = {}

        # 遍历目录，加载每个文件并提取标签
        for label, emotion_folder in enumerate(os.listdir(self.data_folder)):
            emotion_path = os.path.join(self.data_folder, emotion_folder)
            if os.path.isdir(emotion_path):
                for file_name in os.listdir(emotion_path):
                    if file_name.endswith(".png"):
                        # 提取说话人ID (假设ID嵌入文件名，形如 1001_DFA_ANG_XX.png)
                        speaker_id = file_name.split('_')[0]  # 假设文件名中的第一部分是说话人ID
                        self.samples.append((os.path.join(emotion_path, file_name), label, speaker_id))
                        if speaker_id not in self.speaker_ids:
                            self.speaker_ids[speaker_id] = len(self.speaker_ids)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, emotion_label, speaker_id = self.samples[idx]
        image = Image.open(img_path).convert('RGB')

        # 图像预处理
        if self.transform:
            image = self.transform(image)

        speaker_id = self.speaker_ids[speaker_id]  # 转换为数字索引

        return image, emotion_label, speaker_id

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
        speaker_indices (dict): 说话人索引映射
    """
    print(f"加载数据集: {data_folder}")

    # 图像预处理操作
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        # 三通道图像，使用 ImageNet 上常用的标准化
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    # 使用 CustomDataset 加载数据集
    train_dataset = CustomDataset(data_folder, split="train", img_size=img_size, transform=transform)
    val_dataset = CustomDataset(data_folder, split="val", img_size=img_size, transform=transform)
    test_dataset = CustomDataset(data_folder, split="test", img_size=img_size, transform=transform)

    # 创建 DataLoader
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
    os.makedirs(os.path.dirname(class_save_path), exist_ok=True)
    with open(class_save_path, "w") as f:
        json.dump(class_indices, f)

    print(f"类别标签映射已保存: {class_save_path}")

    # 保存说话人标签映射
    speaker_save_path = "../models/label_encoder/CREMA-D_CNN_speaker.json"
    os.makedirs(os.path.dirname(speaker_save_path), exist_ok=True)
    with open(speaker_save_path, "w") as f:
        json.dump(speaker_indices, f)

    print(f"说话人标签映射已保存: {speaker_save_path}")

    return train_loader, val_loader, test_loader, class_indices, speaker_indices


# 对比损失函数：NT-Xent Loss（修改为仅在相同情感类别内进行对比学习）
class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.5, base_temperature=0.07):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(self, features, labels, speaker_ids):
        batch_size = features.shape[0]
        device = features.device

        # 正负样本对的标签：是否属于同一类
        labels = labels.contiguous().view(-1, 1)
        speaker_ids = speaker_ids.contiguous().view(-1, 1)
        
        # 创建情感类别掩码：相同情感类别为1，不同情感类别为0
        emotion_mask = torch.eq(labels, labels.T).float().to(device)
        
        # 创建说话人掩码：相同说话人为1，不同说话人为0
        speaker_mask = torch.eq(speaker_ids, speaker_ids.T).float().to(device)
        
        # 计算特征相似度矩阵
        features = nn.functional.normalize(features, dim=1)  # 特征归一化
        similarity_matrix = torch.matmul(features, features.T)  # [B, B]
        similarity_matrix = similarity_matrix / self.temperature
        
        # 对角线元素设为极小值，避免自身对比
        mask_self = torch.eye(batch_size, dtype=torch.bool).to(device)
        similarity_matrix.masked_fill_(mask_self, -float('inf'))
        
        # 只在相同情感类别内进行对比学习
        # 对于每个情感类别，创建一个掩码
        loss = 0.0
        num_classes = 0
        
        for emotion_label in torch.unique(labels):
            # 找出当前情感类别的样本索引
            emotion_indices = (labels == emotion_label).squeeze().nonzero(as_tuple=True)[0]
            if len(emotion_indices) <= 1:
                continue  # 跳过只有一个样本的情感类别
                
            num_classes += 1
            emotion_features = features[emotion_indices]
            emotion_speaker_ids = speaker_ids[emotion_indices]
            
            # 计算当前情感类别内的相似度矩阵
            emotion_sim = torch.matmul(emotion_features, emotion_features.T) / self.temperature
            
            # 对角线元素设为极小值
            emotion_self_mask = torch.eye(len(emotion_indices), dtype=torch.bool).to(device)
            emotion_sim.masked_fill_(emotion_self_mask, -float('inf'))
            
            # 创建说话人掩码（同一情感类别内）
            emotion_speaker_mask = torch.eq(emotion_speaker_ids, emotion_speaker_ids.T).float().to(device)
            
            # 计算当前情感类别的对比损失
            # 正样本：不同说话人但相同情感
            # 负样本：相同说话人且相同情感
            pos_mask = (1.0 - emotion_speaker_mask) * (1.0 - emotion_self_mask.float())
            
            # 计算 logits
            logits = emotion_sim
            
            # 计算正样本对的损失
            if pos_mask.sum() > 0:
                pos_logits = logits * pos_mask
                pos_logits = pos_logits.sum(dim=1) / (pos_mask.sum(dim=1) + 1e-8)
                emotion_loss = -pos_logits.mean()
                loss += emotion_loss
        
        # 平均所有情感类别的损失
        if num_classes > 0:
            loss = loss / num_classes
        
        return loss

# 自定义 CNN-RNN 模型
class CNN_RNN(nn.Module):
    def __init__(self, num_classes=10, hidden_dim=128, num_layers=1, bidirectional=False):
        super(CNN_RNN, self).__init__()

        # ------------ CNN 特征提取部分 ------------
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        # ------------ LSTM ------------
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        lstm_input_dim = 64 * 28

        self.lstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional
        )

        # ------------ 分类层 ------------
        self.fc = nn.Linear(hidden_dim * (2 if bidirectional else 1), num_classes)

    def forward(self, x):
        # ------------ CNN ------------
        cnn_out = self.cnn(x)
        B, C, H, W = cnn_out.shape
        cnn_out = cnn_out.view(B, C * H, W).permute(0, 2, 1)

        # ------------ LSTM ------------
        lstm_out, _ = self.lstm(cnn_out)

        # ------------ 获取特征向量 ------------
        context = lstm_out.mean(dim=1)  # 获取平均池化后的特征表示
        out = self.fc(context)  # 分类输出

        return out, context  # 返回特征表示用于对比学习

# 准确率计算函数
def calculate_accuracy(outputs, labels):
    _, predicted = torch.max(outputs, 1)
    correct = (predicted == labels).sum().item()
    total = labels.size(0)
    accuracy = correct / total
    return accuracy

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

# 模型训练函数（添加对比学习功能）
def train_model(model, train_loader, val_loader, device, epochs=10, lr=1e-3, resume_training=True, pre_model=None,
                model_output="../models/contrastive_training.pth"):
    criterion = NTXentLoss(temperature=0.5)  # 对比损失函数
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)  # 每5个epoch衰减学习率
    model.to(device)

    # 恢复训练
    if resume_training and pre_model:
        if os.path.exists(pre_model):
            print(f"加载已有模型: {pre_model}")
            model.load_state_dict(torch.load(pre_model))
        else:
            print(f"预训练模型 {pre_model} 不存在，跳过恢复训练。初始化新的模型。")

    print("开始训练模型...")

    for epoch in range(epochs):
        # ------------ 训练 ------------
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for batch_idx, (images, labels, speaker_ids) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            speaker_ids = speaker_ids.to(device)

            optimizer.zero_grad()

            # 获取模型输出和特征
            outputs, features = model(images)

            # 计算对比损失
            contrastive_loss = criterion(features, labels, speaker_ids)

            # 计算分类损失
            classification_loss = nn.CrossEntropyLoss()(outputs, labels)

            # 总损失 = 分类损失 + 对比损失
            loss = classification_loss + contrastive_loss

            loss.backward()
            optimizer.step()

            # 计算训练准确率
            train_acc = calculate_accuracy(outputs, labels)
            running_loss += loss.item() * images.size(0)
            correct_train += (outputs.argmax(1) == labels).sum().item()
            total_train += labels.size(0)

        # 每个epoch的训练损失和准确率
        train_loss = running_loss / len(train_loader.dataset)
        train_accuracy = correct_train / total_train

        # 打印训练过程
        print(f"Epoch [{epoch + 1}/{epochs}], Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}")

        # ------------ 验证 ------------
        model.eval()
        val_running_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for images, labels, speaker_ids in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                speaker_ids = speaker_ids.to(device)
                outputs, features = model(images)

                # 计算对比损失
                contrastive_loss = criterion(features, labels, speaker_ids)

                # 计算分类损失
                classification_loss = nn.CrossEntropyLoss()(outputs, labels)

                # 总损失 = 分类损失 + 对比损失
                val_loss = classification_loss + contrastive_loss
                val_running_loss += val_loss.item() * images.size(0)

                # 计算验证准确率
                correct_val += (outputs.argmax(1) == labels).sum().item()
                total_val += labels.size(0)

        # 每个epoch的验证损失和准确率
        val_loss = val_running_loss / len(val_loader.dataset)
        val_accuracy = correct_val / total_val

        # 打印验证过程
        print(f"Epoch [{epoch + 1}/{epochs}] Summary: Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")

        # 更新学习率
        scheduler.step()

        # 保存模型
        os.makedirs(model_output, exist_ok=True)
        epoch_model_output = os.path.join(model_output, f"Contrastive_Learning_model_epoch_{epoch + 1}.pth")
        print(f"保存模型 (Epoch {epoch + 1}) 到: {epoch_model_output}")
        torch.save(model.state_dict(), epoch_model_output)

# 主程序
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    data_folder = "../features/mel_spectrogram/CREMA-D"
    pre_model = "../models/Contrastive_model.pth"
    model_output = "..\models\Contrastive"

    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_folder = os.path.join(current_dir, data_folder)
    pre_model = os.path.join(current_dir, pre_model)
    model_output = os.path.join(current_dir, model_output)

    batch_size = 64
    num_classes = 6
    epochs = 50
    lr = 1e-3

    train_loader, val_loader, test_loader, class_indices, speaker_ids = load_datasets(
        data_folder=data_folder,
        img_size=(224, 224),
        batch_size=batch_size
    )

    model = CNN_RNN(num_classes=len(class_indices)).to(device)
    train_model(model, train_loader, val_loader, device, epochs=epochs, lr=lr, resume_training=True)