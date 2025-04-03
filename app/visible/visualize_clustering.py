import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from umap.umap_ import UMAP
from torch.utils.data import DataLoader
from torchvision import transforms
import json
from torch.utils.data import Dataset
from PIL import Image
# 在文件开头添加
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))  # 添加项目根目录到Python路径

# 确保从正确的模块导入CNN_RNN
from app.model_training.contrastive_training import CNN_RNN  # 修改为完整路径导入

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
                        speaker_id = file_name.split('_')[0]  # 提取说话人ID
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
        return image, emotion_label, self.speaker_ids[speaker_id]

# 加载数据集
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

    # 获取类别索引
    class_indices = {i: emotion for i, emotion in enumerate(os.listdir(os.path.join(data_folder, "train")))}
    speaker_indices = train_dataset.speaker_ids

    return train_loader, val_loader, test_loader, class_indices, speaker_indices

# 加载数据集和模型
def load_model_and_data(data_folder, model_path, img_size=(224, 224), batch_size=64):
    """
    加载预训练模型和数据集
    """
    # 数据加载
    train_loader, val_loader, test_loader, class_indices, speaker_indices = load_datasets(
        data_folder=data_folder,
        img_size=img_size,
        batch_size=batch_size
    )
    
    # 初始化模型
    num_classes = len(class_indices)
    model = CNN_RNN(num_classes=num_classes)
    
    # 加载预训练权重
    if os.path.exists(model_path):
        print(f"加载预训练模型: {model_path}")
        model.load_state_dict(torch.load(model_path))
    else:
        raise FileNotFoundError(f"模型文件未找到: {model_path}")
    
    model.eval()  # 设置为评估模式
    return model, train_loader, val_loader, test_loader, class_indices, speaker_indices

# 提取特征
def extract_features(model, data_loader, device):
    """
    使用模型提取数据集的特征
    """
    model.to(device)
    features_list = []
    labels_list = []
    speaker_ids_list = []
    
    with torch.no_grad():
        for images, labels, speaker_ids in data_loader:
            images = images.to(device)
            _, features = model(images)  # 获取特征表示
            features_list.append(features.cpu().numpy())
            labels_list.append(labels.numpy())
            speaker_ids_list.append(speaker_ids.numpy())
    
    features_array = np.concatenate(features_list, axis=0)
    labels_array = np.concatenate(labels_list, axis=0)
    speaker_ids_array = np.concatenate(speaker_ids_list, axis=0)
    return features_array, labels_array, speaker_ids_array

# 在文件开头添加
FEATURES_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../clustering_feature/contrastive_training.npz")

def load_or_extract_features(model=None, data_loader=None, device=None):
    """
    检查是否有保存的特征文件，有则加载，没有则提取并保存
    修改为：如果有特征文件，直接加载，不需要模型和数据加载器
    """
    if os.path.exists(FEATURES_FILE):
        print(f"从文件 {FEATURES_FILE} 加载特征...")
        data = np.load(FEATURES_FILE)
        return data['features'], data['labels'], data['speaker_ids']
    else:
        if model is None or data_loader is None or device is None:
            raise ValueError("特征文件不存在，需要提供model、data_loader和device参数")
        print("提取特征中...")
        features, labels, speaker_ids = extract_features(model, data_loader, device)
        print(f"保存特征到文件 {FEATURES_FILE}...")
        os.makedirs(os.path.dirname(FEATURES_FILE), exist_ok=True)
        np.savez(FEATURES_FILE, 
                features=features, 
                labels=labels, 
                speaker_ids=speaker_ids)
        return features, labels, speaker_ids

# 可视化特征
def visualize_clusters(features, labels, speaker_ids, method="tsne", save_path=None):
    """
    使用 t-SNE 或 UMAP 对特征进行降维并可视化
    修改为形状代表情感，颜色代表说话人，只保留情感形状图例
    """
    if method == "tsne":
        print("使用 t-SNE 进行降维...")
        reducer = TSNE(n_components=2, random_state=42, perplexity=30)
    elif method == "umap":
        print("使用 UMAP 进行降维...")
        reducer = UMAP(n_components=2, random_state=42)
    elif method == "pca":
        print("使用 PCA 进行降维...")
        reducer = PCA(n_components=2, random_state=42)
    else:
        raise ValueError("不支持的降维方法，请选择 'tsne', 'umap' 或 'pca'")
    
    # 降维
    reduced_features = reducer.fit_transform(features)
    
    # 可视化 - 形状代表情感，颜色代表说话人
    plt.figure(figsize=(12, 10))
    
    # 获取唯一的说话人ID和情感标签
    unique_speakers = np.unique(speaker_ids)
    unique_emotions = np.unique(labels)
    
    # 为每个说话人分配一个颜色
    cmap = plt.cm.get_cmap('tab20', len(unique_speakers))
    
    # 为每个情感创建一个标记形状
    markers = ['o', 's', '^', 'p', '*', 'h', 'H', 'D', 'd']
    
    # 绘制所有数据点
    for i, speaker in enumerate(unique_speakers):
        speaker_mask = speaker_ids == speaker
        for j, emotion in enumerate(unique_emotions):
            emotion_mask = labels == emotion
            combined_mask = speaker_mask & emotion_mask
            if np.any(combined_mask):
                plt.scatter(reduced_features[combined_mask, 0], 
                           reduced_features[combined_mask, 1],
                           color=cmap(i),
                           marker=markers[j % len(markers)],
                           alpha=0.7)

    # 创建情感图例
    emotion_legend = []
    for j, emotion in enumerate(unique_emotions):
        emotion_legend.append(plt.Line2D([0], [0], 
                                       marker=markers[j % len(markers)], 
                                       color='w', 
                                       label=f'Emotion {emotion}',
                                       markerfacecolor='gray', 
                                       markersize=10))
    
    # 添加情感图例
    plt.legend(handles=emotion_legend, 
              title="Emotions",
              loc='upper right',
              frameon=False)
    
    plt.title(f"Feature Visualization using {method.upper()}")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    
    # 调整布局
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"图像已保存到: {save_path}")
    plt.show()

# 主程序也需要相应修改
if __name__ == "__main__":
    # 初始化路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_folder = os.path.join(current_dir, "../features/mel_spectrogram/CREMA-D")
    model_path = os.path.join(current_dir, "../models/Contrastive_model.pth")
    save_path = os.path.join(current_dir, "../model_visible/tsne_clusters.png")
    
    # 先尝试直接加载特征文件
    try:
        features, labels, speaker_ids = load_or_extract_features()
    except ValueError:
        # 如果特征文件不存在，才加载模型和数据
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, train_loader, _, _, class_indices, speaker_indices = load_model_and_data(
            data_folder=data_folder,
            model_path=model_path,
            img_size=(224, 224),
            batch_size=64
        )
        features, labels, speaker_ids = load_or_extract_features(model, train_loader, device)
    
    # 可视化集群
    print("正在可视化集群...")
    visualize_clusters(features, labels, speaker_ids, method="tsne", save_path=save_path)
