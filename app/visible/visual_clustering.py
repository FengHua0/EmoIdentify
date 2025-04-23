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
from io import BytesIO
import base64
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))  # 添加项目根目录到Python路径

from app.model_training.contrastive_training import CNN_RNN as ContrastiveCNN_RNN
from app.model_training.cnn_rnn_spectrogram import CNN_RNN as SpectrogramCNN_RNN

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

# 添加模型注册表
# 修改模型注册表部分
MODEL_REGISTRY = {
    'contrastive': ContrastiveCNN_RNN,
    'spectrogram': SpectrogramCNN_RNN
}

def visualize_clusters(features, labels, speaker_ids, method="tsne", save_path=None):
    """
    可视化聚类结果
    Args:
        features: 特征数组
        labels: 标签数组
        speaker_ids: 说话人ID数组
        method: 降维方法 ('tsne', 'pca', 'umap')
        save_path: 图片保存路径
    """
    # 降维
    if method == "tsne":
        reducer = TSNE(n_components=2, random_state=42)
    elif method == "pca":
        reducer = PCA(n_components=2)
    elif method == "umap":
        reducer = UMAP(n_components=2, random_state=42)
    else:
        raise ValueError(f"未知的降维方法: {method}")
    
    embeddings = reducer.fit_transform(features)
    
    # 定义形状映射 (每个情感类别对应一个形状)
    shapes = ['o', 's', '^', 'p', '*', 'h', 'H', '+', 'x', 'D', 'd']
    
    # 可视化
    plt.figure(figsize=(12, 10))
    
    # 获取唯一的情感标签和说话人ID
    unique_labels = np.unique(labels)
    unique_speakers = np.unique(speaker_ids)
    
    # 为每个说话人分配颜色
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_speakers)))
    
    # 绘制散点图
    for label in unique_labels:
        # 获取当前情感类别的数据点
        mask = (labels == label)
        # 使用形状区分情感类别，颜色区分说话人
        for speaker in unique_speakers:
            speaker_mask = (speaker_ids == speaker)
            combined_mask = mask & speaker_mask
            if np.any(combined_mask):
                plt.scatter(
                    embeddings[combined_mask, 0], 
                    embeddings[combined_mask, 1],
                    c=[colors[speaker]],  # 颜色区分说话人
                    marker=shapes[label % len(shapes)],  # 形状区分情感
                    alpha=0.6,
                    s=50
                )
    
    # 创建情感图例
    legend_elements = [
        plt.Line2D([0], [0], 
                  marker=shapes[label % len(shapes)], 
                  color='w', 
                  label=f'Emotion {label}',
                  markerfacecolor='gray', 
                  markersize=10)
        for label in unique_labels
    ]
    
    plt.legend(handles=legend_elements, title='Emotion Labels', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title(f"Emotion Clustering ({method.upper()}) - Shapes:Emotion, Colors:Speaker")
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()

def get_base64(fig):
    """
    将matplotlib图形转换为base64编码
    Args:
        fig: matplotlib图形对象
    Returns:
        base64编码的图片字符串
    """
    buf = BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    return img_base64

# 修改导入部分，从各自训练模块导入数据加载函数
from app.model_training.contrastive_training import load_datasets

def load_model_and_data(data_folder, model_path, img_size=(224, 224), batch_size=64, model_type='contrastive'):
    """
    加载预训练模型和数据集
    Args:
        model_type: 模型类型，可以是 'contrastive', 'spectrogram'
    """

    # 数据加载
    train_loader, val_loader, test_loader, class_indices, speaker_indices = load_datasets(
        data_folder=data_folder,
        img_size=img_size,
        batch_size=batch_size
    )
    
    # 从注册表获取模型类
    model_class = MODEL_REGISTRY[model_type]
    model = model_class(num_classes=len(class_indices))
    # 加载预训练权重
    if os.path.exists(model_path):
        print(f"加载预训练模型: {model_path}")
        if model_type in ['contrastive', 'spectrogram']:
            # 明确指定 map_location，并将模型移动到目标设备
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.load_state_dict(torch.load(model_path, map_location=device))
            model = model.to(device)
        else:
            model.load(model_path)
    else:
        raise FileNotFoundError(f"模型文件未找到: {model_path}")
    model.eval()  # 设置为评估模式
    return model, train_loader, val_loader, test_loader, class_indices, speaker_indices

def generate_model_paths(model_type):
    """
    根据模型类型生成相关路径
    Args:
        model_type: 模型类型 ('contrastive', 'spectrogram')
    Returns:
        dict: 包含各种路径的字典
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    if model_type == 'contrastive':
        model_path = os.path.join(current_dir, "../models/Contrastive_model.pth")
    elif model_type == 'spectrogram':
        model_path = os.path.join(current_dir, "../models/cnn_rnn_spectrogram_model.pth")

    # 定义基础路径
    paths = {
        'data_folder': os.path.join(current_dir, "../features/mel_spectrogram/CREMA-D"),
        'model_path': model_path,
        'output_path': os.path.join(current_dir, f"../model_visible/{model_type}_clusters.png"),
        'features_path': os.path.join(current_dir, f"../clustering_feature/{model_type}_features.npz")
    }
    
    return paths


def load_model_and_data(data_folder, model_path, img_size=(224, 224), batch_size=64, model_type='contrastive'):
    """
    加载预训练模型和数据集
    Args:
        model_type: 模型类型，可以是 'contrastive', 'spectrogram', 'svm', 'lgbm'
    """
    # 数据加载
    train_loader, val_loader, test_loader, class_indices, speaker_indices = load_datasets(
        data_folder=data_folder,
        img_size=img_size,
        batch_size=batch_size
    )
    
    # 从注册表获取模型类
    model_class = MODEL_REGISTRY[model_type]
    model = model_class(num_classes=len(class_indices))
    # 加载预训练权重
    if os.path.exists(model_path):
        print(f"加载预训练模型: {model_path}")
        if model_type in ['contrastive', 'spectrogram']:
            # 明确指定 map_location，并将模型移动到目标设备
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.load_state_dict(torch.load(model_path, map_location=device))
            model = model.to(device)
        else:
            model.load(model_path)
    else:
        raise FileNotFoundError(f"模型文件未找到: {model_path}")
    model.eval()  # 设置为评估模式
    return model, train_loader, val_loader, test_loader, class_indices, speaker_indices

# 提取特征
def extract_features(model, data_loader, device, model_type='contrastive'):
    """
    根据模型类型提取特征
    """
    features_list = []
    labels_list = []
    speaker_ids_list = []
    with torch.no_grad():
        for batch in data_loader:
            if model_type in ['contrastive', 'spectrogram']:
                images, labels, speaker_ids = batch
                images = images.to(device)  # 确保输入数据在正确设备上
                labels = labels.to(device)
                speaker_ids = speaker_ids.to(device)
                _, features = model(images)
            else:
                # 处理非神经网络模型的输入
                features, labels, speaker_ids = model.preprocess(batch)
            features_list.append(features.cpu().numpy())
            labels_list.append(labels.cpu().numpy())
            speaker_ids_list.append(speaker_ids.cpu().numpy())
    features_array = np.concatenate(features_list, axis=0)
    labels_array = np.concatenate(labels_list, axis=0)
    speaker_ids_array = np.concatenate(speaker_ids_list, axis=0)
    return features_array, labels_array, speaker_ids_array

def load_or_extract_features(model=None, data_loader=None, device=None, model_type='contrastive'):
    """
    检查是否有保存的特征文件，有则加载，没有则提取并保存
    Args:
        model: 模型对象(可选)
        data_loader: 数据加载器(可选)
        device: 计算设备(可选)
        model_type: 模型类型
    Returns:
        features, labels, speaker_ids
    """
    paths = generate_model_paths(model_type)
    features_file = paths['features_path']
    
    if os.path.exists(features_file):
        print(f"从文件 {features_file} 加载特征...")
        data = np.load(features_file)
        return data['features'], data['labels'], data['speaker_ids']
    else:
        if model is None or data_loader is None or device is None:
            raise ValueError("特征文件不存在，需要提供model、data_loader和device参数")
        print("提取特征中...")
        features, labels, speaker_ids = extract_features(model, data_loader, device, model_type)
        print(f"保存特征到文件 {features_file}...")
        os.makedirs(os.path.dirname(features_file), exist_ok=True)
        np.savez(features_file, 
                features=features, 
                labels=labels, 
                speaker_ids=speaker_ids)
        return features, labels, speaker_ids

from sklearn.metrics.pairwise import cosine_distances

def quantitative_speaker_analysis(features, speaker_ids):
    """
    对身份信息进行定量分析，计算同一说话人和不同说话人之间的特征余弦距离
    Args:
        features: 特征数组 (N, D)
        speaker_ids: 说话人ID数组 (N,)
    Returns:
        dict: 包含同说话人和异说话人距离的均值和标准差
    """
    features = np.asarray(features)
    speaker_ids = np.asarray(speaker_ids)
    n = features.shape[0]
    cos_dists = cosine_distances(features)
    same_speaker_dists = []
    diff_speaker_dists = []
    for i in range(n):
        for j in range(i + 1, n):
            if speaker_ids[i] == speaker_ids[j]:
                same_speaker_dists.append(cos_dists[i, j])
            else:
                diff_speaker_dists.append(cos_dists[i, j])
    same_speaker_dists = np.array(same_speaker_dists)
    diff_speaker_dists = np.array(diff_speaker_dists)
    result = {
        "same_speaker_mean": float(np.mean(same_speaker_dists)),
        "same_speaker_std": float(np.std(same_speaker_dists)),
        "diff_speaker_mean": float(np.mean(diff_speaker_dists)),
        "diff_speaker_std": float(np.std(diff_speaker_dists)),
        "same_count": int(len(same_speaker_dists)),
        "diff_count": int(len(diff_speaker_dists))
    }
    print("身份信息定量分析结果：")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result

def visualize_clustering(model_type='contrastive', data_folder=None, model_path=None, 
                       output_path=None, features_path=None):
    """
    支持多种模型的可视化
    Args:
        model_type: 模型类型 ('contrastive', 'spectrogram')
        data_folder: 数据路径(可选)
        model_path: 模型路径(可选)
        output_path: 输出图片路径(可选)
        features_path: 特征文件路径(可选)
    """
    # 生成路径
    paths = generate_model_paths(model_type)
    data_folder = data_folder or paths['data_folder']
    model_path = model_path or paths['model_path']
    output_path = output_path or paths['output_path']
    features_path = features_path or paths['features_path']
    
    # 加载模型和数据
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, train_loader, _, _, _, _ = load_model_and_data(
        data_folder=data_folder,
        model_path=model_path,
        img_size=(224, 224),
        batch_size=64,
        model_type=model_type
    )
    
    # 使用load_or_extract_features替代直接调用extract_features
    features, labels, speaker_ids = load_or_extract_features(
        model=model,
        data_loader=train_loader,
        device=device,
        model_type=model_type
    )
    
    # 新增：身份信息定量分析
    quantitative_speaker_analysis(features, speaker_ids)
    
    # 可视化集群
    fig = plt.figure(figsize=(12, 10))
    visualize_clusters(features, labels, speaker_ids, method="tsne", save_path=output_path)
    
    # 获取base64编码
    base64_with_coords = get_base64(fig)
    plt.close(fig)
    
    return {
        'feature_name': 'clustering',
        'base64': base64_with_coords
    }

# 修改主程序
if __name__ == "__main__":
    # 定义模型类型
    model_type = "contrastive"  # 可以改为'spectrogram','contrastive'
    
    # 直接调用visualize_clustering，内部会处理路径生成
    result = visualize_clustering(model_type=model_type)
    print("可视化完成，base64编码已生成")
