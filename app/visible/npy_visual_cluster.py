import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from umap.umap_ import UMAP # 使用 umap-learn 库
from torch.utils.data import DataLoader
import json
import argparse
import sys
from pathlib import Path
from io import BytesIO
import base64

# 添加项目根目录到Python路径，以便导入其他模块
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

# 从各自的训练脚本导入模型和数据加载函数
# 使用别名避免命名冲突
from app.model_training.npy_cnn_training import CNN_RNN as NpyCnnCNN_RNN
from app.model_training.npy_cnn_training import load_datasets as load_npy_cnn_datasets
from app.model_training.npy_contrastive_training import CNN_RNN as NpyContrastiveCNN_RNN
from app.model_training.npy_contrastive_training import load_datasets as load_npy_contrastive_datasets

# 模型注册表
MODEL_REGISTRY = {
    'npy_cnn': {
        'model_class': NpyCnnCNN_RNN,
        'load_func': load_npy_cnn_datasets,
        'data_args': {'img_size': (224, 224)} # npy_cnn 需要 img_size
    },
    'npy_contrastive': {
        'model_class': NpyContrastiveCNN_RNN,
        'load_func': load_npy_contrastive_datasets,
        'data_args': {'target_length': 100} # npy_contrastive 需要 target_length
    }
}

def visualize_clusters(features, labels, speaker_ids, method="tsne", save_path=None, title_prefix=""):
    """
    可视化聚类结果 (与 visual_clustering.py 相同)
    Args:
        features: 特征数组
        labels: 标签数组 (情感)
        speaker_ids: 说话人ID数组
        method: 降维方法 ('tsne', 'pca', 'umap')
        save_path: 图片保存路径
        title_prefix: 标题前缀 (例如模型类型)
    """
    print(f"开始使用 {method.upper()} 进行可视化...")
    # 降维
    if method == "tsne":
        reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(features)-1)) # 调整 perplexity
    elif method == "pca":
        reducer = PCA(n_components=2)
    elif method == "umap":
        # UMAP 对 n_neighbors 敏感，确保其小于样本数
        n_neighbors = min(15, len(features) - 1)
        if n_neighbors < 2:
             print(f"警告: UMAP 的 n_neighbors ({n_neighbors}) 过小，可能导致错误或结果不佳。")
             n_neighbors = max(2, n_neighbors) # 至少为2
        reducer = UMAP(n_components=2, random_state=42, n_neighbors=n_neighbors, min_dist=0.1)
    else:
        raise ValueError(f"未知的降维方法: {method}")

    embeddings = reducer.fit_transform(features)

    # 定义形状映射 (每个情感类别对应一个形状)
    shapes = ['o', 's', '^', 'p', '*', 'h', 'H', '+', 'x', 'D', 'd'] # 更多形状以备用

    # 可视化
    plt.figure(figsize=(14, 10)) # 调整图像大小

    # 获取唯一的情感标签和说话人ID
    unique_labels = np.unique(labels)
    unique_speakers = np.unique(speaker_ids)

    # 为每个说话人分配颜色 (使用更丰富的颜色映射)
    # colors = plt.cm.tab20(np.linspace(0, 1, len(unique_speakers)))
    # 使用 'viridis' 或 'plasma' 等连续色谱可能更适合较多说话人
    colors = plt.cm.get_cmap('tab20', len(unique_speakers))

    # 绘制散点图
    for i, label in enumerate(unique_labels):
        label_mask = (labels == label)
        shape = shapes[i % len(shapes)] # 循环使用形状
        for j, speaker in enumerate(unique_speakers):
            speaker_mask = (speaker_ids == speaker)
            combined_mask = label_mask & speaker_mask
            if np.any(combined_mask):
                plt.scatter(
                    embeddings[combined_mask, 0],
                    embeddings[combined_mask, 1],
                    c=[colors(j)],  # 使用 colormap 获取颜色
                    marker=shape,
                    alpha=0.7, # 增加透明度
                    s=60, # 增加点的大小
                    label=f'Emotion {label}, Speaker {speaker}' if i == 0 else "" # 仅为第一个情感类别添加说话人标签示例
                )

    # 创建图例 (区分情感和说话人)
    # 情感图例
    emotion_legend_elements = [
        plt.Line2D([0], [0],
                  marker=shapes[i % len(shapes)],
                  color='w',
                  label=f'Emotion {label}',
                  markerfacecolor='grey', # 使用灰色表示通用情感标记
                  markersize=10)
        for i, label in enumerate(unique_labels)
    ]
    leg1 = plt.legend(handles=emotion_legend_elements, title='Emotions', bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)

    # 说话人图例 (如果说话人数量不多)
    if len(unique_speakers) <= 20: # 限制说话人图例数量避免混乱
        speaker_legend_elements = [
            plt.Line2D([0], [0],
                      marker='o', # 使用统一形状
                      color='w',
                      label=f'Speaker {speaker}',
                      markerfacecolor=colors(j), # 使用对应颜色
                      markersize=10)
            for j, speaker in enumerate(unique_speakers)
        ]
        # 将第二个图例添加到第一个下方
        plt.legend(handles=speaker_legend_elements, title='Speakers', bbox_to_anchor=(1.02, 0.5), loc='center left', borderaxespad=0.)
        # 确保第一个图例仍在图中
        plt.gca().add_artist(leg1)


    plt.title(f"{title_prefix} Clustering ({method.upper()}) - Shapes:Emotion, Colors:Speaker")
    plt.xlabel(f"{method.upper()} Component 1")
    plt.ylabel(f"{method.upper()} Component 2")
    plt.grid(True) # 添加网格

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        print(f"保存图像到: {save_path}")
        plt.savefig(save_path, bbox_inches='tight') # 确保标签完整显示
        plt.close()
    else:
        plt.show()


def get_base64(fig):
    """
    将matplotlib图形转换为base64编码 (与 visual_clustering.py 相同)
    """
    buf = BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    return img_base64

def generate_paths(model_type):
    """
    根据模型类型生成相关路径
    Args:
        model_type: 模型类型 ('npy_cnn', 'npy_contrastive')
    Returns:
        dict: 包含各种路径的字典
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_model_dir = os.path.join(current_dir, "../models") # 指向项目根目录下的 models
    base_features_dir = os.path.join(current_dir, "../features") # 指向项目根目录下的 features
    base_visible_dir = os.path.join(current_dir, "../model_visible") # 指向项目根目录下的 model_visible
    base_clustering_dir = os.path.join(current_dir, "../clustering_feature") # 指向项目根目录下的 clustering_feature

    paths = {
        'data_folder': os.path.join(base_features_dir, "mel_npy/CREMA-D"), # 统一使用 npy 数据
        'model_path': None, # 将在下面设置
        'output_dir': base_visible_dir,
        'features_dir': base_clustering_dir
    }

    if model_type == 'npy_cnn':
        # 修改: 使用 current_dir 替代 base_model_dir
        paths['model_path'] = os.path.join(current_dir, "npy_cnn/npy_cnn_model.pth")

    elif model_type == 'npy_contrastive':
        # 修改: 使用 current_dir 替代 base_model_dir
        # 注意：这里保留了上一轮修改的 "npy_contrastive/" 子目录，如果模型直接在 current_dir 下，请移除 "npy_contrastive/"
        paths['model_path'] = os.path.join(current_dir, "npy_contrastive/npy_contrastive_model.pth")
    else:
        raise ValueError(f"未知的模型类型: {model_type}")

    # 确保所有路径都是绝对路径并处理反斜杠
    for key in paths:
        if paths[key]:
            paths[key] = os.path.abspath(paths[key]).replace("/", "\\")

    # 检查模型文件是否存在
    if not paths['model_path'] or not os.path.exists(paths['model_path']):
         print(f"警告: 模型文件未找到于 {paths['model_path']}。请确保模型已训练并保存。")
         # 可以选择抛出错误或允许脚本继续（如果只想加载数据）
         # raise FileNotFoundError(f"模型文件未找到: {paths['model_path']}")

    print(f"模型路径: {paths['model_path']}")
    print(f"数据路径: {paths['data_folder']}")
    print(f"输出图片目录: {paths['output_dir']}")
    print(f"特征保存目录: {paths['features_dir']}")

    return paths

def load_model_and_data(model_type, data_folder, model_path, batch_size=64):
    """
    加载预训练模型和数据集
    Args:
        model_type (str): 模型类型 ('npy_cnn', 'npy_contrastive')
        data_folder (str): 数据集路径
        model_path (str): 模型权重文件路径
        batch_size (int): 批量大小
    Returns:
        model: 加载的模型
        test_loader: 测试数据加载器
        class_indices: 类别映射
        speaker_indices: 说话人映射
    """
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"未知的模型类型: {model_type}")

    config = MODEL_REGISTRY[model_type]
    model_class = config['model_class']
    load_func = config['load_func']
    data_args = config['data_args']

    # 加载数据 (只需要测试集进行可视化)
    # 注意：load_datasets 返回 train, val, test loaders, class_indices, speaker_indices
    # 我们只关心 test_loader 和 indices
    print(f"使用 {load_func.__name__} 加载数据...")
    _, _, test_loader, class_indices, speaker_indices = load_func(
        data_folder=data_folder,
        batch_size=batch_size,
        **data_args # 传递特定于数据加载器的参数
    )

    # 初始化模型
    # 确保 num_classes 与加载的数据集一致
    num_classes = len(class_indices)
    print(f"初始化模型 {model_class.__name__}，类别数: {num_classes}")
    model = model_class(num_classes=num_classes)

    # 加载预训练权重
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用的设备: {device}")

    if model_path and os.path.exists(model_path):
        print(f"加载预训练模型: {model_path}")
        try:
            # 明确指定 map_location，并将模型移动到目标设备
            model.load_state_dict(torch.load(model_path, map_location=device))
            model = model.to(device)
        except Exception as e:
            print(f"加载模型权重时出错: {e}")
            print("请检查模型架构是否与保存的权重匹配。")
            raise e
    else:
        print(f"警告: 模型文件未找到或未提供: {model_path}。将使用随机初始化的模型。")
        model = model.to(device) # 即使没有加载权重，也要移动到设备

    model.eval()  # 设置为评估模式
    return model, test_loader, class_indices, speaker_indices, device

def extract_features(model, data_loader, device):
    """
    从数据加载器中提取特征、标签和说话人ID
    """
    all_features = []
    all_labels = []
    all_speaker_ids = []
    model.eval()
    with torch.no_grad():
        for i, (inputs, labels, speaker_ids) in enumerate(data_loader):
            inputs = inputs.to(device)
            # 模型 forward 方法返回 (outputs, features)
            _, features = model(inputs)

            all_features.append(features.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_speaker_ids.append(speaker_ids.cpu().numpy()) # 假设 speaker_ids 已经是数值 ID

            print(f"\r提取特征: Batch {i+1}/{len(data_loader)}", end="")
        print("\n特征提取完成.")

    all_features = np.concatenate(all_features, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    all_speaker_ids = np.concatenate(all_speaker_ids, axis=0)

    return all_features, all_labels, all_speaker_ids

# 主程序入口
if __name__ == "__main__":

    # --- 直接在此处定义参数 ---
    model_type = "npy_cnn"  # 在这里直接指定模型类型: 'npy_cnn' 或 'npy_contrastive'
    batch_size = 32         # 使用默认的 batch_size
    methods = ['tsne', 'pca', 'umap'] # 使用默认的降维方法
    skip_feature_extraction = False # 默认不跳过特征提取
    # -------------------------

    # 1. 生成路径 (使用硬编码的 model_type)
    paths = generate_paths(model_type)
    features_filename = f"npy_{model_type}_features.npz"
    features_save_path = os.path.join(paths['features_dir'], features_filename)
    features_save_path = os.path.abspath(features_save_path).replace("/", "\\")


    # 2. 加载模型和数据 / 或加载已有特征 (使用硬编码的参数)
    if not skip_feature_extraction:
        print("-" * 30)
        print(f"加载模型和数据: {model_type}")
        model, test_loader, class_indices, speaker_indices, device = load_model_and_data(
            model_type=model_type,
            data_folder=paths['data_folder'],
            model_path=paths['model_path'],
            batch_size=batch_size # 使用硬编码的 batch_size
        )

        # 3. 提取特征
        print("-" * 30)
        print("开始提取特征...")
        features, labels, speaker_ids = extract_features(model, test_loader, device)

        # 4. 保存特征
        os.makedirs(paths['features_dir'], exist_ok=True)
        print(f"保存提取的特征到: {features_save_path}")
        # 保存时也保存类别和说话人映射信息
        np.savez(features_save_path,
                 features=features,
                 labels=labels,
                 speaker_ids=speaker_ids,
                 class_indices=json.dumps(class_indices), # npz 不能直接存 dict，转为 json 字符串
                 speaker_indices=json.dumps(speaker_indices)) # 同上
    else:
        print("-" * 30)
        print(f"跳过特征提取，尝试加载已有特征文件: {features_save_path}")
        if os.path.exists(features_save_path):
            data = np.load(features_save_path, allow_pickle=True) # allow_pickle 以加载 json 字符串
            features = data['features']
            labels = data['labels']
            speaker_ids = data['speaker_ids']
            # 从 json 字符串恢复字典
            class_indices_json = data['class_indices']
            speaker_indices_json = data['speaker_indices']
            class_indices = json.loads(class_indices_json.item()) if isinstance(class_indices_json.item(), str) else class_indices_json.item()
            speaker_indices = json.loads(speaker_indices_json.item()) if isinstance(speaker_indices_json.item(), str) else speaker_indices_json.item()

            print(f"成功加载 {len(features)} 个特征.")
            # 将字符串键转回整数（如果需要）
            class_indices = {int(k): v for k, v in class_indices.items()}
            # speaker_indices 的键通常是字符串 ID，值是整数索引，可能不需要转换
        else:
            print(f"错误: 特征文件 {features_save_path} 未找到。请先运行不带 --skip_feature_extraction 的脚本。")
            sys.exit(1)


    # 5. 可视化聚类 (使用硬编码的 model_type 和 methods)
    print("-" * 30)
    print("开始生成聚类可视化...")
    title_prefix = f"{model_type.replace('_', ' ').title()}" # e.g., "Npy Cnn"

    for method in methods: # 使用硬编码的 methods 列表
        method = method.lower()
        if method not in ['tsne', 'pca', 'umap']:
            print(f"警告: 跳过未知方法 '{method}'")
            continue

        output_filename = f"npy_{model_type}_{method}_clusters.png"
        output_save_path = os.path.join(paths['output_dir'], output_filename)
        output_save_path = os.path.abspath(output_save_path).replace("/", "\\")


        visualize_clusters(
            features=features,
            labels=labels,
            speaker_ids=speaker_ids,
            method=method,
            save_path=output_save_path,
            title_prefix=title_prefix
        )

    print("-" * 30)
    print("所有可视化完成.")
