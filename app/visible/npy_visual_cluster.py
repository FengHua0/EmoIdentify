# 标准库导入
import os
import sys
import json
from pathlib import Path
from io import BytesIO
import base64

# 第三方库导入
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from sklearn.metrics.pairwise import cosine_distances

# 本地应用/项目导入
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from app.model_training.npy_cnn_training import CNN_RNN as NpyCnnCNN_RNN
from app.model_training.npy_cnn_training import load_datasets as load_npy_cnn_datasets
from app.model_training.npy_contrastive_training import CNN_RNN as NpyContrastiveCNN_RNN
from app.model_training.npy_contrastive_training import load_datasets as load_npy_contrastive_datasets

# 模型注册表
MODEL_REGISTRY = {
    'npy_cnn': {
        'model_class': NpyCnnCNN_RNN,
        'load_func': load_npy_cnn_datasets,
        'data_args': {'target_length': 100},
        # --- 添加 npy_cnn 的模型参数 ---
        'model_args': {'n_mels': 128, 'target_length': 100} # 假设 CNN 也用这些参数
    },
    'npy_contrastive': {
        'model_class': NpyContrastiveCNN_RNN,
        'load_func': load_npy_contrastive_datasets,
        'data_args': {'target_length': 100},
        # --- 添加 npy_contrastive 的模型参数 ---
        'model_args': {'n_mels': 128, 'target_length': 100} # 与 npy_contrastive_training.py 中的默认值匹配
    }
}

def visualize_clusters(features, labels, speaker_ids, save_path=None, title_prefix=""): # 移除 method 参数
    """
    可视化聚类结果 (使用 t-SNE)
    Args:
        features: 特征数组
        labels: 标签数组 (情感)
        speaker_ids: 说话人ID数组
        save_path: 图片保存路径
        title_prefix: 标题前缀 (例如模型类型)
    """
    print(f"开始使用 t-SNE 进行可视化...")
    # 降维
    reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(features)-1))
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


    plt.title(f"{title_prefix} Clustering (t-SNE) - Shapes:Emotion, Colors:Speaker")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
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
        paths['model_path'] = os.path.join(base_model_dir, "npy_cnn_model.pth")

    elif model_type == 'npy_contrastive':
        paths['model_path'] = os.path.join(base_model_dir, "npy_contrastive_model.pth")
    else:
        raise ValueError(f"未知的模型类型: {model_type}")

    # 确保所有路径都是绝对路径并处理反斜杠
    for key in paths:
        if paths[key]:
            paths[key] = os.path.abspath(paths[key]).replace("/", "\\")

    # 检查模型文件是否存在
    if not paths['model_path'] or not os.path.exists(paths['model_path']):
         print(f"警告: 模型文件未找到于 {paths['model_path']}。请确保模型已训练并保存。")

    print(f"模型路径: {paths['model_path']}")
    print(f"输出图片目录: {paths['output_dir']}")

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
    model_args = config.get('model_args', {}) # 获取模型参数，如果未定义则为空字典

    # 加载数据 (只需要测试集进行可视化)
    # 注意：load_datasets 返回 train, val, test loaders, class_indices, speaker_indices
    # 我们只关心 test_loader 和 indices
    _, _, test_loader, class_indices, speaker_indices = load_func(
        data_folder=data_folder,
        batch_size=batch_size,
        **data_args # 传递特定于数据加载器的参数
    )

    # 初始化模型
    # 确保 num_classes 与加载的数据集一致
    num_classes = len(class_indices)
    print(f"初始化模型 {model_class.__name__}，类别数: {num_classes}")
    # --- 修改：初始化模型时传递 model_args ---
    model = model_class(num_classes=num_classes, **model_args)

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

# 身份信息定量分析函数
def _calculate_speaker_distances(features, speaker_ids):
    """辅助函数：计算指定特征和说话人ID的距离统计"""
    cos_dists = cosine_distances(features)
    same_speaker_dists = []
    diff_speaker_dists = []
    
    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            if speaker_ids[i] == speaker_ids[j]:
                same_speaker_dists.append(cos_dists[i, j])
            else:
                diff_speaker_dists.append(cos_dists[i, j])
    
    # 处理没有同类或异类样本对的情况
    if not same_speaker_dists:
        print("警告: 未找到同一说话人的特征对。")
        same_mean, same_std, same_count = np.nan, np.nan, 0
    else:
        same_speaker_dists = np.array(same_speaker_dists)
        same_mean = float(np.mean(same_speaker_dists))
        same_std = float(np.std(same_speaker_dists))
        same_count = int(len(same_speaker_dists))

    if not diff_speaker_dists:
        print("警告: 未找到不同说话人的特征对。")
        diff_mean, diff_std, diff_count = np.nan, np.nan, 0
    else:
        diff_speaker_dists = np.array(diff_speaker_dists)
        diff_mean = float(np.mean(diff_speaker_dists))
        diff_std = float(np.std(diff_speaker_dists))
        diff_count = int(len(diff_speaker_dists))

    return {
        "same_speaker_mean": same_mean,
        "same_speaker_std": same_std,
        "diff_speaker_mean": diff_mean,
        "diff_speaker_std": diff_std,
        "same_count": same_count,
        "diff_count": diff_count
    }

def quantitative_speaker_analysis(features, speaker_ids, labels=None, class_indices=None):
    """
    改进版：支持按情感类别分析说话人距离，并显示情感名称
    Args:
        class_indices: 类别名称映射字典 {label_idx: label_name}
    """
    features = np.asarray(features)
    speaker_ids = np.asarray(speaker_ids)
    if labels is not None:
        labels = np.asarray(labels)
        unique_emotions = np.unique(labels)
    
    n = features.shape[0]
    if n < 2:
        print("警告: 特征数量不足 (< 2)，无法进行定量分析。")
        return None

    results = {}
    
    if labels is not None:
        for emotion in unique_emotions:
            emotion_mask = (labels == emotion)
            emotion_features = features[emotion_mask]
            emotion_speaker_ids = speaker_ids[emotion_mask]
            
            # 获取情感名称（如果有class_indices）
            emotion_name = class_indices.get(emotion, str(emotion)) if class_indices else str(emotion)
            
            emotion_result = _calculate_speaker_distances(emotion_features, emotion_speaker_ids)
            results[f"{emotion_name}"] = emotion_result  # 使用情感名称作为键
    
    global_result = _calculate_speaker_distances(features, speaker_ids)
    results["全局统计"] = global_result
    
    print("-" * 30)
    print("身份信息定量分析结果：")
    print(json.dumps(results, indent=2, ensure_ascii=False))
    return results

# 主程序入口
if __name__ == "__main__":

    model_type = "npy_cnn"  # 在这里直接指定模型类型: 'npy_cnn' 或 'npy_contrastive'
    batch_size = 64

    # 1. 生成路径 (使用硬编码的 model_type)
    paths = generate_paths(model_type)
    features_filename = f"{model_type}_features.npz"
    features_save_path = os.path.join(paths['features_dir'], features_filename)
    features_save_path = os.path.abspath(features_save_path).replace("/", "\\")


    # 2. 检查特征文件是否存在，如果存在则加载，否则提取并保存
    if os.path.exists(features_save_path):
        print("-" * 30)
        print(f"发现已存在的特征文件，直接加载: {features_save_path}")
        try:
            data = np.load(features_save_path, allow_pickle=True) # allow_pickle 以加载 json 字符串
            features = data['features']
            labels = data['labels']
            speaker_ids = data['speaker_ids']
            # 从 json 字符串恢复字典
            class_indices_json = data['class_indices']
            speaker_indices_json = data['speaker_indices']
            # 处理 .item() 可能返回 bytes 的情况
            class_indices_str = class_indices_json.item()
            speaker_indices_str = speaker_indices_json.item()
            if isinstance(class_indices_str, bytes):
                class_indices_str = class_indices_str.decode('utf-8')
            if isinstance(speaker_indices_str, bytes):
                speaker_indices_str = speaker_indices_str.decode('utf-8')

            class_indices = json.loads(class_indices_str)
            speaker_indices = json.loads(speaker_indices_str)

            print(f"成功加载 {len(features)} 个特征.")
            # 将字符串键转回整数（如果需要）
            class_indices = {int(k): v for k, v in class_indices.items()}

        except Exception as e:
            print(f"加载特征文件时出错: {e}")
            print("将尝试重新提取特征。")
            # 设置标记，强制执行特征提取流程
            force_extraction = True
    else:
        # 文件不存在，需要提取
        force_extraction = True
        print(f"特征文件 {features_save_path} 未找到。")

    # 如果文件不存在或加载失败，则执行提取流程
    if not os.path.exists(features_save_path) or 'force_extraction' in locals() and force_extraction:
        print("-" * 30)
        print(f"加载模型和数据以提取特征: {model_type}")
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
        try:
            np.savez(features_save_path,
                     features=features,
                     labels=labels,
                     speaker_ids=speaker_ids,
                     class_indices=json.dumps(class_indices), # npz 不能直接存 dict，转为 json 字符串
                     speaker_indices=json.dumps(speaker_indices)) # 同上
            print("特征保存成功。")
        except Exception as e:
            print(f"保存特征文件时出错: {e}")

    if 'features' in locals() and 'labels' in locals() and 'speaker_ids' in locals():

        # 5. 可视化聚类 (使用加载或新提取的特征)
        print("-" * 30)
        print("开始生成聚类可视化...")
        title_prefix = f"{model_type.replace('_', ' ').title()}" # e.g., "Npy Cnn"
        
        # 定量分析（传入class_indices）
        analysis_results = quantitative_speaker_analysis(
            features, 
            speaker_ids, 
            labels=labels,
            class_indices=class_indices
        )

        # 修改输出文件名格式，移除 method
        output_filename = f"{model_type}_clusters.png"
        output_save_path = os.path.join(paths['output_dir'], output_filename)
        output_save_path = os.path.abspath(output_save_path).replace("/", "\\")

        # 调用 visualize_clusters，移除 method 参数
        visualize_clusters(
            features=features,
            labels=labels,
            speaker_ids=speaker_ids,
            save_path=output_save_path,
            title_prefix=title_prefix
        )

        print("-" * 30)
        print("可视化完成.")
    else:
        print("-" * 30)
        print("错误：未能加载或提取特征，无法进行分析和可视化。")
        sys.exit(1)
