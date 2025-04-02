import os
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import numpy as np

def visualize_clustering(model, dataloader, device, save_dir="../model_visible", use_tsne=True, num_classes=10):
    """
    可视化聚类情况并保存图像

    Args:
        model (nn.Module): 训练的模型
        dataloader (DataLoader): 数据加载器
        device (torch.device): 训练设备
        save_dir (str): 图片保存路径
        use_tsne (bool): 是否使用 t-SNE 降维，默认为 True
        num_classes (int): 类别数量
    """
    # 获取所有样本的特征和标签
    all_features = []
    all_labels = []

    model.eval()
    with torch.no_grad():
        for images, labels, _ in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            # 获取模型输出和特征
            _, features = model(images)
            all_features.append(features.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    # 拼接所有数据
    all_features = np.concatenate(all_features, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # 降维到2D
    if use_tsne:
        print("使用 t-SNE 进行降维...")
        tsne = TSNE(n_components=2, random_state=0)
        reduced_features = tsne.fit_transform(all_features)
    else:
        print("使用 PCA 进行降维...")
        pca = PCA(n_components=2)
        reduced_features = pca.fit_transform(all_features)

    # 可视化数据
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=all_labels, cmap='tab10', s=10)
    plt.colorbar(scatter)
    plt.title(f"Clustering Visualization (t-SNE)" if use_tsne else "Clustering Visualization (PCA)")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")

    # 保存图片
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "clustering_visualization.png")
    plt.savefig(save_path)
    print(f"聚类可视化图已保存到 {save_path}")

    plt.show()