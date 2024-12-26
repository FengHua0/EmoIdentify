import base64
from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np

def visualize_features_as_heatmap(data, title="Feature Heatmap"):
    """
    将二维或一维数据绘制为热力图（heatmap），并将图像编码为 Base64。

    Args:
        data (list of list, ndarray, or list): 特征数据，可以是二维数据或一维数据。
        title (str): 图的标题。

    Returns:
        str: 图像的 Base64 编码字符串。
    """
    try:
        # 如果输入数据是一维的，将其转换为二维数据（形状为 (1, n)）
        if isinstance(data, list) or isinstance(data, np.ndarray):
            data = np.array(data)
            if data.ndim == 1:  # 如果是1D数据
                data = data.reshape(1, -1)  # 转换为 1xN 形式的二维数组

        # 创建图像对象
        fig, ax = plt.subplots(figsize=(12, 8))
        cax = ax.imshow(data, aspect='auto', cmap='viridis', origin='lower')
        fig.colorbar(cax, ax=ax, label="Feature Value")
        ax.set_title(title, fontsize=16)
        ax.set_xlabel("Feature Index", fontsize=12)
        ax.set_ylabel("Time Step", fontsize=12)
        fig.tight_layout()

        # 将图像保存到内存中
        buf = BytesIO()
        fig.savefig(buf, format='png')  # 保存为 PNG 格式
        plt.close(fig)  # 关闭图像以释放内存
        buf.seek(0)

        # 将图像转换为 Base64 编码
        img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        return img_base64

    except Exception as e:
        print(f"可视化时出错: {e}")
        return None