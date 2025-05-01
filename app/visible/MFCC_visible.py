import base64
from io import BytesIO
import matplotlib.pyplot as plt
import numpy as np
import librosa
from pydub import AudioSegment

def extract_mfcc(audio_data):
    """
    从预处理后的音频数据中提取MFCC特征
    
    Args:
        audio_data: 预处理后的音频数据 (numpy数组)
        
    Returns:
        tuple: (原始MFCC特征数组, 每帧MFCC特征的平均值)
    """
    try:
        # 提取MFCC特征 (假设采样率已经是16000Hz)
        mfcc = librosa.feature.mfcc(y=audio_data, sr=16000, n_mfcc=13)
        
        # 转置使时间步为第一维度
        mfcc_transposed = mfcc.T
        
        # 计算每帧MFCC特征的平均值
        mfcc_mean = np.mean(mfcc, axis=1)  # 对时间维度取平均
        
        # 返回原始特征和平均值
        return mfcc_transposed, mfcc_mean
        
    except Exception as e:
        print(f"提取MFCC特征时出错: {e}")
        raise

def mfcc_heatmap(data, title="Feature Heatmap"):
    """
    将二维或一维数据绘制为热力图（heatmap），并返回热力图的 Base64 编码以及特征的名称。

    Args:
        data (list of list, ndarray, or list): 特征数据，可以是二维数据或一维数据。
        title (str): 图的标题。

    Returns:
        dict: 包含以下键值对的字典：
            - 'MFCC_feature_name': 数据名称（例如：“MFCC 特征”）
            - 'mfcc_base64': 图像的 Base64 编码字符串
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

        # 返回包含特征名称和 Base64 编码的字典
        return {
            'feature_name': 'MFCC Features',  # 数据的名称
            'base64': img_base64  # 图像的 Base64 编码
        }

    except Exception as e:
        print(f"可视化时出错: {e}")
        return None
