import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import io
import base64


def waveform_base64(input_data, sr=16000):
    """
    将音频数据（字节流或 NumPy 数组）转换为波形图并返回其 Base64 编码。

    :param input_data: 音频数据，支持字节流（bytes）或 NumPy 数组（numpy.ndarray）
    :param sr: 采样率，默认 16000
    :return: 包含 'feature_name' 和 'base64' 的字典
    """
    # 判断输入类型并加载音频数据
    if isinstance(input_data, bytes):  # 处理字节流
        y, _ = librosa.load(io.BytesIO(input_data), sr=sr, mono=True)
    elif isinstance(input_data, np.ndarray):  # 处理 NumPy 数组
        y = input_data
    else:
        raise ValueError("输入数据必须是字节流 (bytes) 或 NumPy 数组 (numpy.ndarray)。")

    # 创建波形图
    plt.figure(figsize=(12, 6))
    librosa.display.waveshow(y, sr=sr)
    plt.title('Waveform')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')

    # 将图像保存到内存中
    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png')
    img_buf.seek(0)

    # 将图片数据编码为 Base64
    img_base64 = base64.b64encode(img_buf.read()).decode('utf-8')

    # 关闭图像，以节省内存
    plt.close()

    return {
        'feature_name': 'Waveform',  # 数据的名称
        'base64': img_base64  # 图像的 Base64 编码
    }
