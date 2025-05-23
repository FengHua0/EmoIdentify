import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import io
import base64
from matplotlib.backends.backend_template import FigureCanvas


def getbase64():
    img_buf_with_coords = io.BytesIO()
    plt.savefig(img_buf_with_coords, format='png')
    img_buf_with_coords.seek(0)
    base64_with_coords = base64.b64encode(img_buf_with_coords.read()).decode('utf-8')
    plt.close()
    return base64_with_coords

def spectrogram_base64(input_data, sr=16000):
    """
    将音频数据转换为频谱图并返回其 Base64 编码。

    :param audio_bytes: 音频数据的字节流
    :param sr: 采样率，默认 16000
    :return: 包含 'feature_name' 和 'base64' 的字典
    """
    # 判断输入类型
    if isinstance(input_data, str):
        y, _ = librosa.load(input_data, sr=sr)
    elif isinstance(input_data, np.ndarray):
        y = input_data
    elif isinstance(input_data, bytes):  # 如果是字节串类型
        y, sr = librosa.load(io.BytesIO(input_data), sr=sr, mono=True)  # 使用 BytesIO 读取字节数据

    else:
        raise ValueError("输入数据必须是文件路径 (str) 或音频数据 (NumPy 数组)。")

    # 2. 进行短时傅里叶变换（STFT）
    S = librosa.stft(y, n_fft=2048, hop_length=1024)

    # 3. 取幅度谱
    magnitude = np.abs(S)

    # 4. 设定梅尔尺度的频率范围
    mel_low = 0
    mel_high = 2595 * np.log10(1 + (sr / 2) / 700)

    # 5. 创建梅尔滤波器组
    mel_filter_bank = librosa.filters.mel(sr=sr, n_fft=2048, n_mels=128, fmin=mel_low, fmax=mel_high)

    # 6. 将STFT幅度谱映射到梅尔尺度
    S = np.dot(mel_filter_bank, magnitude)

    # 7. 防止数值下溢，保证非零
    S = np.maximum(S, np.finfo(float).eps)

    # 8. 进行对数压缩
    S = 20 * np.log10(S)

    # 有坐标的图片（用于特征展示）：base64_with_coords
    plt.figure(figsize=(12, 6))
    librosa.display.specshow(S, sr=sr, x_axis='time', y_axis='log', cmap='jet')
    plt.title('Spectrogram')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.colorbar(format='%+2.0f dB')
    base64_with_coords = getbase64()

    # 没有坐标的图片（用于训练和预测）
    plt.figure(figsize=(12, 6))
    librosa.display.specshow(S, sr=sr, x_axis='time', y_axis='log', cmap='viridis')
    plt.axis('off')  # 关闭坐标轴
    plt.gca().set_position([0, 0, 1, 1])  # 去除图像边距
    base64_no_coords = getbase64()

    show = {
        'feature_name': 'Spectrogram',  # 数据的名称
        'base64': base64_with_coords  # 图像的 Base64 编码
    }
    train = base64_no_coords

    return show, train

def linear_spectrogram_base64(input_data, sr=16000):
    """
    将音频数据转换为线性频谱图并返回其 Base64 编码。
    不进行梅尔频率转换，使用线性频率刻度。

    :param input_data: 音频数据 (str路径/np数组/bytes)
    :param sr: 采样率，默认 16000
    :return: 包含 'feature_name' 和 'base64' 的字典
    """
    try:
        # 判断输入类型
        if isinstance(input_data, str):
            y, sr = librosa.load(input_data, sr=sr)
        elif isinstance(input_data, np.ndarray):
            y = input_data
        elif isinstance(input_data, bytes):
            y, sr = librosa.load(io.BytesIO(input_data), sr=sr, mono=True)
        else:
            raise ValueError("输入数据必须是文件路径(str)、NumPy数组或bytes")

        # 检查音频长度
        if len(y) < 2048:  # n_fft的默认值
            raise ValueError(f"音频太短({len(y)} samples)，至少需要2048 samples")

        # 使用短时傅里叶变换生成线性频谱图
        D = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))

        # 对数转换
        S = 20 * np.log10(np.maximum(D, np.finfo(float).eps))
        
        # 创建图形
        plt.figure(figsize=(12, 6))
        librosa.display.specshow(S, sr=sr, x_axis='time', y_axis='linear', cmap='viridis')
        plt.title('Linear Spectrogram')
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency (Hz)')
        plt.colorbar(format='%+2.0f dB')
        
        # 生成Base64编码
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        plt.close()
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')

        return {
            'feature_name': 'Linear Spectrogram',
            'base64': img_base64
        }

    except Exception as e:
        print(f"生成线性频谱图时出错: {str(e)}")
        return {
            'error': str(e),
            'feature_name': 'Linear Spectrogram Error'
        }

def save_spectrogram(base64_str, output_path):
    """
    将 Base64 编码的图像保存为 PNG 文件。

    :param base64_str: Base64 编码的图像字符串
    :param output_path: 输出 PNG 文件的路径（包含文件名）
    """
    try:
        # 解码 Base64 字符串
        img_data = base64.b64decode(base64_str)

        # 将解码后的二进制数据写入到文件
        with open(output_path, 'wb') as f:
            f.write(img_data)
    except Exception as e:
        print(f"保存图像时出错: {e}")