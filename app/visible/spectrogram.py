import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import io
import base64

def spectrogram_base64(audio_input, sr=16000):
    """
    将音频数据转换为频谱图并返回其 Base64 编码。

    :param audio_bytes: 音频数据的字节流
    :param sr: 采样率，默认 16000
    :return: 包含 'feature_name' 和 'base64' 的字典
    """
    if isinstance(audio_input, str):
        # 如果是文件路径，直接加载
        y, sr = librosa.load(audio_input, sr=sr, mono=True)
    elif isinstance(audio_input, (bytes, bytearray)):
        # 如果是字节流，将其转换为内存文件并加载
        y, sr = librosa.load(io.BytesIO(audio_input), sr=sr, mono=True)
    else:
        raise ValueError("audio_input 必须是文件路径 (str) 或字节流 (bytes)。")

    # 计算短时傅里叶变换（STFT）
    D = librosa.stft(y)

    # 计算幅值谱
    magnitude = np.abs(D)

    # 转换为对数幅值谱（分贝单位）
    log_magnitude = librosa.amplitude_to_db(magnitude, ref=np.max)

    # 创建频谱图
    plt.figure(figsize=(12, 6))
    librosa.display.specshow(log_magnitude, sr=sr, x_axis='time', y_axis='log', cmap='viridis')
    plt.title('Spectrogram')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.colorbar(format='%+2.0f dB')

    # 有坐标的图片（用于特征展示）：base64_with_coords
    img_buf_with_coords = io.BytesIO()
    plt.savefig(img_buf_with_coords, format='png')
    img_buf_with_coords.seek(0)
    base64_with_coords = base64.b64encode(img_buf_with_coords.read()).decode('utf-8')
    plt.close()

    # 没有坐标的图片（用于训练和预测）
    plt.figure(figsize=(12, 6))
    librosa.display.specshow(log_magnitude, sr=sr, x_axis=None, y_axis=None, cmap='viridis')
    plt.axis('off')  # 关闭坐标轴
    plt.gca().set_position([0, 0, 1, 1])  # 去除图像边距
    img_buf_no_coords = io.BytesIO()
    plt.savefig(img_buf_no_coords, format='png', bbox_inches='tight', pad_inches=0)
    img_buf_no_coords.seek(0)
    base64_no_coords = base64.b64encode(img_buf_no_coords.read()).decode('utf-8')
    plt.close()

    show = {
        'feature_name': 'Spectrogram',  # 数据的名称
        'base64': base64_with_coords  # 图像的 Base64 编码
    }
    train = base64_no_coords

    return show, train


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