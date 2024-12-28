import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import io
import base64

def spectrogram_base64(audio_bytes, sr=16000):
    """
    将音频数据转换为频谱图并返回其 Base64 编码。

    :param audio_bytes: 音频数据的字节流
    :param sr: 采样率，默认 16000
    :return: 包含 'feature_name' 和 'base64' 的字典
    """
    # 使用 librosa 加载音频数据
    y, sr = librosa.load(io.BytesIO(audio_bytes), sr=sr, mono=True)

    # 计算短时傅里叶变换（STFT）
    D = librosa.stft(y)

    # 创建频谱图
    plt.figure(figsize=(12, 6))
    librosa.display.specshow(librosa.amplitude_to_db(np.abs(D), ref=np.max),
                             y_axis='log', x_axis='time', sr=sr)
    plt.title('Spectrogram')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.colorbar(format='%+2.0f dB')

    # 将图像保存到内存中
    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png')
    img_buf.seek(0)

    # 将图片数据编码为 Base64
    img_base64 = base64.b64encode(img_buf.read()).decode('utf-8')

    # 关闭图像，以节省内存
    plt.close()

    return {
        'feature_name': 'Spectrogram',  # 数据的名称
        'base64': img_base64  # 图像的 Base64 编码
    }
