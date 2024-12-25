import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# 加载音频文件
y, sr = librosa.load('../EnglishDataset/ANGERY/1001_DFA_ANG_XX.wav', sr=None)

# 获取音频信号长度
signal_length = len(y)

# 自动计算 n_fft 为信号长度的两倍
n_fft = signal_length * 2

# 将 n_fft 调整为最接近的 2 的幂
n_fft = 2 ** int(np.ceil(np.log2(n_fft)))

# 计算频谱
D = librosa.amplitude_to_db(np.abs(librosa.stft(y, n_fft=n_fft)), ref=np.max)

# 绘制频谱图
plt.figure(figsize=(10, 6))
librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
plt.colorbar(format='%+2.0f dB')
plt.title('Log-frequency power spectrogram')
plt.show()
