import os
import librosa
import numpy as np
import noisereduce as nr
import soundfile as sf

# 1. 预加重
def pre_emphasis(y, coeff=0.97):
    """
    对音频信号进行预加重处理，以增强高频部分。
    input:
    y ： 输入的音频信号（数组）
    coeff ： 预加重系数，默认值为0.97
    Returns:
    预加重后的音频信号（数组）
    """
    return np.append(y[0], y[1:] - coeff * y[:-1])

# 2. 降噪
def noise_reduction(y, sr, frame_length_ms=20, hop_length_ms=10, noise_duration=0.5, noise_factor=0.02):
    """
    基于频谱减法的音频降噪方法。

    :param y: 输入音频信号 (NumPy 数组)
    :param sr: 采样率 (Hz)
    :param frame_length_ms: 每帧时长（毫秒），默认20ms
    :param hop_length_ms: 帧移（毫秒），默认10ms
    :param noise_duration: 用于估计噪声的时间段（秒），默认0.5秒
    :param noise_factor: 噪声谱减法比例，默认0.02
    :return: 降噪后的音频信号 (NumPy 数组)
    """
    # 1. 动态计算帧长和帧移
    n_fft = int(sr * (frame_length_ms / 1000))  # 将帧长从毫秒转为采样点数
    hop_length = int(sr * (hop_length_ms / 1000))  # 将帧移从毫秒转为采样点数

    # 2. 计算短时傅里叶变换 (STFT)
    stft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    magnitude, phase = np.abs(stft), np.angle(stft)  # 分解为幅值谱和相位谱

    # 3. 估计噪声谱
    noise_frames = int((noise_duration * sr) / hop_length)  # 噪声持续时间对应的帧数
    noise_estimation = np.mean(magnitude[:, :noise_frames], axis=1, keepdims=True)  # 取噪声段的平均幅值谱

    # 4. 频谱减法（减去噪声谱，确保非负）
    magnitude_denoised = np.maximum(magnitude - noise_factor * noise_estimation, 0)

    # 5. 重建 STFT 幅值与相位
    stft_denoised = magnitude_denoised * np.exp(1j * phase)

    # 6. 逆短时傅里叶变换 (ISTFT)
    y_denoised = librosa.istft(stft_denoised, hop_length=hop_length)

    return y_denoised

# 3. 音频预处理（降噪 + 预加重）
def preprocess_audio(input_data, sr=16000, pre_emph_coeff=0.97):
    """
    加载音频文件或处理内存中的音频数据，并进行降噪和预加重处理。
    input:
    input_data ： 音频文件路径 (str) 或内存中的音频数据 (NumPy 数组)
    sr ： 音频采样率，默认16000Hz
    pre_emph_coeff ： 预加重系数，默认值为0.97
    Returns:
    处理后的音频信号（NumPy 数组）
    """
    try:
        # 判断输入类型：文件路径或音频数据
        if isinstance(input_data, str):
            y, _ = librosa.load(input_data, sr=sr, mono=True)
        elif isinstance(input_data, np.ndarray):
            y = input_data
        else:
            raise ValueError("输入数据必须是文件路径 (str) 或音频数据 (NumPy 数组)。")

        # 检查音频数据是否为空
        if len(y) == 0:
            print(f"输入音频数据为空，跳过处理...")
            return None

        # 预加重处理
        y_denoised = pre_emphasis(y, coeff=pre_emph_coeff)
        # 降噪处理
        y_preemphasized = noise_reduction(y_denoised, sr)


        return y_preemphasized  # 返回处理后的音频数据

    except Exception as e:
        print(f"处理音频数据时出错: {e}")
        return None

# 4. 批量处理音频文件（按类别）
def process_audio_folder_with_categories(input_folder, output_folder, sr=16000, pre_emph_coeff=0.97):
    """
    批量处理音频文件夹中的音频文件，并保存处理后的结果到输出文件夹。
    输出的音频文件会保存到以 'processed_' 为前缀命名的文件夹中，并保留原始的文件夹结构。

    input:
    input_folder ： 输入文件夹路径，包含多个类别子文件夹
    output_folder ： 输出文件夹路径，处理后的音频将保存到此
    sr ： 音频采样率，默认16000Hz
    pre_emph_coeff ： 预加重系数，默认值为0.97

    Returns:
    str： 返回输出的根文件夹路径（处理后的文件夹路径）
    """
    # 构建输出文件夹的根路径，名字为 'processed_' + input_folder 的文件夹名
    input_folder_name = os.path.basename(os.path.normpath(input_folder))  # 获取输入文件夹的名称
    processed_root_path = os.path.join(output_folder, f"processed_{input_folder_name}")

    if not os.path.exists(processed_root_path):
        os.makedirs(processed_root_path)  # 创建根文件夹

    # 获取所有类别子文件夹
    categories = [cat for cat in os.listdir(input_folder) if os.path.isdir(os.path.join(input_folder, cat))]

    # 遍历每个类别并处理音频
    for category in categories:
        category_input_path = os.path.join(input_folder, category)
        category_output_path = os.path.join(processed_root_path, category)  # 输出路径保留原类别结构

        if not os.path.exists(category_output_path):
            os.makedirs(category_output_path)  # 创建该类别的输出文件夹

        # 获取当前类别下所有音频文件
        audio_files = [f for f in os.listdir(category_input_path) if f.endswith('.wav')]
        print(f"找到 {len(audio_files)} 个音频文件在类别 {category}")

        for audio_file in audio_files:
            input_path = os.path.join(category_input_path, audio_file)
            output_path = os.path.join(category_output_path, audio_file)  # 保持文件名不变

            try:
                # 处理音频文件
                y_processed = preprocess_audio(input_path, sr=sr, pre_emph_coeff=pre_emph_coeff)

                # 如果处理后的音频为空或无效，跳过保存
                if y_processed is None or len(y_processed) == 0:
                    print(f"文件 {audio_file} 处理后为空，跳过保存")
                    continue

                # 保存处理后的音频
                sf.write(output_path, y_processed, sr)

            except Exception as e:
                print(f"处理文件 {audio_file} 时出错: {e}")

        print(f"类别 {category} 处理完毕")

    # 返回处理后的根文件夹路径（即 'processed_' 文件夹）
    return processed_root_path
