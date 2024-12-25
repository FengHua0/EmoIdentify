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
def noise_reduction(y, sr):
    """
    使用音频的前0.5秒作为噪声样本，进行降噪处理。
    input:
    y ： 输入的音频信号（数组）
    sr ： 音频采样率
    Returns:
    降噪后的音频信号（数组）
    """
    noise_sample = y[:int(0.5 * sr)]  # 取前0.5秒作为噪声样本
    if len(noise_sample) == 0 or np.max(np.abs(noise_sample)) < 1e-6:
        print("噪声样本无效，跳过降噪")
        return y  # 如果噪声样本无效，返回原始信号
    try:
        y_denoised = nr.reduce_noise(y=y, sr=sr, y_noise=noise_sample)
        return y_denoised
    except Exception as e:
        print(f"降噪失败，返回原始信号: {e}")
        return y

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

        # 降噪处理
        y_denoised = noise_reduction(y, sr)
        # 预加重处理
        y_preemphasized = pre_emphasis(y_denoised, coeff=pre_emph_coeff)

        return y_preemphasized  # 返回处理后的音频数据

    except Exception as e:
        print(f"处理音频数据时出错: {e}")
        return None

# 4. 批量处理音频文件（按类别）
def process_audio_folder_with_categories(input_folder, output_folder, sr=16000, pre_emph_coeff=0.97):
    """
    批量处理音频文件夹中的音频文件，并保存处理后的结果到输出文件夹。
    input:
    input_folder ： 输入文件夹路径，包含多个类别子文件夹
    output_folder ： 输出文件夹路径，处理后的音频将保存到此
    sr ： 音频采样率，默认16000Hz
    pre_emph_coeff ： 预加重系数，默认值为0.97
    Returns:
    None
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)  # 创建输出文件夹

    # 获取所有类别子文件夹
    categories = [cat for cat in os.listdir(input_folder) if os.path.isdir(os.path.join(input_folder, cat))]
    for category in categories:
        category_input_path = os.path.join(input_folder, category)
        category_output_path = os.path.join(output_folder, category)

        if not os.path.exists(category_output_path):
            os.makedirs(category_output_path)  # 创建类别输出文件夹

        # 获取当前类别下所有音频文件
        audio_files = [f for f in os.listdir(category_input_path) if f.endswith('.wav')]
        print(f"找到 {len(audio_files)} 个音频文件在类别 {category}")

        for audio_file in audio_files:
            input_path = os.path.join(category_input_path, audio_file)
            output_path = os.path.join(category_output_path, audio_file)

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


if __name__ == '__main__':
    input_folder = '../../AudioWAV'  # 输入文件夹路径
    output_folder = '../../EnglishDataset'  # 输出文件夹路径
    process_audio_folder_with_categories(input_folder, output_folder)
    print("所有语音文件预处理完成")
