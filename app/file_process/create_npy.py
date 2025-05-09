import os
import librosa
import numpy as np

def audio_bytes_to_npy(input_data, sr=16000, n_fft=2048, hop_length=512, n_mels=128):
    """
    将音频字节流转换为梅尔频谱npy数据
    
    :param input_data: 音频数据(可以是文件路径str、numpy数组或字节流)
    :param sr: 采样率，默认16000
    :param n_fft: FFT窗口大小，默认2048
    :param hop_length: 帧移，默认512
    :param n_mels: 梅尔滤波器数量，默认128
    :return: 梅尔频谱的numpy数组
    """
    try:
        # 判断输入类型
        if isinstance(input_data, str):
            y, _ = librosa.load(input_data, sr=sr)
        elif isinstance(input_data, np.ndarray):
            y = input_data
        elif isinstance(input_data, bytes):
            y, sr = librosa.load(io.BytesIO(input_data), sr=sr, mono=True)
        else:
            raise ValueError("输入数据必须是文件路径(str)、numpy数组或字节流(bytes)")
            
        # 计算频率范围
        mel_low = 0
        mel_high = 2595 * np.log10(1 + (sr / 2) / 700)

        # 提取梅尔频谱
        mel_spec = librosa.feature.melspectrogram(
            y=y, sr=sr,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            fmin=mel_low,
            fmax=mel_high
        )
        
        # 使用20*np.log10转换分贝
        return 20 * np.log10(np.maximum(mel_spec, np.finfo(float).eps))
        
    except Exception as e:
        print(f"处理音频数据时出错: {e}")
        return None

def create_npy(input_folder, output_folder, sr=16000, n_fft=2048, hop_length=512, n_mels=128):
    """
    循环处理音频数据集文件夹，提取每个文件梅尔频谱并保存为.npy文件
    """
    if not os.path.exists(input_folder):
        print(f"音频文件夹 {input_folder} 不存在。")
        return

    # 获取输入文件夹的名称（不包含路径部分）
    input_folder_name = os.path.basename(os.path.normpath(input_folder))

    # 创建一个新的文件夹来保存输出的.npy文件，名字与输入文件夹相同
    output_dataset_folder = os.path.join(output_folder, input_folder_name)
    os.makedirs(output_dataset_folder, exist_ok=True)

    splits = ["train", "val", "test"]  # 分别处理train/val/test文件夹
    for split in splits:
        split_path = os.path.join(input_folder, split)
        if not os.path.exists(split_path):
            print(f"数据集 {split} 文件夹不存在，跳过...")
            continue

        # 遍历每个类别
        categories = [cat for cat in os.listdir(split_path) 
                     if os.path.isdir(os.path.join(split_path, cat))]
        for category in categories:
            category_path = os.path.join(split_path, category)

            # 创建对应的输出目录
            output_category_folder = os.path.join(output_dataset_folder, split, category)
            os.makedirs(output_category_folder, exist_ok=True)

            # 获取类别下的所有音频文件
            audio_files = [f for f in os.listdir(category_path) 
                          if f.lower().endswith(('.wav', '.mp3'))]
            if not audio_files:
                print(f"类别 {category} 中未找到任何音频文件，跳过...")
                continue

            for audio_file in audio_files:
                file_path = os.path.join(category_path, audio_file)
                output_file_path = os.path.join(
                    output_category_folder, 
                    f"{os.path.splitext(audio_file)[0]}.npy"
                )

                try:
                    mel_db = audio_bytes_to_npy(file_path, sr, n_fft, hop_length, n_mels)
                    if mel_db is not None:
                        np.save(output_file_path, mel_db)
                except Exception as e:
                    print(f"处理文件 {file_path} 时出错: {e}")
            
            print(f"梅尔频谱特征保存完成：{category}")

    # 返回保存.npy文件的文件夹路径
    return output_dataset_folder

if __name__ == '__main__':
    npy_output = '../features/mel_npy'
    input_data = '../../ProcessedDataSet/Split/EmoDB'

    current_dir = os.path.dirname(os.path.abspath(__file__))
    npy_output = os.path.join(current_dir, npy_output)
    input_data = os.path.join(current_dir, input_data)

    create_npy(input_data, npy_output)
