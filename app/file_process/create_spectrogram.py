import os
from app.visible.spectrogram import spectrogram_base64, save_spectrogram


def create_spectrogram(input_folder, output_folder, sr=16000):
    """
    循环处理音频数据集文件夹，提取每个文件频谱图并保存为PNG。

    :param input_folder: 包含 train, val, test 子文件夹的音频数据集路径
    :param output_folder: 保存提取的频谱图的文件夹路径
    :param sr: 采样率，默认16000
    :return: str: 返回保存频谱图的文件夹路径
    """
    if not os.path.exists(input_folder):
        print(f"音频文件夹 {input_folder} 不存在。")
        return

    # 获取输入文件夹的名称（不包含路径部分）
    input_folder_name = os.path.basename(os.path.normpath(input_folder))

    # 创建一个新的文件夹来保存输出的 PNG 文件，名字与输入文件夹相同
    output_dataset_folder = os.path.join(output_folder, input_folder_name)
    os.makedirs(output_dataset_folder, exist_ok=True)

    splits = ["train", "val", "test"]  # 分别处理 train, val, test 文件夹
    for split in splits:
        split_path = os.path.join(input_folder, split)
        if not os.path.exists(split_path):
            print(f"数据集 {split} 文件夹不存在，跳过...")
            continue

        # 遍历每个类别
        categories = [cat for cat in os.listdir(split_path) if os.path.isdir(os.path.join(split_path, cat))]
        for category in categories:
            category_path = os.path.join(split_path, category)

            # 创建对应的输出目录
            output_category_folder = os.path.join(output_dataset_folder, split, category)
            os.makedirs(output_category_folder, exist_ok=True)

            # 获取类别下的所有音频文件
            audio_files = [f for f in os.listdir(category_path) if f.endswith('.wav')]
            if not audio_files:
                print(f"类别 {category} 中未找到任何 .wav 文件，跳过...")
                continue

            for audio_file in audio_files:
                file_path = os.path.join(category_path, audio_file)
                output_file_path = os.path.join(output_category_folder, f"{os.path.splitext(audio_file)[0]}.png")

                try:
                    # 生成频谱图的 Base64 编码
                    _, spectrogram_data = spectrogram_base64(file_path, sr=sr)
                    if spectrogram_data is not None:
                        # 保存频谱图为 PNG 文件
                        save_spectrogram(spectrogram_data, output_file_path)
                except Exception as e:
                    print(f"处理文件 {file_path} 时出错: {e}")
            print(f"频谱图特征保存完成：{category}")

    # 返回保存频谱图的文件夹路径
    return output_dataset_folder

if __name__ == '__main__':
    spectrogram_file = '../features/mel_spectrogram'
    splited_data = '../../ProcessedDataSet/Split/CREMA-D'
    create_spectrogram(splited_data, spectrogram_file)