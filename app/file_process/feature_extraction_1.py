import os
import librosa
import numpy as np
import pandas as pd


import librosa
import numpy as np

def one_features_extract(input_data, sr=16000, n_mfcc=13):
    """
    处理单个音频文件或内存中的音频数据，提取MFCC特征，并对所有帧取平均值。
    input:
    input_data : 音频文件路径 (str) 或 音频数据 (NumPy 数组)
    sr : 采样率，默认16000
    n_mfcc : 提取的MFCC特征数量，默认13
    Returns:
    features : 包含MFCC均值的列表
    """
    try:
        # 判断输入类型
        if isinstance(input_data, str):
            y, _ = librosa.load(input_data, sr=sr)
        elif isinstance(input_data, np.ndarray):
            y = input_data
        else:
            raise ValueError("输入数据必须是文件路径 (str) 或音频数据 (NumPy 数组)。")

        # 检查音频数据是否为空
        if len(y) == 0:
            print("音频数据为空，跳过...")
            return None

        # 提取MFCC特征
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)

        # 对所有帧的MFCC特征取平均值
        mfcc_mean = np.mean(mfccs, axis=1)
        return mfcc_mean.tolist()

    except Exception as e:
        print(f"提取特征时出错: {e}")
        return None

def process_audio_folder(input_folder, output_folder, sr=16000, n_mfcc=13):
    """
    循环处理音频数据集文件夹，提取每个文件的MFCC特征并保存为CSV。
    input:
    input_folder ： 包含 train, val, test 子文件夹的音频数据集路径
    output_folder ： 保存提取特征的CSV文件路径
    sr ： 采样率，默认16000
    n_mfcc ： 提取的MFCC特征数量，默认13
    Returns:
    str: 返回保存特征的文件夹路径
    """
    if not os.path.exists(input_folder):
        print(f"音频文件夹 {input_folder} 不存在。")
        return

    # 获取输入文件夹的名称（不包含路径部分）
    input_folder_name = os.path.basename(os.path.normpath(input_folder))

    # 创建一个新的文件夹来保存输出的 CSV 文件，名字与输入文件夹相同
    output_dataset_folder = os.path.join(output_folder, input_folder_name)
    os.makedirs(output_dataset_folder, exist_ok=True)

    splits = ["train", "val", "test"]  # 分别处理 train, val, test 文件夹
    for split in splits:
        split_path = os.path.join(input_folder, split)
        if not os.path.exists(split_path):
            print(f"数据集 {split} 文件夹不存在，跳过...")
            continue

        features = []  # 存储所有文件的特征

        # 遍历每个类别
        categories = [cat for cat in os.listdir(split_path) if os.path.isdir(os.path.join(split_path, cat))]
        for category in categories:
            category_path = os.path.join(split_path, category)

            # 获取类别下的所有音频文件
            audio_files = [f for f in os.listdir(category_path) if f.endswith('.wav')]
            if not audio_files:
                print(f"类别 {category} 中未找到任何 .wav 文件，跳过...")
                continue

            for audio_file in audio_files:
                file_path = os.path.join(category_path, audio_file)
                # 假设 two_features_extract 是一个提取特征的函数，返回去除平均值后的特征
                file_features = one_features_extract(file_path, sr=sr, n_mfcc=n_mfcc)  # 假设是去平均处理
                if file_features is not None:
                    features.append([audio_file, category] + file_features)

        # 如果没有提取到任何特征，跳过保存
        if not features:
            print(f"数据集 {split} 未提取到任何特征，跳过保存...")
            continue

        # 保存特征到CSV文件
        columns = ["file_name", "category"] + [f"mfcc_{i + 1}" for i in range(n_mfcc)]
        split_df = pd.DataFrame(features, columns=columns)

        # 生成CSV文件的输出路径，文件名为 train.csv, val.csv, test.csv
        output_file = os.path.join(output_dataset_folder, f"{split}.csv")
        split_df.to_csv(output_file, index=False)
        print(f"数据集 {split} 的特征已保存到: {output_file}")

    # 返回保存特征文件的文件夹路径
    return output_dataset_folder
