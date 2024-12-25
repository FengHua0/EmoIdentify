import os
import shutil
import random

def datasetProcessing(dataset_dir, output_dir):
    # 定义划分比例
    train_ratio = 0.85
    val_ratio = 0.10

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(output_dir, split), exist_ok=True)

    # 遍历每个情感类别文件夹
    for category in os.listdir(dataset_dir):
        category_path = os.path.join(dataset_dir, category)
        if not os.path.isdir(category_path):
            continue  # 跳过非文件夹

        # 获取当前类别的所有文件
        files = [f for f in os.listdir(category_path) if os.path.isfile(os.path.join(category_path, f))]
        random.shuffle(files)  # 打乱文件顺序

        # 计算每个划分的数量
        total_files = len(files)
        train_count = int(total_files * train_ratio)
        val_count = int(total_files * val_ratio)

        # 划分数据集
        train_files = files[:train_count]
        val_files = files[train_count:train_count + val_count]
        test_files = files[train_count + val_count:]

        # 将文件复制到对应的文件夹
        for split, split_files in zip(["train", "val", "test"], [train_files, val_files, test_files]):
            split_dir = os.path.join(output_dir, split, category)
            os.makedirs(split_dir, exist_ok=True)
            for file in split_files:
                shutil.copy(os.path.join(category_path, file), os.path.join(split_dir, file))

    print("数据集划分完成")

if __name__ == '__main__':
    dataset_dir = "../../EnglishDataset"  # 原始数据集路径
    # dataset_dir = "../../AudioWAV"  # 原始数据集路径
    output_dir = "../processed_dataset_en"  # 输出目录
    datasetProcessing(dataset_dir, output_dir)