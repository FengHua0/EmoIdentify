import os
import shutil
import random

def datasetProcessing(dataset_dir, output_dir):
    """
    划分数据集为训练集、验证集和测试集，并将它们复制到相应的文件夹中。
    输出文件夹名基于输入文件夹的名称，去掉 'processed_' 前缀。

    input:
    dataset_dir ： 输入数据集文件夹路径，包含多个类别的子文件夹
    output_dir ： 输出数据集文件夹路径，划分后的数据将保存到此

    Returns:
    str： 返回数据集的根文件夹路径（例如 '/CREMA-D'）
    """
    # 获取输入文件夹的名称，去掉 'processed_' 前缀
    dataset_name = os.path.basename(os.path.normpath(dataset_dir))
    if dataset_name.startswith('processed_'):
        dataset_name = dataset_name[len('processed_'):]  # 去掉 'processed_' 前缀

    # 定义划分比例
    train_ratio = 0.85
    val_ratio = 0.10

    # 确保输出目录存在
    output_dataset_dir = os.path.join(output_dir, dataset_name)
    os.makedirs(output_dataset_dir, exist_ok=True)

    # 创建训练集、验证集和测试集的文件夹
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(output_dataset_dir, split), exist_ok=True)

    # 遍历每个类别文件夹，进行数据划分
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
            split_dir = os.path.join(output_dataset_dir, split, category)
            os.makedirs(split_dir, exist_ok=True)
            for file in split_files:
                shutil.copy(os.path.join(category_path, file), os.path.join(split_dir, file))

    print("数据集划分完成")

    # 返回划分后的数据集根文件夹路径
    return output_dataset_dir

