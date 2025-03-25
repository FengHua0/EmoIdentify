import matplotlib.pyplot as plt
import re

def parse_training_log(log_file):
    """
    解析训练日志文件，提取训练损失、准确率和验证损失、准确率数据。

    :param log_file: str, 训练日志文件路径
    :return: list, 解析出的 epochs, train_losses, train_accs, val_losses, val_accs
    """
    epochs = []
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []

    with open(log_file, "r") as f:
        epoch = 1  # 逐行累积
        for line in f:
            match = re.search(r"Train Loss=([\d.]+), Train Acc=([\d.]+) \| Val Loss=([\d.]+), Val Acc=([\d.]+)", line)
            if match:
                train_loss = float(match.group(1))
                train_acc = float(match.group(2))
                val_loss = float(match.group(3))
                val_acc = float(match.group(4))

                epochs.append(epoch)
                train_losses.append(train_loss)
                train_accs.append(train_acc)
                val_losses.append(val_loss)
                val_accs.append(val_acc)

                epoch += 1  # 递增 epoch

    if not epochs:
        raise ValueError("日志文件可能为空，或者格式错误，未能解析出训练数据！")

    return epochs, train_losses, train_accs, val_losses, val_accs


def plot_training_log(log_file):
    """
    读取训练日志并绘制损失和准确率曲线。

    :param log_file: str, 训练日志文件路径
    """
    epochs, train_losses, train_accs, val_losses, val_accs = parse_training_log(log_file)

    plt.figure(figsize=(12, 5))

    # 绘制损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label="Train Loss", color="blue", marker="o")
    plt.plot(epochs, val_losses, label="Val Loss", color="red", marker="s")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Train & Validation Loss")
    plt.legend()
    plt.grid()

    # 绘制准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accs, label="Train Accuracy", color="green", marker="o")
    plt.plot(epochs, val_accs, label="Val Accuracy", color="orange", marker="s")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Train & Validation Accuracy")
    plt.legend()
    plt.grid()

    # 显示图像
    plt.show()


if __name__ == "__main__":
    log_file = "model_visible/rnn.txt"  # 指定训练日志路径
    plot_training_log(log_file)
