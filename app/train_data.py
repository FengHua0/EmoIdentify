import matplotlib.pyplot as plt
import re

# 训练日志文件路径
log_file = "model_visible/cnn_rnn.txt"

# 初始化列表
epochs = []
train_losses, train_accs = [], []
val_losses, val_accs = [], []

# 读取日志文件并解析数据
with open(log_file, "r") as f:
    for line in f:
        # 使用正则表达式提取数值
        match = re.search(r"Epoch (\d+): Train Loss=([\d.]+), Train Acc=([\d.]+) \| Val Loss=([\d.]+), Val Acc=([\d.]+)", line)
        if match:
            epoch = int(match.group(1))
            train_loss = float(match.group(2))
            train_acc = float(match.group(3))
            val_loss = float(match.group(4))
            val_acc = float(match.group(5))

            # 追加到列表
            epochs.append(epoch)
            train_losses.append(train_loss)
            train_accs.append(train_acc)
            val_losses.append(val_loss)
            val_accs.append(val_acc)

# 检查是否正确读取
if not epochs:
    raise ValueError("日志文件可能为空，或者格式错误，未能解析出训练数据！")

# 创建图像
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
