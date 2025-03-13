#TODO 参数存疑，需要调整
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report
from torch.utils.data import Dataset, DataLoader
from scipy.stats import mode
from app.model_training.cnn_rnn_spectrogram import log_results

# 自定义帧级数据集
class EmotionDataset(Dataset):
    def __init__(self, features, labels, file_names):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.file_names = file_names  # 新增文件名列表

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx], self.file_names[idx]


# 定义帧级 MLP 模型
class EmotionClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(EmotionClassifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.fc(x)

# 数据加载函数
def load_features(csv_file):
    """
    加载CSV文件，按帧返回特征和对应的类别标签
    input:
    csv_file ： CSV文件路径
    Returns:
    features ： 帧级特征矩阵 (samples, features_per_frame)
    labels ： 帧级类别标签
    file_names ： 帧对应的文件名
    """
    print(f"加载特征文件: {csv_file}")
    df = pd.read_csv(csv_file)
    features = df.drop(columns=["file_name", "category"]).values
    labels = df["category"].values
    file_names = df["file_name"].values
    return features, labels, file_names

# 多数投票函数
def majority_vote(file_names, y_pred):
    """
    根据帧的预测结果进行多数投票，确定每个文件的最终类别。
    input:
    file_names ： 文件名列表（与每帧对应）
    y_pred ： 帧的预测类别
    Returns:
    file_results ： 每个文件的最终预测类别
    """
    results = pd.DataFrame({"file_name": file_names, "pred_label": y_pred})
    # 对每个文件进行多数投票
    file_results = results.groupby("file_name")["pred_label"].apply(lambda x: mode(x).mode[0])
    return file_results.reset_index()

# 模型训练函数
def train_model(model, criterion, optimizer, train_loader, val_loader, device, model_output, epochs=20):
    model.to(device)
    best_val_acc = 0.0
    for epoch in range(epochs):
        model.train()
        train_loss, correct = 0.0, 0
        for features, labels, _ in train_loader:  # 解包 train_loader 的返回值
            features, labels = features.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()

        train_acc = correct / len(train_loader.dataset)

        # 验证集评估
        model.eval()
        val_loss, val_correct = 0.0, 0
        with torch.no_grad():
            for features, labels, _ in val_loader:  # 解包 val_loader 的返回值
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                val_correct += (outputs.argmax(1) == labels).sum().item()

        val_acc = val_correct / len(val_loader.dataset)

        # 保存最优模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            print(f"保存最优模型 (Epoch {epoch + 1}) 到: {model_output}")
            torch.save(model.state_dict(), model_output)

        print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # 记录日志
        log_file = "../model_visible/mlp_two.txt"
        log_results(log_file, train_loss, train_acc, val_loss, val_acc)


# 测试模型函数
def test_model(model, test_loader, device, label_encoder, model_path):
    # 加载保存的模型
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()

    y_true, y_pred, file_name_list = [], [], []
    with torch.no_grad():
        for features, labels, batch_file_names in test_loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            predictions = outputs.argmax(1).cpu().numpy()

            y_pred.extend(predictions)
            y_true.extend(labels.cpu().numpy())
            file_name_list.extend(batch_file_names)

    # 多数投票
    file_results = majority_vote(file_name_list, label_encoder.inverse_transform(y_pred))
    print("文件级别分类结果：")
    print(file_results)

    # 帧级分类报告
    print("帧级分类结果：")
    print(classification_report(y_true, y_pred, target_names=label_encoder.classes_))


# 主程序
if __name__ == "__main__":
    # 数据路径
    train_file = "../features/Two_dimensional/train_features.csv"
    val_file = "../features/Two_dimensional/val_features.csv"
    test_file = "../features/Two_dimensional/test_features.csv"
    model_output = "../models/mlp_two.pth"  # 保存模型路径

    # 加载数据
    X_train, y_train, train_files = load_features(train_file)
    X_val, y_val, val_files = load_features(val_file)
    X_test, y_test, test_files = load_features(test_file)

    # 数据标准化
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train)
    y_val = label_encoder.transform(y_val)
    y_test = label_encoder.transform(y_test)

    # 数据加载
    train_dataset = EmotionDataset(X_train, y_train, train_files)
    val_dataset = EmotionDataset(X_val, y_val, val_files)
    test_dataset = EmotionDataset(X_test, y_test, test_files)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # 模型定义
    input_size = X_train.shape[1]
    num_classes = len(label_encoder.classes_)
    model = EmotionClassifier(input_size, num_classes)

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_model(model, criterion, optimizer, train_loader, val_loader, device, model_output, epochs=20)

    # 测试模型
    test_model(model, test_loader, device, label_encoder, test_files, model_output)
