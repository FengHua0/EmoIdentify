import os
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report
import joblib  # 用于保存和加载模型

def load_features(csv_file):
    """
    加载CSV文件中的特征和标签。
    input:
    csv_file ： CSV文件路径
    Returns:
    X ： 特征矩阵
    y ： 类别标签
    """
    print(f"加载特征文件: {csv_file}")
    df = pd.read_csv(csv_file)
    y = df["category"]  # 类别标签
    X = df.drop(columns=["file_name", "category"])
    return X, y

def train_svm(train_file, val_file, test_file, model_output):
    """
    训练SVM分类器并保存模型。
    input:
    train_file ： 训练集CSV文件路径
    val_file ： 验证集CSV文件路径
    test_file ： 测试集CSV文件路径
    model_output ： 保存模型的路径
    Returns:
    None
    """
    # 加载数据
    X_train, y_train = load_features(train_file)
    X_val, y_val = load_features(val_file)
    X_test, y_test = load_features(test_file)

    # 标签编码
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_val_encoded = label_encoder.transform(y_val)
    y_test_encoded = label_encoder.transform(y_test)

    # 数据标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # 训练SVM模型
    print("开始训练SVM模型...")
    svm_model = SVC(kernel='linear', random_state=42, probability=True)
    svm_model.fit(X_train_scaled, y_train_encoded)

    # 保存模型和标签编码器
    print(f"保存模型到: {model_output}")
    joblib.dump({"model": svm_model, "scaler": scaler, "label_encoder": label_encoder}, model_output)

    # 在验证集上进行评估
    print("验证集评估结果：")
    y_val_pred = svm_model.predict(X_val_scaled)
    print(classification_report(y_val_encoded, y_val_pred, target_names=label_encoder.classes_))

    # 在测试集上进行评估
    print("测试集评估结果：")
    y_test_pred = svm_model.predict(X_test_scaled)
    print(classification_report(y_test_encoded, y_test_pred, target_names=label_encoder.classes_))

# 主程序
if __name__ == "__main__":
    train_file = "../features/feature_extraction_1/CREMA-D/train.csv"  # 训练集特征文件路径
    val_file = "../features/feature_extraction_1/CREMA-D/val.csv"  # 验证集特征文件路径
    test_file = "../features/feature_extraction_1/CREMA-D/test.csv"  # 测试集特征文件路径
    model_output = "../models/svm_model.joblib"  # 模型保存路径

    # 创建保存模型的文件夹
    os.makedirs(os.path.dirname(model_output), exist_ok=True)

    # 训练模型并保存
    train_svm(train_file, val_file, test_file, model_output)
