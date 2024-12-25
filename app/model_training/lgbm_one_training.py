import os
import pandas as pd
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

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
    X = df.drop(columns=["file_name", "category"])  # 去掉非特征列
    y = df["category"]  # 类别标签
    return X, y

def train_lightgbm(train_file, val_file, test_file, model_output):
    """
    使用LightGBM训练和评估一维特征情感分类模型，并保存模型。
    input:
    train_file ： 训练集CSV文件路径
    val_file ： 验证集CSV文件路径
    test_file ： 测试集CSV文件路径
    model_output ： 保存模型的路径
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

    # 转换为 LightGBM 数据格式
    train_data = lgb.Dataset(X_train_scaled, label=y_train_encoded)
    val_data = lgb.Dataset(X_val_scaled, label=y_val_encoded, reference=train_data)

    # 设置参数
    params = {
        'objective': 'multiclass',  # 多分类任务
        'num_class': len(label_encoder.classes_),  # 类别数量
        'metric': 'multi_logloss',  # 多分类的对数损失
        'boosting_type': 'gbdt',  # 梯度提升
        'learning_rate': 0.05,
        'num_leaves': 31,
        'verbose': -1
    }

    # 训练模型
    print("开始训练 LightGBM 模型...")
    model = lgb.train(params, train_data, valid_sets=[train_data, val_data],
                      num_boost_round=1000)

    # 保存模型
    print(f"保存模型到: {model_output}")
    model.save_model(model_output)

    # 测试模型
    print("测试模型...")
    y_pred = model.predict(X_test_scaled)
    y_pred_labels = [list(row).index(max(row)) for row in y_pred]  # 转换为类别索引
    print("测试集分类结果：")
    print(classification_report(y_test_encoded, y_pred_labels, target_names=label_encoder.classes_))

if __name__ == "__main__":
    # 数据路径
    train_file = "../features/One_dimensional/train_features.csv"
    val_file = "../features/One_dimensional/val_features.csv"
    test_file = "../features/One_dimensional/test_features.csv"
    model_output = "../models/lgbm_model.txt"  # 模型保存路径

    # 训练模型
    train_lightgbm(train_file, val_file, test_file, model_output)
