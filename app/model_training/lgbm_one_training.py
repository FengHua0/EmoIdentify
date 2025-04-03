import os
import pandas as pd
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import joblib

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

class LGBM_Model:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.label_encoder = None

    def load(self, model_path):
        """加载保存的模型"""
        print(f"加载LGBM模型: {model_path}")
        saved_data = joblib.load(model_path)
        self.model = saved_data["model"]
        self.scaler = saved_data["scaler"]
        self.label_encoder = saved_data["label_encoder"]

    def extract_features(self, data_folder):
        """
        从数据文件夹提取特征
        Args:
            data_folder: 包含train.csv, val.csv, test.csv的文件夹
        Returns:
            features, labels, speaker_ids
        """
        # 加载所有数据
        train_file = os.path.join(data_folder, "train.csv")
        val_file = os.path.join(data_folder, "val.csv")
        test_file = os.path.join(data_folder, "test.csv")
        
        X_train, y_train = load_features(train_file)
        X_val, y_val = load_features(val_file)
        X_test, y_test = load_features(test_file)
        
        # 合并所有数据
        X = pd.concat([X_train, X_val, X_test])
        y = pd.concat([y_train, y_val, y_test])
        
        # 标准化特征
        X_scaled = self.scaler.transform(X)
        
        # 生成伪speaker_ids (LGBM模型不需要真实的speaker_ids)
        speaker_ids = np.zeros(len(y))
        
        return X_scaled, self.label_encoder.transform(y), speaker_ids

    def preprocess(self, batch):
        """
        预处理数据用于特征提取
        Args:
            batch: 数据批次 (image, label, speaker_id)
        Returns:
            features, labels, speaker_ids
        """
        # 对于LGBM模型，我们直接从CSV文件加载数据，所以这个方法不会被调用
        raise NotImplementedError("LGBM模型应该直接从CSV文件加载数据")

def train_lightgbm(train_file, val_file, test_file, model_output):
    """训练LGBM分类器并保存模型"""
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

    # 创建并返回LGBM_Model实例
    lgbm_model = LGBM_Model()
    lgbm_model.model = model
    lgbm_model.scaler = scaler
    lgbm_model.label_encoder = label_encoder
    
    # 保存模型对象
    joblib.dump({
        "model": lgbm_model.model,
        "scaler": lgbm_model.scaler,
        "label_encoder": lgbm_model.label_encoder
    }, model_output)
    
    return lgbm_model

# 修改主程序
if __name__ == "__main__":
    train_file = "../features/One_dimensional/train_features.csv"
    val_file = "../features/One_dimensional/val_features.csv"
    test_file = "../features/One_dimensional/test_features.csv"
    model_output = "../models/lgbm_model.joblib"  # 修改为joblib格式

    current_dir = os.path.dirname(os.path.abspath(__file__))
    train_file = os.path.join(current_dir, train_file)
    val_file = os.path.join(current_dir, val_file)
    test_file = os.path.join(current_dir, test_file)
    model_output = os.path.join(current_dir, model_output)

    os.makedirs(os.path.dirname(model_output), exist_ok=True)
    
    # 训练并保存模型
    trained_model = train_lightgbm(train_file, val_file, test_file, model_output)
