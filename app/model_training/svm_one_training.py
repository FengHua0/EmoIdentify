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

class SVM_Model:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.label_encoder = None

    def load(self, model_path):
        """加载保存的模型"""
        print(f"加载SVM模型: {model_path}")
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
        
        # 生成伪speaker_ids (SVM模型不需要真实的speaker_ids)
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
        # 对于SVM模型，我们直接从CSV文件加载数据，所以这个方法不会被调用
        raise NotImplementedError("SVM模型应该直接从CSV文件加载数据")

def train_svm(train_file, val_file, test_file, model_output):
    """训练SVM分类器并保存模型"""
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

# 修改主程序
if __name__ == "__main__":
    train_file = "../features/feature_extraction_1/CREMA-D/train.csv"
    val_file = "../features/feature_extraction_1/CREMA-D/val.csv"
    test_file = "../features/feature_extraction_1/CREMA-D/test.csv"
    model_output = "../models/svm_model.joblib"

    current_dir = os.path.dirname(os.path.abspath(__file__))
    train_file = os.path.join(current_dir, train_file)
    val_file = os.path.join(current_dir, val_file)
    test_file = os.path.join(current_dir, test_file)
    model_output = os.path.join(current_dir, model_output)

    os.makedirs(os.path.dirname(model_output), exist_ok=True)
    
    # 训练并保存模型
    trained_model = train_svm(train_file, val_file, test_file, model_output)
    
    # 保存模型对象
    joblib.dump({
        "model": trained_model.model,
        "scaler": trained_model.scaler,
        "label_encoder": trained_model.label_encoder
    }, model_output)
