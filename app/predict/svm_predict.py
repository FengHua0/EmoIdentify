import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings

def svm_predict(features):
    """
    接收单行特征数组进行预测
    input:
    features: 单行特征数据 (列表或数组)
    returns:
    dict: 包含预测类别的结果
    """
    # 配置路径
    MODEL_PATH = 'models/svm_model.joblib'

    # 加载模型
    model_data = joblib.load(MODEL_PATH)
    model = model_data['model']  # 加载 SVM 模型
    scaler = model_data['scaler']  # 加载标准化器
    label_encoder = model_data['label_encoder']  # 加载标签编码器



    try:
        # 如果输入是单行数组或列表，转换为 DataFrame
        if isinstance(features, (list, tuple)):
            features_df = pd.DataFrame([features])  # 将单行特征转为 DataFrame
        else:
            raise ValueError("输入数据必须是列表或数组！")

        # 忽略警告
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="X does not have valid feature names")
            X_scaled = scaler.transform([features])  # 输入直接是 NumPy 数组

        # 进行预测
        y_pred_encoded = model.predict(X_scaled)
        y_pred = label_encoder.inverse_transform(y_pred_encoded)

        # 返回预测结果（不需要文件名）
        result = {'predicted_category': y_pred[0]}  # 返回预测类别
        return result

    except Exception as e:
        print(f"预测时出错: {e}")
        return None
