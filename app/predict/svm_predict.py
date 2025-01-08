import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
import matplotlib.pyplot as plt
from io import BytesIO
import base64

from app.file_process.feature_extraction_1 import one_features_extract
from app.predict.Base_model import BaseModel
from app.predict.model_factory import register_model


@register_model('svm')
class SVM(BaseModel):
    def __init__(self, processed_audio, sr):
        super().__init__(processed_audio, sr)
        self.MODEL_PATH = 'models/svm_model.joblib'
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.load_model()

    def load_model(self):
        """
        加载训练好的 SVM 模型、标准化器和标签编码器
        """
        try:
            # 加载模型数据
            model_data = joblib.load(self.MODEL_PATH)
            self.model = model_data['model']  # 加载 SVM 模型
            self.scaler = model_data['scaler']  # 加载标准化器
            self.label_encoder = model_data['label_encoder']  # 加载标签编码器
            print("SVM 模型、标准化器和标签编码器加载成功。")
        except Exception as e:
            print(f"加载 SVM 模型时出错: {e}")
            self.model = None
            self.scaler = None
            self.label_encoder = None

    def extract_features(self):
        """
        提取特征的方法
        """
        try:
            self.features = one_features_extract(self.processed_audio, self.sr)
            if not self.features:
                raise ValueError("特征提取失败。")
            return self.features
        except Exception as e:
            print(f"特征提取时出错: {e}")
            return None

    def plot_confidence(self, y_pred_probs):
        """
        绘制类别置信度的柱状图，并返回其 Base64 编码。
        :param y_pred_probs: SVM 模型输出的每个类别的置信度
        :return: str, 图像的 Base64 编码
        """
        try:
            # 获取类别标签
            categories = self.label_encoder.classes_

            # 获取每个类别的置信度（对所有样本的预测都处理）
            confidences = y_pred_probs[0]  # 这里只取第一个样本的预测置信度

            # 绘制柱状图
            plt.figure(figsize=(10, 6))
            plt.bar(categories, confidences, color='skyblue')
            plt.ylabel('Confidence')
            plt.title('Prediction Confidence for Each Emotion')

            # 保存图像到字节流
            img_stream = BytesIO()
            plt.savefig(img_stream, format='png')
            plt.close()
            img_stream.seek(0)

            # 转换为 Base64 编码
            img_base64 = base64.b64encode(img_stream.read()).decode('utf-8')
            return img_base64

        except Exception as e:
            print(f"绘制置信度图时出错: {e}")
            return None

    def predict(self):
        """
        使用训练好的 SVM 模型进行预测，并反编码为情感名称。
        Returns:
            dict: 包含情感名称和置信度图像的结果
        """
        if self.model is None or self.scaler is None or self.label_encoder is None:
            return {'error': '模型、标准化器或标签编码器未正确加载。'}

        try:
            self.features = self.extract_features()
            # 将单行特征转换为 DataFrame
            if isinstance(self.features, (list, tuple)):
                features_df = pd.DataFrame([self.features])
            else:
                raise ValueError("特征数据必须是列表或数组！")

            # 忽略警告
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="X does not have valid feature names")
                X_scaled = self.scaler.transform(features_df)  # 标准化特征

            # 进行预测
            y_pred_encoded = self.model.predict(X_scaled)
            y_pred_probs = self.model.predict_proba(X_scaled)  # 获取所有类别的概率输出
            y_pred = self.label_encoder.inverse_transform(y_pred_encoded)

            # 绘制置信度图并返回图像的 Base64 编码
            confidence_base64 = self.plot_confidence(y_pred_probs)

            confidence = {
                'feature_name': 'Confidence',  # 数据的名称
                'base64': confidence_base64  # 图像的 Base64 编码
            }

            # 返回预测结果和置信度图
            result = {
                'predicted_category': y_pred[0],  # 返回情感名称
                'confidence': confidence  # 返回置信度图的 Base64 编码
            }
            return result

        except Exception as e:
            print(f"预测时出错: {e}")
            return {'error': str(e)}
