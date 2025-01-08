import torch
from torch.nn.utils.rnn import pad_sequence
from app.file_process.feature_extraction_2 import two_features_extract
from app.model_training.RNN_two_training import EmotionClassifier
import joblib
from app.predict.Base_model import BaseModel
from app.predict.model_factory import register_model
import matplotlib.pyplot as plt
from io import BytesIO
import base64

@register_model('rnn')
class RNN(BaseModel):
    def __init__(self, processed_audio, sr):
        super().__init__(processed_audio, sr)
        self.MODEL_PATH = "models/rnn_2.pth"
        self.ENCODER_PATH = "models/label_encoder/CREMA-D_label_encoder.joblib"
        self.model = None
        self.label_encoder = None
        self.load_model()

    def load_model(self):
        """
        加载训练好的模型和标签编码器
        """
        try:
            # 加载 LabelEncoder
            self.label_encoder = joblib.load(self.ENCODER_PATH)

            # 实例化模型
            self.model = EmotionClassifier(input_size=13, hidden_size=128, num_classes=len(self.label_encoder.classes_))
            self.model.load_state_dict(torch.load(self.MODEL_PATH, map_location=torch.device('cpu')))
            self.model.eval()  # 设置模型为评估模式
            print("RNN 模型和编码器加载成功。")
        except Exception as e:
            print(f"加载模型或编码器时出错: {e}")
            self.model = None
            self.label_encoder = None

    def extract_features(self):
        """
        提取特征的方法
        """
        try:
            self.features = two_features_extract(self.processed_audio, self.sr)
            if not self.features:
                raise ValueError("特征提取失败。")
            return self.features
        except Exception as e:
            print(f"特征提取时出错: {e}")
            return None

    def plot_confidence(self, outputs):
        """
        绘制类别置信度的柱状图，并返回其 Base64 编码。
        :param outputs: 模型输出的置信度分数
        :return: str, 图像的 Base64 编码
        """
        try:
            # 获取类别标签
            categories = self.label_encoder.classes_
            confidences = torch.nn.functional.softmax(outputs, dim=1).cpu().detach().numpy()[0]  # 获取置信度

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
        使用训练好的 RNN 模型进行预测，并反编码为情感名称。
        Returns:
            dict: 包含情感名称和置信度图像的结果
        """
        if self.model is None or self.label_encoder is None:
            return {'error': '模型或编码器未正确加载。'}

        try:
            features = self.extract_features()
            # 确保特征是二维列表或数组
            if not isinstance(features, (list, tuple)) or not isinstance(features[0], (list, tuple)):
                raise ValueError("输入数据必须是一个二维列表或数组！")

            # 将每行特征转换为 PyTorch 张量
            features_tensor = [torch.tensor(f, dtype=torch.float32) for f in features]

            # 获取序列长度（只有一个样本，因此长度是一个标量）
            lengths = torch.tensor([len(features_tensor)], dtype=torch.long)

            # 使用 pad_sequence 填充变长序列 (batch_size = 1)
            padded_features = pad_sequence(features_tensor, batch_first=True).unsqueeze(0)  # 增加 batch 维度

            # 模型预测
            with torch.no_grad():
                outputs = self.model(padded_features, lengths)  # 调用模型
                y_pred = outputs.argmax(dim=1).cpu().numpy()  # 获取预测类别（数字）

            # 将数字类别转换为情感名称
            predicted_emotion = self.label_encoder.inverse_transform([y_pred[0]])[0]  # 转换为情感名称

            # 绘制置信度图并返回图像的 Base64 编码
            confidence_base64 = self.plot_confidence(outputs)

            confidence = {
                'feature_name': 'Confidence',  # 数据的名称
                'base64': confidence_base64  # 图像的 Base64 编码
            }

            # 返回预测结果和置信度图
            result = {
                'predicted_category': predicted_emotion,  # 返回情感名称
                'confidence': confidence  # 返回置信度图
            }
            return result

        except Exception as e:
            print(f"预测时出错: {e}")
            return {'error': str(e)}
