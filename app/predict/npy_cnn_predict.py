import torch
import json
import os
import numpy as np
import torch.nn.functional as F
from torchvision import transforms
import matplotlib.pyplot as plt
from io import BytesIO
import base64

from app.predict.Base_model import BaseModel
from app.predict.factory_registry import register_model
from app.model_training.npy_cnn_training import CNN_RNN
from app.file_process.create_npy import audio_bytes_to_npy

class SingleToThreeChannels:
    """将单通道图像复制为三通道"""
    def __call__(self, x):
        return x.repeat(3, 1, 1) if x.size(0) == 1 else x

@register_model('npy_cnn')
class NpyCNN(BaseModel):
    def __init__(self, processed_audio, sr):
        """
        初始化基于.npy输入的CNN-RNN预测器
        :param processed_audio: 预处理后的音频数据(未使用，直接加载.npy)
        :param sr: 采样率(未使用)
        """
        super().__init__(processed_audio, sr)
        self.MODEL_PATH = "models/npy_cnn_model.pth"
        self.LABEL_ENCODER_PATH = "models/label_encoder/npy_CREMA-D_CNN_class.json"

        current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.MODEL_PATH = os.path.join(current_dir, self.MODEL_PATH)
        self.LABEL_ENCODER_PATH = os.path.join(current_dir, self.LABEL_ENCODER_PATH)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.label_mapping = None
        self.transform = transforms.Compose([
            SingleToThreeChannels(),
            transforms.Resize((224, 224)),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        self.load_model()
        self.load_label_mapping()

    def load_model(self):
        """加载训练好的模型"""
        try:
            if not os.path.exists(self.MODEL_PATH):
                raise FileNotFoundError(f"模型文件未找到: {self.MODEL_PATH}")

            num_classes = len(self.label_mapping) if self.label_mapping else 6
            self.model = CNN_RNN(num_classes=num_classes).to(self.device)
            self.model.load_state_dict(torch.load(self.MODEL_PATH, map_location=self.device))
            self.model.eval()
            print("Npy CNN-RNN 模型加载成功！")
        except Exception as e:
            print(f"加载模型时出错: {e}")
            self.model = None

    def load_label_mapping(self):
        """加载类别映射"""
        try:
            if not os.path.exists(self.LABEL_ENCODER_PATH):
                raise FileNotFoundError(f"类别映射文件未找到: {self.LABEL_ENCODER_PATH}")
            with open(self.LABEL_ENCODER_PATH, "r") as f:
                self.label_mapping = json.load(f)
            print("类别映射加载成功！")
        except Exception as e:
            print(f"加载类别映射时出错: {e}")
            self.label_mapping = None

    def plot_confidence(self, predictions):
        """绘制置信度图并返回base64"""
        try:
            categories = list(self.label_mapping.values())
            confidences = predictions.cpu().numpy().flatten()

            plt.figure(figsize=(10, 6))
            plt.bar(categories, confidences, color='skyblue')
            plt.ylabel('Confidence')
            plt.title('Prediction Confidence for Each Emotion')

            img_stream = BytesIO()
            plt.savefig(img_stream, format='png')
            plt.close()
            img_stream.seek(0)
            return base64.b64encode(img_stream.read()).decode('utf-8')
        except Exception as e:
            print(f"绘制置信度图时出错: {e}")
            return None

    def extract_features(self, input_data):
        """
        实现基类要求的特征提取方法
        返回: 处理后的特征张量或None
        """
        try:
            # 使用audio_bytes_to_npy处理输入数据
            mel_db = audio_bytes_to_npy(input_data)
            if mel_db is None:
                raise ValueError("无法从输入数据提取梅尔频谱特征")
                
            # 转换为PyTorch张量并添加必要的维度
            features = torch.from_numpy(mel_db).float().unsqueeze(0)
            return self.transform(features).unsqueeze(0).to(self.device)
            
        except Exception as e:
            print(f"处理输入数据时出错: {e}")
            return None

    def predict(self):
        """
        使用训练好的CNN-RNN模型对.npy特征进行分类预测
        Returns:
            dict: 预测类别名称和置信度图像的Base64编码
        """
        if self.model is None:
            return {'error': '模型未正确加载。'}
        if self.label_mapping is None:
            return {'error': '类别映射未正确加载。'}

        try:
            # 检查是否已处理音频数据
            if not hasattr(self, 'processed_audio') or self.processed_audio is None:
                raise ValueError("未提供音频数据")

            # 提取特征
            features = self.extract_features(self.processed_audio)
            if features is None:
                raise ValueError("特征提取失败")

            # 模型推理
            with torch.no_grad():
                outputs, _ = self.model(features)
                predictions = F.softmax(outputs, dim=1)
                predicted_index = torch.argmax(predictions, dim=1).item()

            # 获取预测类别
            predicted_emotion = self.label_mapping.get(str(predicted_index), "Unknown")

            # 绘制置信度图
            confidence_base64 = self.plot_confidence(predictions)

            # 构建返回结果
            result = {
                'predicted_category': predicted_emotion,
                'confidence': {
                    'feature_name': 'Confidence',
                    'base64': confidence_base64
                } if confidence_base64 else None
            }
            return result

        except Exception as e:
            print(f"预测时出错: {e}")
            return {'error': str(e)}