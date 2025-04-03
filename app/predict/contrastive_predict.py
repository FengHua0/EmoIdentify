import torch
import json
import os
import numpy as np
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms
from io import BytesIO
import base64
import matplotlib.pyplot as plt

from app.predict.Base_model import BaseModel
from app.predict.factory_registry import register_model
from app.visible.spectrogram import spectrogram_base64
from app.visible.visual_clustering import visualize_clustering
from app.model_training.contrastive_training import CNN_RNN

@register_model('contrastive')
class contrastive(BaseModel):
    def __init__(self, processed_audio, sr):
        """
        初始化 PyTorch CNN-RNN 预测器
        :param processed_audio: 预处理后的音频数据
        :param sr: 采样率
        """
        super().__init__(processed_audio, sr)
        self.MODEL_PATH = "models/Contrastive_model.pth"  # 模型路径
        self.LABEL_ENCODER_PATH = "models/label_encoder/CREMA-D_CNN_class.json"  # 类别映射路径
        self.SPEAKER_ENCODER_PATH = "models/label_encoder/CREMA-D_CNN_speaker.json"  # 说话人映射路径

        current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.MODEL_PATH = os.path.join(current_dir, self.MODEL_PATH)
        self.LABEL_ENCODER_PATH = os.path.join(current_dir, self.LABEL_ENCODER_PATH)
        self.SPEAKER_ENCODER_PATH = os.path.join(current_dir, self.SPEAKER_ENCODER_PATH)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = None
        self.label_mapping = None  # 存储类别映射
        self.speaker_mapping = None  # 存储说话人映射
        self.image = None  # 保存频谱图
        self.clustering_image = None  # 保存聚类结果

        self.load_model()
        self.load_label_mapping()

    def load_model(self):
        """
        加载训练好的 PyTorch CNN-RNN 模型
        """
        try:
            if not os.path.exists(self.MODEL_PATH):
                raise FileNotFoundError(f"模型文件未找到: {self.MODEL_PATH}")

            # 加载类别映射文件，获取类别数
            num_classes = len(self.label_mapping) if self.label_mapping else 6  # 默认6类
            self.model = CNN_RNN(num_classes=num_classes).to(self.device)
            self.model.load_state_dict(torch.load(self.MODEL_PATH, map_location=self.device))
            self.model.eval()
            print(f"成功加载模型: {self.MODEL_PATH}")
        except Exception as e:
            print(f"加载模型失败: {str(e)}")
            raise

    def load_label_mapping(self):
        """加载情感类别和说话人ID映射"""
        try:
            # 加载情感类别映射
            with open(self.LABEL_ENCODER_PATH, 'r') as f:
                self.label_mapping = json.load(f)
            print(f"成功加载类别映射: {self.LABEL_ENCODER_PATH}")
            
            # 加载说话人ID映射
            with open(self.SPEAKER_ENCODER_PATH, 'r') as f:
                self.speaker_mapping = json.load(f)
            print(f"成功加载说话人映射: {self.SPEAKER_ENCODER_PATH}")
            
        except Exception as e:
            print(f"加载映射失败: {str(e)}")
            self.label_mapping = None
            self.speaker_mapping = None

    def extract_features(self):
        """
        提取 PyTorch 版本的频谱图特征
        Returns:
            torch.Tensor: 预处理后的图片张量
        """
        try:
            # 获取频谱图的 Base64 编码
            _, spectrogram_base64_str = spectrogram_base64(self.processed_audio, self.sr)
            if not spectrogram_base64_str:
                raise ValueError("频谱图提取失败，返回值为空！")

            # 解码 Base64 编码为字节流
            spectrogram_bytes = base64.b64decode(spectrogram_base64_str)
            # 打开字节流为图像
            spectrogram_image = Image.open(BytesIO(spectrogram_bytes)).convert('RGB')

            # 图像预处理
            transform = transforms.Compose([
                transforms.Resize((224,224)),
                transforms.ToTensor(),
                # 三通道图像，使用 ImageNet 上常用的标准化
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
            ])

            img_tensor = transform(spectrogram_image).unsqueeze(0)  # 添加 batch 维度
            return img_tensor.to(self.device)

        except Exception as e:
            print(f"频谱图提取或预处理时出错: {e}")
            return None

    def plot_confidence(self, predictions):
        """
        绘制类别置信度的柱状图，并返回 Base64 编码
        :param predictions: 模型预测的置信度
        :return: str, 图像的 Base64 编码
        """
        try:
            # 获取类别标签
            categories = list(self.label_mapping.values())
            confidences = predictions.cpu().numpy().flatten()

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
        使用训练好的对比学习模型对频谱图进行分类预测
        Returns:
            dict: 预测类别名称和置信度图像的Base64编码
        """
        if self.model is None:
            return {'error': '模型未正确加载。'}
        if self.label_mapping is None:
            return {'error': '类别映射未正确加载。'}

        try:
            # 提取频谱图特征
            img_tensor = self.extract_features()
            if img_tensor is None:
                raise ValueError("特征提取失败！")

            # 模型推理
            with torch.no_grad():
                outputs, _ = self.model(img_tensor)  # 获取模型输出
                predictions = F.softmax(outputs, dim=1)  # 计算softmax概率
                predicted_index = torch.argmax(predictions, dim=1).item()  # 获取预测类别索引

            # 确保predicted_index为字符串类型以匹配JSON映射
            predicted_emotion = self.label_mapping.get(str(predicted_index), "Unknown")

            # 绘制置信度图
            confidence_base64 = self.plot_confidence(predictions)

            confidence = {
                'feature_name': 'Confidence',  # 数据的名称
                'base64': confidence_base64  # 图像的Base64编码
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

    def get_spectrogram(self):
        """获取频谱图的base64编码"""
        if self.image is None:
            return None
            
        buf = BytesIO()
        self.image.save(buf, format='PNG')
        buf.seek(0)
        return base64.b64encode(buf.read()).decode('utf-8')