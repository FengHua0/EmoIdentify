import json
import os
import numpy as np
import torch
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from app.predict.model_factory import register_model
from app.visible.spectrogram import spectrogram_base64
from app.model_training.cnn_spectrogram import build_cnn_model
from app.predict.Base_model import BaseModel
import base64
import io
import matplotlib.pyplot as plt
from io import BytesIO

@register_model('cnn')
class CNN(BaseModel):
    def __init__(self, processed_audio, sr):
        """
        初始化 CNN 模型。
        :param processed_audio: 预处理后的音频数据
        :param sr: 采样率
        """
        super().__init__(processed_audio, sr)
        self.MODEL_PATH = "models/cnn_spectrogram_model.h5"  # 模型路径
        self.LABEL_ENCODER_PATH = "models/label_encoder/CREMA-D_CNN.json"  # 类别映射文件路径
        self.model = None
        self.label_mapping = None  # 存储类别映射
        self.image = None  # 保存频谱图
        self.load_model()
        self.load_label_mapping()

    def load_model(self):
        """
        加载训练好的 CNN 模型。
        """
        try:
            if not os.path.exists(self.MODEL_PATH):
                raise FileNotFoundError(f"模型文件未找到: {self.MODEL_PATH}")
            self.model = load_model(self.MODEL_PATH)
            print("CNN 模型加载成功！")
        except Exception as e:
            print(f"加载模型时出错: {e}")
            self.model = None

    def load_label_mapping(self):
        """
        加载类别映射文件。
        """
        try:
            if not os.path.exists(self.LABEL_ENCODER_PATH):
                raise FileNotFoundError(f"类别映射文件未找到: {self.LABEL_ENCODER_PATH}")
            with open(self.LABEL_ENCODER_PATH, "r") as f:
                self.label_mapping = json.load(f)
        except Exception as e:
            print(f"加载类别映射时出错: {e}")
            self.label_mapping = None

    def extract_features(self):
        """
        提取频谱图特征。
        Returns:
            np.ndarray: 预处理后的图片数组，用于模型输入
        """
        try:
            # 获取频谱图的 Base64 编码
            _, spectrogram_base64_str = spectrogram_base64(self.processed_audio, self.sr)
            if not spectrogram_base64_str:
                raise ValueError("频谱图提取失败，返回值为空！")

            # 解码 Base64 编码为字节流
            spectrogram_bytes = base64.b64decode(spectrogram_base64_str)
            # 打开字节流为图像
            spectrogram_image = Image.open(io.BytesIO(spectrogram_bytes))

            # 检查图像的模式
            if spectrogram_image.mode != 'RGB':  # 如果图像不是 RGB 模式，则转换
                spectrogram_image = spectrogram_image.convert('RGB')

            # 调整图像大小
            img_size = (224, 224)
            spectrogram_image = spectrogram_image.resize(img_size)

            # 转换为 NumPy 数组
            img_array = img_to_array(spectrogram_image)

            # 增加批次维度
            img_array = np.expand_dims(img_array, axis=0)
            # 归一化到 [0, 1]
            img_array = img_array / 255.0

            return img_array

        except Exception as e:
            print(f"频谱图提取或预处理时出错: {e}")
            return None

    def plot_confidence(self, predictions):
        """
        绘制类别置信度的柱状图，并返回其 Base64 编码。
        :param predictions: 模型预测的置信度
        :return: str, 图像的 Base64 编码
        """
        try:
            # 获取类别标签
            categories = list(self.label_mapping.values())
            confidences = predictions[0]

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
        使用训练好的 CNN 模型对频谱图进行分类预测。
        Returns:
            dict: 包含预测类别名称和置信度图像的 Base64 编码的结果
        """
        if self.model is None:
            return {'error': '模型未正确加载。'}
        if self.label_mapping is None:
            return {'error': '类别映射未正确加载。'}

        try:
            # 提取频谱图特征
            img_array = self.extract_features()
            if img_array is None:
                raise ValueError("特征提取失败！")

            # 模型预测
            predictions = self.model.predict(img_array)
            predicted_index = np.argmax(predictions, axis=1)[0]  # 获取概率最大的类别索引

            # 解码类别索引为标签
            predicted_emotion = self.label_mapping[str(predicted_index)]

            # 绘制置信度图并返回图像的 Base64 编码
            confidence_base64 = self.plot_confidence(predictions)

            confidence = {
                'feature_name': 'Confidence',  # 数据的名称
                'base64': confidence_base64  # 图像的 Base64 编码
            }

            # 返回预测结果和置信度图
            result = {
                'predicted_category': predicted_emotion,  # 返回预测情感类别名称
                'confidence': confidence  # 返回置信度图
            }
            return result

        except Exception as e:
            print(f"预测时出错: {e}")
            return {'error': str(e)}
