import torch
import json
import os
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from io import BytesIO
import base64

from app.predict.Base_model import BaseModel
from app.predict.factory_registry import register_model
# 确保导入的是训练时使用的 CNN_RNN 模型定义
from app.model_training.npy_contrastive_training import CNN_RNN
from app.file_process.create_npy import audio_bytes_to_npy


@register_model('npy_contrastive')
class NpyContrastivePredict(BaseModel):
    def __init__(self, processed_audio, sr, dataset="CASIA"):
        """
        初始化基于对比学习的.npy预测器
        :param processed_audio: 预处理后的音频数据
        :param sr: 采样率(未使用)
        :param dataset: 数据集名称，默认为CASIA
        """
        super().__init__(processed_audio, sr)
        self.dataset = dataset
        self.MODEL_PATH = f"models/{self.dataset}_npy_contrastive_model.pth"  # 修改：使用self.dataset
        self.LABEL_ENCODER_PATH = f"models/label_encoder/{self.dataset}_CNN_class.json"
        self.SPEAKER_ENCODER_PATH = f"models/label_encoder/{self.dataset}_CNN_speaker.json"

        current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.MODEL_PATH = os.path.join(current_dir, self.MODEL_PATH)
        self.LABEL_ENCODER_PATH = os.path.join(current_dir, self.LABEL_ENCODER_PATH)
        self.SPEAKER_ENCODER_PATH = os.path.join(current_dir, self.SPEAKER_ENCODER_PATH)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.label_mapping = None
        self.speaker_mapping = None

        self.load_label_mapping()
        self.load_model() # 确保加载的模型与训练时定义一致

    def load_model(self):
        """加载训练好的对比学习模型"""
        try:
            if not os.path.exists(self.MODEL_PATH):
                raise FileNotFoundError(f"模型文件未找到: {self.MODEL_PATH}")

            num_classes = len(self.label_mapping) if self.label_mapping else 6
            # 确保 CNN_RNN 是最新的定义
            self.model = CNN_RNN(num_classes=num_classes).to(self.device)
            
            # 严格模式加载模型参数
            state_dict = torch.load(self.MODEL_PATH, map_location=self.device)
            self.model.load_state_dict(state_dict, strict=True)
            
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
            raise

    def extract_features(self, input_data):
        """
        从音频数据提取特征，确保与训练时 NPYDataset 处理方式一致
        返回: 处理后的特征张量 [1, 3, n_mels, target_length] 或 None
        """
        try:
            mel_db = audio_bytes_to_npy(input_data)
            if mel_db is None:
                raise ValueError("无法从输入数据提取梅尔频谱特征")

            # 确保特征维度正确 (n_mels, time)
            if mel_db.ndim == 3 and mel_db.shape[0] == 1:
                mel_db = mel_db.squeeze(0) # -> [n_mels, time]

            target_length = 100 # 与训练时 NPYDataset 中的 target_length 保持一致
            # 统一时间维度长度
            if mel_db.shape[1] > target_length:
                mel_db = mel_db[:, :target_length]
            elif mel_db.shape[1] < target_length:
                pad_size = target_length - mel_db.shape[1]
                mel_db = np.pad(mel_db, ((0,0), (0,pad_size)), mode='constant')
                # mel_db shape is now [n_mels, target_length]

            # 转换为PyTorch张量
            features = torch.from_numpy(mel_db).float() # -> [n_mels, target_length]

            # 添加通道维度并复制为三通道，模拟训练时的 NPYDataset 操作
            features = features.unsqueeze(0) # -> [1, n_mels, target_length]
            features = features.repeat(3, 1, 1)  # -> [3, n_mels, target_length]

            # 添加 Batch 维度并移动到设备
            return features.unsqueeze(0).to(self.device) # -> [1, 3, n_mels, target_length]

        except Exception as e:
            print(f"特征提取失败: {str(e)}")
            return None

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
            plt.savefig(img_stream, format='png', bbox_inches='tight')
            plt.close()
            img_stream.seek(0)
            
            return base64.b64encode(img_stream.read()).decode('utf-8')
        except Exception as e:
            print(f"绘制置信度图失败: {str(e)}")
            return None

    def predict(self):
        """执行预测"""
        if self.model is None:
            return {'error': '模型未正确加载。'}
        if self.label_mapping is None:
            return {'error': '类别映射未正确加载。'}

        try:
            if not hasattr(self, 'processed_audio') or self.processed_audio is None:
                raise ValueError("未提供音频数据")

            # 特征提取
            features = self.extract_features(self.processed_audio)
            if features is None:
                raise ValueError("特征提取失败")

            # 模型推理
            with torch.no_grad():
                outputs, features = self.model(features)  # 获取分类输出和特征向量
                predictions = F.softmax(outputs, dim=1)
                predicted_index = torch.argmax(predictions, dim=1).item()

            predicted_emotion = self.label_mapping.get(str(predicted_index), "Unknown")
            confidence_base64 = self.plot_confidence(predictions)

            return {
                'predicted_category': predicted_emotion,
                'confidence': {
                    'feature_name': 'Confidence',
                    'base64': confidence_base64
                } if confidence_base64 else None,
                'features': features.cpu().numpy().tolist()  # 返回特征向量用于后续分析
            }

        except Exception as e:
            print(f"预测失败: {str(e)}")
            return {'error': str(e)}

    def set_dataset(self, dataset):
        """设置数据集并重新加载相关资源"""
        if dataset == self.dataset:
            return  # 如果数据集相同则不需要重新加载
        
        self.dataset = dataset
        # 更新路径
        self.MODEL_PATH = f"models/{self.dataset}_npy_contrastive_model.pth"
        self.LABEL_ENCODER_PATH = f"models/label_encoder/{self.dataset}_CNN_class.json"
        self.SPEAKER_ENCODER_PATH = f"models/label_encoder/{self.dataset}_CNN_speaker.json"
        
        current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.MODEL_PATH = os.path.join(current_dir, self.MODEL_PATH)
        self.LABEL_ENCODER_PATH = os.path.join(current_dir, self.LABEL_ENCODER_PATH)
        self.SPEAKER_ENCODER_PATH = os.path.join(current_dir, self.SPEAKER_ENCODER_PATH)
        
        # 重新加载模型和映射
        self.load_label_mapping()
        self.load_model()