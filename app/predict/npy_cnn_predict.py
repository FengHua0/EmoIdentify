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
from app.model_training.npy_cnn_training import CNN_RNN # 确保导入了正确的模型类
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
        :param processed_audio: 预处理后的音频数据(bytes)
        :param sr: 采样率(未使用)
        """
        super().__init__(processed_audio, sr)
        self.MODEL_PATH = "models/npy_cnn_model.pth"
        self.LABEL_ENCODER_PATH = "models/label_encoder/npy_CREMA-D_CNN_class.json"

        # --- 添加模型参数 ---
        self.n_mels = 128
        self.target_length = 100

        current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.MODEL_PATH = os.path.join(current_dir, self.MODEL_PATH)
        self.LABEL_ENCODER_PATH = os.path.join(current_dir, self.LABEL_ENCODER_PATH)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.label_mapping = None

        # --- 修改 transform: 移除不适用于 Mel 频谱的 Resize 和 Normalize ---
        self.transform = transforms.Compose([
            SingleToThreeChannels()
            # transforms.Resize((224, 224)), # 移除
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # 移除
        ])

        # --- 先加载 label_mapping ---
        self.load_label_mapping() # 需要 num_classes 来加载模型
        self.load_model()

    def load_model(self):
        """加载训练好的模型"""
        try:
            if not os.path.exists(self.MODEL_PATH):
                raise FileNotFoundError(f"模型文件未找到: {self.MODEL_PATH}")
            if self.label_mapping is None:
                 print("错误：无法加载模型，因为类别映射未加载。")
                 self.model = None
                 return

            num_classes = len(self.label_mapping)
            # --- 修改：初始化 CNN_RNN 时传递 n_mels 和 target_length ---
            self.model = CNN_RNN(
                num_classes=num_classes,
                n_mels=self.n_mels,
                target_length=self.target_length
            ).to(self.device)
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
        从音频字节流提取特征，并进行预处理以匹配模型输入。
        返回: 处理后的特征张量或None
        """
        try:
            # 1. 使用 audio_bytes_to_npy 处理输入数据，传递 n_mels
            mel_db = audio_bytes_to_npy(input_data, n_mels=self.n_mels)
            if mel_db is None:
                raise ValueError("无法从输入数据提取梅尔频谱特征")

            # 2. 确保维度正确 (n_mels, time)
            if mel_db.ndim == 3 and mel_db.shape[0] == 1:
                mel_db = mel_db.squeeze(0)
            elif mel_db.ndim != 2:
                 raise ValueError(f"提取的梅尔频谱维度不正确: {mel_db.shape}")

            # 3. 统一时间维度长度 (与训练时 NPYDataset 逻辑一致)
            current_length = mel_db.shape[1]
            if current_length > self.target_length:
                # 从头截取
                features_np = mel_db[:, :self.target_length]
            elif current_length < self.target_length:
                # 使用 constant 模式填充，默认值为 0
                pad_size = self.target_length - current_length
                features_np = np.pad(mel_db, ((0, 0), (0, pad_size)), mode='constant')
            else:
                features_np = mel_db

            # 4. 转换为PyTorch张量并添加通道维度 (1, n_mels, target_length)
            features = torch.from_numpy(features_np).float().unsqueeze(0)

            # 5. 应用 transform (复制通道) -> [3, n_mels, target_length]
            features = self.transform(features)

            # 6. 添加批次维度 -> [1, 3, n_mels, target_length]
            features = features.unsqueeze(0).to(self.device)

            return features

        except Exception as e:
            print(f"处理输入数据时出错: {e}")
            return None

    def predict(self):
        """
        使用训练好的CNN-RNN模型对音频字节流进行分类预测
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

            # 提取特征 (调用更新后的 extract_features)
            features = self.extract_features(self.processed_audio)
            if features is None:
                raise ValueError("特征提取失败")

            # 模型推理
            with torch.no_grad():
                # --- 模型输入现在是 [1, 3, n_mels, target_length] ---
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