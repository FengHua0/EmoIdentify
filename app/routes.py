from flask import Flask, request, jsonify, render_template
import librosa
import numpy as np
from io import BytesIO
from app.file_process.data_processing import preprocess_audio
from app.file_process.feature_extraction_1 import one_features_extract
from app.predict.svm_predict import svm_predict  # 确保函数路径正确

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')  # 返回前端页面

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 检查上传文件是否存在
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file uploaded'}), 400

        audio_file = request.files['audio']  # 获取上传的文件
        if audio_file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        # 更新状态：读取音频文件
        audio_bytes = audio_file.read()  # 将音频文件读入内存
        y, sr = librosa.load(BytesIO(audio_bytes), sr=16000, mono=True)

        # 更新状态：音频预处理
        processed_audio = preprocess_audio(y, sr=sr)  # 预处理音频数据
        if processed_audio is None:
            return jsonify({'error': 'Failed to preprocess audio'}), 500

        # 更新状态：提取音频特征
        features = one_features_extract(processed_audio, sr=sr)  # 调用特征提取函数
        if not features:
            return jsonify({'error': 'Failed to extract features'}), 500

        # 更新状态：调用 SVM 模型进行预测
        result = svm_predict(features)  # 调用 SVM 模型的预测函数
        if result is None:
            return jsonify({'error': 'Prediction failed'}), 500

        # 返回预测结果
        predicted_category = result['predicted_category']
        return jsonify({'predicted_category': predicted_category})

    except Exception as e:
        print(f"处理音频时发生错误: {e}")
        return jsonify({'error': str(e)}), 500
