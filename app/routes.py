from flask import Flask, request, jsonify, render_template
import librosa
import numpy as np
from io import BytesIO
from pydub import AudioSegment
from app.file_process.data_processing import preprocess_audio
from app.file_process.feature_extraction_1 import one_features_extract
from app.file_process.feature_extraction_2 import two_features_extract
from app.predict.svm_predict import svm_predict
from app.predict.RNN_predict import rnn_predict
from app.visible.MFCC_visible import visualize_features_as_heatmap
import os

app = Flask(__name__)

# 允许上传的音频格式
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'webm'}

# 辅助函数：检查文件格式
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

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

        # 获取选择的模型
        model_type = request.form.get('model')  # 获取前端传来的模型类型

        if not model_type:
            return jsonify({'error': 'No model selected'}), 400

        # 更新状态：读取音频文件
        audio_bytes = audio_file.read()  # 将音频文件读入内存

        # 检查文件格式并处理
        file_extension = audio_file.filename.rsplit('.', 1)[1].lower()

        if file_extension == 'webm':
            # 如果是 webm 格式，使用 pydub 转换为 wav 格式
            audio = AudioSegment.from_file(BytesIO(audio_bytes), format='webm')
            wav_io = BytesIO()
            audio.export(wav_io, format='wav')
            wav_bytes = wav_io.getvalue()
            y, sr = librosa.load(BytesIO(wav_bytes), sr=16000, mono=True)
        elif file_extension in ['wav', 'mp3']:
            # 如果是 wav 或 mp3 格式，直接加载
            y, sr = librosa.load(BytesIO(audio_bytes), sr=16000, mono=True)
        else:
            return jsonify({'error': 'Invalid audio file format. Please upload wav, mp3, or webm files.'}), 400

        # 更新状态：音频预处理
        processed_audio = preprocess_audio(y, sr=sr)  # 预处理音频数据
        if processed_audio is None:
            return jsonify({'error': 'Failed to preprocess audio'}), 500

        if model_type == 'svm':
            # 更新状态：提取一维音频特征
            features = one_features_extract(processed_audio, sr=sr)  # 调用特征提取函数
            if not features:
                return jsonify({'error': 'Failed to extract features'}), 500

            # 调用可视化函数生成 Base64 编码
            img_base64 = visualize_features_as_heatmap(features)

            # 调用 SVM 模型进行预测
            result = svm_predict(features)  # 调用 SVM 模型的预测函数
            if result is None:
                return jsonify({'error': 'Prediction failed'}), 500
            predicted_category = result['predicted_category']

        elif model_type == 'rnn':
            # 更新状态：提取二维音频特征
            features = two_features_extract(processed_audio, sr=sr)  # 调用特征提取函数
            if not features:
                return jsonify({'error': 'Failed to extract features'}), 500

            # 调用可视化函数生成 Base64 编码
            img_base64 = visualize_features_as_heatmap(features)

            # 调用 RNN 模型进行预测
            result = rnn_predict(features)  # 调用 RNN 模型的预测函数

            if result is None:
                return jsonify({'error': 'Prediction failed'}), 500
            predicted_category = result['predicted_category']
        else:
            return jsonify({'error': 'Invalid model selection'}), 400

        # 返回预测结果和热力图的 Base64 编码
        return jsonify({
            'predicted_category': predicted_category,
            'mfcc_heatmap': img_base64
        })

    except Exception as e:
        print(f"处理音频时发生错误: {e}")
        return jsonify({'error': str(e)}), 500
