import base64
from io import BytesIO
from flask import Flask, request, jsonify, render_template
from pydub import AudioSegment
import librosa
from app.file_process.data_processing import preprocess_audio
from app.predict import model_factory
from app.visible.spectrogram import spectrogram_base64, linear_spectrogram_base64
from app.visible.waveform import waveform_base64
from app.visible.MFCC_visible import mfcc_heatmap, extract_mfcc

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
        audio_file = request.files.get('audio')
        if not audio_file or audio_file.filename == '':
            print("[ERROR] 没有上传音频文件")  # 新增调试信息
            return jsonify({'error': 'No audio file uploaded'}), 400

        # 获取选择的模型和数据集
        model_type = request.form.get('model')
        dataset_name = request.form.get('dataset')
        print(f"[DEBUG] 获取到模型类型: {model_type}, 数据集: {dataset_name}")  # 新增调试信息
        
        if not model_type:
            print("[ERROR] 没有选择模型")  # 新增调试信息
            return jsonify({'error': 'No model selected'}), 400

        # 读取音频文件
        audio_bytes = audio_file.read()
        file_extension = audio_file.filename.rsplit('.', 1)[1].lower()
        print(f"[DEBUG] 音频文件格式: {file_extension}")  # 新增调试信息
        
        # 处理音频文件，转换为 wav 格式
        if file_extension == 'webm':
            # webm 转换为 wav 格式
            audio_segment = AudioSegment.from_file(BytesIO(audio_bytes), format='webm')
            wav_io = BytesIO()
            audio_segment.export(wav_io, format='wav')
            wav_bytes = wav_io.getvalue()
            y, sr = librosa.load(BytesIO(wav_bytes), sr=16000, mono=True)
        elif file_extension in ['wav', 'mp3']:
            y, sr = librosa.load(BytesIO(audio_bytes), sr=16000, mono=True)
            wav_bytes = audio_bytes  # 对于 wav 或 mp3 格式，直接使用原始字节数据
        else:
            return jsonify({'error': 'Invalid audio file format. Please upload wav, mp3, or webm files.'}), 400
        
        # 预处理
        wav_bytes = preprocess_audio(y)
        
        # 特征提取和可视化
        spectrogram, _ = spectrogram_base64(wav_bytes)
        # linear_spectrogram = linear_spectrogram_base64(wav_bytes)
        waveform = waveform_base64(wav_bytes)
        
        if wav_bytes is None:
            return jsonify({'error': 'Failed to preprocess audio'}), 500
        # 使用工厂模式获取模型实例
        try:
            print(f"[DEBUG] 正在创建模型实例: {model_type}")  # 新增调试信息
            model = model_factory(model_type, wav_bytes, sr)
            
            # 如果是SVM或RNN模型，执行特征提取
            if model_type.lower() in ['svm', 'rnn']:
                print("[DEBUG] 正在提取MFCC特征")  # 新增调试信息
                mfcc_feature = model.extract_features()
                if mfcc_feature is None:
                    print("[ERROR] 特征提取失败")  # 新增调试信息
                    return jsonify({'error': '特征提取失败'}), 500
                    
        except ValueError as ve:
            print(f"[ERROR] 创建模型失败: {str(ve)}")  # 新增调试信息
            return jsonify({'error': str(ve)}), 400

        if model is None:
            print("[ERROR] 模型实例创建失败")  # 新增调试信息
            return jsonify({'error': 'Failed to create model instance'}), 500

        # 执行预测
        print(f"[DEBUG] 正在设置数据集: {dataset_name}")  # 新增调试信息
        model.set_dataset(dataset_name)
        
        print("[DEBUG] 开始执行预测")  # 新增调试信息
        result = model.predict()
        if result is None:
            print("[ERROR] 预测失败")  # 新增调试信息
            return jsonify({'error': 'Prediction failed'}), 500

        # 获取预测类别
        predicted_category = result['predicted_category']
        # MFCC特征可视化
        # mfcc_feature = mfcc_heatmap(mfcc_feature)
        confidence = result['confidence']

        features = []
        # 使用append方法逐个添加特征
        features.append(confidence)

        # features.append(mfcc_feature)
        # features.append(linear_spectrogram)
        
        features.append(spectrogram)
        features.append(waveform)

        # 返回JSON响应
        return jsonify({
            'predicted_category': predicted_category,
            'features': features
        })

    except Exception as e:
        print(f"[ERROR] 预测过程中发生异常: {str(e)}")  # 新增调试信息
        return jsonify({'error': str(e)}), 500
