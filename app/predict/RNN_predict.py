import torch
from torch.nn.utils.rnn import pad_sequence
from app.model_training.RNN_two_training import EmotionClassifier
import joblib


def rnn_predict(features):
    """
    使用训练好的 RNN 模型进行预测，并反编码为情感名称。
    Args:
        features: 二维特征数据 (列表或数组)，每行是一个时间步的特征 (13维)
    Returns:
        dict: 包含情感名称的结果
    """
    # 模型和编码器路径
    MODEL_PATH = "models/rnn_2.pth"
    ENCODER_PATH = "models/label_encoder/CREMA-D_label_encoder.joblib"

    try:
        # 加载 LabelEncoder
        label_encoder = joblib.load(ENCODER_PATH)

        # 实例化模型
        model = EmotionClassifier(input_size=13, hidden_size=128, num_classes=len(label_encoder.classes_))
        model.load_state_dict(torch.load(MODEL_PATH))
        model.eval()  # 设置模型为评估模式

        # 数据预处理
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
            outputs = model(padded_features, lengths)  # 调用模型
            y_pred = outputs.argmax(dim=1).cpu().numpy()  # 获取预测类别（数字）

        # 将数字类别转换为情感名称
        predicted_emotion = label_encoder.inverse_transform([y_pred[0]])[0]  # 转换为情感名称

        # 返回预测结果，只包含情感名称
        result = {
            'predicted_category': predicted_emotion  # 返回情感名称
        }
        return result

    except Exception as e:
        print(f"预测时出错: {e}")
        return None
