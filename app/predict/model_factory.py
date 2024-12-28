# factory.py

from app.predict.svm_predict import SVM
from app.predict.RNN_predict import RNN

def model_factory(model_type, processed_audio, sr):
    """
    根据model_type返回对应的模型实例
    """
    models = {
        'svm': SVM,
        'rnn': RNN,
        # 'cnn': CNNModel,  # 根据需要添加更多模型
    }

    model_class = models.get(model_type.lower())
    if not model_class:
        raise ValueError('无效的模型选择。')

    return model_class(processed_audio, sr)
