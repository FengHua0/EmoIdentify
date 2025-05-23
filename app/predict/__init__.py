from .factory_registry import model_registry
from app.predict.SVM_predict import SVM
from app.predict.RNN_predict import RNN
from app.predict.CNN_RNN_predict import CNN
from app.predict.contrastive_predict import contrastive
from app.predict.npy_cnn_predict import NpyCNN
from app.predict.npy_contrastive_predict import NpyContrastivePredict

def model_factory(model_type, processed_audio, sr):

    print(f"Creating model for: {model_type}")

    model_class = model_registry.get(model_type.lower())

    print(f"Model class found: {model_class}")

    if not model_class:
        raise ValueError(f"无效的模型选择：{model_type}")
    return model_class(processed_audio, sr)