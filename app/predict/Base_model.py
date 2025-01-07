class BaseModel:
    def __init__(self, processed_audio, sr):
        self.processed_audio = processed_audio
        self.sr = sr
        self.features = None

    def extract_features(self):
        """
        特征提取方法，需要在子类中实现
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def predict(self):
        """
        预测方法，需要在子类中实现
        """
        raise NotImplementedError("Subclasses should implement this method.")
