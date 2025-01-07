model_registry = {}

def register_model(model_type):
    def wrapper(cls):
        model_registry[model_type.lower()] = cls
        return cls
    return wrapper


def model_factory(model_type, processed_audio, sr):

    print(f"Creating model for: {model_type}")

    model_class = model_registry.get(model_type.lower())

    print(f"Model class found: {model_class}")

    if not model_class:
        raise ValueError(f"无效的模型选择：{model_type}")
    return model_class(processed_audio, sr)
