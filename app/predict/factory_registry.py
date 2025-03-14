model_registry = {}

def register_model(model_type):
    def wrapper(cls):
        model_registry[model_type.lower()] = cls
        return cls
    return wrapper



