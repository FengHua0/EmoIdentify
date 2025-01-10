import os
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import json

# 主函数定义
def train_cnn_model(data_folder, model_output, epochs=20, batch_size=32, img_size=(224, 224), resume_training=False):
    """
    基于 CNN 的频谱图六分类训练函数。每5轮保存一次模型。

    :param data_folder: 包含 train、val、test 文件夹的根目录
    :param model_output: 模型保存路径
    :param epochs: 训练轮数，默认 20
    :param batch_size: 批量大小，默认 32
    :param img_size: 输入图像尺寸，默认 (224, 224)
    :param resume_training: 是否加载已有模型继续训练，默认 False
    """
    # 检查数据路径
    train_dir = os.path.join(data_folder, "train")
    val_dir = os.path.join(data_folder, "val")
    test_dir = os.path.join(data_folder, "test")
    if not all([os.path.exists(train_dir), os.path.exists(val_dir), os.path.exists(test_dir)]):
        raise FileNotFoundError("数据文件夹结构不完整，请确保 train、val 和 test 文件夹存在。")

    # 数据加载与增强
    datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_generator = datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical'
    )
    val_generator = datagen.flow_from_directory(
        val_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical'
    )
    test_generator = datagen.flow_from_directory(
        test_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical'
    )

    # 获取类别映射
    class_indices = train_generator.class_indices
    print(f"类别映射: {class_indices}")

    # 反转字典用于反编码
    index_to_class = {v: k for k, v in class_indices.items()}

    # 保存路径
    save_path = "../models/label_encoder/CREMA-D_CNN.json"

    # 确保目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # 将类别映射保存为文件
    with open(save_path, "w") as f:
        json.dump(index_to_class, f)
    print(f"类别映射已保存为 {save_path}")

    # 如果需要加载已有模型继续训练
    if resume_training and os.path.exists(model_output):
        print(f"加载已有模型: {model_output}")
        model = load_model(model_output)
    else:
        # 否则创建新模型
        model = build_cnn_model(img_size, train_generator.num_classes)
        model.summary()

    # 编译模型
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # 创建模型检查点回调函数，每5轮保存一次模型
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        model_output,
        save_weights_only=False,
        save_best_only=False,
        save_freq=5 * train_generator.samples // batch_size,  # 每5个epoch保存一次模型
        verbose=1
    )

    # 模型训练
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=val_generator,
        callbacks=[checkpoint_callback]  # 加入回调
    )

    # 模型验证（测试集评估）
    test_loss, test_accuracy = model.evaluate(test_generator)
    print(f"测试集损失: {test_loss:.4f}, 测试集准确率: {test_accuracy:.4f}")

    # 绘制训练曲线
    plot_training_curves(history)


# 模型定义封装
def build_cnn_model(img_size, num_classes):
    """
    定义并返回 CNN 模型。

    :param img_size: 输入图像尺寸 (height, width)
    :param num_classes: 输出类别数
    :return: 已定义的 CNN 模型
    """
    model = Sequential([
        # 卷积层 1
        Conv2D(32, (3, 3), activation='relu', input_shape=(img_size[0], img_size[1], 3)),
        MaxPooling2D(pool_size=(2, 2)),

        # 卷积层 2
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),

        # 卷积层 3
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),

        # 展平
        Flatten(),

        # 全连接层
        Dense(128, activation='relu'),
        Dropout(0.5),

        # 输出层
        Dense(num_classes, activation='softmax')  # 输出类别数
    ])

    return model


# 绘制训练曲线
def plot_training_curves(history):
    """
    绘制训练和验证的损失与准确率曲线。
    """
    # 绘制训练和验证损失
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # 绘制训练和验证准确率
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()


# 主程序
if __name__ == "__main__":
    # 用户提供的路径
    data_folder = "../features/spectrogram/CREMA-D"  # 包含 train、val、test 的文件夹路径
    model_output = "../models/cnn_mel_spectrogram_model.h5"  # 模型保存路径

    train_cnn_model(data_folder, model_output, epochs=10, batch_size=32, img_size=(224, 224), resume_training=True)
