# import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# 加载数据
# MNIST数据集中的图像是28*28像素的手写数字图像，标签是对应的数字（0-9）。这里，我们把数据分为训练集和测试集。
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# 归一化 归一化图像数据，使其在0-1范围内，原因是归一化可以使得不同的特征有相同的尺度，有利于模型的训练。
train_images = train_images / 255.0
test_images = test_images / 255.0

# 转换为 one-hot 格式。同时，把标签转换为one-hot格式，即把0-9的数字转换为10维的向量，对应数字的位置为1，其余位置为0。
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# 定义模型 模型是一个顺序模型（Sequential），包含一个平坦层（用于把28*28的图像转换为784维的向量）、
# 一个全连接层（有128个神经元，使用ReLU激活函数）和一个输出层（有10个神经元，对应10个数字，使用softmax激活函数进行多分类）。
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

# 编译模型
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 训练模型并记录训练过程
history = model.fit(train_images, train_labels, epochs=5, validation_data=(test_images, test_labels))

# 获取训练过程中的准确率数据
train_acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

# 创建x轴数据
epochs = range(1, len(train_acc) + 1)

# 画出准确率曲线图
plt.plot(epochs, train_acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


