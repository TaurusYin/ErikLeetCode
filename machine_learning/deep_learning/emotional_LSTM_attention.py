import numpy as np
from tensorflow.keras.layers import Embedding, LSTM, Dense, Attention, Concatenate, Input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

# 假设的数据和参数
max_length = 100  # 序列的最大长度
vocab_size = 10000  # 词汇表的大小
num_samples = 1000  # 样本数量
num_epochs = 5  # 迭代次数

# 随机生成一些训练数据
X_train = np.random.randint(vocab_size, size=(num_samples, max_length))
y_train = np.random.randint(2, size=(num_samples, 1))  # 二分类问题，标签为0或1

# 输入层
input_layer = Input(shape=(max_length,))

# 嵌入层（将单词转换为向量）
emb_layer = Embedding(vocab_size, 100)(input_layer)

# LSTM层
lstm_layer, hidden_state, cell_state = LSTM(128, return_sequences=True, return_state=True)(emb_layer)

# 注意力层
attention_result = Attention()([hidden_state, lstm_layer])

# 合并注意力输出和LSTM层的输出
concat_result = Concatenate()([lstm_layer, attention_result])

# 输出层（预测任务相关，比如情感分析预测为正面还是负面）
output_layer = Dense(1, activation='sigmoid')(concat_result)

# 创建模型
model = Model(input_layer, output_layer)

# 编译模型
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 训练模型
model.fit(X_train, y_train, epochs=num_epochs)

# 生成一些测试数据
X_test = np.random.randint(vocab_size, size=(100, max_length))
y_test = np.random.randint(2, size=(100, 1))

# 测试模型
test_loss, test_accuracy = model.evaluate(X_test, y_test)

print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")
