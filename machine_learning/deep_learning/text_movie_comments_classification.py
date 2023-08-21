# 导入必要的库
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Embedding
import matplotlib.pyplot as plt
import pickle

def text_to_sequence(text, tokenizer):
    words = text.split()
    sequence = [tokenizer.get(word, 0) for word in words]
    return sequence

# 设定参数
vocab_size = 10000  # 词汇表大小
max_len = 300  # 序列最大长度
embedding_dim = 50  # 词向量维度

# 加载数据
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=vocab_size)

# 序列填充或截断
train_data = pad_sequences(train_data, maxlen=max_len)
test_data = pad_sequences(test_data, maxlen=max_len)

# 构建模型
model = Sequential([
    Embedding(vocab_size, embedding_dim, input_length=max_len),
    Flatten(),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

# 编译模型
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# 训练模型
history = model.fit(train_data, train_labels, epochs=5, validation_data=(test_data, test_labels))

# 画图展示训练过程
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.show()

# 保存tokenizer
tokenizer = imdb.get_word_index()
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

# 测试模型
# 加载tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# 测试评论
test_reviews = ['I really loved this movie. It was so great!', 'I really hate this movie. It is so bad.']

# 将测试评论转换为数字序列
test_sequences = [text_to_sequence(review, tokenizer) for review in test_reviews]

# 对序列进行填充或截断
test_data = pad_sequences(test_sequences, maxlen=max_len)

# 预测评论的情感
predictions = model.predict(test_data)

for i, prediction in enumerate(predictions):
    print(f'Review: {test_reviews[i]}')
    print(f'Predicted sentiment: {"Positive" if prediction > 0.5 else "Negative"}\n')
