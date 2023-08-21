import numpy as np
import pandas as pd

"""
1.2 类别型特征

类别型特征是指具有离散值的特征，如性别、颜色等。处理类别型特征的方法主要包括：

独热编码（One-Hot Encoding）：将类别特征转换为二进制向量，每个类别对应一个独立的维度。
Label Encoding：为每个类别分配一个整数值，将类别型特征转换为数值型特征。
目标编码（Target Encoding）：根据类别特征和目标变量之间的关系为每个类别分配一个值。
"""


def one_hot_encoding(data, column):
    one_hot_encoded = pd.get_dummies(data[column], prefix=column)
    data = pd.concat([data.drop(column, axis=1), one_hot_encoded], axis=1)
    return data


data = pd.DataFrame({'Color': ['Red', 'Green', 'Blue', 'Red', 'Green']})
encoded_data = one_hot_encoding(data, 'Color')
print(encoded_data)

import pandas as pd
from sklearn.preprocessing import LabelEncoder


def label_encoding(data, column):
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    return data


data = pd.DataFrame({'Color': ['Red', 'Green', 'Blue', 'Red', 'Green']})
encoded_data = label_encoding(data, 'Color')
print(encoded_data)

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold

"""
目标编码（Target Encoding），也称为平均数编码（Mean Encoding），是一种处理类别型特征的方法。它主要基于类别特征与目标变量之间的关系，通过计算每个类别在目标变量上的平均值来为类别分配数值。目标编码可以在一定程度上捕捉类别特征与目标变量之间的关系，从而提高模型的预测能力。

目标编码的主要步骤如下：

将数据集划分为训练集和测试集。
对于训练集中的每个类别，计算该类别对应的目标变量的平均值。
用计算得到的平均值替换训练集和测试集中的类别特征。
需要注意的是，为了防止数据泄露，不能直接使用整个训练集的目标变量平均值对类别进行编码。一种常用的解决方法是使用交叉验证（例如K折交叉验证）对训练集进行分组，并在每个分组上分别计算目标编码。

以下是目标编码的示例：

假设我们有如下数据集：

mathematica
Copy code
| Color | Target |
|-------|--------|
| Red   |   0    |
| Green |   1    |
| Blue  |   1    |
| Red   |   0    |
| Green |   1    |
我们需要对Color特征进行目标编码。首先，根据Color特征的每个类别计算Target变量的平均值：

Red：(0 + 0) / 2 = 0
Green：(1 + 1) / 2 = 1
Blue：1 / 1 = 1
然后用这些平均值替换原始类别特征：

lua
Copy code
| Color_target_encoded | Target |
|----------------------|--------|
|          0           |   0    |
|          1           |   1    |
|          1           |   1    |
|          0           |   0    |
|          1           |   1    |
经过目标编码处理后，我们得到了一个数值型特征，可以直接用于训练机器学习模型。

需要注意的是，目标编码可能导致过拟合，特别是在类别数量较多或某些类别样本较少的情况下。为了降低过拟合风险，可以对目标编码进行平滑处理，例如使用贝叶斯平滑。
"""


def target_encoding(data, column, target, folds=5):
    kf = KFold(n_splits=folds, shuffle=True, random_state=42)
    data[f'{column}_target_encoded'] = np.nan

    for train_index, test_index in kf.split(data):
        train_data = data.iloc[train_index]
        test_data = data.iloc[test_index]

        encoding_map = train_data.groupby(column)[target].mean()
        data.loc[test_index, f'{column}_target_encoded'] = test_data[column].map(encoding_map)

    data[f'{column}_target_encoded'].fillna(data[target].mean(), inplace=True)
    data.drop(column, axis=1, inplace=True)
    return data


data = pd.DataFrame({'Color': ['Red', 'Green', 'Blue', 'Red', 'Green'],
                     'Target': [0, 1, 1, 0, 1]})
encoded_data = target_encoding(data, 'Color', 'Target')
print(encoded_data)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences

# 参数设置
vocab_size = 10000  # 词汇表大小
max_length = 500  # 句子最大长度
embedding_dim = 16  # 嵌入维度

# 加载数据
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=vocab_size)

# 将句子截断或填充到相同长度
train_data = pad_sequences(train_data, maxlen=max_length)
test_data = pad_sequences(test_data, maxlen=max_length)

# 构建模型
model = keras.Sequential([
    layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length),
    layers.GlobalAveragePooling1D(),  # 通过平均池化将嵌入向量转换为固定长度的向量
    layers.Dense(16, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # 二元分类
])

# 编译模型
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 训练模型
model.fit(train_data, train_labels, epochs=10, validation_data=(test_data, test_labels))

# 假设我们有一些新的电影评论
new_reviews = ['This movie is fantastic!', 'Awful movie. I will never watch it again.']

# 首先，我们需要将文本转换为整数索引的形式，这通常涉及分词、查找词汇表等步骤，这里我们假设已经完成，得到：
new_reviews_indices = [[1, 4, 11, 9], [25, 3, 1, 13, 48, 19, 22]]

# 然后，我们将整数索引序列填充到相同长度
new_reviews_indices = pad_sequences(new_reviews_indices, maxlen=max_length)

# 最后，我们可以使用模型进行预测
predictions = model.predict(new_reviews_indices)

# 预测结果是每个评论是正面的概率
for i, prediction in enumerate(predictions):
    print('Review: {}\nPositive Probability: {}'.format(new_reviews[i], prediction[0]))
