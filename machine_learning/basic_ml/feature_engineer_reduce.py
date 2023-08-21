"""
1.3 高维组合特征的处理

高维组合特征可能导致维度灾难和过拟合问题。处理方法包括：

降维：通过主成分分析（PCA）等方法将高维特征空间投影到低维空间。
特征选择：通过相关性分析、互信息等方法挑选出与目标变量相关性较高的特征。
正则化：在模型训练过程中引入正则项，如L1正则化、L2正则化等，以防止过拟合。


"""

import numpy as np
from sklearn.decomposition import PCA
"""
PCA的目标是通过线性变换将原始特征投影到新的特征空间，使得新特征空间的各个维度之间的协方差为零。PCA试图寻找最大化方差的方向，使得数据在新的坐标轴上具有更好的区分度。

PCA的原理公式如下：

假设数据矩阵为 X（n x p），其中 n 为样本数，p 为特征数。首先计算数据矩阵 X 的协方差矩阵：

C = (1 / n) * X.T * X

接着计算协方差矩阵 C 的特征值和特征向量。选择前 k 个最大特征值对应的特征向量组成投影矩阵 P（p x k）。最后，将原始数据矩阵 X 乘以投影矩阵 P，得到降维后的数据矩阵 Y（n x k）：

Y = X * P
"""


data = np.random.rand(100, 10)

pca = PCA(n_components=3)
reduced_data = pca.fit_transform(data)

print(reduced_data)


import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
"""
互信息法特征选择：
互信息法（Mutual Information）是度量两个随机变量之间的相关性的一种方法。对于特征选择，互信息法通过计算每个特征与目标变量之间的互信息值来评估特征的重要性。互信息值越大，表示特征与目标变量之间的相关性越强。

互信息公式如下：

I(X, Y) = ΣΣ P(x, y) * log(P(x, y) / (P(x) * P(y)))

其中，X 和 Y 分别为两个随机变量，P(x, y) 是它们的联合概率分布，P(x) 和 P(y) 分别是它们的边缘概率分布。当处理离散型特征时，可以直接计算互信息；当处理连续型特征时，需要先对特征进行离散化。
"""
data = np.random.rand(100, 10)
target = np.random.randint(0, 2, 100)

selector = SelectKBest(mutual_info_classif, k=3)
selected_data = selector.fit_transform(data, target)

print(selected_data)


"""
L2正则化：
L2正则化是一种防止过拟合的方法，通过在损失函数中添加一个正则项来约束模型的复杂度。对于线性模型（例如线性回归或逻辑回归），L2正则化在损失函数中添加了权重向量的L2范数平方项。这个正则项会使得权重向量中的值接近于零，从而降低模型复杂度。

对于逻辑回归模型，损失函数（对数损失）如下：

L(w) = -Σ [y_i * log(p(y_i)) + (1 - y_i) * log(1 - p(y_i))]

其中，y_i 是第 i 个样本的真实类别，p(y_i) 是模型预测的概
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

data = np.random.rand(100, 10)
target = np.random.randint(0, 2, 100)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=42)

# 标准化数据
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 使用带有L2正则化的逻辑回归模型
model = LogisticRegression(penalty='l2', C=1.0, solver='lbfgs')
model.fit(X_train, y_train)

train_accuracy = model.score(X_train, y_train)
test_accuracy = model.score(X_test, y_test)

print(f"Train accuracy: {train_accuracy}, Test accuracy: {test_accuracy}")
