import numpy as np
"""
1.1 特征归一化

特征归一化是数据预处理的一种方法，主要用于消除数据集中不同特征之间的量纲差异。这样做可以加快模型收敛速度，提高模型准确性。常用的特征归一化方法有：

最小-最大归一化（Min-Max Scaling）：将特征值缩放到指定范围（通常为0到1之间）。
均值归一化（Mean Normalization）：将特征值减去均值后除以最大值与最小值之差。
标准化（Standardization）：将特征值减去均值后除以标准差，使得特征值服从均值为0，方差为1的正态分布。
缩放到单位长度（Scaling to Unit Length）：将特征向量除以其模长，使其长度为1。
"""
def min_max_scaling(data):
    min_vals = np.min(data, axis=0)
    max_vals = np.max(data, axis=0)
    scaled_data = (data - min_vals) / (max_vals - min_vals)
    return scaled_data

data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
scaled_data = min_max_scaling(data)
print(scaled_data)

import numpy as np

def mean_normalization(data):
    mean_vals = np.mean(data, axis=0)
    min_vals = np.min(data, axis=0)
    max_vals = np.max(data, axis=0)
    normalized_data = (data - mean_vals) / (max_vals - min_vals)
    return normalized_data

data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
normalized_data = mean_normalization(data)
print(normalized_data)

import numpy as np

def mean_normalization(data):
    mean_vals = np.mean(data, axis=0)
    min_vals = np.min(data, axis=0)
    max_vals = np.max(data, axis=0)
    normalized_data = (data - mean_vals) / (max_vals - min_vals)
    return normalized_data

data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
normalized_data = mean_normalization(data)
print(normalized_data)

import numpy as np

def scaling_to_unit_length(data):
    norms = np.linalg.norm(data, axis=1, keepdims=True)
    unit_length_data = data / norms
    return unit_length_data

data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
unit_length_data = scaling_to_unit_length(data)
print(unit_length_data)
