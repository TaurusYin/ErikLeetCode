import numpy as np


# 0-1损失函数
def zero_one_loss(y_true, y_pred):
    return np.sum(y_true != y_pred) / len(y_true)


# 绝对损失函数
def absolute_loss(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


# 平方损失函数
def squared_loss(y_true, y_pred):
    return np.mean(np.square(y_true - y_pred))


# 对数损失函数/对数似然损失函数
def log_loss(y_true, y_pred_prob):
    epsilon = 1e-15
    y_pred_prob = np.clip(y_pred_prob, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred_prob) + (1 - y_true) * np.log(1 - y_pred_prob))


# Huber损失函数
def huber_loss(y_true, y_pred, delta=1.0):
    error = y_true - y_pred
    abs_error = np.abs(error)
    quadratic = np.minimum(abs_error, delta)
    linear = abs_error - quadratic
    return np.mean(0.5 * np.square(quadratic) + delta * linear)


# Log-Cosh损失函数
def log_cosh_loss(y_true, y_pred):
    error = y_true - y_pred
    return np.mean(np.log(np.cosh(error)))


# 真实值和预测值
y_true = np.array([3.0, -0.5, 2.0, 7.0])
y_pred = np.array([2.5, 0.0, 2.0, 8.0])

# 用于分类问题的真实标签和预测概率
y_true_classification = np.array([1, 0, 1, 0])
y_pred_prob_classification = np.array([0.9, 0.1, 0.8, 0.3])

# 计算各种损失函数的值
print("0-1损失函数:", zero_one_loss(y_true_classification, y_true_classification > 0.5))
print("绝对损失函数:", absolute_loss(y_true, y_pred))
print("平方损失函数:", squared_loss(y_true, y_pred))
print("对数损失函数/对数似然损失函数:", log_loss(y_true_classification, y_pred_prob_classification))
print("Huber损失函数:", huber_loss(y_true, y_pred))
print("Log-Cosh损失函数:", log_cosh_loss(y_true, y_pred))
