import numpy as np
import matplotlib.pyplot as plt

# 创建一些基本的模板序列
templates = [
    np.sin(np.linspace(0, 10, 100)),
    np.cos(np.linspace(0, 10, 100)),
    np.exp(np.linspace(0, 10, 100)),
    np.log(np.linspace(1, 11, 100)),
]

# 生成20条时序序列
n_series = 20
series_length = 100
min_value = -1
max_value = 1
series = np.zeros((n_series, series_length))

for i in range(n_series):
    # 随机选择一个模板序列
    template = np.random.choice(templates)
    # 从模板序列加上一些噪声得到新序列
    series[i] = template + np.random.normal(scale=0.1, size=len(template))
    # 对序列进行归一化，确保值范围在[min_value, max_value]之间
    series[i] = (series[i] - np.min(series[i])) / (np.max(series[i]) - np.min(series[i]))
    series[i] = (max_value - min_value) * series[i] + min_value
    # 计算序列的差分并确保波动方向相似
    series_diff = np.diff(series[i])
    series_diff_sign = np.sign(series_diff)
    for j in range(1, len(series_diff_sign)):
        if series_diff_sign[j] != series_diff_sign[j-1]:
            series[i, j:] *= -1

# 绘制时序序列
fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(series_length)
for i in range(n_series):
    ax.plot(x, series[i], alpha=0.7)
plt.show()
