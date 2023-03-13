import plotly.graph_objects as go
from sklearn.cluster import DBSCAN
import random
import numpy as np

# 生成时序序列
x = list(range(1000))
y = [random.randint(0, 10) for i in x]

# 注入部分异常点
for i in range(20):
    idx = random.randint(0, len(y) - 1)
    y[idx] = random.randint(20, 30)

# 使用DBSCAN进行异常检测
X = np.column_stack((x, y))
db = DBSCAN(eps=10, min_samples=10).fit(X)
labels = db.labels_
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)
print('Estimated number of clusters: %d' % n_clusters_)
print('Estimated number of noise points: %d' % n_noise_)

# 标记异常点
colors = ['rgb(255,0,0)' if label == -1 else 'rgb(0,0,255)' for label in labels]

# 绘制时序图
fig = go.Figure(data=go.Scatter(x=x, y=y, line=dict(color='rgb(0,0,255)')))
fig.add_trace(go.Scatter(x=[idx for idx, label in enumerate(labels) if label == -1],
                         y=[val for val, label in zip(y, labels) if label == -1],
                         mode='markers', marker=dict(color='rgb(255,0,0)')))
fig.show()
