import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from sklearn.preprocessing import MinMaxScaler


class GraphConvLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphConvLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x, adj):
        x = torch.matmul(adj, x)
        x = self.linear(x)
        return x


class Time2Graph(nn.Module):
    def __init__(self, num_nodes, num_features, num_classes):
        super(Time2Graph, self).__init__()
        self.num_nodes = num_nodes
        self.num_features = num_features
        self.num_classes = num_classes
        self.conv1 = GraphConvLayer(num_features, 64)
        self.conv2 = GraphConvLayer(64, 32)
        self.fc = nn.Linear(32 * num_nodes, num_classes)

    def forward(self, x, adj):
        x = F.relu(self.conv1(x, adj))
        x = F.relu(self.conv2(x, adj))
        x = x.view(-1, 32 * self.num_nodes)
        x = self.fc(x)
        return x


# Load and preprocess data
# 读取UCR数据集，例如GunPoint数据集
data = np.loadtxt('data.txt')
data = pd.read_csv('.txt', header=None, delimiter='\t')
# 提取时间序列数据
X = data.iloc[:, 1:].values
scaler = MinMaxScaler()
data = scaler.fit_transform(data)

# Construct graph adjacency matrix
num_nodes = 10
adj = np.zeros((num_nodes, num_nodes))
for i in range(num_nodes):
    for j in range(num_nodes):
        if abs(i - j) <= 2:
            adj[i, j] = 1

# Convert data to graph representation
graph_data = np.zeros((data.shape[0], num_nodes, 1))
for i in range(data.shape[0]):
    for j in range(num_nodes):
        graph_data[i, j, 0] = data[i, j]

# Split data into training and testing sets
train_data = graph_data[:800, :, :]
test_data = graph_data[800:, :, :]

# Define model and optimizer
model = Time2Graph(num_nodes, 1, 2)
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Train model
for epoch in range(100):
    model.train()
    optimizer.zero_grad()
    inputs = torch.tensor(train_data, dtype=torch.float32)
    adj_tensor = torch.tensor(adj, dtype=torch.float32)
    outputs = model(inputs, adj_tensor)
    targets = torch.tensor([0, 1] * 400, dtype=torch.long)
    loss = F.cross_entropy(outputs, targets)
    loss.backward()
    optimizer.step()

# Test model
model.eval()
inputs = torch.tensor(test_data, dtype=torch.float32)
adj_tensor = torch.tensor(adj, dtype=torch.float32)
outputs = model(inputs, adj_tensor)
preds = torch.argmax(outputs, dim=1)
print(preds)
