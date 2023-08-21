import torch
import torch.nn as nn
import torch.optim as optim

# 定义模型
class Model(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Model, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = torch.softmax(self.layer3(x), dim=1)  # 使用softmax激活函数
        return x

# 定义模型参数
num_shards = 10  # 分片数量
input_dim = num_shards  # 输入维度为分片数量，每个分片一个权重
hidden_dim = 64  # 隐藏层维度
output_dim = num_shards  # 输出维度为分片数量，每个分片一个权重

# 创建模型
model = Model(input_dim, hidden_dim, output_dim)

# 创建优化器
optimizer = optim.Adam(model.parameters(), lr=0.01)

num_rounds = 10  # 模拟的轮数
num_epochs = 50  # 每轮的训练次数

for i in range(num_rounds):
    print(f"\nRound {i + 1}:")

    # 模拟输入数据，每个分片的RT
    # RT数据，范围为0.01到1之间
    response_times = torch.rand(num_shards).unsqueeze(0)  # 增加一个维度以匹配模型输入

    for epoch in range(num_epochs):
        # 清空梯度
        optimizer.zero_grad()
        # 使用模型预测
        predictions = model(response_times)
        # 计算损失函数，这里使用方差作为损失函数
        loss = torch.var(predictions)
        # 反向传播
        loss.backward()
        # 更新模型参数
        optimizer.step()

    # 打印每个分片的RT和预测的权重
    with torch.no_grad():
        print(f"Response Times: {response_times.squeeze().numpy()}")
        print(f"Predicted Weights: {predictions.squeeze().numpy()}")
