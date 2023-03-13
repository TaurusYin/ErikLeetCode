import torch
import torch.nn.functional as F
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
from torch_geometric.nn import GCNConv, TopKPooling

# Load a time series dataset
dataset = TUDataset(root='/tmp/ENZYMES', name='ENZYMES')

# Split the dataset into train and test sets
train_dataset, test_dataset = dataset[:500], dataset[500:]
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)


# Define the GCN model
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.pool1 = TopKPooling(hidden_channels, ratio=0.8)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.pool2 = TopKPooling(hidden_channels, ratio=0.8)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.pool3 = TopKPooling(hidden_channels, ratio=0.8)
        self.lin1 = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch):
        # Convert time series to graph
        x = torch.tensor(x, dtype=torch.float)
        edge_index = torch.tensor(edge_index, dtype=torch.long)
        x, edge_index, _, batch, _ = self.pool1(x, edge_index, None, batch)
        x = F.relu(self.conv1(x, edge_index))
        x, edge_index, _, batch, _ = self.pool2(x, edge_index, None, batch)
        x = F.relu(self.conv2(x, edge_index))
        x, edge_index, _, batch, _ = self.pool3(x, edge_index, None, batch)
        x = F.relu(self.conv3(x, edge_index))
        x = self.lin1(x)
        return x


# Train the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN(1, 16, dataset.num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss()
for epoch in range(200):
    model.train()
    loss_all = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data.x, data.edge_index, data.batch)
        loss = criterion(output, data.y)
        loss.backward()
        loss_all += loss.item() * data.num_graphs
        optimizer.step()
    print('Epoch {}, Loss {}'.format(epoch, loss_all / len(train_dataset)))

# Evaluate the model on the test set
model.eval()
correct = 0
total = 0
for data in test_dataset:
    data = data.to(device)
    output = model(data.x, data.edge_index, data.batch)
    _, predicted = torch.max(output, 1)
    total += 1
    correct += (predicted == data.y).sum().item()
print('Accuracy: {}'.format(correct / total))
