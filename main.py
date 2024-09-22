import random
from torch_geometric.loader import DataLoader
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import from_networkx
import networkx as nx
import pandas as pd

data_path = '30550_prod.csv'
data_df = pd.read_csv(data_path)

label_path = 'labels.csv'
label_df = pd.read_csv(label_path)


def masking(graph, masking_rate):
    num_edges = graph.edge_index.size(1)
    num_actual_edges = num_edges // 2

    num_masking = int(masking_rate * num_actual_edges)

    masking_edge = random.sample(range(num_actual_edges), num_masking)

    masking_edge = masking_edge + [i + num_actual_edges for i in masking_edge]

    masked_graph = graph.clone()
    masked_graph.edge_attr[masking_edge] = 0.0

    return masked_graph


def PreProcessing(data_df, label_df):
    patients = sorted(set(col.split('_')[0] for col in data_df.columns if '_' in col), key=int)

    graphs = []
    labels = label_df['Label'].values

    label_idx = 0

    for patient in patients:
        patient_columns = sorted([col for col in data_df.columns if col.startswith(f"{patient}_")], key=lambda x: int(x.split('_')[1]))

        for time_point in patient_columns:
            G = nx.Graph()

            for idx, row in data_df.iterrows():
                node1 = row['node1']
                node2 = row['node2']
                edge_weight = row[time_point]

                G.add_edge(node1, node2, weight=edge_weight)

            data = from_networkx(G)

            edge_weights = [d['weight'] for _, _, d in G.edges(data=True)]
            edge_weights = torch.tensor(edge_weights, dtype=torch.float32).unsqueeze(1)

            data.edge_attr = torch.cat([edge_weights, edge_weights], dim=0)

            num_nodes = data.edge_index.max().item() + 1
            data.x = torch.ones((num_nodes, 1))

            data.y = torch.tensor([labels[label_idx]], dtype=torch.float32)
            graphs.append(data)
            label_idx += 1

    return graphs


class GNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GATConv(1, 32)
        self.conv2 = GATConv(32, 64)
        self.fc = nn.Linear(64, 2)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = global_mean_pool(x, batch)
        x = self.fc(x)

        return x


dataset = PreProcessing(data_df, label_df)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GNN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()

masking_rate = 0.1
previous_acc = 0
max_masking_rate = 0.9


while masking_rate <= max_masking_rate:
    print(f"---------------Running experiment with masking rate: {masking_rate * 100}%---------------")

    non_responder_graphs = [graph for graph in dataset if graph.y.item() == 0]
    responder_graphs = [graph for graph in dataset if graph.y.item() == 1]

    num_non_responders = len(non_responder_graphs)
    num_responders = len(responder_graphs)

    total = (num_non_responders - num_responders) / num_responders

    masked_training_graph = []
    for graph in responder_graphs:
        i = 0
        while i < total:
            masked_training_graph.append(masking(graph, masking_rate))
            i += 1

    training_set = dataset + masked_training_graph
    train_loader = DataLoader(training_set, batch_size=32, shuffle=True)

    masked_testing_graph = []
    for graph in responder_graphs:
        i = 0
        while i < total:
            masked_testing_graph.append(masking(graph, masking_rate))
            i += 1

    testing_set = dataset + masked_testing_graph
    test_loader = DataLoader(testing_set, batch_size=32, shuffle=True)

    accuracy = 0
    for epoch in range(10):
        print(f"Epoch: {epoch+1}\n-----------------------------------")

        model.train()
        training_loss = 0
        correct = 0
        total = 0

        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data)
            loss = criterion(out, data.y.long())
            loss.backward()
            optimizer.step()
            training_loss += loss.item()

            pred = out.argmax(dim=1)
            correct += (pred == data.y).sum().item()
            total += data.y.size(0)

        accuracy = correct / total
        avg_loss = training_loss / len(train_loader)
        print(f"Training Loss: {avg_loss:.4f}, Training Accuracy: {accuracy:.4f}")

        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for data in test_loader:
                data = data.to(device)
                out = model(data)
                pred = out.argmax(dim=1)
                correct += (pred == data.y).sum().item()
                total += data.y.size(0)

            accuracy = correct / total
            print(f"Accuracy: {accuracy:.4f}\n")

    if accuracy < previous_acc:
        print(f"Model's performance started to decline with masking rate {masking_rate * 100}%")
        break

    previous_acc = accuracy
    masking_rate += 0.1














