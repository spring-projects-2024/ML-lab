import pandas as pd
import numpy as np
import networkx as nx
import torch
from gnn import GCN  # Assuming you have a GCN model defined in gnn.py
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import scipy.sparse as sp
from torch_geometric.utils import from_scipy_sparse_matrix
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

# Read and preprocess data
csv_label_file = 'data/TCGA_labels.csv'
df = pd.read_csv(csv_label_file)

def log_and_normalize(df: pd.DataFrame) -> pd.DataFrame:
    features = df.drop(columns=["is_true", "mutation", "Variant_Classification"])
    features = features.apply(lambda x: np.log(1 + x))
    features = (features - features.mean()) / features.std()
    features = pd.concat([features, df[["mutation", "Variant_Classification", "is_true"]]], axis=1)
    return features

def show_graph_repr(df_corr):
    G = nx.Graph()
    for col1 in df_corr.columns:
        for col2 in df_corr.columns:
            if col1 != col2:
                correlation = df_corr.loc[col1, col2]
                G.add_edge(col1, col2, weight=correlation)

    pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=500)
    nx.draw_networkx_labels(G, pos, font_size=12, font_family='sans-serif')
    labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edges(G, pos, edgelist=G.edges(), width=2, alpha=0.6, edge_color='gray')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, font_size=10)
    plt.title('Graph Representation of Correlation Matrix (Distinct Pairs)')
    plt.axis('off')
    plt.show()




scaler = StandardScaler()
Y = df['mutation']
Y = Y.apply(lambda x: 1 if x == 'Missense_Mutation' else 0)
X = df.drop(columns=["is_true", "mutation", "Variant_Classification"])
X_norm = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# Split the data into train and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X_norm, Y, test_size=0.2, random_state=42)

corr_matrix = X_train.corr()
corr_matrix[abs(corr_matrix)<0.05] = 0


# Create sparse adjacency matrix
adj_matrix = sp.coo_matrix(corr_matrix.values)
edge_index, edge_attr = from_scipy_sparse_matrix(adj_matrix)

X_train = np.array(X_train)
X_test = np.array(X_test)
Y_train = np.array(Y_train)
Y_test = np.array(Y_test)

# Create PyTorch Geometric data objects for training set
tr_data = []
for i in range(len(X_train)):
    x = torch.tensor(X_train[i], dtype=torch.float).reshape([-1, 1])
    y = torch.tensor(Y_train[i], dtype=torch.long)
    data = Data(x=x, y=y, edge_index=edge_index)
    tr_data.append(data)

# Create PyTorch Geometric data objects for test set
test_data = []
for i in range(len(X_test)):
    x = torch.tensor(X_test[i], dtype=torch.float).reshape([-1, 1])
    y = torch.tensor(Y_test[i], dtype=torch.long)
    data = Data(x=x, y=y, edge_index=edge_index)
    test_data.append(data)

# DataLoader for batch processing
tr_loader = DataLoader(tr_data, batch_size=1, shuffle=True)
test_loader = DataLoader(test_data, batch_size=1, shuffle=False)

# Model, criterion, optimizer
model = GCN(1, 1, 2, 2)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Training loop
model.train()
for epoch in range(1):
    for batch in tr_loader:
        optimizer.zero_grad()
        out = model(batch)
        loss = criterion(out, batch.y)
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch+1}, Loss: {loss.item()}')

# Evaluation loop
model.eval()
correct = 0
for batch in test_loader:
    with torch.no_grad():
        out = model(batch)
        pred = out.argmax(dim=1)
        correct += (pred == batch.y).sum().item()
val_score = correct / len(Y_test)
print(f'Test Accuracy: {val_score:.4f}')
