import pandas as pd
import numpy as np
import networkx as nx
import torch
from gnn import GCN
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import scipy.sparse as sp
from torch_geometric.utils import from_scipy_sparse_matrix

csv_label_file = 'data/TCGA_labels.csv'
df = pd.read_csv(csv_label_file)
print(df)



def log_and_normalize(df: pd.DataFrame) -> pd.DataFrame:
    # all columns but 'is_true', 'mutation', and 'Variant_Classification'
    features = df.drop(columns=["is_true", "mutation", "Variant_Classification"])
    # log-transform and normalize the features
    features = features.apply(lambda x: np.log(1 + x))
    features = (features - features.mean()) / features.std()
    # add back the non-numeric columns
    features = pd.concat(
        [features, df[["mutation", "Variant_Classification", "is_true"]]], axis=1
    )
    return features


def show_graph_repr(df_corr):
    # Create an undirected graph
    G = nx.Graph()

    # Add edges to the graph from the DataFrame
    for col1 in df_corr.columns:
        for col2 in df_corr.columns:
            if col1 != col2:
                correlation = df_corr.loc[col1, col2]
                G.add_edge(col1, col2, weight=correlation)

    # Draw the graph
    pos = nx.spring_layout(G)  # positions for all nodes

    # Draw nodes and labels
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=500)
    nx.draw_networkx_labels(G, pos, font_size=12, font_family='sans-serif')

    # Draw edges with weights
    labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edges(G, pos, edgelist=G.edges(), width=2, alpha=0.6, edge_color='gray')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, font_size=10)

    # Display the plot
    plt.title('Graph Representation of Correlation Matrix (Distinct Pairs)')
    plt.axis('off')
    plt.show()







#preprocess data : normalize gene expressions
scaler = StandardScaler()
Y = df['is_true'].astype(int).values
print(Y)
X = df.drop(columns=["is_true", "mutation", "Variant_Classification"])
X_norm = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# Split the data into train and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X_norm, Y, test_size=0.2, random_state=42)

corr_matrix = X_train.corr()
print(corr_matrix)


#for efficiency reasons, we just show the graph for first 50 cells  and 50 genes
#show_graph_repr(X_train.iloc[:50, :50].corr())

#for efficiency reasons build a sparse matrix
adj_matrix = sp.coo_matrix(corr_matrix.values)
# Convert to edge_index format for PyTorch Geometric
edge_index, edge_attr = from_scipy_sparse_matrix(adj_matrix)

from torch_geometric.data import Data


X_train = np.array(X_train)
X_test =  np.array(X_test)
Y_train = np.array(Y_train)
Y_test = np.array(Y_test)

#creates our training set
tr_data = []
for i in range(len(X_train)):
    x = torch.tensor(X_train[i], dtype=torch.float).reshape([-1, 1])
    y = torch.tensor(Y_train[i], dtype=torch.long)
    
    # Create Data object
    data = Data(x=x, y=y, edge_index=edge_index)
    tr_data.append(data)

#creates our test set
test_data = []
for i in range(len(X_test)):
    x = torch.tensor(X_test[i], dtype=torch.float).reshape([-1, 1])
    y = torch.tensor(Y_test[i], dtype=torch.long)
    
    data = Data(x=x, y=y, edge_index=edge_index)
    test_data.append(data)



from torch_geometric.loader import DataLoader
# Create a DataLoader for batch processing
tr_loader = DataLoader(tr_data, batch_size=1, shuffle=True)
test_loader = DataLoader(test_data, batch_size=1, shuffle=False)


model = GCN(1, 2, 2, 3)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Training loop
model.train()
for epoch in range(1):  # Number of epochs
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

accuracy = correct / len(Y_test)
print(f'Test Accuracy: {accuracy:.4f}')