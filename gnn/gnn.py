import torch
import networkx as nx
import matplotlib.pyplot as plt
from torch.nn.functional import tanh, log_softmax, cross_entropy, dropout
from torch.nn import Linear
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool
from torch_geometric.datasets import Planetoid



#for NODE PREDICTION

class GCN(torch.nn.Module):
    def __init__(self, n_features, emb_size, n_classes, n_mp_layers):
        """
        Parameters:
            n_features: is the number of nodes in the graph

        """
        super(GCN, self).__init__()
        self.n_mp_layers = n_mp_layers
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(n_features, emb_size))     #trasforms node features to embedding vectors
        for i in range(1, n_mp_layers):
            self.convs.append(GCNConv(emb_size, emb_size))  #message parsing step

        self.out = torch.nn.Sequential(                     #post-message parsing
            Linear(emb_size, n_classes)                     #converts into a vector of same size as the number of classes
        )
        


    def forward(self, batch):
        #data is an element of pytorch dataset, x is the feature matrix  of size (n_nodes X embedding_dim), edge_index is the adjacency list, batch indicates the nodes that belong to given graph
        x, edge_index, batch = batch.x, batch.edge_index, batch.batch
        out = x
        for i in range(self.n_mp_layers-1):
            out = self.convs[i](out, edge_index)
            out = tanh(out)
    
        out = self.convs[self.n_mp_layers-1](out, edge_index)
        out = global_mean_pool(out, batch) #graph level pooling
        out = self.out(out)
        return log_softmax(out, dim=1)
    
    
if __name__=="__main__":

    dataset = Planetoid(root='/tmp/Cora', name='Cora')
    #DataLoader(dataset, batch_size=64)
    model = GCNConv(max(dataset.num_features, 1), 16, dataset.num_classes, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)


    print(dataset.x)



    for epoch in range(200):
        data = dataset[0] 
        model.train()
        optimizer.zero_grad()
        output = model(data.x, data.edge_index)
        loss = cross_entropy(output[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        print(f'Epoch: {epoch}, Loss: {loss.item()}')
    




