import os
import torch
import networkx as nx
import numpy as np
import community as community_louvain
from torch_geometric.data import Data
from models.gcn import GCN
from models.gat import GAT

file_path = os.path.join(os.path.dirname(__file__), "../data/account_graph.pt")
graph_data = torch.load(file_path, weights_only=False)

input_dim = graph_data.x.shape[1]
hidden_dim = 64
output_dim = input_dim 

gcn_model = GCN(input_dim, hidden_dim, output_dim)
gcn_model.load_state_dict(torch.load("../data/gcn_model.pth"))
gcn_model.eval()

gat_model = GAT(input_dim, hidden_dim, output_dim)
gat_model.load_state_dict(torch.load("../data/gat_model.pth"))
gat_model.eval()

with torch.no_grad():
    gcn_embeddings = gcn_model(graph_data.x, graph_data.edge_index)  # GCN embeddings
    gat_embeddings = gat_model(graph_data.x, graph_data.edge_index)  # GAT embeddings

final_embeddings = torch.cat((gcn_embeddings, gat_embeddings), dim=1)  # Concatenate both

final_embeddings_np = final_embeddings.cpu().numpy()

G = nx.Graph()
edge_list = graph_data.edge_index.cpu().numpy().T
G.add_edges_from(edge_list)

partition = community_louvain.best_partition(G, weight='weight')

np.save("../data/louvain_communities.npy", partition)
print("✅ Community Detection Complete! Results Saved.")
np.save("../data/gcn_embeddings.npy", gcn_embeddings.cpu().numpy())
np.save("../data/gat_embeddings.npy", gat_embeddings.cpu().numpy())
print("✅ GCN & GAT embeddings saved as .npy files")
