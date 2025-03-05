import os
import torch
import numpy as np
import networkx as nx
import plotly.graph_objects as go
from community import community_louvain
from torch_geometric.data import Data

torch.serialization.add_safe_globals([Data])

file_path = os.path.join(os.path.dirname(__file__), "../data/account_graph.pt")
graph_data = torch.load(file_path, map_location=torch.device("cpu"), weights_only=False)

edge_index = graph_data.edge_index.cpu().numpy()
G_trans = nx.Graph()
edges = list(zip(edge_index[0, :], edge_index[1, :]))
G_trans.add_edges_from(edges)

communities = community_louvain.best_partition(G_trans, resolution=1.0)

unique_communities = list(set(communities.values()))
np.random.seed(42)
colors = ["red", "sandybrown", "green", "blue", "purple", "orange", "cyan", "pink"]
community_colors = {comm: colors[i % len(colors)] for i, comm in enumerate(unique_communities)}

def plot_network(G, communities, community_colors, title, save_name, max_nodes=1000, max_edges=5000):
    pos = nx.spring_layout(G, seed=42) if len(G.nodes()) < 1000 else nx.random_layout(G, seed=42)
    
    nodes = list(G.nodes())
    if len(nodes) > max_nodes:
        nodes = np.random.choice(nodes, max_nodes, replace=False).tolist()
    
    node_x, node_y, node_colors_list, hover_texts = [], [], [], []
    for node in nodes:
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        comm = communities.get(node, 0)
        node_colors_list.append(community_colors.get(comm, "gray"))
        hover_texts.append(f"Community {comm}")

    edges = list(G.edges())
    if len(edges) > max_edges:
        edge_indices = np.random.choice(len(edges), max_edges, replace=False)
        edges = [edges[i] for i in edge_indices]

    edge_x, edge_y = [], []
    for u, v in edges:
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode="lines", line=dict(width=0.5, color="gray"), hoverinfo="none"))
    fig.add_trace(go.Scatter(x=node_x, y=node_y, mode="markers",
                             marker=dict(size=8, color=node_colors_list, opacity=0.9),
                             text=hover_texts, hoverinfo="text"))
    
    fig.update_layout(title=title, showlegend=False, width=900, height=700, margin=dict(l=10, r=10, t=30, b=10),
                      hovermode="closest", plot_bgcolor="white")
    
    fig.write_html(save_name, include_plotlyjs='cdn')

plot_network(G_trans, communities, community_colors, "GCN Transaction Network", "gcn_visualization.html")
plot_network(G_trans, communities, community_colors, "GAT Transaction Network", "gat_visualization.html")
