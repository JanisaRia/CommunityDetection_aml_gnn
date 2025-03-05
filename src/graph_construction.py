import os
import pandas as pd
import torch
import networkx as nx
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler, LabelEncoder

file_path = os.path.join(os.path.dirname(__file__), "../data/hiPatterns.csv")
df = pd.read_csv(file_path)

le = LabelEncoder()
df['sender_id'] = le.fit_transform(df['sender_id'])
df['receiver_id'] = le.fit_transform(df['receiver_id'])

df = df.select_dtypes(include=['number'])
df.fillna(0, inplace=True)

def normalize_features(df, exclude_cols=[]):
    scaler = StandardScaler()
    return scaler.fit_transform(df.drop(columns=exclude_cols, errors='ignore'))

account_features = normalize_features(df, exclude_cols=['sender_id', 'receiver_id'])
node_features_account = torch.tensor(account_features, dtype=torch.float32)
edge_index_account = torch.tensor(df[['sender_id', 'receiver_id']].values, dtype=torch.long).t().contiguous()

account_graph = Data(x=node_features_account, edge_index=edge_index_account)
torch.save(account_graph, "../data/account_graph.pt")
print(f"Account Graph Saved: {account_graph.x.shape[0]} Nodes, {account_graph.edge_index.shape[1]} Edges")

G_trans = nx.DiGraph()

for _, row in df.iterrows():
    trans_id = f"{row['sender_id']}_{row['receiver_id']}_{row['amount']}"
    G_trans.add_node(trans_id, amount=row['amount'], time=row.get('time', 0))

for _, row in df.iterrows():
    sender = f"{row['sender_id']}_{row['receiver_id']}_{row['amount']}"
    receiver = f"{row['receiver_id']}_{row['sender_id']}_{row['amount']}"
    G_trans.add_edge(sender, receiver, weight=row['amount'])

trans_nodes = list(G_trans.nodes())
trans_edges = list(G_trans.edges())

node_map = {node: i for i, node in enumerate(trans_nodes)}

edge_index_trans = torch.tensor([[node_map[u], node_map[v]] for u, v in trans_edges], dtype=torch.long).t().contiguous()

trans_data = pd.DataFrame(G_trans.nodes(data=True)).T.fillna(0)

if 'amount' in trans_data.columns and 'time' in trans_data.columns:
    trans_data = trans_data[['amount', 'time']]
else:
    trans_data = pd.DataFrame({'amount': [0] * len(G_trans.nodes), 'time': [0] * len(G_trans.nodes)})

trans_data = normalize_features(trans_data)
node_features_trans = torch.tensor(trans_data, dtype=torch.float32)

transaction_graph = Data(x=node_features_trans, edge_index=edge_index_trans)

torch.save(transaction_graph, "../data/transaction_graph.pt")
print(f"Transaction Graph Saved: {transaction_graph.x.shape[0]} Nodes, {transaction_graph.edge_index.shape[1]} Edges")
