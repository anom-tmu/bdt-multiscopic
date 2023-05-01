import pandas as pd
import numpy as np

import torch
from torch_geometric.data import Data

from sklearn.utils import shuffle
import time

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> 

path_training = r'C:\ZOO_DATA\MYCODE\HAND\YOLOv5-livinglab\x_training_posture.csv'

df = pd.read_csv(path_training) #.head()

df_labels = df["label_id"].to_list()

unique, counts = np.unique(df_labels, return_counts=True)
dataset_num_classes = len(unique)

df = shuffle(df)
#print(df)

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> 

dataset = [] 
edge_index = torch.tensor([[0, 1], #[1, 0],
                           [1, 2], #[2, 1],
                           [2, 3], #[3, 2],
                           [0, 4], #[4, 0],
                           [4, 5], #[5, 4],
                           [5, 6], #[6, 5],
                           [0, 7], #[7, 0],
                           [7, 8], #[8, 7],
                           [8, 9], #[9, 8],
                           [0, 10], #[10, 0],
                           [10, 11], #[11, 10],
                           [11, 12], #[12, 11],
                           [0, 13], #[13, 0],
                           [13, 14], #[14, 13],
                           [14, 15] #[15, 14] 
                           ],dtype=torch.long)

i = 0
for i in range(df.shape[0]):
    x = torch.tensor([  [1],
                    [df.iloc[i]['angle01']], [df.iloc[i]['angle02']], [df.iloc[i]['angle03']],
                    [df.iloc[i]['angle11']], [df.iloc[i]['angle12']], [df.iloc[i]['angle13']],
                    [df.iloc[i]['angle21']], [df.iloc[i]['angle22']], [df.iloc[i]['angle23']],
                    [df.iloc[i]['angle31']], [df.iloc[i]['angle32']], [df.iloc[i]['angle33']],                   
                    [df.iloc[i]['angle41']], [df.iloc[i]['angle42']], [df.iloc[i]['angle43']]                   
                    ], dtype=torch.float)                

    y = torch.tensor( [ df.iloc[i]['label_id']] ).to(dtype=torch.int64)
    dataset.append( Data(x=x, edge_index=edge_index.t().contiguous(), y=y) )


#print(data[0].x)
#print(data[0].edge_index)
#print(data[0].y)

data = dataset[0]  # Get the first graph object.
dataset_num_features = data.num_features
dataset_num_node_features = 1

print()
print(f'Dataset Hand Pose:')
print('====================')
print(f'Number of graphs: {len(dataset)}')
print(f'Number of features: {dataset_num_features}')
print(f'Number of classes: {dataset_num_classes}')

print()
#print(data)
print('=============================================================')

# Gather some statistics about the first graph.
print(f'Number of nodes: {data.num_nodes}')
print(f'Number of edges: {data.num_edges}')
print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
print(f'Has isolated nodes: {data.has_isolated_nodes()}')
print(f'Has self-loops: {data.has_self_loops()}')
print(f'Is undirected: {data.is_undirected()}')


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> 

train_dataset = dataset[:525]
test_dataset = dataset[525:]

print(f'Number of training graphs: {len(train_dataset)}')
print(f'Number of test graphs: {len(test_dataset)}')

from torch_geometric.loader import DataLoader

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

for step, data in enumerate(train_loader):
    print(f'Step {step + 1}:')
    print('=======')
    print(f'Number of graphs in the current batch: {data.num_graphs}')
    print(data)
    print()



# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Training a Graph Neural Network (GNN)

from torch.nn import Linear
import torch.nn.functional as F

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Training with GCNConv
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool

""""""
class GCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(GCN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(dataset_num_node_features, hidden_channels)        # dataset.num_node_features
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, dataset_num_classes)                 # dataset.num_classes

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings 
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
        
        return x


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Training with GraphConv

from torch_geometric.nn import GraphConv

class GNN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(GNN, self).__init__()
        torch.manual_seed(12345)

        self.conv1 = GraphConv(dataset_num_node_features, hidden_channels)        # dataset.num_node_features
        self.conv2 = GraphConv(hidden_channels, hidden_channels)
        self.conv3 = GraphConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, dataset_num_classes)                 # dataset.num_classes

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)

        x = global_mean_pool(x, batch)

        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
        
        return x

criterion = torch.nn.CrossEntropyLoss()

def train():
    model.train()

    for data in train_loader:                               # Iterate in batches over the training dataset.
         out = model(data.x, data.edge_index, data.batch)   # Perform a single forward pass.
         loss = criterion(out, data.y)                      # Compute the loss.
         loss.backward()                                    # Derive gradients.
         optimizer.step()                                   # Update parameters based on gradients.
         optimizer.zero_grad()                              # Clear gradients.

def test(loader):
     model.eval()

     correct = 0
     for data in loader:                                    # Iterate in batches over the training/test dataset.
         out = model(data.x, data.edge_index, data.batch)  
         pred = out.argmax(dim=1)                           # Use the class with highest probability.
         correct += int((pred == data.y).sum())             # Check against ground-truth labels.
     return correct / len(loader.dataset)                   # Derive ratio of correct predictions.


#model = GCN(hidden_channels=64)

model = GNN(hidden_channels=64)     #64
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

print(model)

epoch_max = 500
start_time = time.time()

for epoch in range(1, epoch_max+1): #, 1001
    train()
    #if(epoch == epoch_max):
    train_acc = test(train_loader)
    test_acc = test(test_loader)
    
    # Epoch;  Train Acc; Test Acc; 
    print(f'{epoch:03d},{train_acc:.4f},{test_acc:.4f}')

print("--- %s seconds ---" % (time.time() - start_time))

# Save the Model
torch.save(model.state_dict(), 'C:\ZOO_DATA\MYCODE\HAND\YOLOv5-livinglab\model_grasppose_gnn.pkl')


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Create Confusion Matrix
# https://christianbernecker.medium.com/how-to-create-a-confusion-matrix-in-pytorch-38d06a7f04b7

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sn

y_pred = []
y_true = []

# iterate over test data
for data in test_loader:                                                # Iterate in batches over the training/test dataset.
    output = model(data.x, data.edge_index, data.batch)                 # Feed Network
    output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()

    y_pred.extend(output)                                               # Save Prediction
    y_true.extend(data.y.tolist())                                      # Save Truth

# constant for classes
# classes = ('0','1', '2', '3', '4', '5','6', '7', '8', '9', '10', '11', '12', '13', '14', '15')

# Build confusion matrix
cf_matrix = confusion_matrix(y_true, y_pred) #,  normalize='true'
cf_matrix = cf_matrix.astype('float') / cf_matrix.sum(axis=1)[:, np.newaxis]
#df_cm = pd.DataFrame(cf_matrix/np.sum(cf_matrix) *10, index = [i for i in classes],
#                     columns = [i for i in classes])

plt.figure(figsize = (12,6))
sn.heatmap(cf_matrix, annot=True, cmap="Blues")     #df_cm
plt.savefig('output_hand_pose.png')
plt.show()