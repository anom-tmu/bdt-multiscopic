# Reference : 
# https://github.com/python-engineer/pytorch-examples/blob/master/rnn-lstm-gru/main.py

# https://www.pluralsight.com/guides/lstm-versus-gru-units-in-rnn
# https://medium.com/analytics-vidhya/rnn-vs-gru-vs-lstm-863b0b7b1573

############################################### Dependencies

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler 

import matplotlib.pyplot as plt
import time

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset

import pickle

import csv

header = ['data'] 
csvfile = open(r'C:\ZOO_DATA\MYCODE\HAND\YOLOv5-livinglab\training_result.csv', 'w')
writer = csv.writer(csvfile, delimiter = ',', lineterminator='\n')
writer.writerow(header)


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters 
num_classes = 7
num_epochs = 50
batch_size = 1
learning_rate = 0.001

input_size = 15
sequence_length = 10
hidden_size = 128
num_layers = 2

################################################ Load Dataset
# Customized class for data preparation

class TrainDataset(Dataset):
    def __init__(self, file_name):

        # Read csv file and load row data into variables
        file_out = pd.read_csv(file_name)
        x = file_out.iloc[0:840, 0:15].values
        y = file_out.iloc[0:840, 15].values

        # Get odd row data only from 30 to 10 row
        x = x[::3]
        y = y[::3]       

        # Feature scaling and reshape
        sc = StandardScaler()
        new_x = sc.fit_transform(x)
        x_train = new_x.reshape((28, 10, 15))     # 840 / 30 = 28 , 30, 15 
        #x_train = new_x 

        # Feature scaling
        pickle.dump(sc, open(r'C:\ZOO_DATA\MYCODE\HAND\YOLOv5-livinglab\scaler_input.pkl','wb'))
        
        new_y = []
        for i in range(len(y)):
            if i % sequence_length == 0:
                new_y.append(y[i])
        y_train = np.array(new_y)       

        # Converting to torch tensors
        self.x_train = torch.tensor(x_train, dtype=torch.float32)
        self.y_train = torch.tensor(y_train)

    def __len__(self):
        return len(self.y_train)

    def __getitem__(self, idx):
        return  self.x_train[idx], self.y_train[idx]


class TestDataset(Dataset):
    def __init__(self, file_name):

        # Read csv file and load row data into variables
        file_out = pd.read_csv(file_name)
        x = file_out.iloc[0:840, 0:15].values
        y = file_out.iloc[0:840, 15].values

        # Get odd row data only from 30 to 10 row
        x = x[::3]
        y = y[::3]

        # Feature scaling and reshape
        sc = StandardScaler()
        new_x = sc.fit_transform(x)
        x_test = new_x.reshape((28, 10, 15))     # 840 / 30 = 28 , 30, 15  

        new_y = []
        for i in range(len(y)):
            if i % sequence_length == 0:
                new_y.append(y[i])
        y_test = np.array(new_y)        

        # Converting to torch tensors
        self.x_test = torch.tensor(x_test, dtype=torch.float32)
        self.y_test = torch.tensor(y_test)

    def __len__(self):
        return len(self.y_test)

    def __getitem__(self, idx):
        return  self.x_test[idx], self.y_test[idx]

  
# Data loader
train_dataset = TrainDataset(r"C:\ZOO_DATA\MYCODE\HAND\YOLOv5-livinglab\x_training.csv")
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=batch_size, 
                                           shuffle=True)

test_dataset = TestDataset(r"C:\ZOO_DATA\MYCODE\HAND\YOLOv5-livinglab\x_testing.csv")
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                           batch_size=batch_size, 
                                           shuffle=True)



############################################### Design RNN

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        # -> x needs to be: (batch_size, seq, input_size)

        #self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)            # <<<<<<<<<<< RNN
        # or:
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)            # <<<<<<<<<<< GRU
        #self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)           # <<<<<<<<<<< LSTM
        
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        # Set initial hidden states (and cell states for LSTM)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device) 
        #c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)            # <<<<<<<<<<< LSTM
        
        # x: (n, 28, 28), h0: (2, n, 128)
        
        # Forward propagate RNN
        #out, _ = self.rnn(x, h0)                                                            # <<<<<<<<<<< RNN
        # or:
        out, _ = self.gru(x, h0)                                                            # <<<<<<<<<<< GRU
        # or:
        #out, _ = self.lstm(x, (h0,c0))                                                       # <<<<<<<<<<< LSTM
        
        # out: tensor of shape (batch_size, seq_length, hidden_size)
        # out: (n, 28, 128)
        
        # Decode the hidden state of the last time step
        out = out[:, -1, :]
        # out: (n, 128)
         
        out = self.fc(out)
        # out: (n, 10)
        return out

model = RNN(input_size, hidden_size, num_layers, num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  

# Train the model
n_total_steps = len(train_loader)

start_time = time.time()

for epoch in range(num_epochs):
    
    total_loss=0
    
    for i, (images, labels) in enumerate(train_loader):  
        # origin shape: [N, 1, 50, 9]
        # resized: [N, 50, 9]

        images = images.reshape(-1, sequence_length, input_size).to(device)
        labels = labels.to(device)
        labels = labels.long()

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        #if (i+1) % 10 == 0:  #10
            #print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')
            #print (loss.item())
    
    print (total_loss)
    writer.writerow([ total_loss ])

print("--- %s seconds ---" % (time.time() - start_time))

# Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for images, labels in test_loader:

        images = images.reshape(-1, sequence_length, input_size).to(device)
        labels = labels.to(device)
        outputs = model(images)
        
        # max returns (value ,index)
        _, predicted = torch.max(outputs.data, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()

    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy of the network : {acc} %')

# Save the Model
torch.save(model.state_dict(), r'C:\ZOO_DATA\MYCODE\HAND\YOLOv5-livinglab\model_gru.pkl')