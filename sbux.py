import numpy as np
import pandas as pd
import matplotlib.pyplot  as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import torch #pytorch
import torch.nn as nn
from torch.autograd import Variable 
from sklearn.metrics import r2_score as r2s 
import time as time

class LSTM1(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length):
        super(LSTM1, self).__init__()
        self.num_classes = num_classes #number of classes
        self.num_layers = num_layers #number of layers
        self.input_size = input_size #input size
        self.hidden_size = hidden_size #hidden state
        self.seq_length = seq_length #sequence length

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True) #lstm
        self.fc_1 =  nn.Linear(hidden_size, 128) #fully connected 1
        self.fc = nn.Linear(128, num_classes) #fully connected last layer

        self.relu = nn.ReLU()
    
    def forward(self,x):
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)) #hidden state
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)) #internal state
        # Propagate input through LSTM
        _, (hn, _) = self.lstm(x, (h_0, c_0)) #lstm with input, hidden, and internal state
        hn = hn.view(-1, self.hidden_size) #reshaping the data for Dense layer next
        out = self.relu(hn)
        out = self.fc_1(out) #first Dense
        out = self.relu(out) #relu
        out = self.fc(out) #Final Output
        return out
    
class RNN1(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length):
        super(RNN1, self).__init__()
        self.num_classes = num_classes #number of classes
        self.num_layers = num_layers #number of layers
        self.input_size = input_size #input size
        self.hidden_size = hidden_size #hidden state
        self.seq_length = seq_length #sequence length

        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True) #lstm
        self.fc_1 =  nn.Linear(hidden_size, 128) #fully connected 1
        self.fc = nn.Linear(128, num_classes) #fully connected last layer
        self.relu = nn.ReLU()
    
    def forward(self,x):
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)) #hidden state
        # Propagate input through LSTM
        _, hn = self.rnn(x, h_0) #lstm with input, hidden, and internal state
        hn = hn.view(-1, self.hidden_size) #reshaping the data for Dense layer next
        out = self.relu(hn)
        out = self.fc_1(out) #first Dense
        out = self.relu(out) #relu
        out = self.fc(out) #Final Output
        return out
    
class GRU1(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length):
        super(GRU1, self).__init__()
        self.num_classes = num_classes #number of classes
        self.num_layers = num_layers #number of layers
        self.input_size = input_size #input size
        self.hidden_size = hidden_size #hidden state
        self.seq_length = seq_length #sequence length

        self.rnn = nn.GRU(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True) #lstm
        self.fc_1 =  nn.Linear(hidden_size, 128) #fully connected 1
        self.fc = nn.Linear(128, num_classes) #fully connected last layer
        self.relu = nn.ReLU()
    
    def forward(self,x):
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)) #hidden state
        # Propagate input through LSTM
        _, hn = self.rnn(x, h_0) #lstm with input, hidden, and internal state
        hn = hn.view(-1, self.hidden_size) #reshaping the data for Dense layer next
        out = self.relu(hn)
        out = self.fc_1(out) #first Dense
        out = self.relu(out) #relu
        out = self.fc(out) #Final Output
        return out

def train(model, criterion, optimizer, name):
   for epoch in range(num_epochs):
        outputs = model.forward(X_train_tensors_final) #forward pass
        optimizer.zero_grad() #caluclate the gradient, manually setting to 0
        
        # obtain the loss function
        loss = criterion(outputs, y_train_tensors)
        
        loss.backward() #calculates the loss of the loss function
        
        optimizer.step() #improve from loss, i.e backprop
        if epoch % 100 == 0:
            print("Epoch: %d, loss: %1.5f" % (epoch, loss.item())) 
        end = time.time()
def predict(model, train, mm):
    train_predict = model(train)#forward pass
    data_predict = train_predict.data.numpy() #numpy conversion
    data_predict = mm.inverse_transform(data_predict) #reverse transformation
    return data_predict

df = pd.read_csv("sbux.csv", index_col = "Date", parse_dates=True)
plt.style.use("ggplot")
X = df.iloc[:, :-1]
y = df.iloc[:, 5:6] 
mm = MinMaxScaler()
ss = StandardScaler()
X_ss = ss.fit_transform(X)
y_mm = mm.fit_transform(y) 
X_train = X_ss[:200, :]
X_test = X_ss[200:, :]

y_train = y_mm[:200, :]
y_test = y_mm[200:, :] 
X_train_tensors = Variable(torch.Tensor(X_train))
X_test_tensors = Variable(torch.Tensor(X_test))

y_train_tensors = Variable(torch.Tensor(y_train))
y_test_tensors = Variable(torch.Tensor(y_test)) 
#reshaping to rows, timestamps, features
X_train_tensors_final = torch.reshape(X_train_tensors,   (X_train_tensors.shape[0], 1, X_train_tensors.shape[1]))
X_test_tensors_final = torch.reshape(X_test_tensors,  (X_test_tensors.shape[0], 1, X_test_tensors.shape[1])) 

num_epochs = 1000 #1000 epochs
learning_rate = 0.001 #0.001 lr

input_size = 5 #number of features
hidden_size = 2 #number of features in hidden state
num_layers = 1 #number of stacked lstm layers

num_classes = 1 #number of output classes 
lstm1 = LSTM1(num_classes, input_size, hidden_size, num_layers, X_train_tensors_final.shape[1]) #our lstm class 
rnn1 = RNN1(num_classes, input_size, hidden_size, num_layers, X_train_tensors_final.shape[1])
gru1 = GRU1(num_classes, input_size, hidden_size, num_layers, X_train_tensors_final.shape[1])
criterion = torch.nn.MSELoss() 
optimizer_lstm = torch.optim.Adam(lstm1.parameters(), lr=learning_rate) 
optimizer_gru = torch.optim.Adam(gru1.parameters(), lr=learning_rate) 
optimizer_rnn = torch.optim.Adam(rnn1.parameters(), lr=learning_rate) 

start_lstm = time.time()
train(lstm1,criterion,optimizer_lstm, "LSTM")
end_lstm = time.time()
elapse_lstm = end_lstm - start_lstm
print("Время тренировки LSTM: " + str(elapse_lstm))

start_rnn = time.time()
train(rnn1,criterion,optimizer_rnn, "RNN")
end_rnn = time.time()
elapse_rnn = end_rnn - start_rnn

start_gru = time.time()
train(gru1,criterion,optimizer_gru, "GRU")
end_gru = time.time()
elapse_gru = end_gru - start_gru

df_X_ss = ss.transform(df.iloc[:, :-1]) #old transformers
df_y_mm = mm.transform(df.iloc[:, -1:]) #old transformers

df_X_ss = Variable(torch.Tensor(df_X_ss)) #converting to Tensors
df_y_mm = Variable(torch.Tensor(df_y_mm))
#reshaping the dataset
df_X_ss = torch.reshape(df_X_ss, (df_X_ss.shape[0], 1, df_X_ss.shape[1])) 

dataY_plot = df_y_mm.data.numpy()
dataY_lstm = predict(lstm1,df_X_ss,mm)
dataY_rnn = predict(rnn1,df_X_ss,mm)
dataY_gru= predict(gru1,df_X_ss,mm)
dataY_plot = mm.inverse_transform(dataY_plot)
lstm_r2 = r2s(dataY_plot,dataY_lstm)
rnn_r2 = r2s(dataY_plot,dataY_rnn)
gru_r2 = r2s(dataY_plot,dataY_gru)

print("Время тренировки LSTM: " + str(elapse_lstm))
print("Время тренировки RNN: " + str(elapse_rnn))
print("Время тренировки GRU: " + str(elapse_gru))

print("Точность LSTM: " + str(lstm_r2))
print("Точность RNN: " + str(rnn_r2))
print("Точность GRU: " + str(gru_r2))

plt.figure(figsize=(10,6)) #plotting
plt.axvline(x=200, c='r', linestyle='--') #size of the training set

plt.plot(dataY_plot, label='Actual Data') #actual plot
plt.plot(dataY_lstm, label='LSTM Predicted Data') 
plt.plot(dataY_gru, label='GRU Predicted Data') 
plt.plot(dataY_rnn, label='RNN Predicted Data') 
plt.title('Time-Series Prediction')
plt.legend()
plt.show() 