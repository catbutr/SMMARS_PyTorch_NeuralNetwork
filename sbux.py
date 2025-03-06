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
        self.num_classes = num_classes 
        self.num_layers = num_layers 
        self.input_size = input_size 
        self.hidden_size = hidden_size 
        self.seq_length = seq_length 

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True) 
        self.fc_1 =  nn.Linear(hidden_size, 128) 
        self.fc = nn.Linear(128, num_classes)

        self.relu = nn.ReLU()
    
    def forward(self,x):
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)) 
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)) 
        _, (hn, _) = self.lstm(x, (h_0, c_0)) 
        hn = hn.view(-1, self.hidden_size) 
        out = self.relu(hn)
        out = self.fc_1(out) 
        out = self.relu(out) 
        out = self.fc(out) 
        return out
    
class RNN1(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length):
        super(RNN1, self).__init__()
        self.num_classes = num_classes 
        self.num_layers = num_layers 
        self.input_size = input_size 
        self.hidden_size = hidden_size 
        self.seq_length = seq_length 

        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True) 
        self.fc_1 =  nn.Linear(hidden_size, 128) 
        self.fc = nn.Linear(128, num_classes) 
        self.relu = nn.ReLU()
    
    def forward(self,x):
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)) 
        _, hn = self.rnn(x, h_0)
        hn = hn.view(-1, self.hidden_size) 
        out = self.relu(hn)
        out = self.fc_1(out) 
        out = self.relu(out) 
        out = self.fc(out) 
        return out
    
class GRU1(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length):
        super(GRU1, self).__init__()
        self.num_classes = num_classes 
        self.num_layers = num_layers 
        self.input_size = input_size 
        self.hidden_size = hidden_size 
        self.seq_length = seq_length 

        self.rnn = nn.GRU(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True) 
        self.fc_1 =  nn.Linear(hidden_size, 128) 
        self.fc = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
    
    def forward(self,x):
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)) 

        _, hn = self.rnn(x, h_0) 
        hn = hn.view(-1, self.hidden_size) 
        out = self.relu(hn)
        out = self.fc_1(out) 
        out = self.relu(out) 
        out = self.fc(out) 
        return out

def train(model, criterion, optimizer, name):
   for epoch in range(num_epochs):
        outputs = model.forward(X_train_tensors_final) 
        optimizer.zero_grad()

        loss = criterion(outputs, y_train_tensors)
        
        loss.backward() 
        
        optimizer.step() 
        if epoch % 100 == 0:
            print("Epoch: %d, loss: %1.5f" % (epoch, loss.item())) 
        end = time.time()
def predict(model, train, mm):
    train_predict = model(train)
    data_predict = train_predict.data.numpy() 
    data_predict = mm.inverse_transform(data_predict) 
    return data_predict

df = pd.read_csv("sbux.csv", index_col = "Date", parse_dates=True)
plt.style.use("ggplot")
X = df.iloc[:, :-1]
y = df.iloc[:, 5:6] 
mm = MinMaxScaler()
ss = StandardScaler()
X_ss = ss.fit_transform(X)
y_mm = mm.fit_transform(y) 
X_train = X_ss[:180, :]
X_test = X_ss[180:, :]

y_train = y_mm[:180, :]
y_test = y_mm[180:, :] 
X_train_tensors = Variable(torch.Tensor(X_train))
X_test_tensors = Variable(torch.Tensor(X_test))

y_train_tensors = Variable(torch.Tensor(y_train))
y_test_tensors = Variable(torch.Tensor(y_test)) 

X_train_tensors_final = torch.reshape(X_train_tensors,   (X_train_tensors.shape[0], 1, X_train_tensors.shape[1]))
X_test_tensors_final = torch.reshape(X_test_tensors,  (X_test_tensors.shape[0], 1, X_test_tensors.shape[1])) 

num_epochs = 5000 
learning_rate = 0.001 

input_size = 5 
hidden_size = 2 
num_layers = 1 

num_classes = 1 
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

df_X_ss = ss.transform(df.iloc[:, :-1]) 
df_y_mm = mm.transform(df.iloc[:, -1:]) 

df_X_ss = Variable(torch.Tensor(df_X_ss))
df_y_mm = Variable(torch.Tensor(df_y_mm))
df_X_ss = torch.reshape(df_X_ss, (df_X_ss.shape[0], 1, df_X_ss.shape[1])) 

dataY_plot = df_y_mm.data.numpy()
dataY_lstm = predict(lstm1,df_X_ss,mm)
dataY_rnn = predict(rnn1,df_X_ss,mm)
dataY_gru= predict(gru1,df_X_ss,mm)
dataY_plot = mm.inverse_transform(dataY_plot)
lstm_r2 = r2s(dataY_plot,dataY_lstm)
rnn_r2 = r2s(dataY_plot,dataY_rnn)
gru_r2 = r2s(dataY_plot,dataY_gru)

print("Скорость: " + str('%.3f'%elapse_lstm) + "    " + str('%.3f'%elapse_rnn) + "  " + str('%.3f'%elapse_gru))

print("Точность: " + str('%.3f'%lstm_r2) + "    " + str('%.3f'%rnn_r2) + "  " + str('%.3f'%gru_r2))
plt.figure(figsize=(10,6)) #plotting
plt.axvline(x=180, c='r', linestyle='--') #size of the training set

plt.plot(dataY_plot, label='Actual Data') #actual plot
plt.plot(dataY_lstm, label='LSTM Predicted Data') 
plt.plot(dataY_gru, label='GRU Predicted Data') 
plt.plot(dataY_rnn, label='RNN Predicted Data') 
plt.title('Time-Series Prediction')
plt.legend()
plt.show() 