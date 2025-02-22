import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler
from torchsummary import summary
import torch
import seaborn as sns
import torch.nn as nn
import torch.optim as optim
import Blocks.FeedForwardBlock as ffb
from Blocks.NeuralNetworkBlock import NeuralNetworkBlock as NNB
from torch.autograd import Variable 


class LSTMModel(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length):
        super(LSTMModel, self).__init__()
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
        output, (hn, cn) = self.lstm(x, (h_0, c_0)) #lstm with input, hidden, and internal state
        hn = hn.view(-1, self.hidden_size) #reshaping the data for Dense layer next
        out = self.relu(hn)
        out = self.fc_1(out) #first Dense
        out = self.relu(out) #relu
        out = self.fc(out) #Final Output
        return out

#Моноблок
class LSTMBlock(NNB):
    def __init__(self,inputSize=1, hiddenSize=1, numberOfLayers=1,criterion=nn.MSELoss(), numClasses=1,seqLen=1):
        super().__init__(criterion)
        self.model = LSTMModel(num_classes=numClasses,hidden_size=hiddenSize, input_size=inputSize, num_layers=numberOfLayers,seq_length=seqLen)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.optimizer = optim.Adam(params=self.model.parameters())
        self.criterion = criterion

    #Тренировка
    def train(self,num_epochs, train_dataloader):
        self.model.train()
        for epoch in range(num_epochs): # 20 epochs at maximum
            # Print epoch
            print(f'Starting epoch {epoch+1}')
            # Set current loss value
            current_loss = 0.0
            for i, data in enumerate(train_dataloader, 0):
        
                # Get and prepare inputs
                inputs, targets = data
                inputs, targets = inputs.float(), targets.float()
                targets = targets.reshape((targets.shape[0], 1))

                # Zero the gradients
                self.optimizer.zero_grad()

                # Perform forward pass
                outputs = self.model(inputs)

                # Compute loss
                loss = self.criterion(outputs, targets)

                # Perform backward pass
                loss.backward()

                # Perform optimization
                self.optimizer.step()

                # Print statistics
                current_loss += loss.item()
                if i % 10 == 0:
                    print('Loss after mini-batch %5d: %.3f' %
                        (i + 1, current_loss / 500))
                    current_loss = 0.0
    
    # Валидация
    def evaluate(self, val_loader):
        self.model.eval()
        for val_input, val_targets in val_loader:
            val_input, val_targets = val_input.to(self.device), val_targets.to(self.device)
            out = self.model(val_input)
            val_loss = self.criterion(out, val_targets)
        return val_loss


class CommonDataset(torch.utils.data.Dataset):
  '''
  Prepare the Boston dataset for regression
  '''

  def __init__(self, X, y, scale_data=False):
    if not torch.is_tensor(X) and not torch.is_tensor(y):
      if scale_data:
          X = StandardScaler().fit_transform(X)
      self.X = torch.from_numpy(X)
      self.y = torch.from_numpy(y)

  def __len__(self):
      return len(self.X)

  def __getitem__(self, i):
      return self.X[i], self.y[i]

# Проверка устройства
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Загрузка данных
data = sns.load_dataset("dowjones")
print(data)
# Нормализация данных
scaler = MinMaxScaler()
for col in data.columns:
    data[col] = StandardScaler().fit_transform(data[col].to_numpy().reshape(-1,1))
data_scaled = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)
feature_list = ['Price'] # ---Сюда можно дописывать входные признаки
inputs_count = len(feature_list) #это число определяет количество входов в нейронной сети
x = data_scaled[feature_list].to_numpy()
y = data_scaled['Date'].to_numpy()
# Разделение данных на обучающую и тестовую выборки
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42) 
#print("Среднее значение после нормализации:", x_train.mean(axis=0)) # должно быть близкое к 0
#print("Стандартное отклонение после нормализации:", x_train.std(axis=0)) # должно быть близкое к 1
#train_dataset = TensorDataset(torch.from_numpy(x_train).type(torch.float),torch.from_numpy(y_train.values).type(torch.float))
train_dataset = CommonDataset(x_train, y_train)
# Загрузка данных
train_loader = DataLoader(train_dataset,batch_size=10, shuffle=True)
block = LSTMBlock(inputSize=1,hiddenSize=256,numberOfLayers=1, criterion=torch.nn.MSELoss())
block.optimizer = torch.optim.Adam(params=block.model.parameters(),lr=0.001)
#summary(block.model, input_size=(1, 128, 1))

# #Тренировка
block.train(100,train_loader)
predict_y = block.model(torch.from_numpy(x_test).type(torch.float))
x_test = pd.DataFrame(x_test, columns=[feature_list])
print(block.model.parameters)

plt.scatter(x_test['Price'], predict_y.detach().numpy(), color='red', marker='x')
plt.scatter(x_test['Price'], y_test, color='blue', marker='o')
plt.title('Test')
plt.xlabel('Price')
plt.ylabel('Date')
plt.legend(['Predicted', 'Actual'])
plt.show()