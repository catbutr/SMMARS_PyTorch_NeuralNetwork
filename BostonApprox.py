import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler
import torch
import Blocks.FeedForwardBlock as ffb
import Blocks.RecurrentBlock as rb
from torch.utils.data import TensorDataset

class BostonDataset(torch.utils.data.Dataset):
  '''
  Prepare the Boston dataset for regression
  '''

  def __init__(self, X, y, scale_data=False):
    if not torch.is_tensor(X) and not torch.is_tensor(y):
      # Apply scaling if necessary
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
data = pd.read_csv('boston.csv')
# Нормализация данных
scaler = MinMaxScaler()
data_scaled = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)
feature_list = ['LSTAT'] # ---Сюда можно дописывать входные признаки
inputs_count = len(feature_list) #это число определяет количество входов в нейронной сети
x = data_scaled[feature_list].to_numpy()
print(x.size)
y = data_scaled['MEDV'].to_numpy()
print(y.size)
# Разделение данных на обучающую и тестовую выборки
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42) 
x_reshaped = torch.reshape(torch.from_numpy(x_train), (torch.from_numpy(x_train).shape[0], 1, torch.from_numpy(x_train).shape[1])) 
print(type(x_train))
print(type(x_reshaped))
#print("Среднее значение после нормализации:", x_train.mean(axis=0)) # должно быть близкое к 0
#print("Стандартное отклонение после нормализации:", x_train.std(axis=0)) # должно быть близкое к 1
#train_dataset = TensorDataset(torch.from_numpy(x_train).type(torch.float),torch.from_numpy(y_train.values).type(torch.float))
train_dataset = BostonDataset(x_reshaped.numpy(), y_train)
# Загрузка данных
train_loader = DataLoader(train_dataset,batch_size=10, shuffle=True)
block = rb.RecurrentBlock(inputSize=1,hiddenSize=128,outputSize=1,numberOfLayers=1, criterion=torch.nn.MSELoss())
block.optimizer = torch.optim.Adam(params=block.model.parameters(),lr=0.001)
# #Тренировка
block.train(100,train_loader)
# Предугадывание
predict_y = block.model(torch.from_numpy(x_test).type(torch.float))
accuracy = r2_score(y_test,predict_y.detach().numpy())
print(accuracy)
x_test = pd.DataFrame(x_test, columns=[feature_list])
print(block.model.parameters)
print(np.shape(predict_y.detach().numpy()))
print(np.shape(y_test))
plt.scatter(x_test['LSTAT'], predict_y.detach().numpy(), color='red', marker='x')
plt.scatter(x_test['LSTAT'], y_test, color='blue', marker='o')
plt.title('Boston house prices')
plt.xlabel('LSTAT')
plt.ylabel('MEDV')
plt.legend(['Predicted', 'Actual'])
plt.show()