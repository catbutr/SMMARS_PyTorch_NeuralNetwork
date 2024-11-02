import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import torch
import FeedForwardBlock as ffb
import TorchDataset as td

# Проверка устройства
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Загрузка данных
data = pd.read_csv('boston.csv')
x = data[['LSTAT']]
y = data['MEDV']

# Разделение данных на обучающую и тестовую выборки
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42) 

# Нормализация данных
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test) 
#print("Среднее значение после нормализации:", x_train.mean(axis=0)) # должно быть близкое к 0
#print("Стандартное отклонение после нормализации:", x_train.std(axis=0)) # должно быть близкое к 1
train_dataset = TensorDataset(torch.from_numpy(x_train).type(torch.float),torch.from_numpy(y_train.values).type(torch.float))
test_dataset = TensorDataset(torch.from_numpy(x_test).type(torch.float),torch.from_numpy(y_test.values).type(torch.float))
# Загрузка данных
train_loader = DataLoader(train_dataset,256)
test_loader = DataLoader(test_dataset,256)
block = ffb.FeedForwardBlock(inputSize=1,hiddenSize=512,outputSize=1,numberOfLayers=2, optimizer=torch.optim.Adam(lr=0.001))

#Тренировка
block.train(100,block.model,block.criterion,block.optimizer,train_loader,device)
predict_y = block.model(torch.from_numpy(x_test).type(torch.float))
plt.scatter(x_test, predict_y.detach().numpy(), color='red', marker='x')
plt.scatter(x_test, y_test, color='blue', marker='o')
plt.title('Boston house prices')
plt.xlabel('LSTAT')
plt.ylabel('MEDV')
plt.legend(['Predicted', 'Actual'])
plt.show()