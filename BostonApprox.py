import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from torch.utils.data import DataLoader
import TorchDataset as TD
import torch
import FeedForwardBlock as ffb

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
print("Среднее значение после нормализации:", x_train.mean(axis=0)) # должно быть близкое к 0
print("Стандартное отклонение после нормализации:", x_train.std(axis=0)) # должно быть близкое к 1

x_train = torch.tensor(x_train, dtype=torch.float32)
x_test = torch.tensor(x_test, dtype=torch.float32)
# Загрузка данных

train_loader = DataLoader(x_train, 64)
test_loader = DataLoader(x_test, 64)

block = ffb.FeedForwardBlock(1,64,2)

trainedNN = block.fit(500,block.model,block.criterion,block.optimizer,train_loader,test_loader,device)

# plt.scatter(x_test, trainedNN, color='red', marker='x')
plt.scatter(x_test, y_test, color='blue', marker='o')
plt.title('Boston house prices')
plt.xlabel('LSTAT')
plt.ylabel('MEDV')
plt.legend(['Predicted', 'Actual'])
plt.show()