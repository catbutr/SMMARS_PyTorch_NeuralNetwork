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
print("Среднее значение после нормализации:", x_train.mean(axis=0)) # должно быть близкое к 0
print("Стандартное отклонение после нормализации:", x_train.std(axis=0)) # должно быть близкое к 1
train_dataset = TensorDataset(torch.from_numpy(x_train),torch.from_numpy(y_train.values))
test_dataset = TensorDataset(torch.from_numpy(x_test),torch.from_numpy(y_test.values))
# Загрузка данных
train_loader = DataLoader(train_dataset, 64)
test_loader = DataLoader(test_dataset, 64)
block = ffb.FeedForwardBlock(1,64,2)

# Тренировка
#trainedNN = block.train(5,block.model,block.criterion,block.optimizer,train_loader,device)

#plt.scatter(x_test, trainedNN, color='red', marker='x')
plt.scatter(x_test, y_test, color='blue', marker='o')
plt.title('Boston house prices')
plt.xlabel('LSTAT')
plt.ylabel('MEDV')
plt.legend(['Predicted', 'Actual'])
plt.show()

