import torch
import torch.nn as nn
import torchvision.transforms as transforms
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
import numpy as np
from numpy import zeros, newaxis
class BostonDataset(torch.utils.data.Dataset):
  '''
  Prepare the Boston dataset for regression
  '''

  def __init__(self, X, y, scale_data=False):
    if not torch.is_tensor(X) and not torch.is_tensor(y):
      if scale_data:
          X = StandardScaler().fit_transform(X)
      self.X = torch.from_numpy(X).type(torch.float)
      self.y = torch.from_numpy(y).type(torch.LongTensor)

  def __len__(self):
      return len(self.X)
 
  def __getitem__(self, i):
      return self.X[i], self.y[i]

class ConvNN(nn.Module):
    def __init__(self):
         super(ConvNN, self).__init__() 
         self.layer1 = nn.Sequential( nn.Conv1d(10, 32, kernel_size=5, stride=1, padding=2), 
            nn.ReLU()) 
         self.layer2 = nn.Sequential( nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2), 
            nn.ReLU()) 
         self.drop_out = nn.Dropout() 
         self.fc1 = nn.Linear(4, 10) 
         self.fc2 = nn.Linear(10, 1)

    def forward(self, x):
      out = self.layer1(x) 
      out = self.layer2(out) 
      out = out.reshape(out.size(0), -1) 
      out = self.drop_out(out) 
      out = self.fc1(out) 
      out = self.fc2(out) 
      return out

num_epochs = 5 
num_classes = 4 
batch_size = 100 
learning_rate = 0.001
trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]) 
predictors = pd.read_excel("Training Data_predictors.xlsx")
response = pd.read_excel("Training Data_response.xlsx")
data = pd.concat([predictors,response], axis=1)
# Нормализация данных
scaler = MinMaxScaler()
feature_list = ['a','c','P','c/a'] # ---Сюда можно дописывать входные признаки
inputs_count = len(feature_list) #это число определяет количество входов в нейронной сети
x = data[feature_list].to_numpy()
y = data['Kc'].to_numpy()
# Разделение данных на обучающую и тестовую выборки
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42) 
#x_train = x_train[:, :, newaxis]
print(np.shape(x_train))
#print("Среднее значение после нормализации:", x_train.mean(axis=0)) # должно быть близкое к 0
#print("Стандартное отклонение после нормализации:", x_train.std(axis=0)) # должно быть близкое к 1
#train_dataset = TensorDataset(torch.from_numpy(x_train).type(torch.float),torch.from_numpy(y_train.values).type(torch.float))
train_dataset = BostonDataset(x_train, y_train)
test_dataset = BostonDataset(x_test,y_test)
# Загрузка данных
train_loader = DataLoader(train_dataset,batch_size=10, shuffle=True)
test_loader = DataLoader(test_dataset,batch_size=10, shuffle=True)

model = ConvNN()
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
total_step = len(train_loader)
loss_list = []
acc_list = []
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # Прямой запуск
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss_list.append(loss.item())

        # Обратное распространение и оптимизатор
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Отслеживание точности
        total = labels.size(0)
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == labels).sum().item()
        acc_list.append(correct / total)

        if (i + 1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(),
                          (correct / total) * 100))