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
class CNNDataset(torch.utils.data.Dataset):

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
         self.layer1 = nn.Sequential( nn.Conv1d(4, 64, kernel_size=5, stride=1, padding=2), 
            nn.ReLU(), nn.MaxPool1d(1,1)) 
         self.layer2 = nn.Sequential( nn.Conv1d(64, 32, kernel_size=5, stride=1, padding=2), 
            nn.ReLU(), nn.MaxPool1d(1,1)) 
         self.drop_out = nn.Dropout() 
         self.out = nn.Linear(32, 1) 

    def forward(self, x):
      out = self.layer1(x) 
      out = self.layer2(out) 
      out = out.reshape(out.size(0), -1) 
      out = self.drop_out(out) 
      out = self.out(out) 
      return out

num_epochs = 5 

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
x_train = x_train[:, :, newaxis]
x_test = x_test[:, :, newaxis]
print(np.shape(x_train))
train_dataset = CNNDataset(x_train, y_train)
test_dataset = CNNDataset(x_test,y_test)
# Загрузка данных
train_loader = DataLoader(train_dataset,batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset,batch_size=batch_size, shuffle=True)

model = ConvNN()
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
for epoch in range(num_epochs): # 20 epochs at maximum
    # Print epoch
    print(f'Starting epoch {epoch+1}')
    # Set current loss value
    current_loss = 0.0
    for i, data in enumerate(train_loader, 0):
    # Get and prepare inputs
        inputs, targets = data
        inputs, targets = inputs.float(), targets.float()
        targets = targets.reshape((targets.shape[0], 1))
        # Zero the gradients
        optimizer.zero_grad()
        # Perform forward pass
        outputs = model(inputs)

        # Compute loss
        loss = criterion(outputs, targets)
        # Perform backward pass
        loss.backward()
        # Perform optimization
        optimizer.step()
        # Print statistics
        # current_loss += loss.item()
        if i % 10 == 0:
            print('Loss after mini-batch %5d: %.3f' %
                  (i + 1, current_loss / 500))
            current_loss = 0.0

model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Точность: {} %'.format((correct / total) * 100))
