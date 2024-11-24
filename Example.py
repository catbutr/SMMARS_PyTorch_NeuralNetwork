# example is based on https://github.com/christianversloot/machine-learning-articles/blob/main/how-to-create-a-neural-network-for-regression-with-pytorch.md
import torch
from torch import nn
from torch.utils.data import DataLoader
# from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

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
      

class MLP(nn.Module):
  '''
    Multilayer Perceptron for regression.
  '''
  def __init__(self):
    global inputs_count #так делать (глобально) не рекомендуется. Но, если очень хочется...
    super().__init__()
    self.layers = nn.Sequential(
      nn.Linear(inputs_count, 64),#определение количества входов
      nn.ReLU(),
      nn.Linear(64, 32),
      nn.ReLU(),
      nn.Linear(32, 1)
    )


  def forward(self, x):
    '''
      Forward pass
    '''
    return self.layers(x)

  
if __name__ == '__main__':

    # Set fixed random number seed
    torch.manual_seed(42)

    # Load Boston dataset

    data = pd.read_csv('boston.csv')

    #   X, y = load_boston(return_X_y=True)
    # Нормализация данных
    scaler = MinMaxScaler()
    data_scaled = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)
    feature_list = ['LSTAT','RM'] # ---Сюда можно дописывать входные признаки
    inputs_count = len(feature_list) #это число определяет количество входов в нейронной сети
    print(inputs_count)
    x = data_scaled[feature_list].to_numpy()
    y = data_scaled['MEDV'].to_numpy()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42) 
    # Prepare Boston dataset
    dataset = BostonDataset(x_train, y_train)
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=True, num_workers=1)

    # Initialize the MLP
    mlp = MLP()

    # Define the loss function and optimizer
    loss_function = nn.L1Loss()
    optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-4)

    # Run the training loop
    for epoch in range(0, 20): # 20 epochs at maximum

        # Print epoch
        print(f'Starting epoch {epoch+1}')

        # Set current loss value
        current_loss = 0.0

        # Iterate over the DataLoader for training data
        for i, data in enumerate(trainloader, 0):
      
            # Get and prepare inputs
            inputs, targets = data
            inputs, targets = inputs.float(), targets.float()
            targets = targets.reshape((targets.shape[0], 1))

            # Zero the gradients
            optimizer.zero_grad()

            # Perform forward pass
            outputs = mlp(inputs)

            # Compute loss
            loss = loss_function(outputs, targets)

            # Perform backward pass
            loss.backward()

            # Perform optimization
            optimizer.step()

            # Print statistics
            current_loss += loss.item()
            if i % 10 == 0:
                print('Loss after mini-batch %5d: %.3f' %
                    (i + 1, current_loss / 500))
                current_loss = 0.0
    # Process is complete.
    print('Training process has finished.')
    predict_y = mlp(torch.from_numpy(x_test).type(torch.float))
    x_test = pd.DataFrame(x_test, columns=[feature_list])
    plt.scatter(x_test['LSTAT'], predict_y.detach().numpy(), color='red', marker='x')
    plt.scatter(x_test['LSTAT'], y_test, color='blue', marker='o')
    plt.title('Boston house prices')
    plt.xlabel('LSTAT')
    plt.ylabel('MEDV')
    plt.legend(['Predicted', 'Actual'])
    plt.show()