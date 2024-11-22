import torch
import torch.nn as nn
import torch.optim as optim
import FeedForwardNN as ffnn

#Моноблок
class FeedForwardBlock():
    def __init__(self,inputSize=1, hiddenSize=1, outputSize=1, numberOfLayers=1,criterion=nn.MSELoss()):
        super().__init__()
        self.model = ffnn.FeedForwardNN(inputSize=inputSize, hiddenSize=hiddenSize, outputSize=outputSize ,numberOfLayers=numberOfLayers, activationFunction=1)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.optimizer = optim.Adam(params=self.model.parameters())
        self.criterion = criterion

    #Тренировка
    def train(self,num_epochs, model, criterion, optimizer, train_dataloader, device):
        model.train()
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
                current_loss += loss.item()
                if i % 10 == 0:
                    print('Loss after mini-batch %5d: %.3f' %
                        (i + 1, current_loss / 500))
                    current_loss = 0.0
    
    # Валидация
    def evaluate(model,criterion, val_loader, device):
        model.eval()
        for val_input, val_targets in val_loader:
            val_input, val_targets = val_input.to(device), val_targets.to(device)
            out = model(val_input)
            val_loss = criterion(out, val_targets)
        return val_loss