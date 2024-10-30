import torch
import torch.nn as nn
import torch.optim as optim
import FeedForwardNN as ffnn

#Моноблок
class FeedForwardBlock():
    def __init__(self,inputSize=1, numberOfNeurons=1, numberOfLayers=1):
        super().__init__()
        self.model = ffnn.FeedForwardNN(inputSize, numberOfNeurons=numberOfNeurons,numberOfLayers=numberOfLayers, activationFunction=1)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.SGD(params=self.model.parameters(), lr=0.001)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #Тренировка
    def train(self,num_epochs, model, criterion, optimizer, train_dataloader, device):
        model.train()
        for epoch in range(num_epochs):
            for i, (images, labels) in enumerate(train_dataloader):  
                outputs = model(images)
                loss = criterion(outputs, labels)
                print('Epoch: ', epoch, 'Loss: ', loss.item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step() 
    
    # Валидация
    def evaluate(model,criterion, val_loader, device):
        model.eval()
        for val_input, val_targets in val_loader:
            val_input, val_targets = val_input.to(device), val_targets.to(device)
            out = model(val_input)
            val_loss = criterion(out, val_targets)
        return val_loss