import torch
import torch.nn as nn
import torch.optim as optim
import FeedForwardNN as ffnn


class FeedForwardBlock():
    def __init__(self,inputSize=1, outputSize=1, numberOfLayers=1):
        super().__init__()
        self.model = ffnn.FeedForwardNN(inputSize, numberOfNeurons=outputSize,numberOfLayers=numberOfLayers)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.SGD(params=self.model.parameters(), lr=0.001)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def fit(self,num_epochs, model, loss_fn, optimizer, train_dataloader, val_loader, device):
        model.train()
        for epoch in range(num_epochs):
            for inputs, targets in zip(train_dataloader.dataset ,val_loader.dataset):
                inputs, targets = inputs.to(device), targets.to(device)
                # Get predictions.
                preds = model(inputs)
                # Get loss.
                loss = loss_fn(preds, targets)
                # Compute gradients.
                loss.backward()
                # Update model parameters i.e. backpropagation.
                optimizer.step()
                # Reset gradients to zero before the next epoch.
                optimizer.zero_grad()
            if (epoch + 1) % 50 == 0:
                # Get validation loss as well.
                for val_input, val_targets in val_loader:
                    val_input, val_targets = val_input.to(device), val_targets.to(device)
                    out = model(val_input)
                    val_loss = nn.MSELoss(out, val_targets)
                print("Epoch [{}/{}], Training loss: {:.4f}, Validation Loss: {:.4f}".format(epoch + 1, num_epochs, loss.item(), val_loss)) # Report loss value after each epoch.
    
    def evaluate(model, criterion, X, y):
        model.eval()
        with torch.no_grad():
            outputs = model(X)
            loss = criterion(outputs, y)
            rmse = torch.sqrt(loss)
            mae = torch.mean(torch.abs(outputs - y))
        return loss.item(), rmse.item(), mae.item()