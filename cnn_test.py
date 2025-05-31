from tensorflow import keras
from sklearn.model_selection import train_test_split
from torch import nn
import torch
from torch.optim import SGD, RMSprop, Adam

(X_train, Y_train), (X_test, Y_test) = keras.datasets.fashion_mnist.load_data()

X_train, X_test, Y_train, Y_test = torch.tensor(X_train, dtype=torch.float32),\
                                   torch.tensor(X_test, dtype=torch.float32),\
                                   torch.tensor(Y_train, dtype=torch.long),\
                                   torch.tensor(Y_test, dtype=torch.long)

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3,3), padding="same"),
            nn.ReLU(),

            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(3,3), padding="same"),
            nn.ReLU(),

            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=(3,3), padding="same"),
            nn.ReLU(),

            nn.Flatten(),
            nn.Linear(8*28*28, len(classes)),
            #nn.Softmax(dim=1)            
        )

    def forward(self, x_batch):
        preds = self.seq(x_batch)
        return preds

def TrainModelInBatches(model, loss_func, optimizer, X, Y, batch_size=32, epochs=5):
    for i in range(epochs):
        batches = torch.arange((X.shape[0]//batch_size)+1) ### Batch Indices

        losses = [] ## Record loss of each batch
        for batch in batches:
            if batch != batches[-1]:
                start, end = int(batch*batch_size), int(batch*batch_size+batch_size)
            else:
                start, end = int(batch*batch_size), None

            X_batch, Y_batch = X[start:end], Y[start:end] ## Single batch of data

            preds = model(X_batch) ## Make Predictions by forward pass through network

            loss = loss_func(preds, Y_batch) ## Calculate Loss
            losses.append(loss) ## Record Loss

            optimizer.zero_grad() ## Zero weights before calculating gradients
            loss.backward() ## Calculate Gradients
            optimizer.step() ## Update Weights

        print("Categorical Cross Entropy : {:.3f}".format(torch.tensor(losses).mean()))

def MakePredictions(model, input_data, batch_size=32):
    batches = torch.arange((input_data.shape[0]//batch_size)+1) ### Batch Indices

    with torch.no_grad(): ## Disables automatic gradients calculations
        preds = []
        for batch in batches:
            if batch != batches[-1]:
                start, end = int(batch*batch_size), int(batch*batch_size+batch_size)
            else:
                start, end = int(batch*batch_size), None

            X_batch = input_data[start:end]

            preds.append(model(X_batch))

    return preds

X_train, X_test = X_train.reshape(-1,1,28,28), X_test.reshape(-1,1,28,28)

X_train, X_test = X_train/255.0, X_test/255.0

classes =  Y_train.unique()

X_train.shape, X_test.shape, Y_train.shape, Y_test.shape
    
conv_net = ConvNet()

print(conv_net)

preds = conv_net(X_train[:5])

print(preds.shape)

loss = nn.CrossEntropyLoss()

print(loss(preds, Y_train[:5]))

torch.manual_seed(42) ##For reproducibility.This will make sure that same random weights are initialized each time.

epochs = 25
learning_rate = torch.tensor(1/1e3) # 0.001
batch_size=128

conv_net = ConvNet()
cross_entropy_loss = nn.CrossEntropyLoss()
optimizer = SGD(params=conv_net.parameters(), lr=learning_rate)

TrainModelInBatches(conv_net,
                    cross_entropy_loss,
                    optimizer,
                    X_train, Y_train,
                    batch_size=batch_size,
                    epochs=epochs)

test_preds = MakePredictions(conv_net, X_test, batch_size=128) ## Make Predictions on test dataset

test_preds = torch.cat(test_preds) ## Combine predictions of all batches

test_preds = test_preds.argmax(dim=1)

train_preds = MakePredictions(conv_net, X_train, batch_size=128) ## Make Predictions on train dataset

train_preds = torch.cat(train_preds)

train_preds = train_preds.argmax(dim=1)

print(test_preds[:5], train_preds[:5])
print(Y_test[:5], Y_train[:5])