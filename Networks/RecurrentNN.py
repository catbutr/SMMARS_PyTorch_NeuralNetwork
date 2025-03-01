import torch.nn as nn
import Enums.ActivationFunctionEnum as af
import Functions.ChooseActivationFunction as caf
import torch
from torch.autograd import Variable 

class extract_tensor(nn.Module):
    def forward(self,x):
        tensor, _ = x
        return tensor[:, -1, :]

class ReccurentNN(nn.Module):
    def __init__(self, inputSize, hiddenSize, outputSize, numberOfLayers, activationFunction = af.ActivationFunctionEnum.Tahn):
        super().__init__()
        self.hidden_size = hiddenSize
        self.num_layers = numberOfLayers
        stack = []
        stack.append(nn.LSTM(input_size=inputSize,hidden_size=hiddenSize,num_layers=numberOfLayers, batch_first=True))
        stack.append(extract_tensor())
        stack.append(nn.Linear(in_features=inputSize,out_features=outputSize))
        stack.append(caf.chooseActivationFunction(activationFunction))
        seq_stack = nn.Sequential(*stack)
        self.layers = nn.ModuleList(seq_stack)

    def forward(self, x):
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)) #hidden state
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)) #internal state
        # Propagate input through LSTM
        _, (hn, _) = self.layers[0](x, (h_0, c_0)) #lstm with input, hidden, and internal state
        hn = hn.view(-1, self.hidden_size) #reshaping the data for Dense layer next
        out = self.layers[-1](hn)
        out = self.layers[1](x)
        for layer in self.layers[1:]:
            out = layer(out)
        return out