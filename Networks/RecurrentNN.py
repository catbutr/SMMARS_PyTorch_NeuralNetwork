import torch.nn as nn
import Enums.ActivationFunctionEnum as af
import Functions.ChooseActivationFunction as caf

class extract_tensor(nn.Module):
    def forward(self,x):
        tensor, _ = x
        return tensor[:, -1, :]

class ReccurentNN(nn.Module):
    def __init__(self, inputSize, hiddenSize, outputSize, numberOfLayers, activationFunction = af.ActivationFunctionEnum.Tahn):
        super().__init__()
        stack = []
        stack.append(nn.LSTM(input_size=inputSize,hidden_size=hiddenSize,num_layers=numberOfLayers, batch_first=True))
        stack.append(extract_tensor())
        stack.append(nn.Linear(in_features=inputSize,out_features=outputSize))
        stack.append(caf.chooseActivationFunction(activationFunction))
        seq_stack = nn.Sequential(*stack)
        self.layers = nn.ModuleList(seq_stack)

    def forward(self, x):
        out = self.layers[0](x)
        for layer in self.layers[1:]:
            out = layer(out)
        return out