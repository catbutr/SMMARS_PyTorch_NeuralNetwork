import torch.nn as nn
import torch
import ActivationFunctionEnum as af
import ChooseActivationFunction as caf

class FeedForwardNN(nn.Module):
    def __init__(self, inputSize, hiddenSize, outputSize, numberOfLayers, activationFunction = af.ActivationFunctionEnum.ReLU):
        super().__init__()
        stack = []
        halfstack = hiddenSize//2
        stack.append(nn.Linear(inputSize,hiddenSize))
        for i in range(0,numberOfLayers):
            if i%2 == 0:
                stack.append(nn.Linear(hiddenSize,halfstack, dtype=torch.float32))
            else:
                stack.append(nn.Linear(halfstack,hiddenSize, dtype=torch.float32))
            stack.append(caf.chooseActivationFunction(activationFunction))
        if numberOfLayers%2 != 0:
            stack.append(nn.Linear(halfstack,1))
        else:
            stack.append(nn.Linear(hiddenSize,outputSize))
        seq_stack = nn.Sequential(*stack)
        self.layers = nn.ModuleList(seq_stack)

    def forward(self, x):
        out = self.layers[0](x)
        for layer in self.layers[1:]:
            out = layer(out)
        return out