import torch.nn as nn
import torch
import ActivationFunctionEnum as af
import ChooseActivationFunction as caf

class FeedForwardNN(nn.Module):
    def __init__(self, inputSize, numberOfNeurons, numberOfLayers, activationFunction = af.ActivationFunctionEnum.ReLU):
        super().__init__()
        stack = []
        halfstack = numberOfNeurons//2
        stack.append(nn.Linear(inputSize,numberOfNeurons))
        for i in range(0,numberOfLayers):
            if i%2 == 0:
                stack.append(nn.Linear(numberOfNeurons,halfstack, dtype=torch.float32))
            else:
                stack.append(nn.Linear(halfstack,numberOfNeurons, dtype=torch.float32))
            stack.append(caf.chooseActivationFunction(activationFunction))
        if numberOfLayers%2 != 0:
            stack.append(nn.Linear(halfstack,1))
        else:
            stack.append(nn.Linear(numberOfNeurons,1))
        seq_stack = nn.Sequential(*stack)
        self.layers = nn.ModuleList(seq_stack)

    def forward(self, x):
        for layer in self.layers:
            print(layer)
            x = layer(x)
        return x