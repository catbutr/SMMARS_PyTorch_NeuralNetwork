import torch.nn as nn
import ActivationFunctionEnum as af
import ChooseActivationFunction as caf

class FeedForwardNN(nn.Module):
    def __init__(self, inputSize, numberOfNeurons, numberOfLayers, activationFunction = af.ActivationFunctionEnum.ReLU):
        super().__init__()
        stack = []
        stack.append(nn.Linear(inputSize,numberOfNeurons))
        for i in range(0,numberOfLayers):
            stack.append(nn.Linear(numberOfNeurons,numberOfNeurons))
            stack.append(caf.chooseActivationFunction(activationFunction))
        stack.append(nn.Linear(numberOfNeurons,1))
        seq_stack = nn.Sequential(*stack)
        self.layers = nn.ModuleList([nn.Flatten(), seq_stack])

    def forward(self, x):
        x = self.layers[0](x)
        out = self.layers[1](x)
        return out