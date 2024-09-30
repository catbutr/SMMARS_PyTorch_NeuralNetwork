import torch
import torch.nn as nn
def chooseActivationFunction(activationFunction):
    match activationFunction:
        case 1:
            return nn.ReLU()
        case 2:
            return nn.LeakyReLU()
        case 3:
            return nn.Sigmoid()
        case 4:
            return nn.Tanh()
        case 5:
            return nn.Softmax()