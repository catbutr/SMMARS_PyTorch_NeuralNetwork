import torch
import torch.nn as nn
import abc

#Абстрактный класс моноблока
class NeuralNetworkBlock(abc.ABC):
    def __init__(self, criterion=nn.MSELoss()):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.criterion = criterion

    #Тренировка
    @abc.abstractmethod
    def train(self): pass
    
    @abc.abstractmethod
    # Валидация
    def evaluate(self):pass