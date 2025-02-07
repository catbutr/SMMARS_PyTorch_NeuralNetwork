from ActivationFunctionEnum import ActivationFunctionEnum as activationFunction
#Создание кода для создания слоёв модели нейронной сети
class CustomModelScript():
    #script - текст кода
    def __init__(self):
        self.script = ""
    #Добавить линейный слой
    def addLinearLayer(self, inFeatures,outFeatures):
        self.isScriptEmpty()
        self.script = self.script + "stack.append(nn.Linear({},{}))".format(inFeatures,outFeatures)
    #Проверить пустой ли код. Если нет, добавить перенос строки 
    def isScriptEmpty(self):
        if len(self.script) != 0:
            self.script = self.script + "\n"
    #Добавить слой активации
    def addActivationLayer(self,function):
        self.isScriptEmpty()
        match function:
            case activationFunction.ReLU:
                self.script = self.script + "stack.append(nn.ReLU())"
            case activationFunction.LReLU:
                self.script = self.script + "stack.append(nn.LeakyReLU())"
            case activationFunction.Sigmoid:
                self.script = self.script + "stack.append(nn.Sigmoid())"
            case activationFunction.Tahn:
                self.script = self.script + "stack.append(nn.Tanh())"
            case activationFunction.Softmax:
                self.script = self.script + "stack.append(nn.Softmax())"
        