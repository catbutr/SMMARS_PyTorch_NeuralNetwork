import torch.nn as nn
class CustomNN(nn.Module):
    def __init__(self, exec_script):
        super().__init__()
        stack = []
        exec(exec_script)
        seq_stack = nn.Sequential(*stack)
        self.layers = nn.ModuleList(seq_stack)

    def forward(self, x):
        out = self.layers[0](x)
        for layer in self.layers[1:]:
            out = layer(out)
        return out