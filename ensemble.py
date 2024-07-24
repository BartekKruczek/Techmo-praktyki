import torch.nn as nn
import torch

class MyEnsemble(nn.Module):
    def __init__(self, model1, model2, model3):
        self.model1 = model1
        self.model2 = model2
        self.model3 = model3

    def forward(self, x1, x2, x3):
        x1 = self.model1(x1)
        x2 = self.model2(x2)
        x3 = self.model3(x3)

        avg_out = torch.stack([x1, x2, x3]).mean(dim=0)

        return avg_out