import torch.nn as nn
import torch

class MyEnsemble(nn.Module):
    def __init__(self, model1, model2, model3):
        self.model1 = model1
        self.model2 = model2
        self.model3 = model3