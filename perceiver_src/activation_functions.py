import torch 
import torch.nn as nn 
import torch.nn.functional as F 




class mish(nn.Module):
    """
    activation funciton
    Mish: A Self Regularized Non-Monotonic Ativation Function (https://arxiv.org/abs/1908.08681)
    """
    def __init__(self):
        super().__init__()
    
    def forward(self,x):
        return x * torch.tanh(F.softplus(x))
        