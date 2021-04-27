import pandas as pd 
import torch 
import torch.nn as nn  
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader



class dataReader(Dataset):
    def __init__(self):
        pass 

    def __len__(self):
        pass 

    def __getitem__(self,index):
        pass

df = pd.read_csv("./dataset/cleaned_step_1.csv")
