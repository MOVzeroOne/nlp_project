import torch 
import torch.nn as nn 
import torch.optim as optim 
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt 
from dataloader import dataReader, vocabulary
from tqdm import tqdm 




if __name__ == "__main__":
    #hyperparameters
    embedding_dim = 128


    #init 
    

    vocab = vocabulary(embedding_dim=embedding_dim)
    data = dataReader(vocab)
    loader = DataLoader(data,batch_size=1,num_workers=0)


    for input_data, label in loader:
        pass 