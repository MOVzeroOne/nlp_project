import torch 
import torch.nn as nn 
import torch.optim as optim 
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt 
from dataloader import dataReader, vocabulary
from tqdm import tqdm 
from positional_encoding import PositionalEncoding



class lstm_net(nn.Module):
    def __init__(self):
        self.lstm = None 
        self.linear = None 

    def forward(self,x):
        pass 





if __name__ == "__main__":
    #hyperparameters
    embedding_dim = 128
    max_length_sentence = 100
    epochs = 100
    lr=0.01
    #init 
    network = lstm_net()
    optimizer = optim.Adam(network,lr=lr)
    

    vocab = vocabulary(embedding_dim=embedding_dim,max_length_sentence=max_length_sentence)
    data = dataReader(vocab,path="./dataset/splits/cleaned_100.csv")
    loader = DataLoader(data,batch_size=2,num_workers=0)
    pos_encoding = PositionalEncoding(d_model=embedding_dim,max_len=max_length_sentence)

    for i in range(epochs):
        for input_data, label in loader:
            data_with_pos= pos_encoding(input_data)
         