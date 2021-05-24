import torch 
import torch.nn as nn 
import torch.optim as optim 
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt 
from dataloader import dataReader, vocabulary
from tqdm import tqdm 
from positional_encoding import PositionalEncoding
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm 
from collections import deque 


class lstm_net(nn.Module):
    def __init__(self,input_size=128,hidden_size=256,output_size=5):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size,hidden_size =hidden_size,num_layers =2,batch_first =True,bidirectional =False)
        self.linear = nn.Linear(hidden_size,output_size) 
        

    def forward(self,x):
        
        hidden_output, _ = self.lstm(x)

        return nn.Softmax(dim=1)(self.linear(hidden_output)[:,-1,:])


class cross_entopy(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self,output,label):
        label = label.view(-1,1)-1
        e = 0.0000000000001
        return torch.sum(-torch.log(output.gather(1,label)+e))/output.size(0)

class metric(nn.Module):
    def __init__(self,writer,max_memory=1000):
        super().__init__()
        self.max_memory = max_memory
        self.mem = deque(maxlen=self.max_memory)
        self.writer = writer
        self.step = 0
    
    def increment_step(self):
        self.step += 1
        
    def forward(self,output,label):

        self.mem.extend(torch.abs((torch.argmax(output,dim=1)+1)-label))

        memory_contents = torch.tensor(self.mem,dtype=torch.float)

        if(len(self.mem) == self.max_memory):
            mean = torch.mean(memory_contents)
            median = torch.median(memory_contents)
            std = torch.std(memory_contents)

            self.writer.add_scalar("mean error",mean,self.step)
            self.writer.add_scalar("median error",median,self.step)
            self.writer.add_scalar("std error",std,self.step)

            return (mean,median,std)
        else:
            return (None,None, None)
        



if __name__ == "__main__":
    #hyperparameters
    embedding_dim = 128
    max_length_sentence = 100
    epochs = 100
    lr=0.01
    #init 
    writer = SummaryWriter()

    network = lstm_net()
    optimizer = optim.Adam(network.parameters(),lr=lr)
    measurer =  metric(writer)

    vocab = vocabulary(embedding_dim=embedding_dim,max_length_sentence=max_length_sentence)
    
    data = dataReader(vocab,path="./dataset/splits/cleaned_100.csv")
    loader = DataLoader(data,batch_size=10,num_workers=0)
    pos_encoding = PositionalEncoding(d_model=embedding_dim,max_len=max_length_sentence)

    step = 0
    for i in range(epochs):
        for input_data, label in tqdm(loader,ascii=True):
            step += 1
            optimizer.zero_grad()
            #data_with_pos= pos_encoding(input_data) #only needed for a transformer (as attention losses the abitlity to encode position)
            
            output = network(input_data)

            loss = cross_entopy()(output,label)
            loss.backward()
            measurer.increment_step()
            measurer(output,label)
            
            
            writer.add_scalar("cross_entropy",loss.detach().item(),step)


            optimizer.step()
            
            
        