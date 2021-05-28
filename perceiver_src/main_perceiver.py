import torch 
import torch.nn as nn 
import torch.optim as optim 
import torch.nn.utils as utils
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt 
from dataloader import dataReader, vocabulary
from tqdm import tqdm 
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm 
from collections import deque 
from models import perceiver
import random 
import numpy as np
from itertools import chain

class cross_entopy(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self,output,label):
        label = label.view(-1,1)-1
        e = 0.0000000000001
        return torch.sum(-torch.log(output.gather(1,label)+e))/output.size(0)


def entropy(output):
    e = 0.0000000000001
    return -torch.sum(torch.log2(output+e)*output,dim=1)

class metric(nn.Module):
    def __init__(self,writer,max_memory=1000):
        super().__init__()
        self.max_memory = max_memory
        self.mem = deque(maxlen=self.max_memory)
        self.entropy_mem = deque(maxlen=self.max_memory)
        self.writer = writer
        self.step = 0
    
    def increment_step(self):
        self.step += 1
        
    def forward(self,output,label):

        self.mem.extend(torch.abs((torch.argmax(output,dim=1)+1)-label))
        self.entropy_mem.extend(entropy(output))
        memory_contents = torch.tensor(self.mem,dtype=torch.float)
        entropy_memory_contents = torch.tensor(self.entropy_mem,dtype=torch.float)

        if(len(self.mem) == self.max_memory):
            mean = torch.mean(memory_contents)
            std = torch.std(memory_contents)
            mean_entropy = torch.mean(entropy_memory_contents)
            std_entropy = torch.std(entropy_memory_contents)

            self.writer.add_scalar("metric_train/mean_error",mean,self.step)
            self.writer.add_scalar("metric_train/std_error",std,self.step)
            self.writer.add_scalar("entropy_train/mean",mean_entropy,self.step)
            self.writer.add_scalar("entropy_train/std",std_entropy,self.step)

            return (mean,std)
        else:
            return (None, None)
        

def visualize_model_weights(writer,model,step,path_catogory="model_param/"):
    #visualize weights
    for named_param in model.named_parameters():
        name = named_param[0]
        param = named_param[1]
        writer.add_histogram(path_catogory+name, param, step)

def visualize_model_gradients(writer,model,step,path_catogory="model_grads/"):
    #visualize gradients
    for named_param in model.named_parameters():
        name = named_param[0]
        param_grad = named_param[1].grad
        writer.add_histogram(path_catogory+name, param_grad, step)

if __name__ == "__main__":
    #reproducability
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    
    #hyperparameters
    path_train = "./dataset/processed_splits/train_count_837463_cleaned_100.csv"
    path_test = "./dataset/processed_splits/test_count_93052_cleaned_100.csv"
    embedding_dim = 128
    max_length_sentence = 100
    epochs = 1000
    lr=0.001
    batch_size = 100
    steps_till_test = 10 #50 #amount of steps before running on test data 
    amount_steps_test = 10 #amount_steps_test* batch_data = amount tests
    steps_till_weight_log = 10
    steps_till_grad_log = steps_till_weight_log
    #init 
    writer = SummaryWriter(log_dir="runs/perceiver")
    vocab = vocabulary(embedding_dim=embedding_dim,max_length_sentence=max_length_sentence)
    network = perceiver()
    optimizer = optim.Adam(chain(network.parameters(),vocab.parameters()),lr=lr)
    measurer =  metric(writer)

    
    
    train_data = dataReader(vocab,path=path_train)
    test_data = dataReader(vocab,path=path_test)
    
    train_loader = DataLoader(train_data,batch_size=batch_size,num_workers=0,shuffle=True)
    test_loader = DataLoader(test_data,batch_size=batch_size,num_workers=0,shuffle=False)

    

    step = 0
    for i in range(epochs):
        for input_data, label in tqdm(train_loader,ascii=True,desc="train"):
            
            optimizer.zero_grad()
            
            output = network(input_data)
            
            loss = nn.CrossEntropyLoss()(output,label-1)
            loss.backward()
            
            
            measurer.increment_step()
            measurer(output,label)
            
            
            writer.add_scalar("cross_entropy_loss",loss.detach().item(),step)
            #gradient clipping
            utils.clip_grad_norm_(network.parameters(), 1)
            utils.clip_grad_norm_(vocab.parameters(), 1) 

            optimizer.step()
            
            if(step %steps_till_weight_log == 0):
                #log weights
                visualize_model_weights(writer,network,step,path_catogory="network_param/")
                visualize_model_weights(writer,vocab,step,path_catogory="vocab_param/")

                

            if(step % steps_till_grad_log == 0):
                #log gradients
                visualize_model_gradients(writer,network,step,path_catogory="network_grads/")
                visualize_model_gradients(writer,vocab,step,path_catogory="vocab_grads/")




            if(step %steps_till_test == 0):
                #run on test data 
                diff_list = torch.tensor([]) #difference label and output 
                entropy_list = torch.tensor([])
                test_step = 0
                with torch.no_grad():
                    for input_data, label in tqdm(test_loader,ascii=True,desc="test"):
                        
                        output = network(input_data)
                        diff = torch.abs((torch.argmax(output,dim=1)+1)-label)
                        diff_list = torch.cat((diff_list,diff))

                        entropy_test = entropy(output)
                        entropy_list = torch.cat((entropy_list, entropy_test))

                        test_step += 1

                        if(test_step % amount_steps_test == 0):
                            break
                
                diff_zero = torch.sum(diff_list == 0)
                diff_one = torch.sum(diff_list == 1)
                diff_two = torch.sum(diff_list == 2)
                diff_three =torch.sum(diff_list == 3)
                diff_four =  torch.sum(diff_list == 4)
                diff_total = 0*diff_zero + 1*diff_one + 2*diff_two + 3*diff_three + 4*diff_four


                writer.add_scalar("diff_test/zero",diff_zero,step)
                writer.add_scalar("diff_test/one",diff_one,step)
                writer.add_scalar("diff_test/two",diff_two,step)
                writer.add_scalar("diff_test/three",diff_three,step)
                writer.add_scalar("diff_test/four",diff_four,step)
                writer.add_scalar("diff_test/total",diff_total,step)
                
                writer.add_scalar("metric_test/mean_error",torch.mean(diff_list),step)
                writer.add_scalar("metric_test/std_error",torch.std(diff_list),step)

                writer.add_scalar("entropy_test/mean",torch.mean(entropy_list),step)
                writer.add_scalar("entropy_test/std",torch.std(entropy_list),step)
            #increment step
            step += 1

            
    

