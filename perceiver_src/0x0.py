import pandas as pd 
import torch 
import torch.nn as nn  
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader,IterableDataset
from tqdm import tqdm 
from transformers import AutoTokenizer
import math

class vocabulary(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
        self.vocab = nn.Embedding(num_embeddings=self.tokenizer.vocab_size, embedding_dim=embedding_dim)
        

    def forward(self,text):
        """
        input: string 
        output: embeddings (vectors)
        """
        tokens = torch.tensor(self.tokenizer(text)["input_ids"])

        return self.vocab(tokens)


class dataReader(IterableDataset):
    def __init__(self,vocab):
        self.dataset = pd.read_csv("./dataset/cleaned_step_1.csv",chunksize=1) 
        self.vocab = vocab

    def __iter__(self):
        
        for data in self.dataset:
            text = data["reviewText"].item()
            rating = data["overall"].item()
            
            vectors = self.vocab(text)

            yield vectors,rating
        

class self_attention_head(nn.Module):
    """
    self attention 

    x -> Q,V,K

    softmax((Q * K^T)/d_k)*V

    """

    def __init__(self,input_size,d_k,d_v):
        super().__init__()
        
        self.d_k = d_k
        self.d_v = d_v

        self.Q_weights = nn.Linear(input_size,self.d_k,bias=False)
        self.K_weights = nn.Linear(input_size,self.d_k,bias=False) 
        self.V_weights = nn.Linear(input_size,self.d_v,bias=False)

    def forward(self,x):
        Q = self.Q_weights(x)
        K = self.K_weights(x).T
        V = self.V_weights(x)

        attention_map = torch.softmax(torch.matmul(Q,K)/math.sqrt(self.d_k),dim=1)

        return torch.matmul(attention_map,V)


class cross_attention_head(nn.Module):
    """
    cross attention

    text -> K,V
    latent_state -> Q 

    softmax(Q*K^T / d_k)*V
    """
    def __init__(self,latent_size,queried_array_size,d_k=5,d_v=1):
        super().__init__()
        self.d_k = d_k
        self.d_v = d_v

        self.Q_weights = nn.Linear(latent_size,self.d_k,bias=False)
        self.K_weights = nn.Linear(queried_array_size,self.d_k,bias=False) 
        self.V_weights = nn.Linear(queried_array_size,self.d_v,bias=False)

    def forward(self,latent,text):
        Q = self.Q_weights(latent)
        K = self.K_weights(text).T
        V = self.V_weights(text)

        attention_map = torch.softmax(torch.matmul(Q,K)/math.sqrt(self.d_k),dim=1)
        return torch.matmul(attention_map,V)



class self_multi_headed(nn.Module):
    def __init__(self,input_size,amount_heads):
        super().__init__()
        """
        note that input size is not the length of the sequence but the size of the embedding vector (so input_size = |embedding_vec|)

        v_d and k_d have been set to the same size as the input. (v_d is the value dimension, dicates the output size), 
        k_d is the size of the vector for the dot product attention. (bigger the more expressive)
        set to same size as the input
        
        for linear the dimensionality of the input is kept the same
        input_size*amount_heads as the output of each head gets concatenated and then mapped (back to the input size) to the output size

        also contains a residual connection of input -> output (residual )
                                                input -> heads -> linear -> output (standard flow)
                                                total:
                                                input -> heads -> linear -> output + output_residual
        """
        self.heads = nn.ModuleList([self_attention_head(input_size,input_size,input_size) for i in range(amount_heads)])
        self.linear = nn.Linear(input_size*amount_heads,input_size,bias=False)


    def forward(self,x):
        
        return self.linear(torch.cat([head(x) for head in self.heads],dim=1)) + x 
        


class cross_multi_headed(nn.Module):
    def __init__(self,latent_size,embedding_size,amount_heads):
        super().__init__()
        """
        note that embedding_size = |embedding_vec|

        also note that this cross attention module has a residual connection
        """

        self.heads = nn.ModuleList([cross_attention_head(latent_size,embedding_size,embedding_size,latent_size) for i in range(amount_heads)])
        self.linear = nn.Linear(latent_size*amount_heads,latent_size,bias=False)
    
    def forward(self,latent,text):

        return self.linear(torch.cat([head(latent,text) for head in self.heads],dim=1)) + latent 




if __name__ == "__main__":
    #vocab = vocabulary(10)
    #print(vocab("hello world"))
