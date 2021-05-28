import pandas as pd 
import torch 
import torch.nn as nn  
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader,IterableDataset
from tqdm import tqdm 
from transformers import AutoTokenizer
import math
from collections import deque 

class vocabulary(nn.Module):
    def __init__(self, embedding_dim,max_length_sentence= 100,padding='max_length',truncation=True):
        super().__init__()

        """
        padding(variable is  a string)options below:
        -"max_length" (pads with zeros till max length is reached)
        -'longest' (pads with zeros to the size of the longest in that specific batch)
        """

        self.max_length_sentence=max_length_sentence 
        self.padding = padding
        self.truncation = truncation
        self.embedding_dim = embedding_dim
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
        self.vocab = nn.Embedding(num_embeddings=self.tokenizer.vocab_size, embedding_dim=self.embedding_dim)
        

    def forward(self,text):
        """
        input: string 
        output: embeddings (vectors)
        """
        tokens = torch.tensor(self.tokenizer.encode(str(text),max_length=self.max_length_sentence,padding=self.padding ,truncation =self.truncation))

    
        return self.vocab(tokens)


class dataIterator(IterableDataset):
    def __init__(self,vocab,path="./dataset/cleaned_step_1.csv"):
        self.dataset = pd.read_csv(path,chunksize=1) 
        self.vocab = vocab

    def __iter__(self):
        for data in self.dataset: 
            text = data["reviewText"].item()
            rating = data["overall"].item()
            

            vectors = self.vocab(str(text))

            yield vectors,rating
 
class dataReader(Dataset):
    def __init__(self,vocab,path="./dataset/cleaned_step_1.csv"):
        self.dataset = pd.read_csv(path) 
        self.vocab = vocab

    def __getitem__(self,index):
        text = self.dataset["reviewText"].iloc[index]
        rating = self.dataset["overall"].iloc[index]
        vectors = self.vocab(str(text))

        return vectors,rating
    
    def __len__(self):
        return len(self.dataset.index)
