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
        tokens = torch.tensor(self.tokenizer(str(text))["input_ids"])

        return self.vocab(tokens)


class dataReader(IterableDataset):
    def __init__(self,vocab,path="./dataset/cleaned_step_1.csv"):
        self.dataset = pd.read_csv(path,chunksize=1) 
        self.vocab = vocab
    def __iter__(self):
        
        for data in self.dataset: 
            text = data["reviewText"].item()
            rating = data["overall"].item()
            
            vectors = self.vocab(str(text))

            yield vectors,rating
        
