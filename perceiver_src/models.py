import torch 
import torch.nn as nn 
import torch.optim as optim 
from positional_encoding import PositionalEncoding
from activation_functions import mish
from attention import cross_multi_headed, self_multi_headed

class perceiver(nn.Module):
    def __init__(self,output_size=5,recursion_depth = 2, latent_space_sequence_length=20,max_sequence_length=100,embedding_dim = 128):
        super().__init__()
        self.max_sequence_length = max_sequence_length
        self.embedding_dim = embedding_dim
        self.latent_space_size=embedding_dim

        self.pos_encoding = PositionalEncoding(d_model=embedding_dim,max_len=self.max_sequence_length)
        self.latent_space_sequence_length = latent_space_sequence_length

        self.recurrent_perceiver_block = perceiver_block(latent_space_size=self.latent_space_size,embedding_dim=embedding_dim)
        self.recursion_depth = recursion_depth #how many times the recurrent_perceiver_block is repeated
        
        self.end_block = nn.Linear(embedding_dim,output_size)

    def forward(self,x):
   
        data_with_pos = self.pos_encoding(x) #only needed for a transformer (as attention losses the abitlity to encode position)
        latent_vectors = torch.zeros(data_with_pos.size(0),self.latent_space_sequence_length,self.latent_space_size) #batch,sequence_length,embedding_size
        
        for i in range(self.recursion_depth):

            latent_vectors = self.recurrent_perceiver_block(latent_vectors,data_with_pos)
        
        #end 
        averaged_latent = torch.mean(latent_vectors,dim=1)
        return nn.Softmax(dim=1)(self.end_block(averaged_latent))

class perceiver_block(nn.Module):
    def __init__(self,latent_space_size,embedding_dim,self_heads = 1,cross_heads = 1):
        super().__init__()

        self.cross_attention = cross_multi_headed(latent_space_size,embedding_dim,cross_heads)
        self.self_attention = self_multi_headed(embedding_dim,self_heads)
        self.dens_block_1 = nn.Sequential(nn.Linear(latent_space_size,latent_space_size),mish(),nn.Linear(latent_space_size,latent_space_size),mish())
        self.dens_block_2 = nn.Sequential(nn.Linear(latent_space_size,latent_space_size),mish(),nn.Linear(latent_space_size,latent_space_size),mish())
        """
        dense_block_1 follows cross_attention
        dense_block_2 follows the self attention

        as described in https://arxiv.org/pdf/2103.03206.pdf
        note that no dropout or normalization is used in this implementation
        """
        

    def forward(self,latent_vectors,input_sequence):

        latent = torch.tensor([])
        
        for latent_elem,text_elem in zip(latent_vectors,input_sequence): #for elem in batch
            latent_vector = self.cross_attention(latent=latent_elem,text=text_elem) 
            latent_vector = self.dens_block_1(latent_vector)
            latent_vector = self.self_attention(latent_vector)
            latent_vector = self.dens_block_2(latent_vector)
            latent = torch.cat((latent,latent_vector.unsqueeze(dim=0)))
        return latent 
        

