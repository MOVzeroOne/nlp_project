import torch 
import torch.nn as nn 
import torch.optim as optim 
from positional_encoding import PositionalEncoding
from activation_functions import mish
from attention import cross_multi_headed, self_multi_headed

class perceiver(nn.Module):
    def __init__(self,share_parameters=False,batch_size=10,output_size=5,recursion_depth = 7, latent_space_sequence_length=50,max_sequence_length=100,embedding_dim = 128):
        super().__init__()
        self.max_sequence_length = max_sequence_length
        self.embedding_dim = embedding_dim
        self.latent_space_size=embedding_dim

        self.pos_encoding = PositionalEncoding(d_model=embedding_dim,max_len=self.max_sequence_length)
        self.latent_space_sequence_length = latent_space_sequence_length
        self.share_parameters = share_parameters
        self.recursion_depth = recursion_depth #how many times the recurrent_perceiver_block is repeated

        if(self.share_parameters):
            self.recurrent_perceiver_block = perceiver_block(latent_space_size=self.latent_space_size,embedding_dim=embedding_dim)
        else: 
            self.recurrent_perceiver_blocks = nn.ModuleList([perceiver_block(latent_space_size=self.latent_space_size,embedding_dim=embedding_dim) for i in range(self.recursion_depth)])
            

        self.end_block = nn.Linear(embedding_dim,output_size)
        self.latent_text = nn.Parameter(torch.randn(batch_size,self.latent_space_sequence_length,self.latent_space_size)) #batch,sequence_length,embedding_size
    
    def forward(self,x):
   
        data_with_pos = self.pos_encoding(x) #only needed for a transformer (as attention losses the abitlity to encode position)
        latent_vectors = self.latent_text
        
        if(self.share_parameters):
            for i in range(self.recursion_depth):
                latent_vectors = self.recurrent_perceiver_block(latent_vectors,data_with_pos)
        else:
            for block in self.recurrent_perceiver_blocks:
                latent_vectors = block(latent_vectors,data_with_pos)
            
        #end 
        averaged_latent = torch.mean(self.end_block(latent_vectors),dim=1)

        return nn.Softmax(dim=1)(averaged_latent)

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
        

