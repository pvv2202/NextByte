import torch
import torch.nn  as nn
import numpy as np 
import math



class PositionalEncoder(nn.Module):
    """PARAMS:
    
        context length (int): the max number of tokens the model can process at once
        
        d_model (int): length of a token's embedding vector, referred to generally as the 'model dimension'
        
        pdrop, float (Optional): probability of zeroing out an input, default to 0.1
    """
    def __init__(self, context_len, d_model, pdrop=0.1):
        super(PositionalEncoder, self).__init__()
        # encode each position (context_len) with d_model dimensions
        pe = torch.zeros(context_len, d_model) # shape: (context_len x d_model)
        
        self.dropout = nn.Dropout(p=pdrop)
        
        # create a tensor holding every possible position in the input sequence
        position = torch.arange(0, context_len, dtype=torch.float).unsqueeze(1) # shape: (context_len x 1)
        
        # by a bunch of annoying log properties this expression is equal to the dividing term
        # in the paper, and allows for more numerical stability, removing the need to calculate
        # exponents with 10000 as the base
        # 2i just means all the even numbers in d_model the torch.arrange below accomplishes that
        div_term = torch.exp(-1 * (torch.arange(0, d_model, 2) / d_model) * math.log(10000.0))
        
        # apply position * div term to every value in pe matrix, sin for even, cos for odd
        pe[:, 0::2] = torch.sin(position * div_term) # for each row, start at col 0 & skip 2 (even)
        pe[:, 1::2] = torch.cos(position * div_term) # for each row, start at col 1 and skip 2 (odd)
        pe = pe.unsqueeze(0) # add a batch dimension,  shape: (1 x context_len x d_model)

        self.register_buffer('pe', pe) # fixes embeddings, if we want to have them learn we can change
        
    """input x should be a tensor of shape (batch_size, len_input_sequence, d_model)"""
    def forward(self, x):
        # you slice x.shape(1) rows out of the positional encoding matrix to match the length of the input
        # sequence explicitly, used in case input sequence length != context_len. I don't think this would 
        # matter if we pad inputs to max length, but this should help if we decide not to
        x = x + (self.pe[:, :x.shape[1], :]).to(x.device)
        
        # apply dropout to help with overfitting and return
        return self.dropout(x)