import torch
import torch.nn 
import numpy as np 

   
class DecoderLayer(nn.Module):
    """masked mh attention -> add norm -> feedforward -> add norm"""
    def __init__(self, d_model, num_heads):
        super(DecoderLayer, self).__init__()
        self.masked_mh_attention = nn.MultiHeadAttention(d_model, num_heads)
        #
        self.attn_mask = None # TODO: would this be a torch.tril matrix of with the upper triangle zeroed out?
        # layer norm applies to each embedding vector individual, so it needs the size of those vectors (d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        self.ffn = 
        
        pass

    def forward(self, x):
        """ forward method of a mhattention object needs q, k, v (it handles the projection of the inputs),
            and an attention mask
        """
        
        attn_outputs = self.masked_mh_attention(x, x, x, attn_mask = self.attn_mask)
        
    

class NextByteTransformer(nn.Module):
    """there are probably more hyper params to add here"""
    def __init__(self, vocab_size, d_model, hidden_layers, context_length, num_heads):
        super(NextByteTransformer, self).__init__()
        # create the simple embedding layer
        self.emmbedding_layer = nn.Embedding(vocab_size, d_model)
        self.positional_encoder_layer = PositionalEncoder(context_length, d_model)
        
        # TODO: decoder block with masked MHattention + add/norm + ffn + add/norm + linear -> softmax
        