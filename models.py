import torch
import torch.nn 
import numpy as np 
import PositionalEncoder 
import FFN

 
class DecoderLayer(nn.Module):
    
    def __init__(self, d_model, num_heads, d_hidden, num_hidden_layers):
        super(DecoderLayer, self).__init__()
        
        """masked mh attention -> add norm -> feedforward -> add norm"""
        self.masked_mh_attention = nn.MultiHeadAttention(d_model, num_heads)
        
        self.attn_mask = None # TODO: would this be a torch.tril matrix of with the upper triangle zeroed out?
        
        # layer norm applies to each embedding vector individual, so it needs the size of those vectors (d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        
        self.ffn = FFN(d_input=d_model, d_output=d_model, d_hidden=d_hidden, num_hidden_layers=num_hidden_layers) # feed forward
        

    def forward(self, x):
        """ forward method of a mhattention object needs q, k, v (it handles the projection of the inputs),
            and an attention mask
        """
        # COULD CHANGE: decoder flow with residual connections 
        attn_output = self.masked_mh_attention(x, x, x, attn_mask = self.attn_mask)
        residual_one = x + attn_output
        normalized = self.layer_norm(residual_one)
        ffn_output = self.ffn(normalized)
        out = residual_one + ffn_output
        
        return out
    
class NextByteDecoder(nn.Module):
    def __init__(self, num_heads, num_hidden_layers, num_decoders, d_model, d_hidden):
        
        # create a module list of num_decoder DecoderLayer objects
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_hidden, num_hidden_layers)
            for _ in range(num_decoders)
        ])
        
        # OPTIONAL
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        # pass the input through every decoder layer
        for layer in self.decoder_layers:
            x = layer(x)
        
        # optional: return the output after layer normalization  
        return self.layer_norm(x)
        
        
class NextByteTransformer(nn.Module):
    """there are probably more hyper params to add here"""
    def __init__(self, vocab_size, context_length, d_model, d_hidden, num_hidden_layers, num_heads, num_decoders):
        super(NextByteTransformer, self).__init__()
        # create the simple embedding layer
        self.emmbedding_layer = nn.Embedding(vocab_size, d_model)
        self.positional_encoder_layer = PositionalEncoder(context_length, d_model)
        
        # decoder block with masked MHattention + add/norm + ffn 
        self.decoder = NextByteDecoder(
            d_model=d_model,
            d_hidden=d_hidden,
            num_heads=num_heads,
            num_hidden_layers=num_hidden_layers,
            num_decoders=num_decoders
        )
        self.linear = nn.Linear()