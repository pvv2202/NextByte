import torch
import torch.nn as nn
import numpy as np 
from PositionalEncoder import PositionalEncoder
from FFN import FFN

 
class DecoderBlock(nn.Module):
    
    def __init__(self, d_model, num_heads, d_hidden, num_hidden_layers):
        super(DecoderBlock, self).__init__()

        """masked mh attention -> add norm -> feedforward -> add norm"""
        self.masked_mh_attention = nn.MultiheadAttention(d_model, num_heads, batch_first=True) # can add dropout here if we want
        
        # layer norm normalizes each embedding vector individually, so it needs the size of those vectors (d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        
        self.ffn = FFN(d_input=d_model, d_output=d_model, d_hidden=d_hidden, num_hidden_layers=num_hidden_layers) # feed forward
        

    def forward(self, x, padding_mask):
        """ forward method of a mhattention object needs q, k, v (it handles the projection of the inputs),
            and an attention mask
        """
        b, c, d = x.shape
        # context x context mask, upper triangle values set to true, which 
        # tells model to ignore those positions so it cannot cheat.
        mask = torch.triu(torch.ones(c, c, device=x.device), 1).bool()

        # COULD CHANGE: decoder flow with residual connections
        attn_output, att_weights = self.masked_mh_attention(x, x, x, attn_mask = mask, key_padding_mask=padding_mask)
        residual_one = x + attn_output
        normalized = self.layer_norm(residual_one)
        ffn_output = self.ffn(normalized)
        out = residual_one + ffn_output
        
        return out
    
class NextByteDecoder(nn.Module):
    def __init__(self, num_heads, num_hidden_layers, num_decoders, d_model, d_hidden):
        super(NextByteDecoder, self).__init__()
        # create a module list of num_decoder DecoderBlock objects
        self.decoder_blocks = nn.ModuleList([
            DecoderBlock(d_model, num_heads, d_hidden, num_hidden_layers)
            for _ in range(num_decoders)
        ])
        
        # OPTIONAL
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, x, padding_mask):
        # pass the input through every decoder layer
        for block in self.decoder_blocks:
            x = block(x, padding_mask)
        
        # optional: return the output after layer normalization  
        return self.layer_norm(x)
        
        
class NextByteTransformer(nn.Module):
    """there are probably more hyper params to add here"""
    def __init__(self, vocab_size, context_length, d_model, d_hidden, num_hidden_layers, num_heads, num_decoders):
        super(NextByteTransformer, self).__init__()
        # create the simple embedding layer
        self.emmbedding_layer = nn.Embedding(vocab_size, d_model)
        self.positional_encoder = PositionalEncoder(context_length, d_model)
        
        # decoder block with masked MHattention + add/norm + ffn 
        self.decoder = NextByteDecoder(
            d_model=d_model,
            d_hidden=d_hidden,
            num_heads=num_heads,
            num_hidden_layers=num_hidden_layers,
            num_decoders=num_decoders
        )
        # final output projection to all tokens in vocab
        self.to_logits = nn.Linear(d_model, vocab_size)
        # weight tying
        self.to_logits.weight = self.emmbedding_layer.weight
        
    def forward(self, x):
        input_key_mask = x == 0
        
        x = self.emmbedding_layer(x)
        x = self.positional_encoder(x)
        x = self.decoder(x, padding_mask=input_key_mask)
        
        return self.to_logits(x)