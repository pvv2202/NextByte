import torch
import torch.nn as nn
import numpy as np 
from .PositionalEncoder import PositionalEncoder
from .FFN import FFN
import re
import torch.nn.functional as F


 
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
    def __init__(self, vocab_size, context_length, d_model, d_hidden, num_hidden_layers, num_heads, num_decoders, tokenizer):
        super(NextByteTransformer, self).__init__()
        self.tokenizer = tokenizer
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
    
    def generate_recipe(self, input_text, max_new_tokens=100, top_k=10, context_length=512):
        print('generating')
        self.eval() # Set to eval mode
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt") # Tokenize input
        input_ids = input_ids[:, -context_length:] # Cut off if its over context length (not possible here since we go to 400 but worth including)
        generated = input_ids.long()  # Send them to gpu

        with torch.no_grad(): # Don't track gradient
            for _ in range(max_new_tokens):
                if generated.size(1) > context_length:
                    generated = generated[:, -context_length:]

                logits = self(generated.long())[:, -1, :] # Reshape logits for compatibility
                topk_logits, topk_indices = torch.topk(logits, k=top_k, dim=-1) # Get top k
                probs = F.softmax(topk_logits, dim=-1) # Softmax to get probs

                sampled_index = torch.multinomial(probs, num_samples=1) # Actually sampling
                next_token = topk_indices.gather(-1, sampled_index)

                generated = torch.cat([generated, next_token], dim=1).long() # Generated tokens so prev + next

                if next_token.item() == self.tokenizer.eos_token_id: # Stop if we reach the end token for this model. Always "<end>"
                    break

        # Return as text and don't skip special tokens
        output = NextByteTransformer.clean_text(self.tokenizer.decode(generated[0], skip_special_tokens=False))
        title_end = output.find("<end_title>")
        ingredients_end = output.find("<end_ingredients>")
        directions_end = output.find("<end>")

        title = output[len("<start_title>"):title_end].strip()
        ingredients = output[title_end + len("<end_title> <start_ingredients>"):ingredients_end].strip()
        directions = output[ingredients_end + len("<end_ingredients> <start_directions>"):directions_end].strip()

        # Clean up spaces before punctuation
        title = re.sub(r'\s+([.,!?;:])', r'\1', title).strip().capitalize()
        ingredients = re.sub(r'\s+([.,!?;:])', r'\1', ingredients)
        directions = re.sub(r'\s+([.,!?;:])', r'\1', directions)

        # Split ingredients on comma followed by a digit
        ingredients = [i.strip() for i in re.split(r',\s*(?=\d)', ingredients) if i.strip()]
        directions = [s.strip().capitalize() for s in directions.split('.') if s.strip()]
        
        return title, ingredients, directions
    
    @staticmethod 
    def clean_text(text):
        # Necessary for the model pipelines. We need the special tokens to know when to stop so we can't set skip_special_tokens to true.
        # Probably a better way to go about the pipeline but not super important for right now since we aren't deploying this or anything.
        text = text.replace(" ##", "")
        text = text.replace("##", "")
        # for token in fix_tokens:
        #     text = text.replace(token, " ")
        #text = re.sub(r'[^\w\s]', ' ', text)  # Replace punctuation with space
        text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
        return text