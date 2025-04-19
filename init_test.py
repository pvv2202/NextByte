import torch
from models import DecoderBlock, NextByteDecoder, NextByteTransformer
from pathlib import Path

d_model = 66
context_length = 512
batch_size = 4
vocab_size = 20000
test_input = torch.ones(batch_size, context_length, dtype=torch.long)


next_byte = NextByteTransformer(
    d_model=d_model,
    vocab_size=vocab_size,
    context_length=context_length,
    num_heads=2, 
    num_hidden_layers=2, 
    d_hidden=2048, 
    num_decoders=6)


x = next_byte(test_input)
# should output (batch x context x vocab)
# a unormalized probabilities of next word (vocab_size options) for every token in the context
print(x[1][0].shape)

