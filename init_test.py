import torch
from models import DecoderBlock

inputs = torch.ones(4, 20, 512)
test_decoder = DecoderBlock(num_heads=2, d_model=512, num_hidden_layers=2, d_hidden=2048)

x = test_decoder(inputs)
print(x.shape)