import torch
import torch.nn as nn
import numpy as np 


class FFN(nn.Module):
    """PARAMS
       d_input (int): size of embeddings
       d_output (int): should be the same as d_input for a decoder transformer
       num_hidden_layers (int) OPTIONAL : number of hidden layers in feedforward network
       d_hidden (int) OPTIONAL : number of nodes in each hidden layer
       
    """
    def __init__(self, d_input, d_output, num_hidden_layers=4, d_hidden=2048):
        super(FFN, self).__init__()
        
        layers = []

        # first hidden
        layers.append(nn.Linear(d_input, d_hidden))
        layers.append(nn.ReLU())
        
        # all other hidden layers (if necessary )
        for i in range(num_hidden_layers - 1):
            layers.append(nn.Linear(d_hidden, d_hidden))
            layers.append(nn.ReLU())
        
        # final output
        layers.append(nn.Linear(d_hidden, d_output))
        
        # equivalent to passing all layers individually
        self.ffn = nn.Sequential(*layers)
           
    def forward(self, x):
        # just pass it through.
        return self.ffn(x)