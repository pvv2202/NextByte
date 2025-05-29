import torch
from pathlib import Path
from .models import NextByteTransformer
from transformers import PreTrainedTokenizerFast

def init_next_byte():
    # model params
    vocab_size=30000
    context_length = 768
    d_model = 768
    num_heads = 8
    num_hidden_layers = 8
    d_hidden = 3072
    num_decoders = 6

    tokenizer = PreTrainedTokenizerFast.from_pretrained(
        Path("../Tokenizers/nextbyte_tokenizer")
    )
    tokenizer.eos_token_id = tokenizer.convert_tokens_to_ids("<end>")

    model = NextByteTransformer(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        num_heads=num_heads,
        num_hidden_layers=num_hidden_layers,
        d_hidden=d_hidden,
        num_decoders=num_decoders,
        tokenizer=tokenizer
    )

    # load in weights
    model.load_state_dict(
        torch.load(Path("../Models/nextbyte_6.pth"), map_location=torch.device('cpu'))
    )

    return model


