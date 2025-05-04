print('literal top of file')
import torch
import torch.nn
import numpy as np
import pandas as pd
import torch.nn.functional as F
from torcheval.metrics.functional import bleu_score
import random
import csv
from recipe_nlg import RecipeNLGDataset
from pathlib import Path
from models import NextByteTransformer
from transformers import PreTrainedTokenizerFast
from Save_Results import save_bleu

print('at top of script')
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def generate_autoregressive(model, tokenizer, input_text, max_new_tokens=100, top_k=10, context_length=512):
    model.eval()
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
    input_ids = input_ids[:, -context_length:]
    generated = input_ids

    with torch.no_grad():
        for _ in range(max_new_tokens):
            if generated.size(1) > context_length:
                generated = generated[:, -context_length:]
            
            generated = generated.to(device)

            logits = model(generated)[:, -1, :]  # shape: [1, vocab_size]
            topk_logits, topk_indices = torch.topk(logits, k=top_k, dim=-1)
            probs = F.softmax(topk_logits, dim=-1)

            sampled_index = torch.multinomial(probs, num_samples=1).to(device)  # shape: [1, 1]
            next_token = topk_indices.gather(-1, sampled_index)  # shape: [1, 1]

            generated = torch.cat([generated, next_token.to(device)], dim=1)

            if next_token.item() == tokenizer.eos_token_id:
                break

    return tokenizer.decode(generated[0], skip_special_tokens=False)

def get_bleu(models, tokenizers, dataset):    
    
    pred_recipe, ground_truth = [], []
    
    # random index for a random recipe
    rand_index = random.randint(0, dataset.__len__()-1)
    # get the title on its own from the cleaned data frame
    title_prompt = dataset.recipes.iloc[rand_index]['title']
    # get the full recipe string (not tokenized)
    recipe_ref = dataset.recipe_strings.iloc[rand_index]

    if type(models) == list:
        title_to_ingredients_tokenizer = tokenizers[0]
        ingredients_to_directions_tokenizer = tokenizers[1]

        title_to_ingredients_model = models[0]
        ingredients_to_directions_model = models[1]

        output1 = generate_autoregressive(
            model=title_to_ingredients_model,
            tokenizer=title_to_ingredients_tokenizer,
            input_text=title_prompt,
            max_new_tokens=400,
            top_k=10,
            context_length=context_length,
        )
        title_index = output1.find("<end_title>")
        title = output1[:title_index]
        ingredients = output1[title_index + len("<end_title>"):]

        """Run through ingredients to directions model"""
        output2 = generate_autoregressive(
            model=ingredients_to_directions_model,
            tokenizer=ingredients_to_directions_tokenizer,
            input_text=ingredients.replace("<end>", "<end_ingredients>"),
            max_new_tokens=400,
            top_k=10,
            context_length=context_length,
        )
        pred = title+"<end_title>"+output2
    else:
        pred = generate_autoregressive(
            model=models,
            tokenizer=tokenizers,
            input_text= title_prompt,
            max_new_tokens=400,
            top_k=10,
            context_length=context_length,
        )
    
    pred_recipe.append(pred)
    ground_truth.append(recipe_ref)
        
    bleu = bleu_score(input=pred_recipe, target=ground_truth, n_gram=4) # n_gram = max count of sequential words to compare strings on
    
    # bleu is a 1d tensor, extract the value and convert to an np.float64 first
    return bleu.item()



# same params used in all training
context_length = 512
d_model = 512
num_heads = 8
num_hidden_layers = 8
d_hidden = 2048
num_decoders = 2
num_epochs = 8

print('loading models')
# Instantiate and load title_to_ingredients model
title_to_ingredients_model = NextByteTransformer(
    vocab_size=20000,
    context_length=context_length,
    d_model=d_model,
    num_heads=num_heads,
    num_hidden_layers=num_hidden_layers,
    d_hidden=d_hidden,
    num_decoders=num_decoders
)
title_to_ingredients_model.load_state_dict(
    torch.load(Path("Models/title_to_ingredients.pth"), map_location=device)
)
title_to_ingredients_model.to(device)

# Instantiate and load ingredients_to_directions model
ingredients_to_directions_model = NextByteTransformer(
    vocab_size=20000,
    context_length=context_length,
    d_model=d_model,
    num_heads=num_heads,
    num_hidden_layers=num_hidden_layers,
    d_hidden=d_hidden,
    num_decoders=num_decoders
)
ingredients_to_directions_model.load_state_dict(
    torch.load(Path("Models/ingredients_to_directions.pth"), map_location=device)
)
ingredients_to_directions_model.to(device)

# Instantiate and load title_to_all model
title_to_all_model = NextByteTransformer(
    vocab_size=20000,
    context_length=context_length,
    d_model=d_model,
    num_heads=num_heads,
    num_hidden_layers=num_hidden_layers,
    d_hidden=d_hidden,
    num_decoders=num_decoders
)
title_to_all_model.load_state_dict(
    torch.load(Path("Models/all.pth"), map_location=device)
)
title_to_all_model.to(device)
seq_models = [title_to_ingredients_model, ingredients_to_directions_model]

print('loading tokenizers')
# Load tokenizers
title_to_ingredients_tokenizer = PreTrainedTokenizerFast.from_pretrained(
    Path("Tokenizers/title_to_ingredients_tokenizer")
)
title_to_ingredients_tokenizer.eos_token_id = title_to_ingredients_tokenizer.convert_tokens_to_ids("<end>")
ingredients_to_directions_tokenizer = PreTrainedTokenizerFast.from_pretrained(
    Path("Tokenizers/ingredients_to_directions_tokenizer")
)
ingredients_to_directions_tokenizer.eos_token_id = ingredients_to_directions_tokenizer.convert_tokens_to_ids("<end>")
title_to_all_tokenizer = PreTrainedTokenizerFast.from_pretrained(
    Path("Tokenizers/title_to_all_tokenizer")
)
title_to_all_tokenizer.eos_token_id = title_to_all_tokenizer.convert_tokens_to_ids("<end>")
seq_tokenizers = [title_to_ingredients_tokenizer, ingredients_to_directions_tokenizer]

print('loading dataset')
path = '/home/pvandervort25/.cache/kagglehub/datasets/paultimothymooney/recipenlg/versions/1'
# Load the dataset
df = pd.read_csv(path + "/RecipeNLG_dataset.csv", header=0)
dataset = RecipeNLGDataset(df=df, mode='all')

# up to us
num_trials = 10000

# Get results for single model
results_all = ['trial', 'bleu_score']

print('starting test')
for i in range(num_trials):
    results_all.append([i + 1, get_bleu(title_to_all_model, title_to_all_tokenizer, dataset)])
    if i % 100 == 0:
        print(f"Single: {i}")
    
save_bleu(results=results_all, model_name="all")

# Get results for sequence of models
results_seq = ['trial', 'bleu_score']

for i in range(num_trials):
    results_seq.append([i + 1, get_bleu(seq_models, seq_tokenizers, dataset)])
    if i % 100 == 0:
        print(f"Sequence: {i}")

save_bleu(results=results_seq, model_name="seq")