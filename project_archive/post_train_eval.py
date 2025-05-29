print('literal top of file')
import torch
import torch.nn
import numpy as np
import pandas as pd
import re
import torch.nn.functional as F
import nltk
import nltk.translate.bleu_score as bleu
import random
import csv
from bert_score import score
from recipe_nlg import RecipeNLGDataset
from pathlib import Path
from models import NextByteTransformer
from transformers import PreTrainedTokenizerFast
from project_archive.Save_Results import save_bleu

print('at top of script')
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

chenCherry = bleu.SmoothingFunction()

def generate_autoregressive(model, tokenizer, input_text, max_new_tokens=100, top_k=10, context_length=512):
    model.eval() # Set to eval mode
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device) # Tokenize input
    input_ids = input_ids[:, -context_length:] # Cut off if its over context length (not possible here since we go to 400 but worth including)
    generated = input_ids.long().to(device)  # Send them to gpu

    with torch.no_grad(): # Don't track gradient
        for _ in range(max_new_tokens):
            if generated.size(1) > context_length:
                generated = generated[:, -context_length:]

            logits = model(generated.long())[:, -1, :] # Reshape logits for compatibility
            topk_logits, topk_indices = torch.topk(logits, k=top_k, dim=-1) # Get top k
            probs = F.softmax(topk_logits, dim=-1) # Softmax to get probs

            sampled_index = torch.multinomial(probs, num_samples=1).to(device) # Actually sampling
            next_token = topk_indices.gather(-1, sampled_index)

            generated = torch.cat([generated, next_token], dim=1).long().to(device) # Generated tokens so prev + next

            if next_token.item() == tokenizer.eos_token_id: # Stop if we reach the end token for this model. Always "<end>"
                break

    # Return as text and don't skip special tokens
    return tokenizer.decode(generated[0], skip_special_tokens=False)


def clean_text(text, fix_tokens):
    # Necessary for the model pipelines. We need the special tokens to know when to stop so we can't set skip_special_tokens to true.
    # Probably a better way to go about the pipeline but not super important for right now since we aren't deploying this or anything.
    text = text.replace(" ##", "")
    text = text.replace("##", "")
    for token in fix_tokens:
        text = text.replace(token, " ")
    text = re.sub(r'[^\w\s]', ' ', text)  # Replace punctuation with space
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text

def get_metrics(models, tokenizers, dataset, index, mode=None):
    # get the title on its own from the cleaned data frame
    title_prompt = dataset.recipes.iloc[index]['title']
    # get the full recipe string (not tokenized)
    recipe_ref = dataset.recipe_strings.iloc[index]

    if type(models) == list: # Originally just for seq. Now includes title and all
        title_to_ingredients_tokenizer = tokenizers[0]
        ingredients_to_directions_tokenizer = tokenizers[1] # This will actually be all for the title and all model

        title_to_ingredients_model = models[0]
        ingredients_to_directions_model = models[1] # This will also actually be all for the title and all model

        # Generate title to ingredients output
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
        if mode is not None: # Mode set to seq here. This removes the title for the seq model. It gets skipped otherwise
            ingredients = output1[title_index + len("<end_title>"):]

        else:
            ingredients = output1

        # Generate directions with either ingredients to directions or all depending
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
        # If it isn't a list we just do one autoregressive passs
        pred = generate_autoregressive(
            model=models,
            tokenizer=tokenizers,
            input_text= title_prompt,
            max_new_tokens=400,
            top_k=10,
            context_length=context_length,
        )

    # Reformat to get comparable data
    fix_tokens = ["<end>", "<end_ingredients>", "<end_title>"]
    pred_clean = clean_text(pred, fix_tokens)
    ref_clean = clean_text(recipe_ref, fix_tokens)

    bleu_score = bleu.sentence_bleu([ref_clean.split()], pred_clean.split(), smoothing_function=chenCherry.method2) # Just adds 1 to each n-gram to smooth things for low scores
    P, R, F1 = score(
        [pred_clean], [ref_clean],
        lang="en",
        model_type="bert-base-uncased", # Using a small model cause the big one took forever
        batch_size=64,
        verbose=False,
        device=device
    )

    # bleu is a 1d tensor, extract the value and convert to an np.float64 first
    return bleu_score, P[0].item(), R[0].item(), F1[0].item()

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
path = Path.home() / ".cache" / "kagglehub" / "datasets" / "paultimothymooney" / "recipenlg" / "versions" / "1"
# Load the dataset
df = pd.read_csv(path / "RecipeNLG_dataset.csv", header=0)
dataset = RecipeNLGDataset(df=df, mode='all')

# up to us
num_trials = 10000
indices = [random.randint(0, dataset.__len__()-1) for i in range(num_trials)]

# Seq model (title -> ingr, ingr -> dir)
results_seq = []
for i, idx in enumerate(indices):
    bleu_score, precision, recall, f1 = get_metrics([title_to_ingredients_model, ingredients_to_directions_model], [title_to_ingredients_tokenizer, ingredients_to_directions_tokenizer], dataset, idx, mode="seq")
    results_seq.append({"trial": i + 1, "bleu": bleu_score, "precision": precision, "recall": recall, "f1": f1})
    if i % 100 == 0:
        print(f"Sequence: {i}")

save_bleu(results=results_seq, model_name="seq")

# All model (title -> all)
results_all = []
for i, idx in enumerate(indices):
    bleu_score, precision, recall, f1 = get_metrics(title_to_all_model, title_to_all_tokenizer, dataset, idx)
    results_all.append({"trial": i + 1, "bleu": bleu_score, "precision": precision, "recall": recall, "f1": f1})
    if i % 100 == 0:
        print(f"Single: {i}")
    
save_bleu(results=results_all, model_name="all")

# Mixed model (title -> ingr, title -> all)
results_mix = []
for i, idx in enumerate(indices):
    bleu_score, precision, recall, f1 = get_metrics([title_to_ingredients_model, title_to_all_model], [title_to_ingredients_tokenizer, title_to_all_tokenizer], dataset, idx)
    results_mix.append({"trial": i + 1, "bleu": bleu_score, "precision": precision, "recall": recall, "f1": f1})
    if i % 100 == 0:
        print(f"Mix: {i}")

save_bleu(results=results_mix, model_name="mix")