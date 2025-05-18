import torch
import torch.nn as nn
import numpy as np
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_scheduler, PreTrainedTokenizerFast
from tqdm.auto import tqdm
import evaluate
from models import NextByteTransformer
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from recipe_nlg import TokenizedRecipeNLGDataset
from torch.cuda.amp import autocast, GradScaler
from Save_Results import save_results
import os

"""Model & training hyper parameters"""
context_length = 768
d_model = 768
num_heads = 8
num_hidden_layers = 8
d_hidden = 3072
num_decoders = 6
num_epochs = 12
lr = 3e-5
batch_size = 32
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(device)

mode = 'title_to_all'
tokenizer_path = Path('Tokenizers/' + mode + '_tokenizer')

print('loading tokenizer')
tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path, model_max_length=context_length)

# different loss function?
loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

print('loading df..')
path = '/home/pvandervort25/.cache/kagglehub/datasets/paultimothymooney/recipenlg/versions/1'
# Load the dataset
df = pd.read_csv(path + "/RecipeNLG_dataset.csv", header=0)

print('creating dataset..')
train_dataset = TokenizedRecipeNLGDataset(df=df, tokenizer=tokenizer, mode='nextbyte')
values = ['www.kraftrecipes.com', 'recipes-plus.com', 'www.foodgeeks.com', 'allrecipes.com', 'www.cookbooks.com', 'cookeatshare.com', 'www.landolakes.com',  'cookpad.com']
train_dataset.filter('link', values)

print('creating model..')
# declare model
model = NextByteTransformer(
    vocab_size=30000,
    context_length=context_length,
    d_model=d_model,
    num_heads=num_heads,
    num_hidden_layers=num_hidden_layers,
    d_hidden=d_hidden,
    num_decoders=num_decoders
)

print('creating dataloader')
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# check shape
for batch in train_dataloader:
    print(batch['input_ids'].shape)
    break

# # TODO: explain what this is
optimizer = AdamW(model.parameters(), lr=lr)

model.to(device)

# # TODO: explain what this does
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    'linear',
    optimizer=optimizer,
    num_warmup_steps = int(0.05 * num_training_steps),
    num_training_steps=num_training_steps
)

model.train()
scaler = GradScaler()
for epoch in range(num_epochs):
    print(f"EPOCH {epoch}")
    avg_loss = 0
    for batch in tqdm(train_dataloader, unit='batch'):
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()

        with autocast():  # enable mixed precision
            logits = model(input_ids)
            logits = logits.view(-1, logits.size(-1))
            labels = labels.view(-1)
            loss = loss_fn(logits, labels)

        # Prevent NaN/Inf by using gradient scaling
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        lr_scheduler.step()

        avg_loss += loss.item()

    avg_loss /= len(train_dataloader)
    print(f"Average loss: {avg_loss:.6f}")

    os.makedirs("./Models", exist_ok=True)
    torch.save(model.state_dict(), f"./Models/nextbyte_mp_{epoch}.pth")