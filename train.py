import torch
import kagglehub
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_scheduler, PreTrainedTokenizerFast
from tqdm.auto import tqdm
import evaluate
from accelerate import Accelerator
from models import NextByteTransformer
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from recipe_nlg import TokenizedRecipeNLGDataset

"""Model & training hyper parameters"""
context_length = 512
d_model = 512
num_heads = 8
num_hidden_layers = 2
d_hidden = 2048
num_decoders = 8
num_epochs = 15
lr = 3e-5
batch_size = 64

accelerator = Accelerator()
# different loss function
loss_fn = nn.CrossEntropyLoss()

# set mode and tokenizer path
mode = 'title_to_all'
tokenizer_path = Path('Tokenizers/' + mode + 'tokenizer')

print('loading tokenizer')
tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path, model_max_lenth=context_length)


print('loading df..')
path = kagglehub.dataset_download("paultimothymooney/recipenlg")
# Load the dataset
# change this?
df = pd.read_csv(path + "/RecipeNLG_dataset.csv", header=0)

print('splitting into train and test sets')
train_df, eval_df = train_test_split(df, test_size=0.2)

print('creating datasets..')
train_dataset = TokenizedRecipeNLGDataset(df=train_df, mode='all')
eval_dataset = TokenizedRecipeNLGDataset(df=eval_df, mode='all')

print('creating model..')
# declare model
model = NextByteTransformer(
    vocab_size=20000,
    context_length=context_length,
    d_model=d_model,
    num_heads=num_heads,
    num_hidden_layers=num_hidden_layers,
    d_hidden=d_hidden,
    num_decoders=num_decoders
)

print('creating dataloaders')
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=True)

# check shape
for batch in train_dataloader:
    print({k: v.shape for k, v in batch.items()})
    break

# follows https://huggingface.co/learn/llm-course/en/chapter3/4?fw=pt
# # TODO: explain what this is
optimizer = AdamW(model.parameters(), lr=lr)

# # TODO: explain what this does
train_dl, eval_dl, model, optimizer = accelerator.prepare(
    train_dataloader, eval_dataloader, model, optimizer
)

# # TODO: explain what this does
num_training_steps = num_epochs * len(train_dl)
lr_scheduler = get_scheduler(
    'linear',
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps
)

print('starting training')

model.train()
for epoch in range(num_epochs):
    for batch in tqdm(train_dl, unit='batch'):
        input_ids = batch['input_ids']
        labels = batch['labels']
        
        logits = model(input_ids)
        loss = loss_fn(logits, labels)
        
        accelerator.backward(loss)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
  


