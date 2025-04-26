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
from sklearn.metrics import f1_score
from recipe_nlg import TokenizedRecipeNLGDataset

"""Model & training hyper parameters"""
context_length = 512
d_model = 512
num_heads = 8
num_hidden_layers = 8
d_hidden = 2048
num_decoders = 1
num_epochs = 5
lr = 3e-5
batch_size = 16
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(device)

# different loss function?
loss_fn = nn.CrossEntropyLoss()

# set mode and tokenizer path
mode = 'title_to_ingredients'
tokenizer_path = Path('Tokenizers/' + mode + '_tokenizer')

print('loading tokenizer')
tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path, model_max_length=context_length)

print('loading df..')
path = kagglehub.dataset_download("paultimothymooney/recipenlg")
# Load the dataset
df = pd.read_csv(path + "/RecipeNLG_dataset.csv", header=0)

print('splitting into train and test sets')
train_df, eval_df = train_test_split(df, test_size=0.2)

print('creating datasets..')
train_dataset = TokenizedRecipeNLGDataset(df=train_df, tokenizer=tokenizer, mode='title_to_ingredients')
eval_dataset = TokenizedRecipeNLGDataset(df=eval_df, tokenizer=tokenizer, mode='title_to_ingredients')

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
    print(batch['input_ids'].shape)
    break

# # TODO: explain what this is
optimizer = AdamW(model.parameters(), lr=lr)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

# # TODO: explain what this does
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    'linear',
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps
)

def evaluate_model(model, dataloader, device):
    """Takes the model and a dataset. Evaluates the model on the dataset, printing out overall accuracy."""
    # NOTE to make it simple, dataset is a dataloader already
    metric = evaluate.load("accuracy")
    model.eval()
    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        with torch.no_grad():
            logits = model(input_ids)

        predictions = torch.argmax(logits, dim=-1)
        # Flatten predictions and labels
        predictions = predictions.view(-1)  # Shape: [batch_size * seq_len]
        labels = labels.view(-1)  # Shape: [batch_size * seq_len]

        metric.add_batch(predictions=predictions.detach().cpu().numpy(), references=labels.detach().cpu().numpy())
    # average = 'micro' uses a global count of the total TPs, FNs and FPs.
    print(f"F1: {f1_score(y_true=labels.detach().cpu().numpy(), y_pred=predictions.detach().cpu().numpy(), average='micro')}") # average arg needed for multiclass targets
    print(f"ACCURACY: {metric.compute()}")

model.train()
for epoch in range(num_epochs):
    print(f"EPOCH {epoch}")
    for batch in tqdm(train_dataloader, unit='batch'):
        input_ids = batch['input_ids']
        labels = batch['labels']

        logits = model(input_ids)
        # reformat to shape expected by cross entrooy
        logits = logits.view(-1, logits.size(-1))  # (b * seq, v)
        labels = labels.view(-1)  # (b * seq)
        # cross entropy handles the softmax part
        loss = loss_fn(logits, labels)

        # update weights
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

    print("TRAIN METRICS")
    evaluate_model(model, train_dataloader, device)
    print("EVAL METRICS")
    evaluate_model(model, eval_dataloader, device)

torch.save(model.state_dict(), "./Models/title_to_ingredients.pth")