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
from Save_Results import save_results

"""Model & training hyper parameters"""
context_length = 512
d_model = 512
num_heads = 8
num_hidden_layers = 8
d_hidden = 2048
num_decoders = 2
num_epochs = 8
lr = 3e-5
batch_size = 16
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(device)

# different loss function?
loss_fn = nn.CrossEntropyLoss()

# set mode and tokenizer path
mode = 'ingredients_to_directions'
tokenizer_path = Path('Tokenizers/' + mode + '_tokenizer')

print('loading tokenizer')
tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path, model_max_length=context_length)

print('loading df..')
path = kagglehub.dataset_download("paultimothymooney/recipenlg")
# Load the dataset
df = pd.read_csv(path + "/RecipeNLG_dataset.csv", header=0)

print('splitting into train and test sets')
# data split into 70% train, & 15% each for eval and testing
train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)
eval_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

print('creating datasets..')
train_dataset = TokenizedRecipeNLGDataset(df=train_df, tokenizer=tokenizer, mode='ingredients_to_directions')
eval_dataset = TokenizedRecipeNLGDataset(df=eval_df, tokenizer=tokenizer, mode='ingredients_to_directions')
test_dataset = TokenizedRecipeNLGDataset(df=test_df, tokenizer=tokenizer, mode='ingredients_to_directions')

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
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
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
    total_loss = 0
    model.eval()
    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        with torch.no_grad():
            logits = model(input_ids)
            loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))  # Compute loss
            total_loss += loss.item()

        predictions = torch.argmax(logits, dim=-1)
        # Flatten predictions and labels
        predictions = predictions.view(-1)  # Shape: [batch_size * seq_len]
        labels = labels.view(-1)  # Shape: [batch_size * seq_len]

        metric.add_batch(predictions=predictions.detach().cpu().numpy(), references=labels.detach().cpu().numpy())
    # average = 'micro' uses a global count of the total TPs, FNs and FPs.
    avg_loss = total_loss / len(dataloader)
    f1 = f1_score(y_true=labels.detach().cpu().numpy(), y_pred=predictions.detach().cpu().numpy(), average='micro')
    acc = metric.compute()

    return f1, acc, avg_loss

results = {
    'train_acc': [],
    'train_loss': [],
    'train_f1': [],
    'eval_acc': [],
    'eval_loss': [],
    'eval_f1': [],
    'test_acc': 0,
    'test_loss': 0,
    'test_f1': 0
}

model.train()
for epoch in range(num_epochs):
    print(f"EPOCH {epoch}")
    epoch_loss = []
    for batch in tqdm(train_dataloader, unit='batch'):
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)

        logits = model(input_ids)
        # reformat to shape expected by cross entrooy
        logits = logits.view(-1, logits.size(-1))  # (b * seq, v)
        labels = labels.view(-1)  # (b * seq)
        # cross entropy handles the softmax part
        loss = loss_fn(logits, labels)

        # add loss to within epoch list
        epoch_loss.append(loss.item())

        # update weights
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

    # evaluation after every epoch
    # stacks losses of all batches into a num_batches x 1 tensor, gets mean, and converts to py float
    avg_loss_per_epoch = torch.stack(epoch_loss).mean()

    print("TRAIN METRICS")
    f1_t, acc_t, _ = evaluate_model(model, train_dataloader, device=device)  # returns f1, acc, avg_loss in that order
    print(f"F1: {f1_t}")
    print(f"Acc: {acc_t}")
    print(f"Avg Loss: {avg_loss_per_epoch}")
    # keep track of per epoch accuracy/f1/loss on train and eval sets
    results['train_f1'].append(f1_t)
    results['train_acc'].append(acc_t)
    results['train_loss'].append(avg_loss_per_epoch)  # avg training loss/epoch

    print("EVAL METRICS")
    f1_e, acc_e, loss_e = evaluate_model(model, eval_dataloader, device=device)
    print(f"F1: {f1_e}")
    print(f"Acc: {acc_e}")
    print(f"Avg Loss: {loss_e}")

    results['eval_f1'].append(f1_e)
    results['eval_acc'].append(acc_e)
    results['eval_loss'].append(loss_e)

print('Done Training')

f1_test, acc_test, loss_test = evaluate_model(model, test_dataloader, device=device)

results['test_acc'] = acc_test
results['test_f1'] = f1_test
results['test_loss'] = loss_test

save_results(results, model_mode='ingredients_to_directions')

torch.save(model.state_dict(), "./Models/ingredients_to_directions.pth")