import kagglehub
import torch
import torch.nn
import numpy as np
import pandas as pd
from torcheval.metrics.functional import bleu_score
import random
from recipe_nlg import RecipeNLGDataset
from models import NextByteTransformer
from Save_Results import save_bleu



def get_bleu(model, num_prompts_per_trial, dataset):
    
    pred_recipes, ground_truth = [], []
    
    for i in range(num_prompts_per_trial):
        # random index for a random recipe
        rand_index = random.randint(dataset.__len__())
        # get the title on its own from the cleaned data frame
        title_prompt = dataset.recipes.iloc[rand_index]['title']
        # get the full recipe string (not tokenized)
        recipe_ref = dataset.recipe_strings.iloc[rand_index]
        pred = model.generate(title_prompt) # replace with actual generation method
        
        pred_recipes.append(pred)
        ground_truth.append(recipe_ref)
        
    bleu = bleu_score(input=pred_recipes, target=ground_truth, n_gram=4) # n_gram = max count of sequential words to compare strings on
    
    # bleu is a 1d tensor, extract the value and convert to an np.float64 first
    return bleu.item()


if __name__ in "__main__":
    
    # same params used in all training
    context_length = 512
    d_model = 512
    num_heads = 8
    num_hidden_layers = 8
    d_hidden = 2048
    num_decoders = 2
    num_epochs = 8
    
    # instantiate plain model
    model = NextByteTransformer(
        vocab_size=20000,
        context_length=context_length,
        d_model=d_model,
        num_heads=num_heads,
        num_hidden_layers=num_hidden_layers,
        d_hidden=d_hidden,
        num_decoders=num_decoders
    ) 
    
    # LOAD MODEL HERE
    model_name = ""
    state_dict_path = ""
    model.load_state_dict(state_dict_path)
      
    
    path = '/home/pvandervort25/.cache/kagglehub/datasets/paultimothymooney/recipenlg/versions/1'
    # Load the dataset
    df = pd.read_csv(path + "/RecipeNLG_dataset.csv", header=0)
    dataset = RecipeNLGDataset(df=df, mode='all')
    
    # up to us
    num_prompts_per_trial = 10000
    num_trials = 10000
    
    results = ['trial', 'bleu_score']
    
    
    for i in range(num_trials):
        results.append([i + 1, get_bleu(model, num_prompts_per_trial, dataset)])
        
    save_bleu(results=results, model_name=model_name)
        
    
    
    
    
    