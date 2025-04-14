import kagglehub
import pandas as pd
from recipe_nlg import RecipeNLGDataset

# Download latest version
path = kagglehub.dataset_download("paultimothymooney/recipenlg")
# Load the dataset
df = pd.read_csv(path + "/RecipeNLG_dataset.csv", header=0)

# Create an instance of the RecipeNLGDataset class
dataset = RecipeNLGDataset(df=df, tokenizer=None, mode='all')

# Print unique values in certain columns
print(dataset.__getitem__(0)[0])
print(dataset.__getitem__(0)[1])