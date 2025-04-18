import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split
import re

class TokenizedRecipeNLGDataset(Dataset):
    """Dataset for the Tokenized RecipeNLG dataset"""
    def __init__(self, df, tokenizer, mode='all'):
        self.df = df
        self.columns = self.df.columns.tolist()

        # Get only relevant columns
        self.recipes = self.df[['title', 'ingredients', 'directions']].map(self.clean_text)

        # Reformat ingredients
        self.recipes['ingredients'] = self.recipes.apply(
            lambda row: ','.join(i.strip(' "').lower() for i in row['ingredients'].split('",')),
            axis=1
        )

        # Reformat directions
        self.recipes['directions'] = self.recipes.apply(
            lambda row: ' '.join(d.strip(' "').lower() for d in row['directions'].split('",')),
            axis=1
        )

        # Format strings
        match mode:
            case 'all':
                self.recipe_strings = self.recipes.apply(
                    lambda row: (
                        f"{row['title'].lower()}<end_title>"
                        f"{row['ingredients']}<end_ingredients>"
                        f"{row['directions']}<end>"
                    ),
                    axis=1
                )
            case 'title_to_ingredients':
                self.recipe_strings = self.recipes.apply(
                    lambda row: (
                        f"{row['title'].lower()}<end_title>"
                        f"{row['ingredients']}<end>"
                    ),
                    axis=1
                )
            case 'ingredients_to_directions':
                self.recipe_strings = self.recipes.apply(
                    lambda row: (
                        f"{row['ingredients']}<end_ingredients>"
                        f"{row['directions']}<end>"
                    ),
                    axis=1
                )

        self.tokenized_recipes = tokenizer(
            text=self.recipe_strings.tolist(),
            padding=True,
            truncation=True,
            return_tensors='pt'
        )

    def __len__(self):
        return len(self.recipes)

    def __getitem__(self, idx):
        """Return a training sample"""
        sample = self.tokenized_recipes[idx]
        return sample[:-1], sample[1:]

    @staticmethod
    def clean_text(text):
        """Clean to fix recipes with "None" values or non-latin characters"""
        if pd.isna(text):
            return ""
        text = text.lower() # All lowercase
        text = re.sub(r'[^a-z0-9\s.,:;!?()\'"-]', '', text) # Substitute anything that isn't a-z, 0-9, or punctuation
        return text

    def filter(self, columns, values):
        """Filter the DataFrame to keep only certain columns and values from the columns.
        columns[i] corresponds to values[i]"""
        if len(columns) != len(values):
            raise ValueError("Columns and values should be the same length to filter")

        for col, val in zip(columns, values):
            if col in self.columns:
                self.df = self.df[self.df[col].str.contains(val, na=False)]
            else:
                print(f"Column '{col}' isn't in the dataset")

class RecipeNLGDataset(Dataset):
    """Dataset for the RecipeNLG dataset"""
    def __init__(self, df, mode='all'):
        self.df = df
        self.columns = self.df.columns.tolist()

        # Get only relevant columns
        self.recipes = self.df[['title', 'ingredients', 'directions']].map(self.clean_text)

        # Reformat ingredients
        self.recipes['ingredients'] = self.recipes.apply(
            lambda row: ','.join(i.strip(' "').lower() for i in row['ingredients'].split('",')),
            axis=1
        )

        # Reformat directions
        self.recipes['directions'] = self.recipes.apply(
            lambda row: ' '.join(d.strip(' "').lower() for d in row['directions'].split('",')),
            axis=1
        )

        # Format strings
        match mode:
            case 'all':
                self.recipe_strings = self.recipes.apply(
                    lambda row: (
                        f"{row['title'].lower()}<end_title>"
                        f"{row['ingredients']}<end_ingredients>"
                        f"{row['directions']}<end>"
                    ),
                    axis=1
                )
            case 'title_to_ingredients':
                self.recipe_strings = self.recipes.apply(
                    lambda row: (
                        f"{row['title'].lower()}<end_title>"
                        f"{row['ingredients']}<end>"
                    ),
                    axis=1
                )
            case 'ingredients_to_directions':
                self.recipe_strings = self.recipes.apply(
                    lambda row: (
                        f"{row['ingredients']}<end_ingredients>"
                        f"{row['directions']}<end>"
                    ),
                    axis=1
                )

    def __len__(self):
        return len(self.recipes)

    def __getitem__(self, idx):
        """Return a training sample"""
        sample = self.recipe_strings.iloc[idx]
        return sample[:-1], sample[1:]

    @staticmethod
    def clean_text(text):
        """Clean to fix recipes with "None" values or non-latin characters"""
        if pd.isna(text):
            return ""
        text = text.lower() # All lowercase
        text = re.sub(r'[^a-z0-9\s.,:;!?()\'"-]', '', text) # Substitute anything that isn't a-z, 0-9, or punctuation
        return text

    def print_columns(self):
        """Print the column names of the DataFrame."""
        print("Column Names:")
        for col in self.columns:
            print(col)

    def print_unique_values(self, column, keys=None):
        """Print unique values within a certain column. Keys should be a list of two strings. Unique values
        will be between those strings"""
        if column in self.columns:
            print(f"Unique values in {column}:")
            unique_values = self.df[column].unique()
            if keys is None:
                for value in unique_values:
                    print(value)
            else:
                for value in unique_values:
                    if keys in value:
                        print(value)
        else:
            print(f"Column '{column}' isn't in the dataset")

    def filter(self, columns, values):
        """Filter the DataFrame to keep only certain columns and values from the columns.
        columns[i] corresponds to values[i]
        """
        if len(columns) != len(values):
            raise ValueError("Columns and values should be the same length to filter")

        for col, val in zip(columns, values):
            if col in self.columns:
                self.df = self.df[self.df[col].str.contains(val, na=False)]
            else:
                print(f"Column '{col}' isn't in the dataset")