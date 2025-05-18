import pandas as pd
import torch
import re
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split

class TokenizedRecipeNLGDataset(Dataset):
    """Dataset for the Tokenized RecipeNLG dataset"""
    def __init__(self, df, tokenizer, mode='all'):
        self.df = df
        self.columns = self.df.columns.tolist()
        self.tokenizer = tokenizer
        self.mode = mode
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"RECIPE DEVICE: {self.device}")
        self.recipes = None
        self.recipe_strings = None
        self.generate_strings()

    def __len__(self):
        return len(self.recipes)

    def __getitem__(self, idx):
        """Return a training sample"""
        sample = self.recipe_strings.iloc[idx]
        tokens = self.tokenizer(
            text=sample,
            padding='max_length',
            max_length = 512,
            truncation=True,
            return_tensors='pt'
        )['input_ids'].squeeze(0)

        # inputs, labels
        return {
            'input_ids': tokens[:-1],
            'labels': tokens[1:]
        }

    def generate_strings(self):
        """Return the recipe strings"""
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
        if self.mode == 'all':
            self.recipe_strings = self.recipes.apply(
                lambda row: (
                    f"{row['title'].lower()}<end_title>"
                    f"{row['ingredients']}<end_ingredients>"
                    f"{row['directions']}<end>"
                ),
                axis=1
            )
        elif self.mode == 'title_to_ingredients':
            self.recipe_strings = self.recipes.apply(
                lambda row: (
                    f"{row['title'].lower()}<end_title>"
                    f"{row['ingredients']}<end>"
                ),
                axis=1
            )
        elif self.mode == 'ingredients_to_directions':
            self.recipe_strings = self.recipes.apply(
                lambda row: (
                    f"{row['ingredients']}<end_ingredients>"
                    f"{row['directions']}<end>"
                ),
                axis=1
            )
        elif self.mode == 'nextbyte':
            # Random mask to decide order (seeded for consistency)
            rng = np.random.default_rng(seed=2)
            order_mask = pd.Series(rng.random(len(self.recipes)) < 0.5, index=self.recipes.index)

            def format_row(row, use_title_first):
                if use_title_first:
                    return (
                        f"<start_title>{row['title'].lower()}<end_title>"
                        f"<start_ingredients>{row['ingredients']}<end_ingredients>"
                        f"<start_directions>{row['directions']}<end>"
                    )
                else:
                    return (
                        f"<start_ingredients>{row['ingredients']}<end_ingredients>"
                        f"<start_title>{row['title'].lower()}<end_title>"
                        f"<start_directions>{row['directions']}<end>"
                    )

            self.recipe_strings = self.recipes.apply(
                lambda row: format_row(row, use_title_first=order_mask.loc[row.name]), axis=1
            )

    @staticmethod
    def clean_text(text):
        """Clean to fix recipes with "None" values or non-latin characters"""
        if pd.isna(text):
            return ""
        text = text.lower() # All lowercase
        text = re.sub(r'[^a-z0-9\s.,:;!?()\'"\/%+=\-<>$&#*°~#]', '', text)
        return text

    def filter(self, column, values):
        """Filter the DataFrame to keep only certain values from the columns.
        columns[i] corresponds to values[i]
        """
        for val in values:
            if column in self.columns:
                self.df = self.df[~self.df[column].str.contains(val, na=False)]
            else:
                print(f"Column '{column}' isn't in the dataset")

        self.generate_strings()

class RecipeNLGDataset(Dataset):
    """Dataset for the RecipeNLG dataset"""
    def __init__(self, df, mode='all'):
        self.df = df
        self.mode = mode
        self.columns = self.df.columns.tolist()
        self.recipes = None
        self.recipe_strings = None
        self.generate_strings()

    def __len__(self):
        return len(self.recipes)

    def __getitem__(self, idx):
        """Return a training sample"""
        sample = self.recipe_strings.iloc[idx]
        return sample[:-1], sample[1:]

    def generate_strings(self):
        """Return the recipe strings"""
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
        if self.mode == 'all':
            self.recipe_strings = self.recipes.apply(
                lambda row: (
                    f"{row['title'].lower()}<end_title>"
                    f"{row['ingredients']}<end_ingredients>"
                    f"{row['directions']}<end>"
                ),
                axis=1
            )
        elif self.mode == 'title_to_ingredients':
            self.recipe_strings = self.recipes.apply(
                lambda row: (
                    f"{row['title'].lower()}<end_title>"
                    f"{row['ingredients']}<end>"
                ),
                axis=1
            )
        elif self.mode == 'ingredients_to_directions':
            self.recipe_strings = self.recipes.apply(
                lambda row: (
                    f"{row['ingredients']}<end_ingredients>"
                    f"{row['directions']}<end>"
                ),
                axis=1
            )
        elif self.mode == 'nextbyte':
            # Random mask to decide order (seeded for consistency)
            rng = np.random.default_rng(seed=2)
            order_mask = pd.Series(rng.random(len(self.recipes)) < 0.5, index=self.recipes.index)

            def format_row(row, use_title_first):
                if use_title_first:
                    return (
                        f"<start_title>{row['title'].lower()}<end_title>"
                        f"<start_ingredients>{row['ingredients']}<end_ingredients>"
                        f"<start_directions>{row['directions']}<end>"
                    )
                else:
                    return (
                        f"<start_ingredients>{row['ingredients']}<end_ingredients>"
                        f"<start_title>{row['title'].lower()}<end_title>"
                        f"<start_directions>{row['directions']}<end>"
                    )

            self.recipe_strings = self.recipes.apply(
                lambda row: format_row(row, use_title_first=order_mask.loc[row.name]), axis=1
            )

    @staticmethod
    def clean_text(text):
        """Clean to fix recipes with "None" values or non-latin characters"""
        if pd.isna(text):
            return ""
        text = text.lower() # All lowercase
        text = re.sub(r'[^a-z0-9\s.,:;!?()\'"\/%+=\-<>$&#*°~#]', '', text)
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

    def filter(self, column, values):
        """Filter the DataFrame to keep only certain values from the columns.
        columns[i] corresponds to values[i]
        """
        for val in values:
            if column in self.columns:
                self.df = self.df[~self.df[column].str.contains(val, na=False)]
            else:
                print(f"Column '{column}' isn't in the dataset")

        self.generate_strings()