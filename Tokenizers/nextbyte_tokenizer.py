from tokenizers import Tokenizer, models, trainers, pre_tokenizers, normalizers
from recipe_nlg import RecipeNLGDataset, TokenizedRecipeNLGDataset
from torch.utils.data import DataLoader
from tokenizers.processors import TemplateProcessing
from transformers import PreTrainedTokenizerFast
from pathlib import Path
import torch
import pandas as pd
import kagglehub

if __name__ == "__main__":
    # Download latest version
    path = kagglehub.dataset_download("paultimothymooney/recipenlg")
    # Load the dataset
    df = pd.read_csv(path + "/RecipeNLG_dataset.csv", header=0)

    tokenizer_path = Path("nextbyte_tokenizer")

    if tokenizer_path.exists():
        print("Loading tokenizer")
        hf_tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
    else:
        # Initialize WordPiece tokenizer. Based off of https://huggingface.co/learn/llm-course/en/chapter6/8
        tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]"))

        tokenizer.normalizer = normalizers.Sequence(
            [normalizers.NFD(), normalizers.Lowercase(), normalizers.StripAccents()]
        )

        tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
            [pre_tokenizers.WhitespaceSplit(), pre_tokenizers.Punctuation()]
        )

        # Special tokens and trainer
        special_tokens = ["[PAD]", "[UNK]", "<start_title>", "<end_title>", "<start_ingredients>", "<end_ingredients>", "<start_directions>", "<end>"]
        trainer = trainers.WordPieceTrainer(vocab_size=30000, special_tokens=special_tokens)

        string_dataset = RecipeNLGDataset(df, mode='nextbyte')
        values = ['www.kraftrecipes.com', 'recipes-plus.com', 'www.foodgeeks.com', 'allrecipes.com', 'www.cookbooks.com', 'cookeatshare.com', 'www.landolakes.com', 'cookpad.com']
        string_dataset.filter('link', values)

        # Train the tokenizer
        tokenizer.train_from_iterator(string_dataset, trainer=trainer)

        # Save tokenizer
        hf_tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer)
        hf_tokenizer.add_special_tokens({
            "pad_token": "[PAD]",
            "unk_token": "[UNK]",
            "additional_special_tokens": [
                "<start_title>", "<end_title>", "<start_ingredients>", "<end_ingredients>", "<start_directions>", "<end>"
            ]
        })
        hf_tokenizer.save_pretrained("nextbyte_tokenizer")

    # Test tokenizer
    print("Testing tokenizer")

    test_string = "Chicken tikka masala"

    # Tokenize and encode
    tokenized_ids = hf_tokenizer.encode(test_string)
    tokens = hf_tokenizer.convert_ids_to_tokens(tokenized_ids)

    print("Token IDs:     ", tokenized_ids)
    print("Tokens:        ", tokens)

    # Decode to check round-trip
    decoded = hf_tokenizer.decode(tokenized_ids)
    print("Decoded string:", decoded)

    for token, idx in hf_tokenizer.get_vocab().items():
        print(f"{idx}: {token}")

