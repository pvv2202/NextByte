# NextByte
This is the code base documentation for our Natural Language Processing Final Project: NextByte. We begin by outlining each sub-directory, followed by a description of all files within the root directory.

SUB FOLDERS

/Models:
Holds the .pth files for the three models we trained. '.pth' is the PyTorch file extension for a model's state dict, making it a lighter weight option to save a model and load it at a later date.

/R-code:
R is arguably still the forefront language for statistical analysis and visualization. Since we had some experience with R, we used it to create visualizations of training and post-training results (included in both the paper and poster). We also conducted independent samples t-tests to compare the generative ability of our models on a variety of relevant metrics

 --- data: holds all training and post-training data collected by our python scripts
	- all_res.csv: the post-train generation data for the 'all' model
	- seq_res.csv: the post-train generation data for the 'seq' model pipeline
	- nextbyte_training_results.csv: comprehensive results of training all three models on train, val, and test sets on a variety of metrics
 --- analysis.R : where the t-tests were conducted
 --- visualizations.R: using ggplot, created the by epoch training results plots available in the paper and presentation 

/Tests: scripts in here are not to be run, they were created to test various components of our project

/Tokenizers: holds the tokenizer files for each model and the separate scripts used to create each one

---/ingredients_to_directions_tokenizer, /title_to_all_tokenizer, /title_to_ingredients_tokenzier:
   -holds the tokenizer data their respective model (it's in the name)
---ingredients_to_directions.py, title_to_all.py, title_to_ingredients.py: Not meant to be executed at this point
   - the scripts that created the tokenizers. Each trained a Word Piece tokenizer on distinct components of each recipe in the dataset

ROOT DIRECTORY FILES:

- Save_Results.py: contains some handy methods for saving the training results and generation results to txt and csv files

- requirements.txt: our list of dependencies, created with pipreqs

- recipe_nlg.py: holds the dataset class objects that clean, format, and tokenize the original recipe dataframe into the desired structure for each model.
 	-RecipeNLGDataset.py: the original class before we created tokenizers
 	-TokenizedRecipeNLGDataset.py: what we actually used for each model. Its __get_item__ method tokenizes recipes on the fly, used by the dataloader objects at training 	 time

- MODEL CODE:
	- PositionalEncoder.py: a from scratch implementation of the positional encoding layer of a transformer
	- FFN.py: a custom feedforward neural net created with the PyTorch nn Module
	- models.py: holds the rest of the components of our NextByte Transformer
	  -- class DecoderBlock: a flexible implementation of one decoder block with masked multi head attention, layer normalization, and a multi layer perceptron (FFN.py)
	  -- class NextByteDecoder: a wrapper that stacks many instances of the DecoderBlock class. Applies layer norm to the output at the end of the stack
	  -- class NextByteTransformer: the class that puts it all together:
		-- an embedding layer -> PositionalEncoder -> NextByteDecoder -> logits, following the standard flow of the input through decoder architecture

- TRAINING CODE:
	- all_training.py: the script that trained the title to all model on a gpu: 
	  instantiates the model and dataset, loads the correct tokenizer, and runs for 8 epochs, evaluating the model at the end of each epoch on the train and validation s	  ets. At the end, it evaluates the model on the test set and saves the comprehensive results on accuracy, loss, and f1, as well as the model state
	- all_training.sb: the slurm script to run all_training.py on FrostByte
	- ingredients_to_directions.py: same as above but for the ingredients to directions model 
	- ingredients_to_directions.sb: slurm script for running the above training file on FrostByte
	- title_to_ingredients.py: same training script but for title to ingredients model
	- title_to_ingredients.sb: slurm script for running the above training file on FrostByte

- post_train_eval.py: this script tests the generative abilities of the 'All' and 'Seq' models on 10,000 recipe title prompts, evaluating performance with BLEU, precision, 
  recall, and f1. It also contains the method for autoregressive generation. The results for each model were saved to a csv file for analysis in R.

- post_train_eval.sb: the slurm script for running the post train evaluation on FrostByte. NLG takes time!

