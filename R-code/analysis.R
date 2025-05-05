library(tidyverse)
library(rstatix)
library(broom)


#===============================================================================
# randomized selection bleu scores
#===============================================================================
all_bleu_rand <- read.csv('./data/all_bleu_rand.csv')
seq_bleu_rand <- read_csv('./data/seq_bleu_rand.csv')

all_bleu_rand <- all_bleu_rand |>
  select(Trial, Bleu) |>
  mutate(Trial = "rand") |>
  mutate(Model = 'all') |>
  drop_na()

seq_bleu_rand <- seq_bleu_rand |>
  select(Trial, Bleu) |>
  mutate(Trial = "rand") |>
  mutate(Model = 'seq') |>
  drop_na()

#===============================================================================
# first 10,000 selection bleu scores
#===============================================================================
all_bleu_ord <- read.csv('./data/all_bleu_ord.csv')
seq_bleu_ord <- read_csv('./data/seq_bleu_ord.csv')

all_bleu_ord <- all_bleu_ord |>
  select(Trial, Bleu) |>
  mutate(Trial = "ord") |>
  mutate(Model = 'all') |>
  drop_na()

seq_bleu_ord <- seq_bleu_ord |>
  select(Trial, Bleu) |>
  mutate(Trial = "ord") |>
  mutate(Model = 'seq') |>
  drop_na()

#===============================================================================
# Analysis
#===============================================================================

rand_trial_comb <- bind_rows(all_bleu_rand, seq_bleu_rand)

ord_trial_comb <- bind_rows(all_bleu_ord, seq_bleu_ord)

levene_rand <- levene_test(Bleu~Model, data=rand_trial_comb)

levene_ord <- levene_test(Bleu~Model, data=ord_trial_comb)

t.test(Bleu ~ Model, data = rand_trial_comb, var.equal = FALSE)

t.test(Bleu ~ Model, data = ord_trial_comb, var.equal = TRUE)
