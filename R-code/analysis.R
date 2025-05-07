library(tidyverse)
library(rstatix)
library(broom)
library(kableExtra)
#===============================================================================
# first 10,000 selection bleu scores
#===============================================================================

all_res <- read.csv('./data/all_res.csv') |>
    mutate(model = 'all')

seq_res <- read_csv('./data/seq_res.csv') |>
 mutate(model = 'seq')

#===============================================================================
# Analysis
#===============================================================================

combined <- bind_rows(all_res, seq_res) |>
  select(-trial)

combined |>
  group_by(model)|>
  summarize(
     n =n(),
     avg_bl = mean(bleu),
     avg_p = mean(precision),
     avg_r = mean(recall),
     avg_f1 = mean(f1)
) 

levene_test(bleu~model, data=combined)
levene_test(precision~model, data=combined)
levene_test(recall~model,data=combined)
levene_test(f1~model, data=combined)

t.test(bleu~model, data = combined, var.equal = TRUE)
t.test(precision~model, data = combined, var.equal = TRUE)
t.test(recall~model, data = combined, var.equal = TRUE)
t.test(f1~model, data = combined, var.equal = TRUE)

# visualize

