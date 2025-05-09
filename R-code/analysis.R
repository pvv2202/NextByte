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

mix_res <- read.csv('./data/mix_res.csv') |>
  mutate(model = 'mix')



#===============================================================================
# Analysis
#===============================================================================

combined <- bind_rows(all_res, seq_res, mix_res) |>
  select(-trial)

summary_stat <- combined |>
  group_by(model)|>
  summarize(
     n =n(),
     avg_bl = mean(bleu),
     std_bl = sd(bleu),
     avg_p = mean(precision),
     std_p = sd(precision),
     avg_r = mean(recall),
     std_r = sd(recall),
     avg_f1 = mean(f1),
     std_f1 = sd(f1)
) 

levene_test(bleu~model, data=combined)
levene_test(precision~model, data=combined)
levene_test(recall~model,data=combined)
levene_test(f1~model, data=combined)

aov_bleu <- aov(bleu~model, data=combined)
summary(aov_bleu)

aov_p <- aov(precision~model, data=combined)
summary(aov_p)

aov_r <- aov(recall~model, data=combined)
summary(aov_r)

aov_f1 <- aov(f1~model, data=combined)
summary(aov_f1)



TukeyHSD(aov_bleu)

TukeyHSD(aov_p)

TukeyHSD(aov_r)

TukeyHSD(aov_f1)

# t.test(bleu~model, data = combined, var.equal = TRUE)
# t.test(precision~model, data = combined, var.equal = TRUE)
# t.test(recall~model, data = combined, var.equal = TRUE)
# t.test(f1~model, data = combined, var.equal = TRUE)

# visualize

