library(tidyverse)
library(RColorBrewer)
library(ggthemes)
library(ggrepel)
library(kableExtra)



train_results <- read.csv('./data/nextbyte_training_results.csv')

val_data <- train_results |>
  filter(set == 'val')

# we want one graph each for accuracy, avg loss, and f1 across models,
# using the validation data
  
#===============================================================================
# ACCURACY PLOT
#===============================================================================
val_data |>
  ggplot() +
  geom_line(aes(x = epoch, y = accuracy, color = model)) +
  geom_point(aes(x = epoch, y = accuracy, shape=model, color=model), size=2.5) +
  scale_shape_discrete(
    name = "Model",
    labels= c('Ingr -> Dir', 'Title -> All', 'Title -> Ingr')
  ) + 
  scale_color_discrete(
    name = "Model",
    labels= c('Ingr -> Dir', 'Title -> All', 'Title -> Ingr')
  ) + 
  labs(
    x = "Epoch",
    y = "Next Token Prediction Accuracy",
    color = "Model",
    title = "A Cross-Model Accuracy Comparison",
    subtitle = "Performance by epoch on validation set",
    shape = "Model"
  ) +
  theme_minimal()

#===============================================================================
# Avg_loss PLOT
#===============================================================================

val_data |>
  ggplot() +
  geom_line(aes(x = epoch, y = avg_loss, color = model)) +
  geom_point(aes(x = epoch, y = avg_loss, shape=model, color=model), size=2.5) +
  scale_shape_discrete(
    name = "Model",
    labels= c('Ingr -> Dir', 'Title -> All', 'Title -> Ingr')
  ) + 
  scale_color_discrete(
    name = "Model",
    labels= c('Ingr -> Dir', 'Title -> All', 'Title -> Ingr')
  ) + 
  labs(
    x = "Epoch",
    y = "Average Loss",
    color = "Model",
    title = "A Cross-Model Average Loss Comparison",
    subtitle = "Cross entroby loss by epoch on validation set",
    shape = "Model"
  ) +
  theme_minimal()

#===============================================================================
# F1 PLOT
#===============================================================================

val_data |>
  ggplot() +
  geom_line(aes(x = epoch, y = f1, color = model)) +
  geom_point(aes(x = epoch, y = f1, shape=model, color=model), size=2.5) +
  scale_shape_discrete(
    name = "Model",
    labels= c('Ingr -> Dir', 'Title -> All', 'Title -> Ingr')
  ) + 
  scale_color_discrete(
    name = "Model",
    labels= c('Ingr -> Dir', 'Title -> All', 'Title -> Ingr')
  ) + 
  labs(
    x = "Epoch",
    y = "F1",
    color = "Model",
    title = "A Cross-Model F1 Comparison",
    subtitle = "Performance by epoch on validation set",
    shape = "Model"
  ) +
  theme_minimal()


#===============================================================================
# In text summary table of training data
#===============================================================================


#===============================================================================
# Full Training results
#===============================================================================
library(flextable)

results <- train_results 

title_all <- results |>
  filter(model == 'title_all') |>
  select(-model)

flextable(title_all) |>
  merge_v(j='set') |>
  theme_vanilla() |>
  set_header_labels(
    values = list(
      set = 'Set',
      epoch = 'Epoch',
      accuracy = 'Accuracy',
      avg_loss = "Avg Loss",
      f1 = "F1"
    )
  )

title_ingr <- results |>
  filter(model == 'title_ing') |>
  select(-model)

flextable(title_ingr) |>
  merge_v(j='set') |>
  theme_vanilla() |>
  set_header_labels(
    values = list(
      set = 'Set',
      epoch = 'Epoch',
      accuracy = 'Accuracy',
      avg_loss = "Avg Loss",
      f1 = "F1"
    )
  )

ingr_dir <- results |>
  filter(model == 'ing_dir') |>
  select(-model)

flextable(ingr_dir) |>
  merge_v(j='set') |>
  theme_vanilla() |>
  set_header_labels(
    values = list(
      set = 'Set',
      epoch = 'Epoch',
      accuracy = 'Accuracy',
      avg_loss = "Avg Loss",
      f1 = "F1"
    )
  )
 





# geom_point(x = 8, y = test_rs$Accuracy) +
#   geom_label_repel(data=test_rs, aes(x = 8, y = Accuracy,
#                                      label = "test set"),
#                    nudge_y = -0.005) + 
