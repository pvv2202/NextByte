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
library(textutils)

results <- train_results |>
  filter(set != 'test')

test_row <- train_results |> 
  filter(set == "test")  


title_all <- results |>
  filter(model == 'title_all') |>
  select(-model) |>
  pivot_wider(names_from = "set", values_from = c(
    'accuracy', 'avg_loss', 'f1'
  )) 

textutils::TeXencode(title_all)

title_all |>
  kable(format = "html", digits = 4, 
        col.names = c('Epoch', 'Train', 'Val', 'Train', 'Val', 'Train', 'Val')) |>
  add_header_above(c(" " = 1, 
                     "Accuracy" = 2, 
                     "Avg Loss" = 2, 
                     "F1 Score" = 2)) |>
  kable_styling(bootstrap_options = c("striped", "hover", "condensed", "responsive")) |>
  save_kable(file = 'title_all_res.png')




title_ingr <- results |>
  filter(model == 'title_ing') |>
  select(-model)

title_ingr |>
  pivot_wider(names_from = "set", values_from = c(
    'accuracy', 'avg_loss', 'f1'
  )) |>
  kable(format = "html", digits = 4, 
        col.names = c('Epoch', 'Train', 'Val', 'Train', 'Val', 'Train', 'Val')) |>
  add_header_above(c(" " = 1, 
                     "Accuracy" = 2, 
                     "Avg Loss" = 2, 
                     "F1 Score" = 2)) |>
  kable_styling(bootstrap_options = c("striped", "hover", "condensed", "responsive")) |>
  save_kable(file = 'title_ingr_res.png')



ingr_dir <- results |>
  filter(model == 'ing_dir') |>
  select(-model)

ingr_dir |>
  pivot_wider(names_from = "set", values_from = c(
    'accuracy', 'avg_loss', 'f1'
  )) |>
  kable(format = "html", digits = 4, 
        col.names = c('Epoch', 'Train', 'Val', 'Train', 'Val', 'Train', 'Val')) |>
  add_header_above(c(" " = 1, 
                     "Accuracy" = 2, 
                     "Avg Loss" = 2, 
                     "F1 Score" = 2)) |>
  kable_styling(bootstrap_options = c("striped", "hover", "condensed", "responsive")) |>
  save_kable(file = 'ingr_dir_res.png')

toLatex()








# geom_point(x = 8, y = test_rs$Accuracy) +
#   geom_label_repel(data=test_rs, aes(x = 8, y = Accuracy,
#                                      label = "test set"),
#                    nudge_y = -0.005) + 
