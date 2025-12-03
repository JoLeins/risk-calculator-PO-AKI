# ==========================================================================
# Additional model performance analysis, extraction of native Feature 
# Importances from time point specific RFC risk scores
# ==========================================================================

# MIT License
# Copyright (c) 2025 Hendrik Meyer and Jonas Leins
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

rm(list = ls())

# ----
# functions

# ==========
# code to install required packages

count_unavailable_packages <- function(vector_package_names){ 
  
  #'@title count packages that are unavailable
  #'@param vector_package_names a vector containing the names of required packages as character strings.
  #'@return the length of a vector containing the names of all required packages that are not installed.
  
  return (length(setdiff(vector_package_names, rownames(installed.packages()))))  
}

install_packages_if_unavailable <- function(vector_package_names){
  
  #'@title Install all missing packages
  #'@description Calls count_unavailable_packages() and installs packages via install.packages()
  #'if any required packages are missing. If after installation some packages are still missing,
  #'the function stops and prints an error listing the unmet dependencies.
  #'@param vector_package_names vector with names of required packages.
  
  if (count_unavailable_packages(vector_package_names) > 0) {
    install.packages(vector_package_names)
  }
  if (count_unavailable_packages(vector_package_names) > 0) {
    
    stop(paste0("The following packages could not be installed: ",
                setdiff(vector_package_names, rownames(installed.packages()))))
  }
}

load_packages <- function(vector_package_names){
  
  #'@title Load all required packages
  #'@description Ensures packages are installed by calling install_packages_if_unavailable(),
  #'then loads each package with library().
  #'@param vector_package_names vector containing package names as character strings.
  
  install_packages_if_unavailable(vector_package_names)
  
  for(pack in vector_package_names){
    library(pack, character.only = TRUE)
  }
}



# setup
# ----

load_packages(c(
  "tidyverse", "dplyr", "caret", "pROC", "tidymodels", 
  "ranger", "kernlab", "gglasso", "glmnet"
))  # "data.table",
set.seed(3010)





# main
# ----


## Load trained models ######
models <- readRDS("/your/path/models_time_points.rds")

#### Output of performance on the validation dataset ####

# List to store performance metrics
performance_metrics <- list()

# Compute performance for each time point
for (time_point in names(timepoints_var)) {
  # Load the trained model for the current time point
  RF <- models[[time_point]]
  
  # Directly access the validation dataset from the list
  validation_dataset <- timepoints_var[[time_point]][["val"]]
  X_valid <- validation_dataset %>% select(-c(AKI_bin))
  y_valid <- validation_dataset$AKI_bin
  
  # Align y_valid levels with model labels
  y_valid <- factor(y_valid, levels = c("0", "1"), labels = c("no_event", "event"))
  
  # Predict class probabilities
  y_pred <- predict(RF, newdata = X_valid, type = "prob")
  
  # Convert probabilities to binary class labels
  y_pred_binary <- factor(ifelse(y_pred[, "event"] >= 0.5, "event", "no_event"),
                          levels = c("no_event", "event"))
  
  # Compute confusion matrix
  cm <- confusionMatrix(y_pred_binary, y_valid, positive = "event")
  
  # Additional metrics: F1-score and balanced accuracy
  f1_score <- cm$byClass["F1"]
  balanced_accuracy <- cm$byClass["Balanced Accuracy"]
  
  # Prepare data frame for ROC and AUC
  yscore <- data.frame(y_pred[, "event"])
  rdb <- cbind(as.factor(y_valid), yscore)
  colnames(rdb) <- c('y', 'yscore')
  
  # Compute ROC statistics
  roc_stat <- roc_curve(rdb, y, yscore)
  roc_stat$specificity <- 1 - roc_stat$specificity
  auc <- roc_auc(rdb, y, yscore, event_level = "second")$.estimate
  roc_title <- paste('ROC Curve Random Forest (AUC = ', toString(round(auc, 2)), ')', sep = '')
  
  # Plot ROC curve
  ROC <- ggplot(roc_stat, aes(x = sensitivity, y = specificity)) + 
    geom_line(linewidth = 1, color = "blue") +
    labs(x = "False Positive Rate", y = "True Positive Rate") +
    geom_abline(intercept = 0, slope = 1, linetype = "dashed") +
    theme_minimal() +
    theme(legend.title = element_blank()) +
    ggtitle(roc_title)
  
  pdf(paste0("/your/path/ROC_Random_Forest_", time_point, ".pdf"),
      height = 4, width = 6)
  ROC
  dev.off()
  
  # Compute precision–recall statistics
  pr_stat <- pr_curve(rdb, y, yscore, event_level = "second")
  pauc <- pr_auc(rdb, y, yscore, event_level = "second")$.estimate
  pr_title <- paste('Precision Recall Plot Random Forest (AUC = ', toString(round(pauc, 2)), ')', sep = '')
  baseline <- sum(y_valid == "event") / length(y_valid)
  
  # Plot precision–recall curve
  PR <- ggplot(pr_stat, aes(x = recall, y = precision)) + 
    geom_line(linewidth = 1, color = "blue") +
    labs(x = "Recall", y = "Precision") +
    theme_minimal() +
    geom_hline(yintercept = baseline, linetype = "dashed") +
    theme(legend.title = element_blank()) +
    ggtitle(pr_title)
  
  pdf(paste0("/your/path/PR_Random_Forest_", time_point, ".pdf"),
      height = 4, width = 6)
  PR
  dev.off()
}

# Extract feature importances with variable names
feature_importances <- as.data.frame(varImp(RF)$importance)
feature_importances$Variable <- rownames(feature_importances)  # Add variable names
feature_importances <- feature_importances[order(-feature_importances$Overall), ]  # Sort by importance
rownames(feature_importances) <- NULL  # Remove row names for clarity
colnames(feature_importances)[1] <- "Importance"  # Rename column for consistency

# Save feature importances as CSV
write.csv(feature_importances,
          paste0("/your/path/Feature_Importances_RF_", time_point, ".csv"),
          row.names = FALSE)

# Save performance metrics
Performance_RF <- list(
  "pred" = rdb,
  "ConfusionMatrix" = cm,
  "auc" = auc,
  "roc" = roc_stat,
  "roc_plot" = ROC,
  "pr" = pr_stat,
  "pr_plot" = PR,
  "F1_Score" = f1_score,
  "Balanced_Accuracy" = balanced_accuracy,
  "Feature_Importances" = feature_importances
)

saveRDS(Performance_RF,
        paste0("/your/path/Performance_RF_", time_point, ".rds"))

# Store the results in the list
performance_metrics[[time_point]] <- Performance_RF


# Save full performance metrics object
saveRDS(performance_metrics,
        "/your/path/Performance_Metrics_All_Timepoints.rds")

performance_metrics <- readRDS("/your/path/Performance_Metrics_All_Timepoints.rds")

# Prepare data for joint plotting

# Libraries
library(ggplot2)
library(dplyr)
library(ggpubr) # for ggarrange()

# Generate colour palette
gg_color_hue <- function(n) {
  hues = seq(15, 375, length = n + 1)
  hcl(h = hues, l = 65, c = 100)[1:n]
}
pal <- gg_color_hue(9)
pal[6:8] <- pal[5:7]
pal[5] <- "#000000" 

# Time points & plot titles
time_points <- c("praeop", "einleitung", "einlplus30", "einlplus60", "einlplus90", "einlplus120", "einlplus150", "postop")
titles <- c("Preoperative", 
            "Induction", 
            "Induction + 30 min", 
            "Induction + 60 min",
            "Induction + 90 min",
            "Induction + 120 min",
            "Induction + 150 min",
            "Postoperative")

# Empty data frames for ROC and PR curves
roc_combined <- data.frame()
pr_combined <- data.frame()

# Extract ROC and PR data from performance_metrics
for (i in seq_along(time_points)) {
  
  time_point <- time_points[i]
  model_name <- titles[i]  # Display name for the model
  
  # Add ROC data
  roc_temp <- performance_metrics[[time_point]][["roc"]]
  roc_temp$Model <- model_name  # Add model name
  roc_combined <- rbind(roc_combined, roc_temp)
  
  # Add PR data
  pr_temp <- performance_metrics[[time_point]][["pr"]]
  pr_temp$Model <- model_name  # Add model name
  pr_combined <- rbind(pr_combined, pr_temp)
}

# Ensure that time points appear in the intended order
roc_combined$Model <- factor(roc_combined$Model, levels = titles)
pr_combined$Model  <- factor(pr_combined$Model,  levels = titles)

# ================
# ROC curve comparison
# ================
roc_plot <- ggplot(roc_combined, aes(x = sensitivity, y = specificity, color = Model)) + 
  geom_line(linewidth = 0.5) +
  labs(x = "False Positive Rate", y = "True Positive Rate") +
  theme_minimal() +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed") +
  theme(legend.title = element_blank()) +
  theme(legend.text = element_text(size = 10)) +
  scale_color_manual(values = pal)

# Save as PDF
pdf("/your/path/ROC_Comparison.pdf", height = 4, width = 6)
print(roc_plot)
dev.off()

# ================
# Precision–recall curve comparison
# ================
baseline_precision <- 0.1344701  # Adjust here if the baseline changes

pr_plot <- ggplot(pr_combined, aes(x = recall, y = precision, color = Model)) + 
  geom_line(linewidth = 0.5) +
  labs(x = "Recall", y = "Precision") +
  theme_minimal() +
  geom_hline(yintercept = baseline_precision, linetype = "dashed") +
  theme(legend.title = element_blank()) +
  theme(legend.text = element_text(size = 10)) +
  scale_color_manual(values = pal)

# Save as PDF
pdf("/your/path/PR_Comparison.pdf", height = 4, width = 6)
print(pr_plot)
dev.off()

# ================
# Combined plots (ROC + PR)
# ================
# Combined plot
combined <- ggarrange(
  roc_plot, pr_plot, 
  ncol = 2, 
  labels = c("a", "b"), 
  common.legend = TRUE, 
  legend = "bottom"
)

# --- FIRST: save as PDF (vector format) ---
# px -> inches at given DPI (2244x1496 px @ 356 dpi)
pdf_width_in  <- 2244 / 356
pdf_height_in <- 1496 / 356

pdf("/your/path/ROC_PR_combined_timepoints.pdf",
    width = pdf_width_in, height = pdf_height_in, useDingbats = FALSE)
print(combined)
dev.off()

# --- THEN: save as TIF (raster format) ---
tiff("/your/path/ROC_PR_combined.tif",
     units = "px", width = 2244, height = 1496, res = 356, compression = "none")
print(combined)
dev.off()







# ==============================================================================
#### Feature importances #####

plot_top_features <- function(performance_metrics, time_points, titles, output_file) {
  
  top_features_set <- c()
  all_features_list <- list()
  
  rename_feature <- function(var_name) {
    if (grepl("skewness", var_name)) {
      return("skewness (timepoint)")
    } else if (grepl("trend", var_name)) {
      return("trend (timepoint)")
    } else if (grepl("kurtosis", var_name)) {
      return("kurtosis (timepoint)")
    } else if (grepl("std", var_name)) {
      return("std (timepoint)")
    } else if (grepl("mean", var_name)) {
      return("mean (timepoint)")
    } else if (grepl("aucMAPunder65", var_name)) {
      return("aucMAPunder65 (timepoint)")
    } else if (grepl("minMAPcumu1Min", var_name)) {
      return("minMAPcumu1Min (timepoint)")
    } else if (grepl("minMAPcumu5Min", var_name)) {
      return("minMAPcumu5Min (timepoint)")
    } else {
      return(var_name)
    }
  }
  
  for (time_point in time_points) {
    feature_importances <- performance_metrics[[time_point]]$Feature_Importances
    feature_importances$TimePoint <- time_point
    
    feature_importances$Variable <- sapply(feature_importances$Variable, rename_feature)
    
    # Compute ranks
    feature_importances$Rank <- rank(-feature_importances$Importance, ties.method = "min")
    
    # Add only top 5
    top_features_set <- union(
      top_features_set,
      head(feature_importances$Variable[order(-feature_importances$Importance)], 5)
    )
    
    all_features_list[[time_point]] <- feature_importances
  }
  
  # Combine all time points
  all_features_all <- do.call(rbind, all_features_list)
  filtered_features <- all_features_all[all_features_all$Variable %in% top_features_set, ]
  filtered_features$TimePoint <- factor(filtered_features$TimePoint, levels = time_points, labels = titles)
  
  # Colour palette
  gg_color_hue <- function(n) {
    hues <- seq(15, 375, length = n + 1)
    hcl(h = hues, l = 65, c = 100)[1:n]
  }
  pal <- gg_color_hue(length(unique(filtered_features$Variable)))
  
  # Build plot
  feature_importance_plot <- ggplot(filtered_features, aes(x = TimePoint, y = Rank, color = Variable, group = Variable)) +
    geom_line(size = 0.5) +
    geom_point(size = 2) +
    scale_y_reverse(breaks = seq(1, max(filtered_features$Rank)),
                    limits = c(max(filtered_features$Rank), 1)) +
    labs(x = "Time Point", y = "Rank", color = "Variable") +
    theme_minimal() +
    theme(
      panel.grid.major.y = element_line(),
      panel.grid.minor.y = element_blank(),
      legend.position = "right",
      legend.text = element_text(size = 16),
      legend.title = element_text(size = 18),
      axis.text.x = element_text(angle = 45, hjust = 1, size = 14),
      axis.text.y = element_text(size = 14),
      axis.title.x = element_text(size = 18),
      axis.title.y = element_text(size = 18)
    ) +
    scale_color_manual(values = pal)
  
  # Save as PDF
  ggsave(
    filename = sub("\\.tif$", ".pdf", output_file),
    plot = feature_importance_plot,
    width = 13, height = 7, units = "in", dpi = 300
  )
  
  # Save as TIFF
  tiff(output_file, units = "px", width = 2244, height = 1496, res = 356, compression = "none")
  print(feature_importance_plot)
  dev.off()
  
  return(feature_importance_plot)
}

plot_top_features(
  performance_metrics = performance_metrics,
  time_points = time_points,
  titles = titles,
  output_file = "/your/path/Feature_Importance_Selected_Top5_alltimepoints.tif"
)


#### Output feature importance table ####
library(dplyr)
library(tidyr)
library(readr)
library(kableExtra)
library(stringr)

# ---- Logically group feature names ----
rename_feature <- function(var_name) {
  if (grepl("skewness", var_name)) {
    return("skewness (timepoint)")
  } else if (grepl("trend", var_name)) {
    return("trend (timepoint)")
  } else if (grepl("entropy", var_name)) {
    return("entropy (timepoint)")
  } else if (grepl("kurtosis", var_name)) {
    return("kurtosis (timepoint)")
  } else if (grepl("std", var_name)) {
    return("std (timepoint)")
  } else if (grepl("mean", var_name)) {
    return("mean (timepoint)")
  } else if (grepl("aucMAPunder65", var_name)) {
    return("aucMAPunder65 (timepoint)")
  } else if (grepl("minMAPcumu1Min", var_name)) {
    return("minMAPcumu1Min (timepoint)")
  } else if (grepl("minMAPcumu5Min", var_name)) {
    return("minMAPcumu5Min (timepoint)")
  } else {
    return(var_name)
  }
}

# ---- LaTeX-compatible wrapping of variable names ----
latex_wrap_variable <- function(varname) {
  varname <- str_replace_all(varname, "_", "\\\\_")
  
  if (str_detect(varname, "Induplus")) {
    wrapped <- str_replace(varname, "Induplus", "\\\\Induplus")
  } else if (str_detect(varname, "Bolus")) {
    wrapped <- str_replace(varname, "Bolus", "\\\\Bolus")
  } else if (str_detect(varname, "Perfu")) {
    wrapped <- str_replace(varname, "Perfu", "\\\\Perfu")
  } else if (nchar(varname) > 18) {
    first <- substr(varname, 1, 18)
    second <- substr(varname, 19, nchar(varname))
    wrapped <- paste0(first, "\\\\", second)
  } else {
    wrapped <- varname
  }
  
  return(paste0("\\makecell[l]{", wrapped, "}"))
}

# ---- Collect feature importances across time points ----
all_features_list <- list()

for (time_point in time_points) {
  feature_importances <- performance_metrics[[time_point]]$Feature_Importances %>%
    select(Variable, Importance) %>%
    rename(!!time_point := Importance)
  
  all_features_list[[time_point]] <- feature_importances
}

# ---- Wide join across time points ----
feature_importance_wide <- Reduce(function(x, y) full_join(x, y, by = "Variable"), all_features_list)
colnames(feature_importance_wide) <- c("Variable", titles)
feature_importance_wide[is.na(feature_importance_wide)] <- 0
feature_importance_wide[, -1] <- round(feature_importance_wide[, -1], 3)

# ---- Grouping by logical feature categories ----
feature_importance_wide <- feature_importance_wide %>%
  mutate(GroupKey = sapply(Variable, rename_feature))

feature_importance_grouped <- feature_importance_wide %>%
  select(-Variable) %>%
  group_by(GroupKey) %>%
  summarise(across(everything(), ~ max(.x, na.rm = TRUE)), .groups = "drop") %>%
  rename(Variable = GroupKey)

# ---- Compute ranks per time point ----
feature_importance_wide_ranked <- feature_importance_grouped
rank_matrix <- matrix(0, nrow = nrow(feature_importance_grouped), ncol = length(titles))

for (i in seq_along(titles)) {
  title <- titles[i]
  ranks <- rank(-feature_importance_grouped[[title]], ties.method = "min")
  rank_matrix[, i] <- ranks
  feature_importance_wide_ranked[[title]] <- paste0(
    formatC(feature_importance_grouped[[title]], format = "f", digits = 3),
    " (", ranks, ")"
  )
}

# ---- Sort by overall rank (sum of ranks across time points) ----
total_ranks <- rowSums(rank_matrix)
feature_importance_wide_ranked <- feature_importance_wide_ranked %>%
  mutate(SortKey = total_ranks) %>%
  arrange(SortKey) %>%
  select(-SortKey)

# ---- Make variable names LaTeX compatible ----
feature_importance_wide_ranked$Variable <- sapply(
  feature_importance_wide_ranked$Variable,
  latex_wrap_variable
)

# ---- Build LaTeX table ----
latex_table <- feature_importance_wide_ranked %>%
  kbl(format = "latex", booktabs = TRUE, longtable = TRUE,
      escape = FALSE,
      linesep = "",
      caption = "Feature importances and ranks across time points") %>%
  kable_styling(latex_options = c("hold_position", "repeat_header"))

# ---- Save LaTeX table ----
writeLines(latex_table,
           con = "/your/path/Feature_Importance_alltimepoints_sorted.tex")







# ==============================================================================
# Visualization of Bootstrapped performance variation
# ==============================================================================

## Determine confidence intervals for bootstraps

# Load packages
library(dplyr)
library(readr)
library(tidyr)
library(stringr)

# Titles for time points in desired order
titles <- c("Preoperative", 
            "Induction", 
            "Induction + 30 min", 
            "Induction + 60 min",
            "Induction + 90 min",
            "Induction + 120 min",
            "Induction + 150 min",
            "Postoperative")

# Matching file identifiers in corresponding order
file_order <- c("preop", "induc", "ind30", "ind60", "ind90", "ind120", "ind150", "postop")

# Path to the CSV files
files <- list.files(path = "/Users/hendrikmeyer/Desktop/Dr. Arbeit/Publikation/25_02_19_JL_TP", 
                    pattern = "*.csv", full.names = TRUE)

# Relevant metrics
metric_columns <- c("AUC", "Sensitivity", "Specificity", "Pos Pred Value", 
                    "Neg Pred Value", "Precision", "Recall", "F1", 
                    "Prevalence", "Detection Rate", "Detection Prevalence", 
                    "Balanced Accuracy")

# Function to read a CSV with locale and basic cleaning
read_my_csv <- function(file) {
  df <- read_delim(file, 
                   delim = ";", 
                   locale = locale(decimal_mark = ",", grouping_mark = ".")) %>%
    slice(1:100) %>%
    select(-starts_with("..."))
  return(df)
}

# Function to compute confidence intervals
compute_ci <- function(x) {
  lower <- quantile(x, probs = 0.025, na.rm = TRUE)
  upper <- quantile(x, probs = 0.975, na.rm = TRUE)
  mean_val <- mean(x, na.rm = TRUE)
  return(data.frame(Mean = mean_val, CI_lower = lower, CI_upper = upper))
}

# Collect all results
all_ci_results <- list()

for (file in files) {
  # Extract time point from file name
  timepoint_raw <- gsub(".*performance_|\\.csv", "", basename(file))
  title_index <- match(timepoint_raw, file_order)
  if (is.na(title_index)) next
  
  readable_title <- titles[title_index]
  
  # Read and persist data under a descriptive object name
  df_raw <- read_my_csv(file)
  
  assign(paste0("df_raw_", gsub(" ", "_", tolower(readable_title))), df_raw, envir = .GlobalEnv)
  
  # Basic console check
  message("Loaded: ", readable_title, " (", nrow(df_raw), " rows, ", ncol(df_raw), " columns)")
  
  # Restrict to relevant metric columns
  df_metrics <- df_raw[, metric_columns]
  
  # Compute confidence intervals
  ci_df <- df_metrics %>%
    summarise(across(everything(), compute_ci, .names = "{.col}")) %>%
    pivot_longer(cols = everything(), names_to = "Metric", values_to = "Result") %>%
    unnest_wider(Result) %>%
    mutate(Timepoint = readable_title)
  
  all_ci_results[[readable_title]] <- ci_df
}

# Combine and sort CI table
final_ci_table <- bind_rows(all_ci_results) %>%
  mutate(Timepoint = factor(Timepoint, levels = titles)) %>%
  arrange(Timepoint, Metric)

# Preview
print(final_ci_table)

# New column with formatted string: "Mean (CI_lower–CI_upper)"
final_ci_table_formated <- final_ci_table %>%
  mutate(KI_Text = sprintf("%.4f (%.4f – %.4f)", Mean, CI_lower, CI_upper)) %>%
  select(Timepoint, Metric, KI_Text) %>%
  pivot_wider(names_from = Metric, values_from = KI_Text) %>%
  arrange(factor(Timepoint, levels = titles))

write_csv(final_ci_table_formated, "/your/path/Summary_Metrics_Bootstrapping.csv")

#### Plot temporal course of metrics ####
library(ggplot2)
library(ggpubr)

# --- Exact means (from final_TP_perf_red.csv) ---
AUC <- c(0.722, 0.720, 0.726, 0.725, 0.725, 0.722, 0.716, 0.738)
F1  <- c(0.321, 0.317, 0.323, 0.322, 0.322, 0.325, 0.324, 0.326)
Balanced_Accuracy <- c(0.661, 0.659, 0.665, 0.664, 0.663, 0.665, 0.666, 0.670)

# --- Exact CI bounds (from final_TP_CI_red.csv) ---
AUC_lower <- c(0.703, 0.702, 0.703, 0.704, 0.705, 0.702, 0.696, 0.716)
AUC_upper <- c(0.746, 0.740, 0.746, 0.744, 0.745, 0.739, 0.737, 0.755)

F1_lower  <- c(0.299, 0.290, 0.295, 0.298, 0.300, 0.302, 0.298, 0.301)
F1_upper  <- c(0.346, 0.340, 0.345, 0.344, 0.345, 0.348, 0.348, 0.351)

BA_lower  <- c(0.642, 0.643, 0.645, 0.644, 0.643, 0.647, 0.648, 0.648)
BA_upper  <- c(0.683, 0.678, 0.681, 0.680, 0.680, 0.681, 0.683, 0.689)

# Build dataframe
df <- data.frame(
  Timepoint = factor(c(
    "Preoperative", "Induction", "Induction +30 min", "Induction + 60 min", 
    "Induction + 90 min", "Induction +120 min", "Induction + 150 min", "Postoperative"
  ), levels = c(
    "Preoperative", "Induction", "Induction +30 min", "Induction + 60 min", 
    "Induction + 90 min", "Induction +120 min", "Induction + 150 min", "Postoperative"
  )),
  AUC, AUC_lower, AUC_upper,
  F1, F1_lower, F1_upper,
  Balanced_Accuracy, BA_lower, BA_upper
)

# Sanity check
print(df)


# Individual plots with error bars
auc_plot <- ggplot(df, aes(x = Timepoint, y = AUC, group = 1)) +
  geom_line(color = pal[5], size = 1.2) +
  geom_point(color = pal[5], size = 2.5) +
  geom_errorbar(aes(ymin = AUC_lower, ymax = AUC_upper), width = 0.2, color = pal[5]) +
  ylim(min(df$AUC_lower) - 0.01, max(df$AUC_upper) + 0.01) +
  labs(title = "AUC-ROC perioperative", x = "Timepoint", y = "AUC-ROC") +
  theme_minimal() +
  theme(
    plot.title  = element_text(size = 16),
    axis.title.x = element_text(size = 16),
    axis.title.y = element_text(size = 16),
    axis.text.x  = element_text(angle = 45, hjust = 1, size = 12),
    axis.text.y  = element_text(size = 12)
  )

f1_plot <- ggplot(df, aes(x = Timepoint, y = F1, group = 1)) +
  geom_line(color = pal[6], size = 1.2) +
  geom_point(color = pal[6], size = 2.5) +
  geom_errorbar(aes(ymin = F1_lower, ymax = F1_upper), width = 0.2, color = pal[6]) +
  ylim(min(df$F1_lower) - 0.005, max(df$F1_upper) + 0.005) +
  labs(title = "F1 score perioperative", x = "Timepoint", y = "F1 score") +
  theme_minimal() +
  theme(
    plot.title  = element_text(size = 16),
    axis.title.x = element_text(size = 16),
    axis.title.y = element_text(size = 16),
    axis.text.x  = element_text(angle = 45, hjust = 1, size = 12),
    axis.text.y  = element_text(size = 12)
  )

ba_plot <- ggplot(df, aes(x = Timepoint, y = Balanced_Accuracy, group = 1)) +
  geom_line(color = pal[7], size = 1.2) +
  geom_point(color = pal[7], size = 2.5) +
  geom_errorbar(aes(ymin = BA_lower, ymax = BA_upper), width = 0.2, color = pal[7]) +
  ylim(min(df$BA_lower) - 0.005, max(df$BA_upper) + 0.005) +
  labs(title = "BACC perioperative",
       x = "Timepoint",
       y = "Balanced Accuracy") +
  theme_minimal() +
  theme(
    plot.title  = element_text(size = 16),
    axis.title.x = element_text(size = 16),
    axis.title.y = element_text(size = 16),
    axis.text.x  = element_text(angle = 45, hjust = 1, size = 12),
    axis.text.y  = element_text(size = 12)
  )

# Combined plot
combined_plot <- ggarrange(auc_plot, f1_plot, ba_plot,
                           ncol = 3,
                           labels = c("a", "b", "c"),
                           common.legend = FALSE)

# Show plot
print(combined_plot)

# Optionally save as PDF
ggsave("/your/path/combined_metrics_plot_perioperative.pdf", 
       plot = combined_plot, width = 13, height = 5)

# Export as TIF
tiff("/your/path/combined_metrics_plot_perioperative.tif",
     units = "px", width = 2244, height = 1496, res = 356, compression = "none")

print(combined_plot)

dev.off()

# end
# ----
