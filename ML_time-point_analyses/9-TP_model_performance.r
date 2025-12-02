# ==============================================================================
# Evaluation and Visualization of Performances for all final models in 
# Time Point Analyses
# ==============================================================================

# MIT License
# Copyright (c) 2025 Jonas Leins and Hendrik Meyer
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
#functions

# ==========
# code to install needed packages

count_unavailable_packages <- function(vector_package_names){ 
  
  #'@title count packages that are unavailable
  #'@param vector_package_names is a vector containing the names of needed packages as string variables.
  #'@return the lenght of a vector containing the names of all needed packages that are not within the list of installed packages as integer value.
  
  return (length(setdiff(vector_package_names, rownames(installed.packages()))))  
}

install_packages_if_unavailable <- function(vector_package_names){
  
  #'@title A function to install all packages that are not yet installed.
  #'@description The function calls the function count_unavailable_packages and pastes the
  #'input parameter into the install.packages() function if the output of the former isnt equal to 0.
  #'If after the call of install.packages the output of the function count_unavailable_packages still isnt 
  #'equal to zero, this function stops and returns an error message containing not installed packages.
  #'@param vector_package_names is a vector containing the names of needed packages as string variables.
  
  if (count_unavailable_packages(vector_package_names) > 0) {
    install.packages(vector_package_names)
  }
  if (count_unavailable_packages(vector_package_names) > 0) {
    
    stop(paste0("Folgende Pakete konnten nicht installiert werden: ",
                setdiff(vector_package_names, rownames(installed.packages()))))
  }
}

load_packages <- function(vector_package_names){
  
  #'@title a function to load all required packages.
  #'@description The function calls the function install_packages_if_unavailable() to make
  #'sure that all required packages  (param) are installed, and then proceeds to loop through 
  #'the list of packages, calling the library() function to load each element in the input vector.
  #'@param vector_package_names is a vector containing the names of needed packages as string variables. 
  
  install_packages_if_unavailable(vector_package_names)
  
  for(pack in vector_package_names){
    library(pack,character.only = TRUE)
  }
}

calculate_metrics <- function(y_pred, y_pred_class, y_test, pos) {
  # NOTE THAT PRED AND y_test NEED TO HAVE THE SAME LEVELS
  # Calculate AUC
  yscore <- data.frame(y_pred)
  rdb <- cbind(as.factor(y_test),yscore)
  colnames(rdb) = c('y','yscore')
  auc <- roc_auc(rdb, y, yscore, event_level = "second")
  auc <- auc$.estimate
  
  # Calculate confusion matrix
  confusion <- confusionMatrix(as.factor(y_pred_class), as.factor(y_test), positive = pos)
  cm_perf <- as.numeric(confusion$byClass)
  Perf <- append(auc, cm_perf)
  
  # Return metrics
  return(Perf)
}



bootstrap_performance_RF <- function(model, X_val, y_val, n_boot){
  # Initialize storage for performance metrics
  performance <- data.frame()
  
  # Bootstrap process
  for (i in 1:n_boot) {
    
    # Generate bootstrap sample
    bootstrap_indices <- sample(1:nrow(X_val), replace = TRUE)
    bootstrap_X_val <- X_val[bootstrap_indices, ]
    bootstrap_y_val <- y_val[bootstrap_indices, ]
    levels(bootstrap_y_val) <- c("no_event", "event")
    
    # Predict bootstrapped validation dataset
    y_pred <- predict(model, 
                      bootstrap_X_val, 
                      type = "prob") # predict to obtain class probabilities
    
    y_pred_class <- predict(model,
                            bootstrap_X_val, 
                            type = "raw") # re-predicting in order to obtain predicted classes
    
    metrics <- calculate_metrics(y_pred[,2], y_pred_class, bootstrap_y_val, "event")
    row <- append(i, metrics)
    performance <- rbind(performance, row)
  }
  colnames(performance) <- c("Bootstrap Sample", "AUC", "Sensitivity", "Specificity",
                             "Pos Pred Value", "Neg Pred Value", "Precision",
                             "Recall", "F1", "Prevalence", "Detection Rate", 
                             "Detection Prevalence", "Balanced Accuracy")
  performance <- rbind(performance, c("Mean", colMeans(performance[,2:13])))# add mean row
  performance[,2:13] <- as.data.frame(lapply(performance[,2:13], function(x) as.numeric(as.character(x))))
  performance <- rbind(performance,c("SD", sapply(performance[1:n_boot,2:13],sd)))# add standard deviation row
  performance[,2:13] <- as.data.frame(lapply(performance[,2:13], function(x) as.numeric(as.character(x))))
  performance <- rbind(performance, c("Upper", sapply(performance[1:n_boot, 2:13], quantile, probs = 0.975)))
  performance[,2:13] <- as.data.frame(lapply(performance[,2:13], function(x) as.numeric(as.character(x))))
  performance <- rbind(performance, c("Lower", sapply(performance[1:n_boot, 2:13], quantile, probs = 0.025)))
  performance[,2:13] <- as.data.frame(lapply(performance[,2:13], function(x) as.numeric(as.character(x))))
  
  return(performance)
}



# ----
# setup

#load packages

load_packages( c("tidyverse", "dplyr", "data.table", "caret", "pROC", "ggplot2", "ggpubr", "tidymodels", "glmnet", "gglasso"))
setwd("/your/path/")






# ----
# main

# ==========
# load final models



models <- readRDS("/your/path/models_time_points.rds")

# ==========
# load validation dataset
dsets <- readRDS("/your/path/Timepoint_datasets.rds")



# ==========
# extract performances per model and join them in a table

# Example data
set.seed(3010)

# Define number of bootstrap samples
n_boot <- 100


# ----
# Random Forest
# ----

# Generate Full Performance Data Frames
valds <- dsets$praeop$val
X_val <- valds %>% select(-AKI_bin)
y_val <- valds %>% select(AKI_bin)
y_val$AKI_bin <- as.factor(y_val$AKI_bin)
levels(y_val) <- c("no_event", "event") 
bs_perf_preop <- bootstrap_performance_RF(models$praeop,
                                             X_val,
                                             y_val,
                                             n_boot)



valds <- dsets$einleitung$val
X_val <- valds %>% select(-AKI_bin)
y_val <- valds %>% select(AKI_bin)
y_val$AKI_bin <- as.factor(y_val$AKI_bin)
levels(y_val) <- c("no_event", "event") 
bs_perf_induc <- bootstrap_performance_RF(models$einleitung,
                                               X_val,
                                               y_val,
                                               n_boot)

valds <- dsets$einlplus30$val
X_val <- valds %>% select(-AKI_bin)
y_val <- valds %>% select(AKI_bin)
y_val$AKI_bin <- as.factor(y_val$AKI_bin)
levels(y_val) <- c("no_event", "event") 
bs_perf_ind30 <- bootstrap_performance_RF(models$einlplus30,
                                          X_val,
                                          y_val,
                                          n_boot)

valds <- dsets$einlplus60$val
X_val <- valds %>% select(-AKI_bin)
y_val <- valds %>% select(AKI_bin)
y_val$AKI_bin <- as.factor(y_val$AKI_bin)
levels(y_val) <- c("no_event", "event") 
bs_perf_ind60 <- bootstrap_performance_RF(models$einlplus60,
                                          X_val,
                                          y_val,
                                          n_boot)

valds <- dsets$einlplus90$val
X_val <- valds %>% select(-AKI_bin)
y_val <- valds %>% select(AKI_bin)
y_val$AKI_bin <- as.factor(y_val$AKI_bin)
levels(y_val) <- c("no_event", "event") 
bs_perf_ind90 <- bootstrap_performance_RF(models$einlplus90,
                                          X_val,
                                          y_val,
                                          n_boot)

valds <- dsets$einlplus120$val
X_val <- valds %>% select(-AKI_bin)
y_val <- valds %>% select(AKI_bin)
y_val$AKI_bin <- as.factor(y_val$AKI_bin)
levels(y_val) <- c("no_event", "event") 
bs_perf_ind120 <- bootstrap_performance_RF(models$einlplus120,
                                          X_val,
                                          y_val,
                                          n_boot)

valds <- dsets$einlplus150$val
X_val <- valds %>% select(-AKI_bin)
y_val <- valds %>% select(AKI_bin)
y_val$AKI_bin <- as.factor(y_val$AKI_bin)
levels(y_val) <- c("no_event", "event") 
bs_perf_ind150 <- bootstrap_performance_RF(models$einlplus150,
                                          X_val,
                                          y_val,
                                          n_boot)

valds <- dsets$postop$val
X_val <- valds %>% select(-AKI_bin)
y_val <- valds %>% select(AKI_bin)
y_val$AKI_bin <- as.factor(y_val$AKI_bin)
levels(y_val) <- c("no_event", "event") 
bs_perf_postop <- bootstrap_performance_RF(models$postop,
                                          X_val,
                                          y_val,
                                          n_boot)




# aggregate dataframes in list
bs_performance <- list(bs_perf_preop, bs_perf_induc,
                       bs_perf_ind30, bs_perf_ind60,
                       bs_perf_ind90, bs_perf_ind120, 
                       bs_perf_ind150, bs_perf_postop
                       )

saveRDS(bs_performance, "/your/path/bs_perf.rds")

rm(bs_perf_preop, bs_perf_induc,
   bs_perf_ind30, bs_perf_ind60,
   bs_perf_ind90, bs_perf_ind120, 
   bs_perf_ind150, bs_perf_postop)

# create summary dataframes for Mean and Standard deviation

assign <- data.frame(Model = c("PreOP", "indu", "indu+30", "indu+60", "indu+90", "indu+120", "indu+150", "PostOP"))
                     

df_mean <- bind_rows(lapply(bs_performance, function(df) df %>% filter(`Bootstrap Sample` == "Mean")), .id = "Source")
df_mean <- df_mean %>% select(-Source, -`Bootstrap Sample`)
df_mean <- cbind(assign, df_mean)
df_sd <- bind_rows(lapply(bs_performance, function(df) df %>% filter(`Bootstrap Sample` == "SD")), .id = "Source")
df_sd <- df_sd %>% select(-Source, -`Bootstrap Sample`)
df_sd <- cbind(assign, df_sd)


# combined table

means_rnd <- round(df_mean[,-c(1)],3)
sds_rnd <- round(df_sd[,-c(1)],3)

# Combine the rounded means and SDs with a ± sign element-wise
combined_df <- as.data.frame(mapply(function(m, s) paste(m, "±", s), means_rnd, sds_rnd))
combined_df <- cbind(assign, combined_df)

combined_df_reduced <- combined_df %>% select(-c("Pos Pred Value", "Recall", "Prevalence", "Detection Prevalence", "Detection Rate"))


# save data frames

write.csv2(combined_df, "/your/path/final_TP_perf_full.csv")
write.csv2(combined_df_reduced, "/your/path/final_TP_perf_red.csv")




# ==========
# Confidence Interals
# ==========


# Assuming you already have assign:
assign <- data.frame(Model = c("PreOP", "indu", "indu+30", "indu+60", "indu+90", "indu+120", "indu+150", "PostOP"))

# Prepare Lower CI dataframe
df_lower <- bind_rows(lapply(bs_performance, function(df) df %>% filter(`Bootstrap Sample` == "Lower")), .id = "Source")
df_lower <- df_lower %>% select(-Source, -`Bootstrap Sample`)
df_lower <- cbind(assign, df_lower)

# Prepare Upper CI dataframe
df_upper <- bind_rows(lapply(bs_performance, function(df) df %>% filter(`Bootstrap Sample` == "Upper")), .id = "Source")
df_upper <- df_upper %>% select(-Source, -`Bootstrap Sample`)
df_upper <- cbind(assign, df_upper)

# Round both Lower and Upper values
lower_rnd <- round(df_lower[,-c(1)], 3)
upper_rnd <- round(df_upper[,-c(1)], 3)

# Combine the Lower and Upper CI with a "—" (en-dash) separator
combined_ci_df <- as.data.frame(mapply(function(l, u) paste(l, "—", u), lower_rnd, upper_rnd))
combined_ci_df <- cbind(assign, combined_ci_df)

# Optionally, reduce columns if needed (like before)
combined_ci_df_reduced <- combined_ci_df %>%
  select(-c("Pos Pred Value", "Recall", "Prevalence", "Detection Prevalence", "Detection Rate"))


# save data frames

write.csv2(combined_ci_df, "/your/path/final_TP_CI_full.csv")
write.csv2(combined_ci_df_reduced, "/your/path/final_TP_CI_red.csv")



# ----
# end
