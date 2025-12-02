# ==============================================================================
# Evaluation and Visualization of Performances for all of the final models
# 100 bootstrap replicates are performed for variance estimation
# ==============================================================================

# MIT License
# Copyright (c) 2025 Jonas Leins
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

bootstrap_performance_SVM <- function(model, X_val, y_val, n_boot){
  # Initialize storage for performance metrics
  performance <- data.frame()
  
  # Bootstrap process
  for (i in 1:n_boot) {
    
    # Generate bootstrap sample
    bootstrap_indices <- sample(1:nrow(X_val), replace = TRUE)
    bootstrap_X_val <- X_val[bootstrap_indices, ]
    bootstrap_y_val <- y_val[bootstrap_indices,]
    levels(bootstrap_y_val) <- c("no_event", "event")
    
    # Predict bootstrapped validation dataset
    y_pred <- predict(model, 
                      bootstrap_X_val, 
                      type = "prob") # predict to obtain class probabiities
    
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

bootstrap_performance_xGBM <- function(model, X_val, y_val, n_boot){
  # Initialize storage for performance metrics
  performance <- data.frame()
  
  # Bootstrap process
  for (i in 1:n_boot) {
    
    # Generate bootstrap sample
    bootstrap_indices <- sample(1:nrow(X_val), replace = TRUE)
    bootstrap_X_val <- X_val[bootstrap_indices, ]
    bootstrap_y_val <- y_val[bootstrap_indices, ]
    levels(bootstrap_y_val) <- c("0", "1")
    
    # Predict bootstrapped validation dataset
    y_pred <- predict(model, 
                      bootstrap_X_val, 
                      type = "prob") # predict to obtain class probabiities
    
    y_pred_class <- as.factor(ifelse(y_pred$`1` > 0.5, 1, 0)) # no direkt implementation available
    
    metrics <- calculate_metrics(y_pred$`1`, y_pred_class, bootstrap_y_val, "1")
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

bootstrap_performance_grLASSO <- function(model, X_val, y_val, n_boot){
  # Initialize storage for performance metrics
  performance <- data.frame()
  
  # Bootstrap process
  for (i in 1:n_boot) {
    # Generate bootstrap sample
    bootstrap_indices <- sample(1:nrow(X_val), replace = TRUE)
    bootstrap_X_val <- X_val[bootstrap_indices, ]
    bootstrap_y_val <- y_val[bootstrap_indices, ]
    levels(bootstrap_y_val) <- c("-1", "1")
    
    # Predict bootstrapped validation dataset
    y_pred <- predict(model, 
                      s = model$lambda.min,
                      bootstrap_X_val, 
                      type = "link") # predict to obtain class probabiities
    
    y_pred_class <- predict(model,
                            s = model$lambda.min,
                            bootstrap_X_val, 
                            type = "class") # re-predicting in order to obtain predicted classes
    
    metrics <- calculate_metrics(y_pred, y_pred_class, bootstrap_y_val, "1")
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

bootstrap_performance_ngrLASSO <- function(model, X_val, y_val, n_boot){
  # Initialize storage for performance metrics
  performance <- data.frame()
  
  # Bootstrap process
  for (i in 1:n_boot) {
    # Generate bootstrap sample
    bootstrap_indices <- sample(1:nrow(X_val), replace = TRUE)
    bootstrap_X_val <- X_val[bootstrap_indices, ]
    bootstrap_y_val <- y_val[bootstrap_indices, ]
    
    # Predict bootstrapped validation dataset
    y_pred <- predict(model, 
                      s = model$lambda.min,
                      bootstrap_X_val, 
                      type = "response") # predict to obtain class probabiities
    
    y_pred_class <- predict(model,
                            s = model$lambda.min,
                            bootstrap_X_val, 
                            type = "class") # re-predicting in order to obtain predicted classes
    
    metrics <- calculate_metrics(y_pred, y_pred_class, bootstrap_y_val, "1")
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
# load and aggregate final models



models <- readRDS("/your/path/final_models.rds")

# ==========
# load validation dataset
valds <- readRDS("/your/path/valds_prep.rds")
X_val <- valds %>% select(-AKI_bin)
y_val <- valds %>% select(AKI_bin)

# ==========
# extract performances per model and join them in a table

# Example data
set.seed(3010)

# Define number of bootstrap samples
n_boot <- 100 



# ----
# SVM
# ----

# Generate Full Performance Data Frames
bs_perf_bal_SVM <- bootstrap_performance_SVM(models$balanced$svm,
                                             X_val,
                                             y_val,
                                             n_boot)

saveRDS(bs_perf_bal_SVM, "/your/path/SVM_bal_bs.rds")

bs_perf_unbal_SVM <- bootstrap_performance_SVM(models$unbalanced$svm,
                                             X_val,
                                             y_val,
                                             n_boot)
saveRDS(bs_perf_unbal_SVM, "/your/path/SVM_unbal_bs.rds")



# ----
# Random Forest
# ----

# Generate Full Performance Data Frames
bs_perf_bal_RF <- bootstrap_performance_RF(models$balanced$r_forest,
                                             X_val,
                                             y_val,
                                             n_boot)
saveRDS(bs_perf_bal_RF, "/your/path/RF_bal_bs.rds")

bs_perf_unbal_RF <- bootstrap_performance_RF(models$unbalanced$r_forest,
                                               X_val,
                                               y_val,
                                               n_boot)
saveRDS(bs_perf_unbal_RF, "/your/path/RF_unbal_bs.rds")



# ----
# xGBM
# ----

# ==========
# separate continuous and discrete predictors

continuousNames <- c("baselineMAP",
                     "inductDurat",
                     "anestDurat",
                     "age",
                     "height",
                     "bmi",
                     "doseInduEtomidateBolus",
                     "doseInduPropoBolus",
                     "doseInduPropoPerfu",
                     "doseInduRemifPerfu",
                     "doseInduSufenBolus",
                     "doseInduSufenPerfu",
                     "doseInduThiopBolus",
                     "doseSurgGelafInf",
                     "doseSurgSteroInf",
                     "maxSevoExp",
                     "eGFRPreSurg",
                     "minMAPcumu1MinAnes",
                     "minMAPcumu5MinAnes",
                     "aucMAPunder65Anes",
                     "meanAnes",              
                     "stdAnes",                
                     "entropyAnes",            
                     "trendAnes",             
                     "kurtosisAnes",           
                     "skewnessAnes") # vector containing the names of all continuous features

cont_val <- X_val %>% select(all_of(continuousNames))
disc_val <- X_val %>% select(- c(all_of(continuousNames)))

rm(continuousNames)

# encoding discrete data in dummy variables
dummies <- fastDummies::dummy_cols(disc_val[, 2:5], remove_first_dummy = TRUE)
disc_val <- disc_val[,-c(2:5)] # remove multi-level features
disc_val <- cbind(disc_val, dummies[, 5:18])

# convert predictor dataframes (X_train, X_test) to matrix
X_val <- as.matrix(cbind(disc_val, cont_val))

# convert Matrix from character to numeric
X_val <- apply(X_val, 2 ,as.numeric)



# Generate Full Performance Data Frames

# ======
# linear xGBM
bs_perf_bal_xGBM_lin <- bootstrap_performance_xGBM(models$balanced$lin_xGBM,
                                             X_val,
                                             y_val,
                                             n_boot)
saveRDS(bs_perf_bal_xGBM_lin, "/your/path/xGBM_lin_bal_bs.rds")

bs_perf_unbal_xGBM_lin <- bootstrap_performance_xGBM(models$unbalanced$lin_xGBM,
                                               X_val,
                                               y_val,
                                               n_boot)
saveRDS(bs_perf_unbal_xGBM_lin, "/your/path/xGBM_lin_unbal_bs.rds")

# ======
# tree-based xGBM
bs_perf_bal_xGBM_tree <- bootstrap_performance_xGBM(models$balanced$tree_xGBM,
                                                   X_val,
                                                   y_val,
                                                   n_boot)
saveRDS(bs_perf_bal_xGBM_tree, "/your/path/xGBM_tree_bal_bs.rds")

bs_perf_unbal_xGBM_tree <- bootstrap_performance_xGBM(models$unbalanced$tree_xGBM,
                                                     X_val,
                                                     y_val,
                                                     n_boot)
saveRDS(bs_perf_unbal_xGBM_tree, "/your/path/xGBM_tree_unbal_bs.rds")



# ----
# Regression
# ----

# Generate Full Performance Data Frames

# =====
# Group-LASSO
bs_perf_bal_grLASSO <- bootstrap_performance_grLASSO(models$balanced$group_lasso,
                                                   X_val,
                                                   y_val,
                                                   n_boot)
saveRDS(bs_perf_bal_grLASSO, "/your/path/grLASSO_bal_bs.rds")

bs_perf_unbal_grLASSO <- bootstrap_performance_grLASSO(models$unbalanced$group_lasso,
                                                     X_val,
                                                     y_val,
                                                     n_boot)
saveRDS(bs_perf_unbal_grLASSO, "/your/path/grLASSO_unbal_bs.rds")

# =====
# non-Group-LASSO
bs_perf_bal_ngrLASSO <- bootstrap_performance_ngrLASSO(models$balanced$lasso,
                                                     X_val,
                                                     y_val,
                                                     n_boot)
saveRDS(bs_perf_bal_ngrLASSO, "/your/path/LASSO_bal_bs.rds")

bs_perf_unbal_ngrLASSO <- bootstrap_performance_ngrLASSO(models$unbalanced$lasso,
                                                       X_val,
                                                       y_val,
                                                       n_boot)
saveRDS(bs_perf_unbal_ngrLASSO, "/your/path/LASSO_unbal_bs.rds")

# =====
# Ridge
bs_perf_bal_Ridge <- bootstrap_performance_ngrLASSO(models$balanced$ridge,
                                                       X_val,
                                                       y_val,
                                                       n_boot)
saveRDS(bs_perf_bal_Ridge, "/your/path/Ridge_bal_bs.rds")

bs_perf_unbal_Ridge <- bootstrap_performance_ngrLASSO(models$unbalanced$ridge,
                                                         X_val,
                                                         y_val,
                                                         n_boot)
saveRDS(bs_perf_unbal_Ridge, "/your/path/Ridge_unbal_bs.rds")

# =====
# Elastic Net
bs_perf_bal_Enet <- bootstrap_performance_ngrLASSO(models$balanced$enet,
                                                       X_val,
                                                       y_val,
                                                       n_boot)
saveRDS(bs_perf_bal_Enet, "/your/path/Enet_bal_bs.rds")

bs_perf_unbal_Enet <- bootstrap_performance_ngrLASSO(models$unbalanced$enet,
                                                         X_val,
                                                         y_val,
                                                         n_boot)
saveRDS(bs_perf_unbal_Enet, "/your/path/Enet_unbal_bs.rds")




# aggregate dataframes in list
bs_performance <- list(bs_perf_bal_grLASSO, bs_perf_bal_ngrLASSO,
                       bs_perf_bal_Ridge, bs_perf_bal_Enet,
                       bs_perf_bal_SVM, bs_perf_bal_xGBM_lin, 
                       bs_perf_bal_xGBM_tree, bs_perf_bal_RF,
                       bs_perf_unbal_grLASSO, bs_perf_unbal_ngrLASSO,
                       bs_perf_unbal_Ridge, bs_perf_unbal_Enet,
                       bs_perf_unbal_SVM, bs_perf_unbal_xGBM_lin, 
                       bs_perf_unbal_xGBM_tree, bs_perf_unbal_RF)
saveRDS(bs_performance, "/your/path/All_performance_bs.rds")

rm(bs_perf_bal_grLASSO, bs_perf_bal_ngrLASSO,
   bs_perf_bal_Ridge, bs_perf_bal_Enet,
   bs_perf_bal_SVM, bs_perf_bal_xGBM_lin, 
   bs_perf_bal_xGBM_tree, bs_perf_bal_RF,
   bs_perf_unbal_grLASSO, bs_perf_unbal_ngrLASSO,
   bs_perf_unbal_Ridge, bs_perf_unbal_Enet,
   bs_perf_unbal_SVM, bs_perf_unbal_xGBM_lin, 
   bs_perf_unbal_xGBM_tree, bs_perf_unbal_RF)

# create summary dataframes for Mean and Standard deviation

assign <- data.frame(Data=c("Bal", "Bal", "Bal", "Bal", "Bal", "Bal", "Bal", "Bal",
                            "Unbal", "Unbal", "Unbal", "Unbal", "Unbal", "Unbal", "Unbal", "Unbal"),
                     Model = c("Group LASSO", "LASSO", "Ridge", "Elastic Net", "SVM", "Linear xGBM", "Tree xGBM", "RF",
                               "Group LASSO", "LASSO", "Ridge", "Elastic Net", "SVM", "Linear xGBM", "Tree xGBM", "RF"))
                     

df_mean <- bind_rows(lapply(bs_performance, function(df) df %>% filter(`Bootstrap Sample` == "Mean")), .id = "Source")
df_mean <- df_mean %>% select(-Source, -`Bootstrap Sample`)
df_mean <- cbind(assign, df_mean)
df_sd <- bind_rows(lapply(bs_performance, function(df) df %>% filter(`Bootstrap Sample` == "SD")), .id = "Source")
df_sd <- df_sd %>% select(-Source, -`Bootstrap Sample`)
df_sd <- cbind(assign, df_sd)


# combined table

means_rnd <- round(df_mean[,-c(1:2)],3)
sds_rnd <- round(df_sd[,-c(1:2)],3)

# Combine the rounded means and SDs with a ± sign element-wise
combined_df <- as.data.frame(mapply(function(m, s) paste(m, "±", s), means_rnd, sds_rnd))
combined_df <- cbind(assign, combined_df)

combined_df_reduced <- combined_df %>% select(-c("Pos Pred Value", "Recall", "Prevalence", "Detection Prevalence", "Detection Rate"))


# save data frames

write.csv2(combined_df, "/your/path/final_m_perf_full.csv")
write.csv2(combined_df_reduced, "/your/path/final_m_perf_red.csv")


# Prepare Lower CI dataframe
df_lower <- bind_rows(lapply(bs_performance, function(df) df %>% filter(`Bootstrap Sample` == "Lower")), .id = "Source")
df_lower <- df_lower %>% select(-Source, -`Bootstrap Sample`)
df_lower <- cbind(assign, df_lower)

# Prepare Upper CI dataframe
df_upper <- bind_rows(lapply(bs_performance, function(df) df %>% filter(`Bootstrap Sample` == "Upper")), .id = "Source")
df_upper <- df_upper %>% select(-Source, -`Bootstrap Sample`)
df_upper <- cbind(assign, df_upper)

# Round both Lower and Upper values
lower_rnd <- round(df_lower[,-c(1:2)],3)
upper_rnd <- round(df_upper[,-c(1:2)],3)

# Combine the Lower and Upper CI with a "—" (en-dash) separator
combined_ci_df <- as.data.frame(mapply(function(l, u) paste(l, "—", u), lower_rnd, upper_rnd))
combined_ci_df <- cbind(assign, combined_ci_df)

# Optionally, reduce columns if needed (like before)
combined_ci_df_reduced <- combined_ci_df %>%
  select(-c("Pos Pred Value", "Recall", "Prevalence", "Detection Prevalence", "Detection Rate"))


# save data frames

write.csv2(combined_ci_df, "/your/path/final_m_CI_full.csv")
write.csv2(combined_ci_df_reduced, "/your/path/final_m_CI_red.csv")





# ==========
# load other performance data for visualisation

# balanced
log_reg_bal <- readRDS("/your/path/Performance_logistic_regression_bal.rds")
rdm_for_bal <- readRDS("/your/path/Performance_RF_balanced.rds")
svm_bal <- readRDS("/your/path/Performance_SVM_bal.rds")
xgbm_bal <- readRDS("/your/path/Performance_xGBM_balanced.rds")

# unbalanced
log_reg_unbal <- readRDS("/your/path/Performance_logistic_regression_unbal.rds")
rdm_for_unbal <- readRDS("/your/path/Performance_RF_unbalanced.rds")
svm_unbal <- readRDS("/your/path/Performance_SVM_unbal.rds")
xgbm_unbal <- readRDS("/your/path/Performance_xGBM_unbalanced.rds")

# create name vectors
mnames_vec <- c("Group LASSO Regression balanced", 
                "Non-Group LASSO Regression balanced",
                "Ridge Regression balanced",
                "Elastic Net Regression balanced",
                "Group LASSO Regression unbalanced",
                "Non-Group LASSO Regression unbalanced",
                "Ridge Regression unbalanced",
                "Elastic Net Regression unbalanced",
                "Random Forest balanced", 
                "Random Forest unbalanced",
                "Support Vector Machine balanced", 
                "Support Vector Machine unbalanced",
                "Linear xGBM balanced", 
                "Tree xGBM balanced",
                "Linear xGBM unbalanced",
                "Tree xGBM unbalanced")

colnames_performance <- append( "AUC", names(log_reg_bal$group_lasso$ConfusionMatrix$byClass))


# create dataframe
performances <- data.frame()

# Logistic Regression Models
for(model in names(log_reg_bal)){
  perf <- as.numeric(log_reg_bal[[model]]$ConfusionMatrix$byClass)
  auc <- log_reg_bal[[model]]$auc
  row <- append(auc, perf)
  performances <- rbind(performances, row)
}

for(model in names(log_reg_unbal)){
  perf <- as.numeric(log_reg_unbal[[model]]$ConfusionMatrix$byClass)
  auc <- log_reg_unbal[[model]]$auc
  row <- append(auc, perf)
  performances <- rbind(performances, row)
}



# Random Forest model
perf <- as.numeric(rdm_for_bal$ConfusionMatrix$byClass)
auc <- rdm_for_bal$auc
row <- append(auc, perf)
performances <- rbind(performances, row)

perf <- as.numeric(rdm_for_unbal$ConfusionMatrix$byClass)
auc <- rdm_for_unbal$auc
row <- append(auc, perf)
performances <- rbind(performances, row)



# SVM
perf <- as.numeric(svm_bal$ConfusionMatrix$byClass)
auc <- svm_bal$auc
row <- append(auc, perf)
performances <- rbind(performances, row)

perf <- as.numeric(svm_unbal$ConfusionMatrix$byClass)
auc <- svm_unbal$auc
row <- append(auc, perf)
performances <- rbind(performances, row)



# Extreme Gradient Boosting Models
for(model in names(xgbm_bal)){
  perf <- as.numeric(xgbm_bal[[model]]$ConfusionMatrix$byClass)
  auc <- xgbm_bal[[model]]$auc
  row <- append(auc, perf)
  performances <- rbind(performances, row)
}

for(model in names(xgbm_unbal)){
  perf <- as.numeric(xgbm_unbal[[model]]$ConfusionMatrix$byClass)
  auc <- xgbm_unbal[[model]]$auc
  row <- append(auc, perf)
  performances <- rbind(performances, row)
}



rownames(performances) <- mnames_vec
colnames(performances) <- colnames_performance

rm(auc, model, perf, row, colnames_performance, mnames_vec)


# save table

write.csv2(performances, "/your/path/Performance_final_models.csv")



# ==============================================================================
# Visualize Roc and PR together

gg_color_hue <- function(n) {
  hues = seq(20, 350, length = n + 1)
  hcl(h = hues, l = 65, c = 100, fixup = TRUE)[1:n]
}
pal <- gg_color_hue(9)
pal[6:8] <- pal[5:7]
pal[5] <- "#000000"


#gg_custom_colors <- function(n) {
#  hues <- seq(60, 240, length = n)  # 120 = green, 60 = yellow, 240 = blue
#  hcl(h = hues, l = 75, c = 100)
#}

#pal <- gg_custom_colors(9)
#pal[5] <- "#000000"  # Keep black at index 5 if needed


#gg_custom_gradient <- colorRampPalette(c("green", "yellow", "blue"))
#pal <- gg_custom_gradient(9)
#pal[5] <- "#000000"  # Set fifth color to black

show_palette <- function(pal) {
  barplot(rep(1, length(pal)), col = pal, border = NA, space = 0, axes = FALSE)
}

#library(RColorBrewer)
#pal <- brewer.pal(9, "Set1")
#pal[6:9] <- pal[5:8]
#pal[5] <- "#000000"
show_palette(pal)

#library(viridis)
#pal <- viridis(9, option = "D")
#show_palette(pal)

# ==============================================================================
# Balanced
# ==============================================================================

# ==========
# ROC Curves


# create Dataframe of ROC Curves
roc1_df <- data.frame(fpr = log_reg_bal$group_lasso$roc$sensitivity, 
                      tpr = log_reg_bal$group_lasso$roc$specificity, Model = "Group Lasso")
roc2_df <- data.frame(fpr = log_reg_bal$ngr_lasso$roc$sensitivity, 
                      tpr = log_reg_bal$ngr_lasso$roc$specificity, Model = "Lasso")
roc3_df <- data.frame(fpr = log_reg_bal$ridge$roc$sensitivity, 
                      tpr = log_reg_bal$ridge$roc$specificity, Model = "Ridge")
roc4_df <- data.frame(fpr = log_reg_bal$elastic_net$roc$sensitivity, 
                      tpr = log_reg_bal$elastic_net$roc$specificity, Model = "Elastic Net")
roc5_df <- data.frame(fpr = rdm_for_bal$roc$sensitivity, 
                      tpr = rdm_for_bal$roc$specificity, Model = "Random Forest")
roc6_df <- data.frame(fpr = svm_bal$roc$sensitivity, 
                      tpr = svm_bal$roc$specificity, Model = "SVM")
roc7_df <- data.frame(fpr = xgbm_bal$Linear$roc$sensitivity, 
                      tpr = xgbm_bal$Linear$roc$specificity, Model = "Linear xGBM")
roc8_df <- data.frame(fpr = xgbm_bal$Tree$roc$sensitivity, 
                      tpr = xgbm_bal$Tree$roc$specificity, Model = "Tree xGBM")


roc_df <- rbind(roc1_df, roc2_df, roc3_df, roc4_df, roc5_df,roc6_df, roc7_df, roc8_df)



# Draw Curves
ROC_bal <- ggplot(roc_df, aes(x = fpr, y = tpr, color = Model)) + 
  geom_line(linewidth = .7) +
  labs(x = "False Positive Rate", y = "True Positive Rate") +
  theme_minimal() +
  geom_abline(aes(intercept = 0, slope = 1, colour = "Random Classifier"), linetype = "dashed")+
  theme(legend.title = element_blank())+
  scale_color_manual(values = pal)

ROC_bal

#pdf("/your/path/ROC_all_models_balanced.pdf", height = 6, width = 9)
tiff("/your/path/ROC_all_models_balanced.tif", 
     units="px", width=2244, height=1795, res=356, compression = 'none') #1496

ROC_bal

dev.off()

# ==========
# Precision Recall Plot


# create Dataframe of PR plots

pr1_df <- data.frame(prec = log_reg_bal$group_lasso$pr$precision, 
                     rec = log_reg_bal$group_lasso$pr$recall, model = "Group Lasso")
pr2_df <- data.frame(prec = log_reg_bal$ngr_lasso$pr$precision, 
                     rec = log_reg_bal$ngr_lasso$pr$recall, model = "Lasso")
pr3_df <- data.frame(prec = log_reg_bal$ridge$pr$precision, 
                     rec = log_reg_bal$ridge$pr$recall, model = "Ridge")
pr4_df <- data.frame(prec = log_reg_bal$elastic_net$pr$precision, 
                     rec = log_reg_bal$elastic_net$pr$recall, model = "Elastic Net")
pr5_df <- data.frame(prec = rdm_for_bal$pr$precision, 
                     rec = rdm_for_bal$pr$recall, model = "Random Forest")
pr6_df <- data.frame(prec = svm_bal$pr$precision, 
                     rec = svm_bal$pr$recall, model = "SVM")
pr7_df <- data.frame(prec = xgbm_bal$Linear$pr$precision, 
                     rec = xgbm_bal$Linear$pr$recall, model = "Linear xGBM")
pr8_df <- data.frame(prec = xgbm_bal$Tree$pr$precision, 
                     rec = xgbm_bal$Tree$pr$recall, model = "Tree xGBM")


pr_df <- rbind(pr1_df, pr2_df, pr3_df, pr4_df, pr5_df,pr6_df, pr7_df, pr8_df)



# Draw Curves
PR_bal <- ggplot(pr_df, aes(x = rec, y = prec, color = model)) + 
  geom_line(size = 0.7) +
  labs(x = "Recall", y = "Precision") +
  theme_minimal() +
  geom_hline(aes(yintercept = 0.1344701, colour = "Random Classifier"), linetype="dashed") +
  theme(legend.title = element_blank())+
  scale_color_manual(values = pal)

PR_bal

#pdf("/your/path/PR_all_models_balanced.pdf", height = 6, width = 9)
tiff("/your/path/PR_all_models_balanced.tif", 
     units="px", width=2244, height=1795, res=356, compression = 'none') #1496

PR_bal

dev.off()









# ==============================================================================
# Unbalanced
# ==============================================================================

# ==========
# ROC Curves


# create Dataframe of ROC Curves
roc1_df <- data.frame(fpr = log_reg_unbal$group_lasso$roc$sensitivity, 
                      tpr = log_reg_unbal$group_lasso$roc$specificity, Model = "Group Lasso")
roc2_df <- data.frame(fpr = log_reg_unbal$ngr_lasso$roc$sensitivity, 
                      tpr = log_reg_unbal$ngr_lasso$roc$specificity, Model = "Lasso")
roc3_df <- data.frame(fpr = log_reg_unbal$ridge$roc$sensitivity, 
                      tpr = log_reg_unbal$ridge$roc$specificity, Model = "Ridge")
roc4_df <- data.frame(fpr = log_reg_unbal$elastic_net$roc$sensitivity, 
                      tpr = log_reg_unbal$elastic_net$roc$specificity, Model = "Elastic Net")
roc5_df <- data.frame(fpr = rdm_for_unbal$roc$sensitivity, 
                      tpr = rdm_for_unbal$roc$specificity, Model = "Random Forest")
roc6_df <- data.frame(fpr = svm_unbal$roc$sensitivity, 
                      tpr = svm_unbal$roc$specificity, Model = "SVM")
roc7_df <- data.frame(fpr = xgbm_unbal$Linear$roc$sensitivity, 
                      tpr = xgbm_unbal$Linear$roc$specificity, Model = "Linear xGBM")
roc8_df <- data.frame(fpr = xgbm_unbal$Tree$roc$sensitivity, 
                      tpr = xgbm_unbal$Tree$roc$specificity, Model = "Tree xGBM")


roc_df <- rbind(roc1_df, roc2_df, roc3_df, roc4_df, roc5_df,roc6_df, roc7_df, roc8_df)



# Draw Curves
ROC_unbal <- ggplot(roc_df, aes(x = fpr, y = tpr, color = Model)) + 
  geom_line(size = 0.7) +
  labs(x = "False Positive Rate", y = "True Positive Rate") +
  theme_minimal() +
  geom_abline(aes(intercept = 0, slope = 1, colour = "Random Classifier"), linetype = "dashed")+
  theme(legend.title = element_blank())+
  scale_color_manual(values = pal)

ROC_unbal

pdf("/your/path/ROC_all_models_unbalanced.pdf", height = 6, width = 9)
#tiff("/your/path/ROC_all_models_unbalanced.tif", 
#     units="px", width=2244, height=1795, res=356, compression = 'none')

ROC_unbal

dev.off()

# ==========
# Precision Recall Plot


# create Dataframe of PR plots

pr1_df <- data.frame(prec = log_reg_unbal$group_lasso$pr$precision, 
                     rec = log_reg_unbal$group_lasso$pr$recall, model = "Group Lasso")
pr2_df <- data.frame(prec = log_reg_unbal$ngr_lasso$pr$precision, 
                     rec = log_reg_unbal$ngr_lasso$pr$recall, model = "Lasso")
pr3_df <- data.frame(prec = log_reg_unbal$ridge$pr$precision, 
                      rec = log_reg_unbal$ridge$pr$recall, model = "Ridge")
pr4_df <- data.frame(prec = log_reg_unbal$elastic_net$pr$precision, 
                     rec = log_reg_unbal$elastic_net$pr$recall, model = "Elastic Net")
pr5_df <- data.frame(prec = rdm_for_unbal$pr$precision, 
                     rec = rdm_for_unbal$pr$recall, model = "Random Forest")
pr6_df <- data.frame(prec = svm_unbal$pr$precision, 
                     rec = svm_unbal$pr$recall, model = "SVM")
pr7_df <- data.frame(prec = xgbm_unbal$Linear$pr$precision, 
                     rec = xgbm_unbal$Linear$pr$recall, model = "Linear xGBM")
pr8_df <- data.frame(prec = xgbm_unbal$Tree$pr$precision, 
                     rec = xgbm_unbal$Tree$pr$recall, model = "Tree xGBM")


pr_df <- rbind(pr1_df, pr2_df, pr3_df, pr4_df, pr5_df, pr6_df, pr7_df, pr8_df)



# Draw Curves
PR_unbal <- ggplot(pr_df, aes(x = rec, y = prec, color = model)) + 
  geom_line(size = 0.7) +
  labs(x = "Recall", y = "Precision") +
  theme_minimal() +
  geom_hline(aes(yintercept = 0.1344701, colour = "Random Classifier"), linetype="dashed") +
  theme(legend.title = element_blank())+
  scale_color_manual(values = pal)

PR_unbal

pdf("/your/path/PR_all_models_unbalanced.pdf", height = 6, width = 9)
#tiff("/your/path/PR_all_models_unbalanced.tif", 
#     units="px", width=2244, height=1795, res=356, compression = 'none')

PR_unbal

dev.off()



# arrange plots


balanced <- ggarrange(ROC_bal, PR_bal, ncol = 2, labels = c("a", "b"), common.legend = TRUE, legend = "bottom")
unbalanced <- ggarrange(ROC_unbal, PR_unbal, ncol = 2, labels = c("a", "b"), common.legend = TRUE, legend = "bottom")
unbalanced


pdf("/your/path/Performance_unbalanced.pdf", height = 6, width = 9)
#png("/your/path/Performance_unbalanced.png", 
#     units="px", width=2244, height=1496, res=356)#, compression = 'none'

unbalanced

dev.off()

pdf("/your/path/Performance_balanced2.pdf", height = 2, width = 3)
#png("/your/path/Performance_balanced.png", 
#     units="px", width=2244, height=1496, res=356)#, compression = 'none'

balanced

dev.off()





# ----
#end

