# ==========================================================================
# Training of RFC models on time-point specific feature sets.
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



RF_nested_cv <- function(X, y, ntree_vec, k){
  
  # Initialize list to store the optimal model for each outer fold
  outer_models <- list()
  
  # Outer k-fold cross-validation
  outer_folds <- createFolds(y, k = k, list = TRUE)
  
  
  # ______________________________________________________________________________
  #                                outer CV
  
  for (fold in seq_along(outer_folds)) {
    cat("Processing outer fold", fold, "\n")
    
    # Split into training and test folds
    test_indices     <- outer_folds[[fold]]
    X_train_fold     <- X[-test_indices,]
    y_train_fold     <- y[-test_indices]
    X_test_fold      <- X[test_indices,]
    y_test_fold      <- y[test_indices]
    
    # List to store inner models (each representing tuned params for a specific ntree value)
    inner_models <- list()
    
    # Inner cross-validation loop
    for (num_trees in num_trees_vector) {
      cat("  Tuning num.trees =", num_trees, "\n")
      
      # ______________________________________________________________________________
      #                                inner CV
      
      # Define inner CV settings
      inner_control <- trainControl(
        method = "CV",
        number = 5,
        verboseIter = FALSE,
        classProbs = TRUE, 
        allowParallel = TRUE
      )
      
      # Define tuning grid
      tuneGrid <- expand.grid(
        mtry          = c(seq(5, 25, 5)),
        splitrule     = c("gini"),
        min.node.size = c(1, 3, 5, 8)
      )
      
      # Train with ranger using inner CV
      model <- train(
        x = X_train_fold,
        y = y_train_fold,
        method      = "ranger",
        metric      = "Kappa",
        tuneGrid    = tuneGrid,
        trControl   = inner_control,
        importance  = "impurity",
        num.trees   = num_trees,
        num.threads = 6
      )
      
      # Save inner model
      inner_models[[as.character(num_trees)]] <- model
    }
    
    # ______________________________________________________________________________
    #                outer CV performance estimation
    
    # Evaluate each tuned inner model on the held-out test fold
    performances <- list()
    
    for(mindex in 1:length(inner_models)){
      y_pred <- predict(
        inner_models[[mindex]],
        newdata = X_test_fold,
        type = "prob"
      )
      
      threshold <- 0.5
      
      performances[[as.character(mindex)]] <- confusionMatrix(
        factor(y_pred$event > threshold),
        factor(y_test_fold == "event"),
        positive = "TRUE"
      )$overall["Kappa"]
      
      best_model <- inner_models[[which.max(performances)]]
    }
    
    outer_models[[fold]] <- best_model
    
    cat("Finished outer fold", fold, "\n")
    cat("Best model with Kappa = ", as.character(performances[[which.max(performances)]]), "\n")
  }
  
  return(outer_models)
}

load_packages(c(
  "tidyverse", "dplyr", "caret", "pROC", "tidymodels", 
  "ranger", "kernlab", "gglasso", "glmnet"
))  # "data.table",
set.seed(3010)


### Load time point datasets with variance estimation

timepoints_var <- readRDS("/your/path/Timepoint_datasets.rds")


#### Model training #####

# Number of trees for nested CV
num_trees_vector <- c(50, 100, 200, 300, 400, 500, 800)

# Set hyperparameter tuning control parameters
hpt_control <- trainControl(method = "CV",
                            number = 5,
                            verboseIter = FALSE,
                            classProbs = TRUE,
                            allowParallel = TRUE)

# Hyperparameter tuning grid
hpt_tuneGrid <- expand.grid(mtry = c(seq(5, 25, 5)), 
                            splitrule = c("gini"), 
                            min.node.size = c(1, 3, 5, 9))

# List to store the trained models
models <- list()

output_dir <- "/your/path/"
if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
}

# Extract all time points
timepoints <- names(timepoints_var)

# Model training for each time point
for (time_point in timepoints) {
  
  # Load the corresponding training dataset
  training_dataset <- timepoints_var[[time_point]]$train
  
  # Prepare training data
  X_devds <- training_dataset %>% dplyr::select(-AKI_bin)
  y_devds <- as.factor(training_dataset$AKI_bin)
  levels(y_devds) <- c("no_event", "event")
  
  # Nested cross-validation
  start_time <- Sys.time()
  nc_RF <- RF_nested_cv(X_devds, y_devds, num_trees_vector, k = 5)
  print(Sys.time() - start_time)
  
  # Save nested CV object if needed
  # saveRDS(nc_RF, paste0("/your/path/CV_trees_", time_point, ".rds"))
  
  # Load nested CV object if needed
  # nc_RF <- readRDS(paste0("/your/path/CV_trees_", time_point, ".rds"))
  
  # Average number of trees across outer folds
  ntree <- sapply(nc_RF, function(x) x$finalModel$num.trees)
  
  # Final model training
  start_time <- Sys.time()
  RF <- train(x = X_devds,
              y = y_devds,
              method = "ranger",
              metric = "Kappa",
              tuneGrid = hpt_tuneGrid,
              trControl = hpt_control,
              importance = "impurity",
              num.tree = round(mean(ntree)),
              num.threads = 6)
  print(Sys.time() - start_time)
  
  # Store model
  models[[time_point]] <- RF
}

# Save trained models
saveRDS(models, "/your/path/models_time_points.rds")

