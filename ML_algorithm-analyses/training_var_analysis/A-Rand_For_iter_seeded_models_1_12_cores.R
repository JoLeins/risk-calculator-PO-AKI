# ==============================================================================
# Jonas Leins 
# Training Random Forest Models on PO-AKI Dataset using different seeds for 
# variance estimation and preprocessing selection [[Part 1](Split for parallel submission)]
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


rm( list = ls())


# ----
#functions


# code to install needed packages

# ----
#functions


# code to install needed packages

count_unavailable_packages <- function(vector_package_names){ 
  
  #'@title count packages that are unavailable
  #'@param vector_package_names is a vector containing the names of needed packages as string variables.
  #'@return the lenght of a vector containing the names of all needed packages that are not within the list of installed packages as integer value.
  
  return (length(setdiff(vector_package_names, rownames(installed.packages(lib.loc = "/your/path/")))))  
}


install_packages_if_unavailable <- function(vector_package_names){
  
  #'@title A function to install all packages that are not yet installed.
  #'@description The function calls the function count_unavailable_packages and pastes the
  #'input parameter into the install.packages() function if the output of the former isnt equal to 0.
  #'If after the call of install.packages the output of the function count_unavailable_packages still isnt 
  #'equal to zero, this function stops and returns an error message containing not installed packages.
  #'@param vector_package_names is a vector containing the names of needed packages as string variables.
  
  if (count_unavailable_packages(vector_package_names) > 0) {
    install.packages(vector_package_names, lib = "/your/path/", repos="https://ftp.gwdg.de/pub/misc/cran/")
  }
  if (count_unavailable_packages(vector_package_names) > 0) {
    
    stop(paste0("Folgende Pakete konnten nicht installiert werden: ",
                setdiff(vector_package_names, rownames(installed.packages(lib.loc = "/your/path/")))))
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
    library(pack,character.only = TRUE, lib.loc = "/your/path/")
  }
}


RF_nested_cv <- function(X, y, ntree_vec, k){
  
  # Initialize a list to save ideal models per fold
  outer_models <- list()
  
  # Outer 10-fold cross-validation
  outer_folds <- createFolds(y, k = k, list = TRUE)
  
  
  # ______________________________________________________________________________
  
  #                                outer CV
  
  
  for (fold in seq_along(outer_folds)) {
    cat("Processing outer fold", fold, "\n")
    
    # Split data into training and validation sets
    test_indices <- outer_folds[[fold]]
    X_train_fold <- X[-test_indices,]
    y_train_fold <- y[-test_indices]
    X_test_fold <- X[test_indices,]
    y_test_fold <- y[test_indices]
    
    # Initialize a list for inner models (one for each ntree with optimal params in inner cv)
    inner_models <- list()
    
    # Inner cross-validation for tuning
    for (num_trees in num_trees_vector) {
      cat("  Tuning num.trees =", num_trees, "\n")
      
      
      # ______________________________________________________________________________
      
      #                               inner CV
      
      
      
      # Define control for inner cross-validation
      inner_control <- trainControl(method = "CV",
                                    number = 5,
                                    verboseIter = FALSE,
                                    classProbs = TRUE, 
                                    allowParallel = TRUE)
      
      # Define the tuning grid for the current number of trees
      tuneGrid <- expand.grid(mtry = c(seq(5,25,5)), # number of predictor variables to sample at each split
                              splitrule = c("gini"),
                              min.node.size = c(1, 3, 5, 8)) # minimum size of leaf nodes
      
      # Train the model using ranger with inner cross-validation
      model <- train(x = X_train_fold,
                     y = y_train_fold,
                     method = "ranger",
                     metric = "Kappa",
                     tuneGrid = tuneGrid,
                     trControl = inner_control,
                     importance = "impurity",
                     num.trees = num_trees,
                     num.threads = 12)
      
      # Save the trained model in the inner list
      inner_models[[as.character(num_trees)]] <- model
    }
    
    # ______________________________________________________________________________
    
    #                           outer cv performance est.
    
    
    # Select the best ntree model based on performance on outer test set
    
    #loop through inner models to find best performing model on test ds
    performances <- list()
    for(mindex in 1:length(inner_models)){
      y_pred <- predict(inner_models[[mindex]], 
                        newdata = X_test_fold, 
                        type = "prob") # predict in order to obtain predicted class probabilities
      
      threshold <- 0.5
      performances[[as.character(mindex)]] <- confusionMatrix(factor(y_pred$event>threshold), 
                                                              factor(y_test_fold=="event"), 
                                                              positive="TRUE")$overall["Kappa"]
      best_model <- inner_models[[which.max(performances)]]
    }
    outer_models[[fold]] <- best_model
    
    cat("Finished outer fold", fold, "\n")
    cat("Best model", "with Kappa = ", as.character(performances[[which.max(performances)]]))
  }
  
  return(outer_models)
}

seeded_RF <- function(X, y, vector_of_seeds, tuneGrid, control, num_trees_vector){
  
  list_models <- list()
  for(i in vector_of_seeds){
    
    name <- as.character(i)
    
    # split the data into train and test
    set.seed(i)
    data <- cbind(y, X)
    trainIndex <- createDataPartition(y = data[,1], p = .8,
                                      list = FALSE,
                                      times = 1)
    X_train <- X[trainIndex,]
    y_train <- y[trainIndex]
    
    
    list_models[[name]] <- list()
    
    # store train indices
    list_models[[name]]$train_ids <- trainIndex 
    
    # tune ntree
    nc_RF <- RF_nested_cv(X, y, num_trees_vector, k=5) # This is by far the most time consuming step (nested cv)
    
    ntree <- c()
    for(i in seq_along(nc_RF)){
      ntree <- append(ntree, nc_RF[[i]]$finalModel$num.trees)
    }
    
    # train model and save to list
    list_models[[name]]$model <- train(x = X_train,
                                       y = y_train,
                                       method = "ranger",
                                       metric = "Kappa",
                                       tuneGrid = tuneGrid,
                                       trControl = control,
                                       importance= "impurity",
                                       num.tree = round(mean(ntree)),
                                       num.threads = 12)
    print(paste("seed", i, "is done."))
  }
  return(list_models)
}






# ----
# setup

#load packages

load_packages( c("tidyverse","dplyr", "caret",  "ranger", "ggplot2", "tidymodels", "pROC"))



#setwd("./../Desktop/_masterthesis/data/")
set.seed(3010)

# create vector of seeds for variance estimation
vector_of_seeds_RF  <- sample(1:25000, 50, replace = FALSE)







# ----
# main





# ==============================================================================
# loading datasets
# ==============================================================================



# ==========
# load data

devds_unbal <- readRDS("/your/path/devds_original_prep.rds")
devds_bal <- readRDS("/your/path/devds_balanced_prep.rds")



# ==========
# separate predictors (X) and outcome (y)

X_devds_unbal <- devds_unbal %>% select(-c(AKI_bin))
y_devds_unbal <- as.factor(devds_unbal$AKI_bin)
X_devds_bal <- devds_bal %>% select(-c(AKI_bin))
y_devds_bal <- as.factor(devds_bal$AKI_bin)


levels(y_devds_unbal) <- c("no_event", "event")
levels(y_devds_bal) <- c("no_event", "event")



# ==============================================================================
# Random Forest Classifier
# ==============================================================================



# ==============================================================================
#  Set parameters
# ==============================================================================

# set hpt control parameters
hpt_control <- trainControl(method = "CV",
                            number = 5,
                            verboseIter = FALSE,
                            classProbs = TRUE, 
                            allowParallel = TRUE)

# calibrate hp tuning grid
hpt_tuneGrid <- expand.grid(mtry = c(seq(5,25,5)), # number of predictor variables to sample at each split
                        splitrule = c("gini"),
                        min.node.size = c(1, 3, 5, 8)) # minimum size of leaf nodes


# vector of ntree to be tuned

num_trees_vector <- c(50, 100, 200, 300, 400, 500, 800)








# ==============================================================================

# Variance estimation

# ==============================================================================




# ==============================================================================
#  Train models
# ==============================================================================


start_time <- Sys.time() # Track start time
#_______________________________________________________________________________

# train model (ATU)
list_RF_unbal <- seeded_RF(X_devds_unbal, y_devds_unbal, vector_of_seeds_RF, hpt_tuneGrid, hpt_control, num_trees_vector)

#save model (ATU)
saveRDS(list_RF_unbal, "/your/path/RF_unbalanced_1.rds")

#__________________________  _____________________________________________________
end_time <- Sys.time() # Track end time
end_time - start_time # Calculate the difference


# run duration: 1.37722 hours



start_time <- Sys.time() # Track start time
#_______________________________________________________________________________

# train model (ATB)
list_RF_bal <- seeded_RF(X_devds_bal, y_devds_bal, vector_of_seeds_RF, hpt_tuneGrid, hpt_control, num_trees_vector)

#save model (ATB)
saveRDS(list_RF_bal, "/your/path/RF_balanced_1.rds")

#__________________________  _____________________________________________________
end_time <- Sys.time() # Track end time
time_taken <- end_time - start_time # Calculate the difference
print(time_taken)# Print the time taken

# run duration: 24.76981 mins




  

end_time <- Sys.time() # Track end time
time_taken <- end_time - start_time # Calculate the difference




# ----
# end

# 4.2 days estimated
