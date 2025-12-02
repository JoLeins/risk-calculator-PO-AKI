# ==============================================================================
# training of 16 ML Algorithms on full development datasets.
# for following the publication remove all training and evaluation steps labelled "unbalanced" 
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
                     num.threads = 6)
      
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



# ----
# setup

load_packages( c("tidyverse", "dplyr", "caret", "pROC", "tidymodels", "ranger", "kernlab", "gglasso", "glmnet" )) #"data.table",
set.seed(3010)
setwd("/Your/Path/")







################################################################################
################################################################################

################################################################################
################################################################################



# ______________________________________________________________________________

#                           SUPPORT VECTOR MACHINES

# ______________________________________________________________________________



# ==============================================================================
# loading and preparing dataset
# ==============================================================================



# ==========
# load data
devds_unbal <- readRDS("/Your/Path/devds_original_prep.rds")
devds_bal <- readRDS("/Your/Path/devds_balanced_prep.rds")
valds <- readRDS("/Your/Path/valds_prep.rds")



print("Preparation started")

# ==========
# reencode levels for application of kernlab based train

levels(devds_unbal$AKI_bin) <- c("no_event", "event")
levels(devds_bal$AKI_bin) <- c("no_event", "event")
levels(valds$AKI_bin) <- c("no_event", "event")

#split valds in predictors (X) and response(y)
X_valid <- valds %>% select(-c(AKI_bin))
y_valid <- valds %>% select(c(AKI_bin))



# ==============================================================================
#                           HP tuning and Training
# ==============================================================================

### NOTE THAT SVM CONVERTS FACTORS INTO DUMMIES AUTOMATICALLY AND CAN RUN INTO ISSUES IF DUMMIES ARE CREATED MANUALLY!!!!!

# ==========
# define hp tuning parameters
control <- trainControl(method="CV", 
                        number=5, 
                        verboseIter = FALSE,
                        classProbs = TRUE)
tuneLength <- 15

# ____
start_time <- Sys.time()

# =========
# tune hyperparameters and train models similar to iterative training

SVM_bal <- train(AKI_bin ~ .,
                 data = devds_bal,
                 method = "svmRadial",
                 metric = "Kappa",
                 trControl = control,
                 tuneLength = tuneLength
)

Sys.time() - start_time # 2.595057 mins

start_time <- Sys.time()
SVM_unbal <- train(AKI_bin ~ .,
                   data = devds_unbal,
                   method = "svmRadial",
                   metric = "Kappa",
                   trControl = control,
                   tuneLength = tuneLength
)
Sys.time() - start_time #39.22929 mins
#save models
saveRDS(SVM_bal, "/Your/Path/SVM_bal_final_balanced.rds")
saveRDS(SVM_unbal, "/Your/Path/SVM_unbal_final_unbalanced.rds")







# ==============================================================================
#                           Model Performance
# ==============================================================================


# ______________________________________________________________________________
#                                 BALANCED

# ==========
# predict class probabilities with optimal model
y_pred <- predict(SVM_bal, 
                  X_valid,
                  type = "prob")

# ==========
# predict classes with optimal model
y_pred_class <- predict(SVM_bal, 
                        X_valid,
                        type = "raw")


# ==========
# calculate Confusion Matrix
y_valid_class <- y_valid$AKI_bin
levels(y_valid_class) <- c("no_event", "event")

cm <- confusionMatrix(y_pred_class, y_valid_class, positive = "event")
cm



# ==========
# Draw ROC Curve

y_valid <- y_valid$AKI_bin

# create matrix of predicted and true classes
yscore <- data.frame(y_pred$event)
rdb <- cbind(as.factor(y_valid),yscore)
colnames(rdb) = c('y', "yscore")

# calculate roc and auc
roc_stat <- roc_curve(rdb, y, yscore)
roc_stat$specificity <- 1 - roc_stat$specificity
auc = roc_auc(rdb, y, yscore, event_level = "second")
auc = auc$.estimate
title = paste('ROC Curve SVM (AUC = ',toString(round(auc,2)),')',sep = '')

# drawing ROC-Curve
ROC <- ggplot(roc_stat, aes(x = sensitivity, y = specificity)) + 
  geom_line(size = 1, color="blue") +
  labs(x = "False Positive Rate", y = "True Positive Rate") +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", ) +
  theme_minimal() +
  theme(legend.title = element_blank())+
  ggtitle(title)

pdf("/Your/Path/ROC_SVM_bal_final.pdf", height = 4, width = 6)

ROC

dev.off()

# ==========
# Draw Precision Recall Plot

#calculate pr(Precision-Recall) and aps (Average Precision Score)
pr_stat <- pr_curve(rdb, y, yscore, event_level = "second")
pauc <- pr_auc(rdb, y, yscore, event_level = "second")
pauc <- pauc$.estimate
title = paste('Precision Recall Plot SVM (AUC = ',toString(round(pauc,2)),')',sep = '')

#calculate baseline Precision to be expected if the model guesses randomly
baseline <- sum(y_valid == 1) / length(y_valid)

# draw presicion recall plot
PR <- ggplot(pr_stat, aes(x = recall, y = precision)) + 
  geom_line(size = 1, color="blue") +
  labs(x = "Recall", y = "Precision") +
  theme_minimal() +
  geom_hline(yintercept = baseline, linetype = "dashed") +
  theme(legend.title = element_blank())+
  ggtitle(title)

pdf("/Your/Path/PR_SVM_bal_final.pdf", height = 4, width = 6)

PR

dev.off()

# ==========
# save performance measures and graphs in List
Performance_SVM_bal <- list("pred" = rdb,
                            "ConfusionMatrix" = cm, 
                            "auc" = auc, 
                            "roc" = roc_stat,
                            "roc_plot" = ROC, 
                            "pr" = pr_stat, 
                            "pr_plot" = PR )

saveRDS(Performance_SVM_bal, "/Your/Path/Performance_SVM_bal.rds")



# ______________________________________________________________________________
#                                 UNBALANCED

# ==========
# predict class probabilities with optimal model
y_pred <- predict(SVM_unbal, 
                  X_valid,
                  type = "prob")

# ==========
# predict classes with optimal model
y_pred_class <- predict(SVM_unbal, 
                        X_valid,
                        type = "raw")


# ==========
# calculate Confusion Matrix

levels(y_valid_class) <- c("no_event", "event")

cm <- confusionMatrix(y_pred_class, y_valid_class, positive = "event")
cm



# ==========
# Draw ROC Curve

# create matrix of predicted and true classes
yscore <- data.frame(y_pred$event)
rdb <- cbind(as.factor(y_valid),yscore)
colnames(rdb) = c('y', "yscore")

# calculate roc and auc
roc_stat <- roc_curve(rdb, y, yscore)
roc_stat$specificity <- 1 - roc_stat$specificity
auc = roc_auc(rdb, y, yscore, event_level = "second")
auc = auc$.estimate
title = paste('ROC Curve SVM (AUC = ',toString(round(auc,2)),')',sep = '')

# drawing ROC-Curve
ROC <- ggplot(roc_stat, aes(x = sensitivity, y = specificity)) + 
  geom_line(size = 1, color="blue") +
  labs(x = "False Positive Rate", y = "True Positive Rate") +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", ) +
  theme_minimal() +
  theme(legend.title = element_blank())+
  ggtitle(title)

pdf("/Your/Path/ROC_SVM_unbal_final.pdf", height = 4, width = 6)

ROC

dev.off()

# ==========
# Draw Precision Recall Plot

#calculate pr(Precision-Recall) and aps (Average Precision Score)
pr_stat <- pr_curve(rdb, y, yscore, event_level = "second")
pauc <- pr_auc(rdb, y, yscore, event_level = "second")
pauc <- pauc$.estimate
title = paste('Precision Recall Plot SVM (AUC = ',toString(round(pauc,2)),')',sep = '')

#calculate baseline Precision to be expected if the model guesses randomly
baseline <- sum(y_valid == 1) / length(y_valid)

# draw presicion recall plot
PR <- ggplot(pr_stat, aes(x = recall, y = precision)) + 
  geom_line(size = 1, color="blue") +
  labs(x = "Recall", y = "Precision") +
  theme_minimal() +
  geom_hline(yintercept = baseline, linetype = "dashed") +
  theme(legend.title = element_blank())+
  ggtitle(title)

pdf("/Your/Path/PR_SVM_unbal_final.pdf", height = 4, width = 6)

PR

dev.off()

# ==========
# save performance measures and graphs in List
Performance_SVM_unbal <- list( "pred" = rdb,
                             "ConfusionMatrix" = cm, 
                             "auc" = auc, 
                             "roc" = roc_stat,
                             "roc_plot" = ROC, 
                             "pr" = pr_stat, 
                             "pr_plot" = PR )

saveRDS(Performance_SVM_unbal, "/Your/Path/Performance_SVM_unbal.rds")










################################################################################
################################################################################

################################################################################
################################################################################



# ______________________________________________________________________________

#                               RANDOM FOREST


# ______________________________________________________________________________



# ==============================================================================
# loading datasets
# ==============================================================================



# ==========
# separate predictors (X) and outcome (y)

X_devds_unbal <- devds_unbal %>% select(-c(AKI_bin))
y_devds_unbal <- as.factor(devds_unbal$AKI_bin)
X_devds_bal <- devds_bal %>% select(-c(AKI_bin))
y_devds_bal <- as.factor(devds_bal$AKI_bin)


#levels(y_devds_unbal) <- c("no_event", "event")
#levels(y_devds_bal) <- c("no_event", "event")





# ==============================================================================
#                           HP tuning and Training
# ==============================================================================


# ______________________________________________________________________________


# Nested Cross val.

library(caret)
library(ranger)

# ______________________________________________________________________________

#                               Nested CV Setup




# Define your number of trees to tune
num_trees_vector <- c(50, 100, 200, 300, 400, 500, 800)  # Adjust this vector as needed

start_time <- Sys.time() # Track start time
nc_RF_unbal <- RF_nested_cv(X_devds_unbal, y_devds_unbal, num_trees_vector, k=5)
Sys.time() - start_time # Calculate the difference 1.180826 hours

start_time <- Sys.time() # Track start time
nc_RF_bal <- RF_nested_cv(X_devds_bal, y_devds_bal, num_trees_vector, k=5)
Sys.time() - start_time # Calculate the difference 20.11978 mins

saveRDS(nc_RF_bal, "/Your/Path/CV_trees_bal.rds")
saveRDS(nc_RF_unbal, "/Your/Path/CV_trees_unbal.rds")

nc_RF_unbal <- readRDS("/Your/Path/CV_trees_unbal.rds")
nc_RF_bal <- readRDS("/Your/Path/CV_trees_bal.rds")

# ==============================================================================
#  Set parameters
# ==============================================================================

# set hyperparametertuning control parameters
hpt_control <- trainControl(method = "CV",
                            number = 5,
                            verboseIter = FALSE,
                            classProbs = TRUE,
                            allowParallel = TRUE)

# calibrate hyperparameter tuning grid
hpt_tuneGrid <- expand.grid(mtry = c(seq(5,25,5)), # number of predictor variables to sample at each split
                            splitrule = c("gini"),
                            min.node.size = c(1, 3, 5, 9)) # minimum size of 

ntree_bal <- c()
for(i in seq_along(nc_RF_bal)){
  ntree_bal <- append(ntree_bal, nc_RF_bal[[i]]$finalModel$num.trees)
}

ntree_unbal <- c()
for(i in seq_along(nc_RF_unbal)){
  ntree_unbal <- append(ntree_unbal, nc_RF_unbal[[i]]$finalModel$num.trees)
}


# ==============================================================================
#  Train model
# ==============================================================================

start_time <- Sys.time() # Track start time
#_______________________________________________________________________________

# train random forest classifier

RF_bal <- train(x = X_devds_bal,
                y = y_devds_bal,
                method = "ranger",
                metric = "Kappa",
                tuneGrid = hpt_tuneGrid,
                trControl = hpt_control,
                importance = "impurity",
                num.tree = round(mean(ntree_bal)),
                num.threads = 6)

#__________________________  _____________________________________________________
Sys.time() - start_time # Calculate the difference 48.43856 secs

start_time <- Sys.time() # Track start time
#_______________________________________________________________________________

# train random forest classifier

RF_unbal <- train(x = X_devds_unbal,
                y = y_devds_unbal,
                method = "ranger",
                metric = "Kappa",
                tuneGrid = hpt_tuneGrid,
                trControl = hpt_control,
                importance = "impurity",
                num.tree = round(mean(ntree_unbal)),
                num.threads = 6)

#__________________________  _____________________________________________________
Sys.time() - start_time # Calculate the difference .378908 mins



#save models 
saveRDS(RF_bal, "/Your/Path/RF_final_balanced.rds")
saveRDS(RF_unbal, "/Your/Path/RF_final_unbalanced.rds")





# ==============================================================================
#                           Model Performance
# ==============================================================================





# ______________________________________________________________________________
#                                 BALANCED

# ==========
# predict class probabilities with optimal model

y_pred <- predict(RF_bal, 
                  newdata = X_valid, 
                  type = "prob") # predict to obtain class probabiities



# adapt level encoding of y_valid
levels(valds$AKI_bin) <- c("0", "1")
y_valid <- (valds$AKI_bin)
# ==========
# calculate Confusion Matrix

cm <- confusionMatrix(as.factor(ifelse(y_pred[, 2] >= 0.5, 1, 0)), as.factor(y_valid), positive = "1")
cm


# ==========
# Draw ROC Curve

# create matrix of predicted and true classes
yscore <- data.frame(y_pred[, 2])
rdb <- cbind(as.factor(y_valid),yscore)
colnames(rdb) = c('y','yscore')

# calculate roc and auc
roc_stat <- roc_curve(rdb, y, yscore)
roc_stat$specificity <- 1 - roc_stat$specificity
auc = roc_auc(rdb, y, yscore, event_level = "second")
auc = auc$.estimate
title = paste('ROC Curve Random Forest (AUC = ',toString(round(auc,2)),')',sep = '')

# drawing ROC-Curve
ROC <- ggplot(roc_stat, aes(x = sensitivity, y = specificity)) + 
  geom_line(size = 1, color="blue") +
  labs(x = "False Positive Rate", y = "True Positive Rate") +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", ) +
  theme_minimal() +
  theme(legend.title = element_blank())+
  ggtitle(title)

pdf("/Your/Path/ROC_Random_Forest_balanced_final.pdf", height = 4, width = 6)

ROC

dev.off()




# ==========
# Draw Precision Recall Plot

#calculate pr(Precision-Recall) and aps (Average Precision Score)
pr_stat <- pr_curve(rdb, y, yscore, event_level = "second")
pauc <- pr_auc(rdb, y, yscore, event_level = "second")
pauc <- pauc$.estimate
title = paste('Precision Recall Plot Random Forest (AUC = ',toString(round(pauc,2)),')',sep = '')

#calculate baseline Precision to be expected if the model guesses randomly
baseline <- sum(y_valid == 1) / length(y_valid)

# draw presicion recall plot
PR <- ggplot(pr_stat, aes(x = recall, y = precision)) + 
  geom_line(size = 1, color="blue") +
  labs(x = "Recall", y = "Precision") +
  theme_minimal() +
  geom_hline(yintercept = baseline, linetype = "dashed") +
  theme(legend.title = element_blank())+
  ggtitle(title)

pdf("/Your/Path/PR_Random_Forest_balanced_final.pdf", height = 4, width = 6)

PR

dev.off()


# ==========
# save performance measures and plotly graphs in List

Performance_RF_bal <- list( "pred" = rdb,
                          "ConfusionMatrix" = cm,
                          "auc" = auc, 
                          "roc" = roc_stat,
                          "roc_plot" = ROC, 
                          "pr" = pr_stat, 
                          "pr_plot" = PR )


saveRDS(Performance_RF_bal, "/Your/Path/Performance_RF_balanced.rds")





# ______________________________________________________________________________
#                                 UNBALANCED

# ==========
# predict class probabilities with optimal model

y_pred <- predict(RF_unbal, 
                  newdata = X_valid, 
                  type = "prob") # predict to obtain class probabiities

# ==========
# calculate Confusion Matrix

cm <- confusionMatrix(as.factor(ifelse(y_pred[, 2] >= 0.5, 1, 0)), as.factor(y_valid), positive = "1")
cm


# ==========
# Draw ROC Curve

# create matrix of predicted and true classes
yscore <- data.frame(y_pred[, 2])
rdb <- cbind(as.factor(y_valid),yscore)
colnames(rdb) = c('y','yscore')

# calculate roc and auc
roc_stat <- roc_curve(rdb, y, yscore)
roc_stat$specificity <- 1 - roc_stat$specificity
auc = roc_auc(rdb, y, yscore, event_level = "second")
auc = auc$.estimate
title = paste('ROC Curve Random Forest (AUC = ',toString(round(auc,2)),')',sep = '')

# drawing ROC-Curve
ROC <- ggplot(roc_stat, aes(x = sensitivity, y = specificity)) + 
  geom_line(size = 1, color="blue") +
  labs(x = "False Positive Rate", y = "True Positive Rate") +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", ) +
  theme_minimal() +
  theme(legend.title = element_blank())+
  ggtitle(title)

pdf("/Your/Path/ROC_Random_Forest_unbalanced_final.pdf", height = 4, width = 6)

ROC

dev.off()




# ==========
# Draw Precision Recall Plot

#calculate pr(Precision-Recall) and aps (Average Precision Score)
pr_stat <- pr_curve(rdb, y, yscore, event_level = "second")
pauc <- pr_auc(rdb, y, yscore, event_level = "second")
pauc <- pauc$.estimate
title = paste('Precision Recall Plot Random Forest (AUC = ',toString(round(pauc,2)),')',sep = '')

#calculate baseline Precision to be expected if the model guesses randomly
baseline <- sum(y_valid == 1) / length(y_valid)

# draw presicion recall plot
PR <- ggplot(pr_stat, aes(x = recall, y = precision)) + 
  geom_line(size = 1, color="blue") +
  labs(x = "Recall", y = "Precision") +
  theme_minimal() +
  geom_hline(yintercept = baseline, linetype = "dashed") +
  theme(legend.title = element_blank())+
  ggtitle(title)

pdf("/Your/Path/PR_Random_Forest_unbalanced_final.pdf", height = 4, width = 6)

PR

dev.off()


# ==========
# save performance measures and plotly graphs in List

Performance_RF_unbal <- list( "pred" = rdb,
                            "ConfusionMatrix" = cm,
                            "auc" = auc, 
                            "roc" = roc_stat,
                            "roc_plot" = ROC, 
                            "pr" = pr_stat, 
                            "pr_plot" = PR )


saveRDS(Performance_RF_unbal, "/Your/Path/Performance_RF_unbalanced.rds")












################################################################################
################################################################################

################################################################################
################################################################################



# ______________________________________________________________________________

#                   extreme GRADIENT BOOSTING MACHINES

# ______________________________________________________________________________



# ==========
# reverse previous reencoding

levels(devds_unbal$AKI_bin) <- c("0", "1")
levels(devds_bal$AKI_bin) <- c("0", "1")
levels(valds$AKI_bin) <- c("0", "1")


# ==========
# separate predictors (X) and outcome (y)
X_devds_unbal <- devds_unbal %>% select(-c(AKI_bin))
y_devds_unbal <- devds_unbal$AKI_bin
X_devds_bal <- devds_bal %>% select(-c(AKI_bin))
y_devds_bal <- devds_bal$AKI_bin
X_valid <- valds %>% select(-c(AKI_bin))
y_valid <- valds$AKI_bin



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

cont_unbal <- X_devds_unbal %>% select(all_of(continuousNames))
disc_unbal <- X_devds_unbal %>% select(- c(all_of(continuousNames)))
cont_bal <- X_devds_bal %>% select(all_of(continuousNames))
disc_bal <- X_devds_bal %>% select(- c(all_of(continuousNames)))
cont_val <- X_valid %>% select(all_of(continuousNames))
disc_val <- X_valid %>% select(- c(all_of(continuousNames)))

rm(continuousNames)



# ==========
# encoding discrete data in dummy variables
dummies <- fastDummies::dummy_cols(disc_unbal[, 2:5], remove_first_dummy = TRUE)
disc_unbal <- disc_unbal[,-c(2:5)] # remove multi-level features
disc_unbal <- cbind(disc_unbal, dummies[, 5:18])
dummies <- fastDummies::dummy_cols(disc_bal[, 2:5], remove_first_dummy = TRUE)
disc_bal <- disc_bal[,-c(2:5)] # remove multi-level features
disc_bal <- cbind(disc_bal, dummies[, 5:18])
dummies <- fastDummies::dummy_cols(disc_val[, 2:5], remove_first_dummy = TRUE)
disc_val <- disc_val[,-c(2:5)] # remove multi-level features
disc_val <- cbind(disc_val, dummies[, 5:18])



# ==========
# convert predictor dataframes (X_train, X_test) to matrix
X_unbal <- as.matrix(cbind(disc_unbal, cont_unbal))
X_bal <- as.matrix(cbind(disc_bal, cont_bal))
X_val <- as.matrix(cbind(disc_val, cont_val))



# convert Matrix from character to numeric
X_unbal <- apply(X_unbal, 2 ,as.numeric)
X_bal <- apply(X_bal, 2 ,as.numeric)
X_val <- apply(X_val, 2 ,as.numeric)




# ==============================================================================
#                           HP tuning and Training
# ==============================================================================


# specify cross validation methods
xgb_trcontrol = trainControl(
  method = "cv",
  number = 5,  
  allowParallel = TRUE,
  verboseIter = FALSE,
  returnData = FALSE
)


# specify hyperparametergrids
xgbTREEGrid <- expand.grid(nrounds = c(50, 100, 200, 300),  # number of boosting rounds
                           max_depth = c(10, 15, 25), # maximum depth of a tree # 15, 20
                           colsample_bytree = seq(0.4, 0.95, length.out = 9), # subsample ratio of columns when constructing each tree
                           eta = c(0.1, 0.01), # learning rate
                           gamma= 0, # minimum loss reduction 
                           min_child_weight = 1, # minimum sum of instance weight (hessian) needed ina child
                           subsample = 1 # subsample ratio of the training instances
)


xgbLINGrid <- expand.grid(nrounds = c(50, 100, 200, 300),  # number of boosting rounds
                          lambda = seq(0.01, 1, length.out = 6), # L2 regularization
                          alpha = seq(0.01, 1, length.out = 6), # L1 regularization 
                          eta = c(0.1, 0.01) # learning rate
)


# =========
# tune hyperparameters and train models


# ==========
# Linear

start_time <- Sys.time() # Track start time
#_______________________________________________________________________________

xgbLIN_bal <- train(X_bal, 
                    as.factor(y_devds_bal),
                    trControl = xgb_trcontrol,
                    tuneGrid = xgbLINGrid,
                    method = "xgbLinear"
)

#__________________________  _____________________________________________________
Sys.time() - start_time # Calculate the difference

# run duration: 27.28883 mins


start_time <- Sys.time() # Track start time 
#_______________________________________________________________________________

xgbLIN_unbal <- train(X_unbal, 
                    as.factor(y_devds_unbal),
                    trControl = xgb_trcontrol,
                    tuneGrid = xgbLINGrid,
                    method = "xgbLinear"
)

#__________________________  _____________________________________________________
Sys.time() - start_time # Calculate the difference

# run duration: 49.90591 mins



# ==========
# Tree

start_time <- Sys.time() # Track start time
#_______________________________________________________________________________

xgbTREE_bal <- train(X_bal, 
                    as.factor(y_devds_bal),
                    trControl = xgb_trcontrol,
                    tuneGrid = xgbTREEGrid,
                    method = "xgbTree"
)

#__________________________  _____________________________________________________
Sys.time() - start_time # Calculate the difference

# run duration: 38.66736 mins



start_time <- Sys.time() # Track start time
#_______________________________________________________________________________

xgbTREE_unbal <- train(X_unbal, 
                     as.factor(y_devds_unbal),
                     trControl = xgb_trcontrol,
                     tuneGrid = xgbTREEGrid,
                     method = "xgbTree"
)

#__________________________  _____________________________________________________
Sys.time() - start_time # Time difference of 

# run duration: 1.14245 hours

xgbModels_bal <- list("Linear" = xgbLIN_bal, 
                      "Tree" = xgbTREE_bal) 
xgbModels_unbal <- list("Linear" = xgbLIN_unbal, 
                      "Tree" = xgbTREE_unbal) 

saveRDS(xgbModels_bal, "/Your/Path/xGBM_balanced_final.rds")
saveRDS(xgbModels_unbal, "/Your/Path/xGBM_unbalanced_final.rds")





################################################################################

#                           Model Performance

################################################################################





# ______________________________________________________________________________
#                                 BALANCED

# ==========
# Linear


# ==========
# predict class probabilities with optimal model

y_pred <- predict(xgbLIN_bal, 
                  newdata = X_val,
                  type = "prob")


# ==========
# calculate Confusion Matrix

cm <- confusionMatrix(as.factor(ifelse(y_pred$`1` > 0.5, 1, 0)), as.factor(y_valid), positive = "1")
cm



# ==========
# Draw ROC Curve

# create matrix of predicted and true classes
yscore <- data.frame(y_pred$`1`)
rdb <- cbind(as.factor(y_valid),yscore)
colnames(rdb) = c('y', "yscore")

# calculate roc and auc
roc_stat <- roc_curve(rdb, y, yscore)
roc_stat$specificity <- 1 - roc_stat$specificity
auc = roc_auc(rdb, y, yscore, event_level = "second")
auc = auc$.estimate
title = paste('ROC Curve xGBM Linear (AUC = ',toString(round(auc,2)),')',sep = '')

# drawing ROC-Curve
ROC <- ggplot(roc_stat, aes(x = sensitivity, y = specificity)) + 
  geom_line(size = 1, color="blue") +
  labs(x = "False Positive Rate", y = "True Positive Rate") +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", ) +
  theme_minimal() +
  theme(legend.title = element_blank())+
  ggtitle(title)

pdf("/Your/Path/ROC_XBGM_LIN_balanced_final.pdf", height = 4, width = 6)

ROC

dev.off()




# ==========
# Draw Precision Recall Plot

#calculate pr(Precision-Recall) and aps (Average Precision Score)
pr_stat <- pr_curve(rdb, y, yscore, event_level = "second")
pauc <- pr_auc(rdb, y, yscore, event_level = "second")
pauc <- pauc$.estimate
title = paste('Precision Recall Plot xGBM Linear (AUC = ',toString(round(pauc,2)),')',sep = '')

#calculate baseline Precision to be expected if the model guesses randomly
baseline <- sum(y_valid == 1) / length(y_valid)

# draw presicion recall plot
PR <- ggplot(pr_stat, aes(x = recall, y = precision)) + 
  geom_line(size = 1, color="blue") +
  labs(x = "Recall", y = "Precision") +
  theme_minimal() +
  geom_hline(yintercept = baseline, linetype = "dashed") +
  theme(legend.title = element_blank())+
  ggtitle(title)

pdf("/Your/Path/PR_XGBM_LIN_bal_final.pdf", height = 4, width = 6)

PR

dev.off()


# ==========
# save performance measures and graphs in List

Performance_xgbLINEAR_bal <- list("pred" = rdb,
                                  "ConfusionMatrix" = cm, 
                                  "auc" = auc, 
                                  "roc" = roc_stat,
                                  "roc_plot" = ROC, 
                                  "pr" = pr_stat, 
                                  "pr_plot" = PR )



rm( rdb, cm, auc, roc_stat, ROC, pr_stat, PR, y_pred)

################################################################################



# ==========
# Tree


# ==========
# predict class probabilities with optimal model

y_pred <- predict(xgbTREE_bal, 
                  newdata = X_val,
                  type = "prob")


# ==========
# calculate Confusion Matrix

cm <- confusionMatrix(as.factor(ifelse(y_pred$`1` > 0.5, 1, 0)), as.factor(y_valid), positive = "1")
cm



# ==========
# Draw ROC Curve

# create matrix of predicted and true classes
yscore <- data.frame(y_pred$`1`)
rdb <- cbind(as.factor(y_valid),yscore)
colnames(rdb) = c('y', "yscore")

# calculate roc and auc
roc_stat <- roc_curve(rdb, y, yscore)
roc_stat$specificity <- 1 - roc_stat$specificity
auc = roc_auc(rdb, y, yscore, event_level = "second")
auc = auc$.estimate
title = paste('ROC Curve xGBM Tree (AUC = ',toString(round(auc,2)),')',sep = '')

# drawing ROC-Curve
ROC <- ggplot(roc_stat, aes(x = sensitivity, y = specificity)) + 
  geom_line(size = 1, color="blue") +
  labs(x = "False Positive Rate", y = "True Positive Rate") +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", ) +
  theme_minimal() +
  theme(legend.title = element_blank())+
  ggtitle(title)

pdf("/Your/Path/ROC_XGBM_TREE_bal_final.pdf", height = 4, width = 6)

ROC

dev.off()




# ==========
# Draw Precision Recall Plot

#calculate pr(Precision-Recall) and aps (Average Precision Score)
pr_stat <- pr_curve(rdb, y, yscore, event_level = "second")
pauc <- pr_auc(rdb, y, yscore, event_level = "second")
pauc <- pauc$.estimate
title = paste('Precision Recall Plot xGBM Tree (AUC = ',toString(round(pauc,2)),')',sep = '')

#calculate baseline Precision to be expected if the model guesses randomly
baseline <- sum(y_valid == 1) / length(y_valid)

# draw presicion recall plot
PR <- ggplot(pr_stat, aes(x = recall, y = precision)) + 
  geom_line(size = 1, color="blue") +
  labs(x = "Recall", y = "Precision") +
  theme_minimal() +
  geom_hline(yintercept = baseline, linetype = "dashed") +
  theme(legend.title = element_blank())+
  ggtitle(title)

pdf("/Your/Path/PR_XGBM_TREE_bal_final.pdf", height = 4, width = 6)

PR

dev.off()


# ==========
# save performance measures and plotly graphs in List

Performance_xgbTREE_bal <- list("pred" = rdb,
                                "ConfusionMatrix" = cm, 
                                "auc" = auc, 
                                "roc" = roc_stat,
                                "roc_plot" = ROC, 
                                "pr" = pr_stat, 
                                "pr_plot" = PR )





# ==========
# join performace lists for different models in list and save

xGBM_performances_bal <- list("Linear" = Performance_xgbLINEAR_bal, 
                          "Tree" = Performance_xgbTREE_bal) 

saveRDS(xGBM_performances_bal, "/Your/Path/Performance_xGBM_balanced.rds")





# ______________________________________________________________________________
#                                 UNBALANCED

# ==========
# Linear


# ==========
# predict class probabilities with optimal model

y_pred <- predict(xgbLIN_unbal, 
                  newdata = X_val,
                  type = "prob")


# ==========
# calculate Confusion Matrix

cm <- confusionMatrix(as.factor(ifelse(y_pred$`1` > 0.5, 1, 0)), as.factor(y_valid), positive = "1")
cm



# ==========
# Draw ROC Curve

# create matrix of predicted and true classes
yscore <- data.frame(y_pred$`1`)
rdb <- cbind(as.factor(y_valid),yscore)
colnames(rdb) = c('y', "yscore")

# calculate roc and auc
roc_stat <- roc_curve(rdb, y, yscore)
roc_stat$specificity <- 1 - roc_stat$specificity
auc = roc_auc(rdb, y, yscore, event_level = "second")
auc = auc$.estimate
title = paste('ROC Curve xGBM Linear (AUC = ',toString(round(auc,2)),')',sep = '')

# drawing ROC-Curve
ROC <- ggplot(roc_stat, aes(x = sensitivity, y = specificity)) + 
  geom_line(size = 1, color="blue") +
  labs(x = "False Positive Rate", y = "True Positive Rate") +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", ) +
  theme_minimal() +
  theme(legend.title = element_blank())+
  ggtitle(title)

pdf("/Your/Path/ROC_XBGM_LIN_unbalanced_final.pdf", height = 4, width = 6)

ROC

dev.off()




# ==========
# Draw Precision Recall Plot

#calculate pr(Precision-Recall) and aps (Average Precision Score)
pr_stat <- pr_curve(rdb, y, yscore, event_level = "second")
pauc <- pr_auc(rdb, y, yscore, event_level = "second")
pauc <- pauc$.estimate
title = paste('Precision Recall Plot xGBM Linear (AUC = ',toString(round(pauc,2)),')',sep = '')

#calculate baseline Precision to be expected if the model guesses randomly
baseline <- sum(y_valid == 1) / length(y_valid)

# draw presicion recall plot
PR <- ggplot(pr_stat, aes(x = recall, y = precision)) + 
  geom_line(size = 1, color="blue") +
  labs(x = "Recall", y = "Precision") +
  theme_minimal() +
  geom_hline(yintercept = baseline, linetype = "dashed") +
  theme(legend.title = element_blank())+
  ggtitle(title)

pdf("/Your/Path/PR_XGBM_LIN_unbal_final.pdf", height = 4, width = 6)

PR

dev.off()


# ==========
# save performance measures and graphs in List

Performance_xgbLINEAR_unbal <- list("pred" = rdb,
                                  "ConfusionMatrix" = cm, 
                                  "auc" = auc, 
                                  "roc" = roc_stat,
                                  "roc_plot" = ROC, 
                                  "pr" = pr_stat, 
                                  "pr_plot" = PR )



rm( rdb, cm, auc, roc_stat, ROC, pr_stat, PR, y_pred)

################################################################################



# ==========
# Tree


# ==========
# predict class probabilities with optimal model

y_pred <- predict(xgbTREE_unbal, 
                  newdata = X_val,
                  type = "prob")


# ==========
# calculate Confusion Matrix

cm <- confusionMatrix(as.factor(ifelse(y_pred$`1` > 0.5, 1, 0)), as.factor(y_valid), positive = "1")
cm



# ==========
# Draw ROC Curve

# create matrix of predicted and true classes
yscore <- data.frame(y_pred$`1`)
rdb <- cbind(as.factor(y_valid),yscore)
colnames(rdb) = c('y', "yscore")

# calculate roc and auc
roc_stat <- roc_curve(rdb, y, yscore)
roc_stat$specificity <- 1 - roc_stat$specificity
auc = roc_auc(rdb, y, yscore, event_level = "second")
auc = auc$.estimate
title = paste('ROC Curve xGBM Tree (AUC = ',toString(round(auc,2)),')',sep = '')

# drawing ROC-Curve
ROC <- ggplot(roc_stat, aes(x = sensitivity, y = specificity)) + 
  geom_line(size = 1, color="blue") +
  labs(x = "False Positive Rate", y = "True Positive Rate") +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", ) +
  theme_minimal() +
  theme(legend.title = element_blank())+
  ggtitle(title)

pdf("/Your/Path/ROC_XGBM_TREE_unbal_final.pdf", height = 4, width = 6)

ROC

dev.off()




# ==========
# Draw Precision Recall Plot

#calculate pr(Precision-Recall) and aps (Average Precision Score)
pr_stat <- pr_curve(rdb, y, yscore, event_level = "second")
pauc <- pr_auc(rdb, y, yscore, event_level = "second")
pauc <- pauc$.estimate
title = paste('Precision Recall Plot xGBM Tree (AUC = ',toString(round(pauc,2)),')',sep = '')

#calculate baseline Precision to be expected if the model guesses randomly
baseline <- sum(y_valid == 1) / length(y_valid)

# draw presicion recall plot
PR <- ggplot(pr_stat, aes(x = recall, y = precision)) + 
  geom_line(size = 1, color="blue") +
  labs(x = "Recall", y = "Precision") +
  theme_minimal() +
  geom_hline(yintercept = baseline, linetype = "dashed") +
  theme(legend.title = element_blank())+
  ggtitle(title)

pdf("/Your/Path/PR_XGBM_TREE_unbal_final.pdf", height = 4, width = 6)

PR

dev.off()


# ==========
# save performance measures and plotly graphs in List

Performance_xgbTREE_unbal <- list("pred" = rdb,
                                "ConfusionMatrix" = cm, 
                                "auc" = auc, 
                                "roc" = roc_stat,
                                "roc_plot" = ROC, 
                                "pr" = pr_stat, 
                                "pr_plot" = PR )





# ==========
# join performace lists for different models in list and save

xGBM_performances_unbal <- list("Linear" = Performance_xgbLINEAR_unbal, 
                              "Tree" = Performance_xgbTREE_unbal) 

saveRDS(xGBM_performances_unbal, "/Your/Path/Performance_xGBM_unbalanced.rds")










# ______________________________________________________________________________

#                             LOGISTIC REGRESSION

# ______________________________________________________________________________



# ==============================================================================
# group regression (LASSO)
# ==============================================================================

print("Unbalanced Group LASSO started")

# ==========
#create group index for X variables
v.group <- c(1:21, 
             22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 
             23, 23, 23, 
             24, 24, 24, 
             25, 25, 25, 25, 
             26, 26, 26, 26, 
             27:52) 
length(X_bal[1 ,])==length(v.group)

# recode response into -1 and 1
y_bal_grp <- ifelse(y_devds_bal == 1, 1, -1)# recode response to be numeric -1 and 1
y_unbal_grp <- ifelse(y_devds_unbal == 1, 1, -1)# recode response to be numeric -1 and 1


# Train models

start_time <- Sys.time() # Track start time
#_______________________________________________________________________________

# Balanced data set
Group_lasso_bal <- cv.gglasso(x = X_bal, 
                              y = y_bal_grp, # cv.gglasso requires response variable to be numeric -1 and 1
                              group = v.group, 
                              loss = "logit", 
                              pred.loss = "misclass", # classification error set as loss function for cv error 
                              intercept = TRUE, 
                              nfolds = 5)
#__________________________  _____________________________________________________
Sys.time() - start_time # Time difference of 

# run duration:


start_time <- Sys.time() # Track start time
#_______________________________________________________________________________

# Unbalanced data set
Group_lasso_unbal <- cv.gglasso(x = X_unbal, 
                                y = y_unbal_grp, # cv.gglasso requires response variable to be numeric -1 and 1
                                group = v.group, 
                                loss = "logit", 
                                pred.loss = "misclass", # classification error set as loss function for cv error 
                                intercept = TRUE, 
                                nfolds = 5)
#__________________________  _____________________________________________________
Sys.time() - start_time # Time difference of 

# run duration: # 6 min



# ==============================================================================
# non-group LASSO regression
# ==============================================================================


start_time <- Sys.time() # Track start time
#_______________________________________________________________________________

ngr_Lasso_bal <- cv.glmnet(X_bal, 
                           y_devds_bal,
                           type.measure = "class",
                           alpha = 1, 
                           family= "binomial",
                           nfolds = 5)
#__________________________  _____________________________________________________
Sys.time() - start_time # Time difference of 

# run duration:


start_time <- Sys.time() # Track start time
#_______________________________________________________________________________

ngr_Lasso_unbal <- cv.glmnet(X_unbal, 
                             y_devds_unbal,
                             type.measure = "class",
                             alpha = 1, 
                             family= "binomial",
                             nfolds = 5)
#__________________________  _____________________________________________________
Sys.time() - start_time # Time difference of 

# run duration: ALL IN SECONDS




# ==============================================================================
# non-group Ridge regression
# ==============================================================================



start_time <- Sys.time() # Track start time
#_______________________________________________________________________________

ngr_Ridge_bal <- cv.glmnet(X_bal, 
                           y_devds_bal,
                           type.measure = "class",
                           alpha = 0, 
                           family= "binomial",
                           nfolds = 5)
#__________________________  _____________________________________________________
Sys.time() - start_time # Time difference of 

# run duration:


start_time <- Sys.time() # Track start time
#_______________________________________________________________________________

ngr_Ridge_unbal <- cv.glmnet(X_unbal, 
                             y_devds_unbal,
                             type.measure = "class",
                             alpha = 0, 
                             family= "binomial",
                             nfolds = 5)
#__________________________  _____________________________________________________
Sys.time() - start_time # Time difference of 

# run duration:





# ==============================================================================
# non-group Elastic Net regression
# ==============================================================================



start_time <- Sys.time() # Track start time
#_______________________________________________________________________________

ngr_ENet_bal <- cv.glmnet(X_bal, 
                          y_devds_bal,
                          type.measure = "class",
                          alpha = 0.5, 
                          family= "binomial",
                          nfolds = 5)
#__________________________  _____________________________________________________
Sys.time() - start_time # Time difference of 

# run duration:


start_time <- Sys.time() # Track start time
#_______________________________________________________________________________

ngr_ENet_unbal <- cv.glmnet(X_unbal, 
                            y_devds_unbal,
                            type.measure = "class",
                            alpha = 0.5, 
                            family= "binomial",
                            nfolds = 5)
#__________________________  _____________________________________________________
Sys.time() - start_time # Time difference of 

# run duration:





################################################################################

#                           Model Performance

################################################################################


# ==============================================================================
#                                 BALANCED
# ==============================================================================



# ==============================================================================
# group regression (LASSO)
# ==============================================================================


# ==========
# predict class probabilities with optimal model

y_pred <- predict(Group_lasso_bal, 
                  s = Group_lasso_bal$lambda.min,
                  newx = X_val, 
                  type = "link") # predict to obtain class probabiities


# ==========
# predict classes with optimal model

y_pred_class <- predict(Group_lasso_bal,
                        s = Group_lasso_bal$lambda.min,
                        newx = X_val, 
                        type = "class") # re-predicting in order to obtain predicted classes


# ==========
# calculate Confusion Matrix

y_valid_class <- y_valid
y_valid_class <- ifelse(y_valid == 1, 1, -1)

cm <- confusionMatrix(as.factor(y_pred_class), as.factor(y_valid_class), positive = "1")
cm



# ==========
# Draw ROC Curve

# create matrix of predicted and true classes
yscore <- data.frame(y_pred)
rdb <- cbind(as.factor(y_valid),yscore)
colnames(rdb) = c('y','yscore')

# calculate roc and auc
roc_stat <- roc_curve(rdb, y, yscore)
roc_stat$specificity <- 1 - roc_stat$specificity
auc = roc_auc(rdb, y, yscore, event_level = "second")
auc = auc$.estimate
title = paste('ROC Curve Group LASSO Regression (AUC = ',toString(round(auc,2)),')',sep = '')

# drawing ROC-Curve
ROC <- ggplot(roc_stat, aes(x = sensitivity, y = specificity)) + 
  geom_line(size = 1, color="blue") +
  labs(x = "False Positive Rate", y = "True Positive Rate") +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", ) +
  theme_minimal() +
  theme(legend.title = element_blank())+
  ggtitle(title)


pdf("/Your/Path/ROC_Group_LASSO_bal_final.pdf", height = 4, width = 6)

ROC

dev.off()


# ==========
# Draw Precision Recall Plot

#calculate pr(Precision-Recall) and aps (Average Precision Score)
pr_stat <- pr_curve(rdb, y, yscore, event_level = "second")
pauc <- pr_auc(rdb, y, yscore, event_level = "second")
pauc <- pauc$.estimate
title = paste('Precision Recall Plot Group LASSO Regression (AUC = ',toString(round(pauc,2)),')',sep = '')

#calculate baseline Precision to be expected if the model guesses randomly
baseline <- sum(y_valid == 1) / length(y_valid)

# draw presicion recall plot
PR <- ggplot(pr_stat, aes(x = recall, y = precision)) + 
  geom_line(size = 1, color="blue") +
  labs(x = "Recall", y = "Precision") +
  theme_minimal() +
  geom_hline(yintercept = baseline, linetype = "dashed") +
  theme(legend.title = element_blank())+
  ggtitle(title)


pdf("/Your/Path/PR_Group_LASSO_bal_final.pdf", height = 4, width = 6)

PR

dev.off()


# ==========
# save performance measures and plotly graphs in List

Performance_group_lasso_bal <- list("pred" = rdb,
                                "ConfusionMatrix" = cm, 
                                "auc" = auc, 
                                "roc" = roc_stat,
                                "roc_plot" = ROC, 
                                "pr" = pr_stat, 
                                "pr_plot" = PR )


rm(rdb, cm, auc, roc_stat, ROC, pr_stat, PR)









# ==========
# preparing the VALIDATION data set for non-group regression


# ==============================================================================
# non-group LASSO regression
# ==============================================================================


####################

# ==========
# predict class probabilities with optimal model

y_pred <- predict(ngr_Lasso_bal, 
                  s = ngr_Lasso_bal$lambda.min,
                  newx = X_val, 
                  type = "response") # predict to obtain class probabiities


# ==========
# predict classes with optimal model

y_pred_class <- predict(ngr_Lasso_bal,
                        s = ngr_Lasso_bal$lambda.min,
                        newx = X_val, 
                        type = "class") # re-predicting in order to obtain predicted classes


# ==========
# calculate Confusion Matrix

cm <- confusionMatrix(as.factor(y_pred_class), as.factor(y_valid), positive = "1")
cm



# ==========
# Draw ROC Curve

# create matrix of predicted and true classes
yscore <- data.frame(y_pred)
rdb <- cbind(as.factor(y_valid),yscore)
colnames(rdb) = c('y','yscore')

# calculate roc and auc
roc_stat <- roc_curve(rdb, y, yscore)
roc_stat$specificity <- 1 - roc_stat$specificity
auc = roc_auc(rdb, y, yscore, event_level = "second")
auc = auc$.estimate
title = paste('ROC Curve Non-Group LASSO Regression (AUC = ',toString(round(auc,2)),')',sep = '')

# drawing ROC-Curve
ROC <- ggplot(roc_stat, aes(x = sensitivity, y = specificity)) + 
  geom_line(size = 1, color="blue") +
  labs(x = "False Positive Rate", y = "True Positive Rate") +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", ) +
  theme_minimal() +
  theme(legend.title = element_blank())+
  ggtitle(title)


pdf("/Your/Path/ROC_non-Group_LASSO_bal_final.pdf", height = 4, width = 6)

ROC

dev.off()


# ==========
# Draw Precision Recall Plot

#calculate pr(Precision-Recall) and aps (Average Precision Score)
pr_stat <- pr_curve(rdb, y, yscore, event_level = "second")
pauc <- pr_auc(rdb, y, yscore, event_level = "second")
pauc <- pauc$.estimate
title = paste('Precision Recall Plot Non-Group LASSO Regression (AUC = ',toString(round(pauc,2)),')',sep = '')

#calculate baseline Precision to be expected if the model guesses randomly
baseline <- sum(y_valid == 1) / length(y_valid)

# draw presicion recall plot
PR <- ggplot(pr_stat, aes(x = recall, y = precision)) + 
  geom_line(size = 1, color="blue") +
  labs(x = "Recall", y = "Precision") +
  theme_minimal() +
  geom_hline(yintercept = baseline, linetype = "dashed") +
  theme(legend.title = element_blank())+
  ggtitle(title)


pdf("/Your/Path/PR_non-Group_LASSO_bal_final.pdf", height = 4, width = 6)

PR

dev.off()



# ==========
# save performance measures and plotly graphs in List

Performance_non_group_lasso_bal <- list("pred" = rdb,
                                    "ConfusionMatrix" = cm, 
                                    "auc" = auc, 
                                    "roc" = roc_stat,
                                    "roc_plot" = ROC, 
                                    "pr" = pr_stat, 
                                    "pr_plot" = PR )


rm(rdb, cm, auc, roc_stat, ROC, pr_stat, PR)









# ==============================================================================
# non group Ridge Regression
# ==============================================================================


# ==========
# predict class probabilities with optimal model

y_pred <- predict(ngr_Ridge_bal, 
                  s = ngr_Ridge_bal$lambda.min,
                  newx = X_val, 
                  type = "response") # predict to obtain class probabiities


# ==========
# predict classes with optimal model

y_pred_class <- predict(ngr_Ridge_bal,
                        s = ngr_Ridge_bal$lambda.min,
                        newx = X_val, 
                        type = "class") # re-predicting in order to obtain predicted classes


# ==========
# calculate Confusion Matrix

cm <- confusionMatrix(as.factor(y_pred_class), as.factor(y_valid), positive = "1")
cm



# ==========
# Draw ROC Curve

# create matrix of predicted and true classes
yscore <- data.frame(y_pred)
rdb <- cbind(as.factor(y_valid),yscore)
colnames(rdb) = c('y','yscore')

# calculate roc and auc
roc_stat <- roc_curve(rdb, y, yscore)
roc_stat$specificity <- 1 - roc_stat$specificity
auc = roc_auc(rdb, y, yscore, event_level = "second")
auc = auc$.estimate
title = paste('ROC Curve Ridge Regression (AUC = ',toString(round(auc,2)),')',sep = '')

# drawing ROC-Curve
ROC <- ggplot(roc_stat, aes(x = sensitivity, y = specificity)) + 
  geom_line(size = 1, color="blue") +
  labs(x = "False Positive Rate", y = "True Positive Rate") +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", ) +
  theme_minimal() +
  theme(legend.title = element_blank())+
  ggtitle(title)


pdf("/Your/Path/ROC_Ridge_bal_final.pdf", height = 4, width = 6)

ROC

dev.off()


# ==========
# Draw Precision Recall Plot

#calculate pr(Precision-Recall) and aps (Average Precision Score)
pr_stat <- pr_curve(rdb, y, yscore, event_level = "second")
pauc <- pr_auc(rdb, y, yscore, event_level = "second")
pauc <- pauc$.estimate
title = paste('Precision Recall Plot Ridge Regression (AUC = ',toString(round(pauc,2)),')',sep = '')

#calculate baseline Precision to be expected if the model guesses randomly
baseline <- sum(y_valid == 1) / length(y_valid)

# draw presicion recall plot
PR <- ggplot(pr_stat, aes(x = recall, y = precision)) + 
  geom_line(size = 1, color="blue") +
  labs(x = "Recall", y = "Precision") +
  theme_minimal() +
  geom_hline(yintercept = baseline, linetype = "dashed") +
  theme(legend.title = element_blank())+
  ggtitle(title)


pdf("/Your/Path/PR_Ridget_bal_final.pdf", height = 4, width = 6)

PR

dev.off()



# ==========
# save performance measures and plotly graphs in List

Performance_Ridge_bal <- list("pred" = rdb,
                              "ConfusionMatrix" = cm, 
                              "auc" = auc, 
                              "roc" = roc_stat,
                              "roc_plot" = ROC, 
                              "pr" = pr_stat, 
                              "pr_plot" = PR )


rm(rdb, cm, auc, roc_stat, ROC, pr_stat, PR)










# ==============================================================================
# non group Elastic Net Regression
# ==============================================================================


# ==========
# predict class probabilities with optimal model

y_pred <- predict(ngr_ENet_bal, 
                  s = ngr_ENet_bal$lambda.min,
                  newx = X_val, 
                  type = "response") # predict to obtain class probabiities


# ==========
# predict classes with optimal model

y_pred_class <- predict(ngr_ENet_bal,
                        s = ngr_ENet_bal$lambda.min,
                        newx = X_val, 
                        type = "class") # re-predicting in order to obtain predicted classes


# ==========
# calculate Confusion Matrix

cm <- confusionMatrix(as.factor(y_pred_class), as.factor(y_valid), positive = "1")
cm



# ==========
# Draw ROC Curve

# create matrix of predicted and true classes
yscore <- data.frame(y_pred)
rdb <- cbind(as.factor(y_valid),yscore)
colnames(rdb) = c('y','yscore')

# calculate roc and auc
roc_stat <- roc_curve(rdb, y, yscore)
roc_stat$specificity <- 1 - roc_stat$specificity
auc = roc_auc(rdb, y, yscore, event_level = "second")
auc = auc$.estimate
title = paste('ROC Curve Elastic Net Regression (AUC = ',toString(round(auc,2)),')',sep = '')

# drawing ROC-Curve
ROC <- ggplot(roc_stat, aes(x = sensitivity, y = specificity)) + 
  geom_line(size = 1, color="blue") +
  labs(x = "False Positive Rate", y = "True Positive Rate") +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", ) +
  theme_minimal() +
  theme(legend.title = element_blank())+
  ggtitle(title)


pdf("/Your/Path/ROC_Elastic_Net_bal_final.pdf", height = 4, width = 6)

ROC

dev.off()


# ==========
# Draw Precision Recall Plot

#calculate pr(Precision-Recall) and aps (Average Precision Score)
pr_stat <- pr_curve(rdb, y, yscore, event_level = "second")
pauc <- pr_auc(rdb, y, yscore, event_level = "second")
pauc <- pauc$.estimate
title = paste('Precision Recall Plot Elastic Net Regression (AUC = ',toString(round(pauc,2)),')',sep = '')

#calculate baseline Precision to be expected if the model guesses randomly
baseline <- sum(y_valid == 1) / length(y_valid)

# draw presicion recall plot
PR <- ggplot(pr_stat, aes(x = recall, y = precision)) + 
  geom_line(size = 1, color="blue") +
  labs(x = "Recall", y = "Precision") +
  theme_minimal() +
  geom_hline(yintercept = baseline, linetype = "dashed") +
  theme(legend.title = element_blank())+
  ggtitle(title)


pdf("/Your/Path/PR_Elastic_Net_bal_final.pdf", height = 4, width = 6)

PR

dev.off()



# ==========
# save performance measures and plotly graphs in List

Performance_elastic_net_bal <- list("pred" = rdb,
                                "ConfusionMatrix" = cm, 
                                "auc" = auc, 
                                "roc" = roc_stat,
                                "roc_plot" = ROC, 
                                "pr" = pr_stat, 
                                "pr_plot" = PR )


rm(rdb, cm, auc, roc_stat, ROC, pr_stat, PR)


# ==========
# join performace lists for different models in list and save

logistic_regression_performances_bal <- list("group_lasso" = Performance_group_lasso_bal, 
                                             "ngr_lasso" = Performance_non_group_lasso_bal,
                                             "ridge" = Performance_Ridge_bal,
                                             "elastic_net" = Performance_elastic_net_bal) 

saveRDS(logistic_regression_performances_bal, "/Your/Path/Performance_logistic_regression_bal.rds")










# ==============================================================================
#                                 UNBALANCED
# ==============================================================================



# ==============================================================================
# group regression (LASSO)
# ==============================================================================


# ==========
# predict class probabilities with optimal model

y_pred <- predict(Group_lasso_unbal, 
                  s = Group_lasso_unbal$lambda.min,
                  newx = X_val, 
                  type = "link") # predict to obtain class probabiities


# ==========
# predict classes with optimal model

y_pred_class <- predict(Group_lasso_unbal,
                        s = Group_lasso_bal$lambda.min,
                        newx = X_val, 
                        type = "class") # re-predicting in order to obtain predicted classes


# ==========
# calculate Confusion Matrix

y_valid_class <- y_valid
y_valid_class <- ifelse(y_valid == 1, 1, -1)

cm <- confusionMatrix(as.factor(y_pred_class), as.factor(y_valid_class), positive = "1")
cm



# ==========
# Draw ROC Curve

# create matrix of predicted and true classes
yscore <- data.frame(y_pred)
rdb <- cbind(as.factor(y_valid),yscore)
colnames(rdb) = c('y','yscore')

# calculate roc and auc
roc_stat <- roc_curve(rdb, y, yscore)
roc_stat$specificity <- 1 - roc_stat$specificity
auc = roc_auc(rdb, y, yscore, event_level = "second")
auc = auc$.estimate
title = paste('ROC Curve Group LASSO Regression (AUC = ',toString(round(auc,2)),')',sep = '')

# drawing ROC-Curve
ROC <- ggplot(roc_stat, aes(x = sensitivity, y = specificity)) + 
  geom_line(size = 1, color="blue") +
  labs(x = "False Positive Rate", y = "True Positive Rate") +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", ) +
  theme_minimal() +
  theme(legend.title = element_blank())+
  ggtitle(title)


pdf("/Your/Path/ROC_Group_LASSO_unbal_final.pdf", height = 4, width = 6)

ROC

dev.off()


# ==========
# Draw Precision Recall Plot

#calculate pr(Precision-Recall) and aps (Average Precision Score)
pr_stat <- pr_curve(rdb, y, yscore, event_level = "second")
pauc <- pr_auc(rdb, y, yscore, event_level = "second")
pauc <- pauc$.estimate
title = paste('Precision Recall Plot Group LASSO Regression (AUC = ',toString(round(pauc,2)),')',sep = '')

#calculate baseline Precision to be expected if the model guesses randomly
baseline <- sum(y_valid == 1) / length(y_valid)

# draw presicion recall plot
PR <- ggplot(pr_stat, aes(x = recall, y = precision)) + 
  geom_line(size = 1, color="blue") +
  labs(x = "Recall", y = "Precision") +
  theme_minimal() +
  geom_hline(yintercept = baseline, linetype = "dashed") +
  theme(legend.title = element_blank())+
  ggtitle(title)


pdf("/Your/Path/PR_Group_LASSO_unbal_final.pdf", height = 4, width = 6)

PR

dev.off()


# ==========
# save performance measures and plotly graphs in List

Performance_group_lasso_unbal <- list("pred" = rdb,
                                    "ConfusionMatrix" = cm, 
                                    "auc" = auc, 
                                    "roc" = roc_stat,
                                    "roc_plot" = ROC, 
                                    "pr" = pr_stat, 
                                    "pr_plot" = PR )


rm(rdb, cm, auc, roc_stat, ROC, pr_stat, PR)









# ==========
# preparing the VALIDATION data set for non-group regression


# ==============================================================================
# non-group LASSO regression
# ==============================================================================


####################

# ==========
# predict class probabilities with optimal model

y_pred <- predict(ngr_Lasso_unbal, 
                  s = ngr_Lasso_unbal$lambda.min,
                  newx = X_val, 
                  type = "response") # predict to obtain class probabiities


# ==========
# predict classes with optimal model

y_pred_class <- predict(ngr_Lasso_unbal,
                        s = ngr_Lasso_unbal$lambda.min,
                        newx = X_val, 
                        type = "class") # re-predicting in order to obtain predicted classes


# ==========
# calculate Confusion Matrix

cm <- confusionMatrix(as.factor(y_pred_class), as.factor(y_valid), positive = "1")
cm



# ==========
# Draw ROC Curve

# create matrix of predicted and true classes
yscore <- data.frame(y_pred)
rdb <- cbind(as.factor(y_valid),yscore)
colnames(rdb) = c('y','yscore')

# calculate roc and auc
roc_stat <- roc_curve(rdb, y, yscore)
roc_stat$specificity <- 1 - roc_stat$specificity
auc = roc_auc(rdb, y, yscore, event_level = "second")
auc = auc$.estimate
title = paste('ROC Curve Non-Group LASSO Regression (AUC = ',toString(round(auc,2)),')',sep = '')

# drawing ROC-Curve
ROC <- ggplot(roc_stat, aes(x = sensitivity, y = specificity)) + 
  geom_line(size = 1, color="blue") +
  labs(x = "False Positive Rate", y = "True Positive Rate") +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", ) +
  theme_minimal() +
  theme(legend.title = element_blank())+
  ggtitle(title)


pdf("/Your/Path/ROC_non-Group_LASSO_unbal_final.pdf", height = 4, width = 6)

ROC

dev.off()


# ==========
# Draw Precision Recall Plot

#calculate pr(Precision-Recall) and aps (Average Precision Score)
pr_stat <- pr_curve(rdb, y, yscore, event_level = "second")
pauc <- pr_auc(rdb, y, yscore, event_level = "second")
pauc <- pauc$.estimate
title = paste('Precision Recall Plot Non-Group LASSO Regression (AUC = ',toString(round(pauc,2)),')',sep = '')

#calculate baseline Precision to be expected if the model guesses randomly
baseline <- sum(y_valid == 1) / length(y_valid)

# draw presicion recall plot
PR <- ggplot(pr_stat, aes(x = recall, y = precision)) + 
  geom_line(size = 1, color="blue") +
  labs(x = "Recall", y = "Precision") +
  theme_minimal() +
  geom_hline(yintercept = baseline, linetype = "dashed") +
  theme(legend.title = element_blank())+
  ggtitle(title)


pdf("/Your/Path/PR_non-Group_LASSO_unbal_final.pdf", height = 4, width = 6)

PR

dev.off()



# ==========
# save performance measures and plotly graphs in List

Performance_non_group_lasso_unbal <- list("pred" = rdb,
                                        "ConfusionMatrix" = cm, 
                                        "auc" = auc, 
                                        "roc" = roc_stat,
                                        "roc_plot" = ROC, 
                                        "pr" = pr_stat, 
                                        "pr_plot" = PR )


rm(rdb, cm, auc, roc_stat, ROC, pr_stat, PR)









# ==============================================================================
# non group Ridge Regression
# ==============================================================================


# ==========
# predict class probabilities with optimal model

y_pred <- predict(ngr_Ridge_unbal, 
                  s = ngr_Ridge_unbal$lambda.min,
                  newx = X_val, 
                  type = "response") # predict to obtain class probabiities


# ==========
# predict classes with optimal model

y_pred_class <- predict(ngr_Ridge_unbal,
                        s = ngr_Ridge_unbal$lambda.min,
                        newx = X_val, 
                        type = "class") # re-predicting in order to obtain predicted classes


# ==========
# calculate Confusion Matrix

cm <- confusionMatrix(as.factor(y_pred_class), as.factor(y_valid), positive = "1")
cm



# ==========
# Draw ROC Curve

# create matrix of predicted and true classes
yscore <- data.frame(y_pred)
rdb <- cbind(as.factor(y_valid),yscore)
colnames(rdb) = c('y','yscore')

# calculate roc and auc
roc_stat <- roc_curve(rdb, y, yscore)
roc_stat$specificity <- 1 - roc_stat$specificity
auc = roc_auc(rdb, y, yscore, event_level = "second")
auc = auc$.estimate
title = paste('ROC Curve Ridge Regression (AUC = ',toString(round(auc,2)),')',sep = '')

# drawing ROC-Curve
ROC <- ggplot(roc_stat, aes(x = sensitivity, y = specificity)) + 
  geom_line(size = 1, color="blue") +
  labs(x = "False Positive Rate", y = "True Positive Rate") +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", ) +
  theme_minimal() +
  theme(legend.title = element_blank())+
  ggtitle(title)


pdf("/Your/Path/ROC_Ridge_unbal_final.pdf", height = 4, width = 6)

ROC

dev.off()


# ==========
# Draw Precision Recall Plot

#calculate pr(Precision-Recall) and aps (Average Precision Score)
pr_stat <- pr_curve(rdb, y, yscore, event_level = "second")
pauc <- pr_auc(rdb, y, yscore, event_level = "second")
pauc <- pauc$.estimate
title = paste('Precision Recall Plot Ridge Regression (AUC = ',toString(round(pauc,2)),')',sep = '')

#calculate baseline Precision to be expected if the model guesses randomly
baseline <- sum(y_valid == 1) / length(y_valid)

# draw presicion recall plot
PR <- ggplot(pr_stat, aes(x = recall, y = precision)) + 
  geom_line(size = 1, color="blue") +
  labs(x = "Recall", y = "Precision") +
  theme_minimal() +
  geom_hline(yintercept = baseline, linetype = "dashed") +
  theme(legend.title = element_blank())+
  ggtitle(title)


pdf("/Your/Path/PR_Ridget_unbal_final.pdf", height = 4, width = 6)

PR

dev.off()



# ==========
# save performance measures and plotly graphs in List

Performance_Ridge_unbal <- list("pred" = rdb,
                              "ConfusionMatrix" = cm, 
                              "auc" = auc, 
                              "roc" = roc_stat,
                              "roc_plot" = ROC, 
                              "pr" = pr_stat, 
                              "pr_plot" = PR )


rm(rdb, cm, auc, roc_stat, ROC, pr_stat, PR)










# ==============================================================================
# non group Elastic Net Regression
# ==============================================================================


# ==========
# predict class probabilities with optimal model

y_pred <- predict(ngr_ENet_unbal, 
                  s = ngr_ENet_unbal$lambda.min,
                  newx = X_val, 
                  type = "response") # predict to obtain class probabiities


# ==========
# predict classes with optimal model

y_pred_class <- predict(ngr_ENet_unbal,
                        s = ngr_ENet_unbal$lambda.min,
                        newx = X_val, 
                        type = "class") # re-predicting in order to obtain predicted classes


# ==========
# calculate Confusion Matrix

cm <- confusionMatrix(as.factor(y_pred_class), as.factor(y_valid), positive = "1")
cm



# ==========
# Draw ROC Curve

# create matrix of predicted and true classes
yscore <- data.frame(y_pred)
rdb <- cbind(as.factor(y_valid),yscore)
colnames(rdb) = c('y','yscore')

# calculate roc and auc
roc_stat <- roc_curve(rdb, y, yscore)
roc_stat$specificity <- 1 - roc_stat$specificity
auc = roc_auc(rdb, y, yscore, event_level = "second")
auc = auc$.estimate
title = paste('ROC Curve Elastic Net Regression (AUC = ',toString(round(auc,2)),')',sep = '')

# drawing ROC-Curve
ROC <- ggplot(roc_stat, aes(x = sensitivity, y = specificity)) + 
  geom_line(size = 1, color="blue") +
  labs(x = "False Positive Rate", y = "True Positive Rate") +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", ) +
  theme_minimal() +
  theme(legend.title = element_blank())+
  ggtitle(title)


pdf("/Your/Path/ROC_Elastic_Net_unbal_final.pdf", height = 4, width = 6)

ROC

dev.off()


# ==========
# Draw Precision Recall Plot

#calculate pr(Precision-Recall) and aps (Average Precision Score)
pr_stat <- pr_curve(rdb, y, yscore, event_level = "second")
pauc <- pr_auc(rdb, y, yscore, event_level = "second")
pauc <- pauc$.estimate
title = paste('Precision Recall Plot Elastic Net Regression (AUC = ',toString(round(pauc,2)),')',sep = '')

#calculate baseline Precision to be expected if the model guesses randomly
baseline <- sum(y_valid == 1) / length(y_valid)

# draw presicion recall plot
PR <- ggplot(pr_stat, aes(x = recall, y = precision)) + 
  geom_line(size = 1, color="blue") +
  labs(x = "Recall", y = "Precision") +
  theme_minimal() +
  geom_hline(yintercept = baseline, linetype = "dashed") +
  theme(legend.title = element_blank())+
  ggtitle(title)


pdf("/Your/Path/PR_Elastic_Net_unbal_final.pdf", height = 4, width = 6)

PR

dev.off()



# ==========
# save performance measures and plotly graphs in List

Performance_elastic_net_unbal <- list("pred" = rdb,
                                    "ConfusionMatrix" = cm, 
                                    "auc" = auc, 
                                    "roc" = roc_stat,
                                    "roc_plot" = ROC, 
                                    "pr" = pr_stat, 
                                    "pr_plot" = PR )


rm(rdb, cm, auc, roc_stat, ROC, pr_stat, PR)


# ==========
# join performace lists for different models in list and save

logistic_regression_performances_unbal <- list("group_lasso" = Performance_group_lasso_unbal, 
                                             "ngr_lasso" = Performance_non_group_lasso_unbal,
                                             "ridge" = Performance_Ridge_unbal,
                                             "elastic_net" = Performance_elastic_net_unbal) 

saveRDS(logistic_regression_performances_unbal, "/Your/Path/Performance_logistic_regression_unbal.rds")



balanced <- list("group_lasso" = Group_lasso_bal,
                 "lasso" = ngr_Lasso_bal,
                 "ridge" = ngr_Ridge_bal,
                 "enet" = ngr_ENet_bal,
                 "r_forest" = RF_bal,
                 "svm" = SVM_bal,
                 "lin_xGBM" = xgbLIN_bal,
                 "tree_xGBM" = xgbTREE_bal)

unbalanced <- list("group_lasso" = Group_lasso_unbal,
                 "lasso" = ngr_Lasso_unbal,
                 "ridge" = ngr_Ridge_unbal,
                 "enet" = ngr_ENet_unbal,
                 "r_forest" = RF_unbal,
                 "svm" = SVM_unbal,
                 "lin_xGBM" = xgbLIN_unbal,
                 "tree_xGBM" = xgbTREE_unbal)

final_models <- list("balanced" = balanced,
                     "unbalanced" = unbalanced)

saveRDS(final_models, "/Your/Path/final_models.rds")



# end
# ----