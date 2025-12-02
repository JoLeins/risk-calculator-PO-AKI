# ==============================================================================
# Training SVM Models on PO-AKI Dataset using different seeds for 
# variance estimation and preprocessing selection [[Part1](Split for parallel submission)]
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

seeded_SVM <- function(data, vector_of_seeds, tunelength, control){
  
  list_models <- list()
  for(i in vector_of_seeds){
    start <- Sys.time()
    name <- as.character(i)
    
    # split the data into train and test
    set.seed(i)
    trainIndex <- createDataPartition(y = data$AKI_bin, p = .8,
                                      list = FALSE,
                                      times = 1)
    train <- data[trainIndex,]
    
    
    list_models[[name]] <- list()
    
    # store train indices
    list_models[[name]]$train_ids <- trainIndex 
    
    # tune hyperparameters and save resulting objects
    list_models[[name]]$tune <- train(AKI_bin ~ ., 
                                      data = train, 
                                      method = "svmRadial", 
                                      metric = "Kappa", 
                                      trControl = control,
                                      tuneLength = tunelength
                                      )
    end <- Sys.time()
    time <- end-start
    
#    print(paste("seed ", i, " was done in ", round(time, digits = 2), "mins"))
  }
  return(list_models)
}








# ----
# setup

#load packages

load_packages( c("tidyverse","dplyr", "caret")) # "kernlab",


#setwd("~/../Desktop/_masterthesis/data/")
set.seed(3010)
vector_of_seeds <- sample(1:50000, 100, replace = FALSE)








# ----
# main




# ==============================================================================
# loading and preparing dataset
# ==============================================================================



# ==========
# load data
devds_unbal <- readRDS("/your/path/devds_original_prep.rds")
devds_bal <- readRDS("/your/path/devds_balanced_prep.rds")




print("Preparation started")

# ==========
# reencode levels for application of kernlab based train

levels(devds_unbal$AKI_bin) <- c("no_event", "event")
levels(devds_bal$AKI_bin) <- c("no_event", "event")



print("Preparation finished")

################################################################################

#                           HP tuning and Training

################################################################################

### NOTE THAT SVM CONVERTS FACTORS INTO DUMMIES AUTOMATICALLY AND CAN RUN INTO ISSUES IF DUMMIES ARE CREATED MANUALLY!!!!!



# ==========
# define hp tuning parameters

train_control <- trainControl(method="CV", 
                              number=5, 
                              verboseIter = FALSE,
                              classProbs = TRUE)

tuneLength <- 15



# ==========
# generate lists of models



# ____
ptm <- proc.time()

print("Unbalanced SVM started")

# train model (ATU)
list_SVM_unbal <- seeded_SVM(devds_unbal, vector_of_seeds, tunelength = tuneLength, control = train_control)

#save model (ATU)
saveRDS(list_SVM_unbal, "/your/path/list_SVM_unbal.rds")

print(proc.time() - ptm) # 

print("Unbalanced SVM finished")


# ____
ptm <- proc.time()

print("Balanced SVM started")

# train model (ATB)
list_SVM_bal <- seeded_SVM(devds_bal, vector_of_seeds, tunelength = tuneLength, control = train_control)

#save model (ATB)
saveRDS(list_SVM_bal, "/your/path/list_SVM_bal.rds")

print(proc.time() - ptm) #

print("Balanced SVM finished")



################################################################################


# ----
# end


# ca. 3.5 days estimated

    



