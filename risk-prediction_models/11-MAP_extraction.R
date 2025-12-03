# ==============================================================================
# Extraction of univariate Features from Mean Arterial Pressure Time Series
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
# function


# function to compute min MAP for a certain number of cumulative (non-consecutive) minutes
calculate_lowest_map <- function(data, minutes) {
  t(apply(data, 1, function(row) {
    sorted_values <- sort(row[!is.na(row)])
    sapply(minutes, function(x) if (length(sorted_values) >= x) sorted_values[x] else NA)
  }))
}


# function to compute the AUC below a threshold
compute_auc <- function(zeitreihe, threshold) {
  # time series until first NA
  zeitreihe <- zeitreihe[!is.na(zeitreihe)]
  
  # AUC between curve and threshold
  differenzen <- ifelse(zeitreihe < threshold, threshold - zeitreihe, 0)
  
  sum(differenzen)
}


# create data frame with computed AUC values
compute_MAP_data_auc <- function(MAP_df, thresholds) {
  MAP_data_auc <- data.frame(matrix(ncol = length(thresholds), nrow = nrow(MAP_df)))
  
  # set colnames
  colnames(MAP_data_auc) <- paste0("aucMAPunder", thresholds, "Anes")
  
  # compute AUC for all thresholds
  for (i in seq_along(thresholds)) {
    MAP_data_auc[, i] <- apply(MAP_df[, 0:ncol(MAP_df)], 1, compute_auc, threshold = thresholds[i])
  }
  
  return(MAP_data_auc)
}


# function to cumpute entropy of time series
entropy <- function(time_series) {
  time_series <- time_series[!is.na(time_series)]  # remove NA´s
  individual_values <- unique(time_series)  # filter only unique values
  N <- length(time_series)  # length of time series
  p_s <- 0
  
  for(x in individual_values) {
    n_x <- length(time_series[time_series == x])  # count of x instances
    p_x <- n_x / N  # probability of value x
    p_s <- p_s + p_x * log(p_x)  # sum of probabilities multiplied with their respective logarithm
  }
  
  ntrop <- -p_s  # negative summed entropy
  scaled_ntrop <- ntrop / log(length(individual_values))  # entropy scaled to ts length
  return(scaled_ntrop)
}


# function to compute all statistical MAP features
calculate_features <- function(time_series) {
  time_series <- time_series[!is.na(time_series)]  # remove NAs
  mean_val <- mean(time_series, na.rm = TRUE)  # mean
  std_val <- sd(time_series, na.rm = TRUE)  # standard deviation
  entropy_val <- entropy(time_series)  # entropy
  trend_val <- if (length(time_series) >= 3) mk.test(time_series)$estimates[["tau"]] else NA  # trend (needs at least  3 different vales)
  kurtosis_val <- kurtosis(time_series, na.rm = TRUE)  # kurtosis
  skewness_val <- skewness(time_series, na.rm = TRUE)  # skewness
  
  return(c(mean_val, std_val, entropy_val, trend_val, kurtosis_val, skewness_val))
}


# ----
# setup

#install.packages("dplyr")
#install.packages("zoo")
#install.packages("ggplot2")
#install.packages("reshape2")
#install.packages("moments")
#install.packages("trend")
#install.packages("tidyr")

library(dplyr)
library(zoo)
library(ggplot2)
library(reshape2)
library(moments)
library(trend)
library(tidyr)

setwd("/your/workspace/")




# ----
# main

# load MAP time-series dataframe
MAP_df <- read.csv("/your/dataframe.csv") #cols = OP minutes, rows = patients. all values after end of time series should be NA

# change accession column to rownames if necessary
rownames(MAP_df) <- MAP_df$X
MAP_df <- select(MAP_df, -c(X))

# create Feature data frame
MAP_features <- data.frame(matrix(ncol = 0, nrow = nrow(MAP_df)))
rownames(MAP_features) <- rownames(MAP_df)
#rownames(MAP_features) <- MAP_df$X # add rownames as e.g. patient identifiers 


# ______________________________________________________________________________
#                               Minimal values

# define cumulative minutes
cumulative_minutes <- c(1, 5)

# add cumulative minimal values to feature data frame
lowest_map <- calculate_lowest_map(MAP_df, cumulative_minutes)
colnames(lowest_map) <- paste0("minMAPcumu", cumulative_minutes, "MinAnes:")
MAP_features <- cbind(MAP_features, lowest_map)
rm(lowest_map, cumulative_minutes)


# ______________________________________________________________________________
#                                   AUC

# Berechnungen für jeden DataFrame durchführen und Ergebnisse zum bestehenden DataFrame hinzufügen
MAP_data_auc <- compute_MAP_data_auc(MAP_df, 65)
MAP_features <- cbind(MAP_features, MAP_data_auc)
rm(MAP_data_auc)


# ______________________________________________________________________________
#             Mean, Std, Entropy, Skewness, Kurtosis and Trend

# calculate statistical features
features <- t(apply(MAP_df, 1, calculate_features))
colnames(features) <- c("meanAnes", "stdAnes", "entropyAnes", "trendAnes", 
                        "kurtosisAnes", "skewnessAnes")
MAP_features <- cbind(MAP_features, features)
rm(features)


# ______________________________________________________________________________
#                               Baseline value

# extract baseline values
MAP_features$Baseline_MAP <- MAP_df[, 1]



# save MAP_features data frame
write.csv(MAP_features, file = "./MAP_Features.csv",row.names = TRUE)


# ----
# end
