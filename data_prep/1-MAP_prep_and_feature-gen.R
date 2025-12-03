# ==================================================================== 
# Script for initial preprocessing
# and for generating various features from the MAP time series 
# ====================================================================

# MIT License
# Copyright (c) 2025 by Hendrik Meyer and Jonas Leins
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

# Load data and perform initial cleaning ####
rm(list = ls())


library(dplyr)
library(zoo)
library(ggplot2)
library(reshape2)
library(moments)
library(trend)
library(tidyr)

# Print package versions without a colon after "version"
cat("version", as.character(packageVersion("dplyr")), "\n")
cat("version", as.character(packageVersion("zoo")), "\n")
cat("version", as.character(packageVersion("ggplot2")), "\n")
cat("version", as.character(packageVersion("reshape2")), "\n")
cat("version", as.character(packageVersion("moments")), "\n")
cat("version", as.character(packageVersion("trend")), "\n")
cat("version", as.character(packageVersion("tidyr")), "\n")

# Load the dataset (Datensatz_PO-AKI_preprocessed_25_01_26.csv)
daten <- read.csv("/your/path/full_dataset.csv")

# Build a new table containing only the MAP values
map_daten <- daten[paste0("map.", 0:719)]

rownames(map_daten) <- daten$Protokollnummer
names(map_daten) <- as.integer(sub("map.", "", names(map_daten)))

rownames(daten) <- daten$Protokollnummer

# Remove all columns whose names start with "map."
daten <- daten[, !grepl("^map\\.", names(daten))]

daten <- daten[,-c(1,2)] # remove redundant protocol number columns from 'daten'

# Build a table with summary statistics related to the MAP values
map_daten_analyse <- daten %>%
  select(grep("Dauer|ref", names(daten))) 

rownames(map_daten_analyse) <- daten$Protokollnummer

# Function to determine the number of non-NA values in each row
anzahl_MAP_Werte <- function(dataframe) {
  apply(dataframe, 1, function(zeile) sum(!is.na(zeile)))
}

anzahl_MAP_Werte <- anzahl_MAP_Werte(map_daten)

map_daten_analyse <- cbind(map_daten_analyse, anzahl_MAP_Werte)

# MAP time series begins at defAneStart (= start of anaesthesia induction)

# Some MAP time series end before anaesthesia end (time series are continuous without intermediate NAs)
print(sum(map_daten_analyse$refAneEnde - map_daten_analyse$refDefAneStart > map_daten_analyse$anzahl_MAP_Werte))  # 12 cases out of 24,000
# MAP time series may also end before surgery end
print(sum(map_daten_analyse$refNaht - map_daten_analyse$refDefAneStart > map_daten_analyse$anzahl_MAP_Werte))  # 12 cases out of 24,000

# Remove rows where refNaht - refDefAneStart > anzahl_MAP_Werte (i.e., MAP data not available for the full surgery duration)
map_daten_analyse <- map_daten_analyse[
  !(map_daten_analyse$refNaht - map_daten_analyse$refDefAneStart > map_daten_analyse$anzahl_MAP_Werte), ]

# Filter map_daten to keep only rows present in map_daten_analyse
gemeinsame_indizes <- intersect(rownames(map_daten), rownames(map_daten_analyse))
map_daten <- map_daten[gemeinsame_indizes, ]
daten <- daten[gemeinsame_indizes, ]

# For feature engineering, inspect various time periods ####
# Therefore, build dataframes containing MAP data for specific phases

mean(map_daten_analyse$einlDauer)   # 24.9 min
mean(map_daten_analyse$opDauer)     # 125.7 min
mean(map_daten_analyse$narkoseDauer)# 199.8746 min

# Plot distribution of opDauer
ggplot(map_daten_analyse, aes(x = opDauer)) +
  geom_histogram(binwidth = 5, fill = "blue", color = "black", alpha = 0.7) +
  labs(title = "Distribution of opDauer",
       x = "Surgery duration (min)",
       y = "Frequency") +
  theme_minimal()

ggplot(map_daten_analyse, aes(x = opDauer)) +
  geom_histogram(binwidth = 2, fill = "red", color = "black", alpha = 0.7) +
  labs(title = "Distribution of opDauer",
       x = "Surgery duration (min)",
       y = "Frequency") +
  xlim(0, 300) +
  theme_minimal()

# Build an induction-phase MAP dataframe (MAP values only from induction; all others NA)
# Initialize map_daten_einl as a copy with all NA values
map_daten_einl <- as.data.frame(matrix(NA, nrow = nrow(map_daten), ncol = ncol(map_daten)))
rownames(map_daten_einl) <- rownames(map_daten)
names(map_daten_einl) <- names(map_daten)

# Fill map_daten_einl with the corresponding MAP data from map_daten
for (i in 1:nrow(map_daten_analyse)) {
  start <- 0 + 1  ## Add 1 to column indices because R indexing starts at 1
  ende <- map_daten_analyse$refOpF[i] - map_daten_analyse$refDefAneStart[i] + 1  # MAP recording begins at defAneStart
  
  # Ensure that start and end indices are valid
  if (start >= 1 && ende <= ncol(map_daten) && start <= ende) {
    map_daten_einl[i, start:ende] <- map_daten[i, start:ende]
  }
}

# Function to create dataframes that include additional minutes after the end of induction
create_extended_df <- function(minutes) {
  extended_df <- as.data.frame(matrix(NA, nrow = nrow(map_daten), ncol = ncol(map_daten)))
  rownames(extended_df) <- rownames(map_daten)
  names(extended_df) <- names(map_daten)
  
  for (i in 1:nrow(map_daten_analyse)) {
    start <- 0 + 1  # Add 1 to the column indices because R starts indexing at 1
    ende <- map_daten_analyse$refOpF[i] - map_daten_analyse$refDefAneStart[i] + 1 + minutes
    
    # Ensure that start and end indices are valid
    if (start >= 1 && ende <= ncol(map_daten) && start <= ende) {
      extended_df[i, start:ende] <- map_daten[i, start:ende]
    }
  }
  
  return(extended_df)
}

# Create additional dataframes
map_daten_einlplus30  <- create_extended_df(30)
map_daten_einlplus60  <- create_extended_df(60)
map_daten_einlplus90  <- create_extended_df(90)
map_daten_einlplus120 <- create_extended_df(120)
map_daten_einlplus150 <- create_extended_df(150)

# Build an OR-phase MAP dataframe (MAP values only during surgery; all others NA)
map_daten_OP <- as.data.frame(matrix(NA, nrow = nrow(map_daten), ncol = ncol(map_daten)))
rownames(map_daten_OP) <- rownames(map_daten)
names(map_daten_OP) <- names(map_daten)

# Fill map_daten_OP with the corresponding MAP values from map_daten
for (i in 1:nrow(map_daten_analyse)) {
  start <- map_daten_analyse$refSchnitt[i] - map_daten_analyse$refDefAneStart[i] + 1  ## Add 1 because R indexing starts at 1
  ende  <- map_daten_analyse$refNaht[i]   - map_daten_analyse$refDefAneStart[i] + 1
  
  # Ensure that start and end indices are valid
  if (start >= 1 && ende <= ncol(map_daten) && start <= ende) {
    map_daten_OP[i, start:ende] <- map_daten[i, start:ende]
  }
}

# Build a MAP dataframe for the entire anaesthesia duration (induction start to anaesthesia end; others NA)
map_daten_Narkose <- as.data.frame(matrix(NA, nrow = nrow(map_daten), ncol = ncol(map_daten)))
rownames(map_daten_Narkose) <- rownames(map_daten)
names(map_daten_Narkose) <- names(map_daten)

# Fill map_daten_Narkose with the corresponding MAP values from map_daten
for (i in 1:nrow(map_daten_analyse)) {
  start <- 0 + 1  ## Add 1 to the column indices because R starts indexing at 1
  ende <- map_daten_analyse$refAneEnde[i] - map_daten_analyse$refDefAneStart[i] + 1
  
  # Ensure that start and end indices are valid
  if (start >= 1 && ende <= ncol(map_daten) && start <= ende) {
    map_daten_Narkose[i, start:ende] <- map_daten[i, start:ende]
  }
}

# Calculate various absolute MAP features across multiple intervals ####

# List of dataframes to be analysed and their corresponding names
dataframes <- list(
  map_daten_Narkose, map_daten_einl, map_daten_einlplus30, map_daten_einlplus60,
  map_daten_einlplus90, map_daten_einlplus120, map_daten_einlplus150, map_daten_OP
)
names <- c("Narkose", "einl", "einlplus30", "einlplus60", "einlplus90", "einlplus120", "einlplus150", "OP")

# Define cumulative minutes
cumulative_minutes <- c(1, 3, 5, 10)

# Function to compute the lowest MAP values for the specified cumulative minutes
calculate_lowest_map <- function(data, minutes) {
  t(apply(data, 1, function(row) {
    sorted_values <- sort(row[!is.na(row)])
    sapply(minutes, function(x) if (length(sorted_values) >= x) sorted_values[x] else NA)
  }))
}

# Loop through the dataframes and append the results to the existing dataframe
for (i in seq_along(dataframes)) {
  lowest_map <- calculate_lowest_map(dataframes[[i]], cumulative_minutes)
  colnames(lowest_map) <- paste("Lowest MAP for cumulative", cumulative_minutes, "minutes:", names[i])
  map_daten_analyse <- cbind(map_daten_analyse, lowest_map)
}

# Define intervals
sustained_minutes <- c(1, 3, 5, 10)

# Function to find the maximum value within a sliding window
finde_maximalwert <- function(zeitreihe, intervall) {
  # Restrict time series to values before the first NA
  zeitreihe <- zeitreihe[!is.na(zeitreihe)]
  
  # Skip calculation if the interval is longer than the time series
  if (length(zeitreihe) < intervall) {
    return(NA)
  }
  
  maxima <- sapply(seq_len(length(zeitreihe) - intervall + 1), function(i) {
    max(zeitreihe[i:(i + intervall - 1)], na.rm = TRUE)
  })
  
  return(min(maxima, na.rm = TRUE))
}

# Function to create a dataframe with the calculated values
berechne_map_daten_sustained <- function(map_daten, datensatz_name) {
  map_daten_sustained <- data.frame(matrix(ncol = length(sustained_minutes), nrow = nrow(map_daten)))
  
  # Column names based on intervals and dataset name
  colnames(map_daten_sustained) <- paste0("Lowest MAP for sustained ", sustained_minutes, " minutes: ", datensatz_name)
  
  # Compute the maximum value for each interval and time series
  for (i in seq_along(sustained_minutes)) {
    map_daten_sustained[, i] <- apply(map_daten[, 0:719], 1, finde_maximalwert, intervall = sustained_minutes[i])
  }
  
  return(map_daten_sustained)
}

# Compute values for each dataframe and append results to the existing dataframe
for (i in seq_along(dataframes)) {
  map_daten_sustained <- berechne_map_daten_sustained(dataframes[[i]], names[i])
  map_daten_analyse <- cbind(map_daten_analyse, map_daten_sustained)
}

# Define thresholds
thresholds <- c(75, 70, 65, 60, 55, 50)

# Function to calculate minutes below each threshold
berechne_minutes_under_threshold <- function(map_daten, datensatz_name) {
  minutes_under_threshold <- sapply(thresholds, function(threshold) rowSums(map_daten < threshold, na.rm = TRUE))
  
  # Name the result columns
  colnames(minutes_under_threshold) <- paste("Minutes of MAP under absolute threshold ", thresholds, ": ", datensatz_name)
  
  return(as.data.frame(minutes_under_threshold))
}

# Compute values for each dataframe and append results to the existing dataframe
for (i in seq_along(dataframes)) {
  minutes_under <- berechne_minutes_under_threshold(dataframes[[i]], names[i])
  map_daten_analyse <- cbind(map_daten_analyse, minutes_under)
}


# Define thresholds for AUC calculation
thresholds <- c(75, 70, 65, 60, 55, 50)

# Function to calculate the AUC for a given threshold
berechne_auc <- function(zeitreihe, threshold) {
  # Restrict time series to values before the first NA
  zeitreihe <- zeitreihe[!is.na(zeitreihe)]
  
  # AUC where the value is below the threshold
  differenzen <- ifelse(zeitreihe < threshold, threshold - zeitreihe, 0)
  
  sum(differenzen)
}

# Function to generate a dataframe with the calculated AUC values
berechne_map_daten_auc <- function(map_daten, datensatz_name) {
  map_daten_auc <- data.frame(matrix(ncol = length(thresholds), nrow = nrow(map_daten)))
  
  # Column names based on thresholds and dataset name
  colnames(map_daten_auc) <- paste0("AUC of MAP under ", thresholds, ": ", datensatz_name)
  
  # Calculate AUC for each threshold and time series
  for (i in seq_along(thresholds)) {
    map_daten_auc[, i] <- apply(map_daten[, 0:719], 1, berechne_auc, threshold = thresholds[i])
  }
  
  return(map_daten_auc)
}

# Compute AUC for each dataframe and append to results
for (i in seq_along(dataframes)) {
  map_daten_auc <- berechne_map_daten_auc(dataframes[[i]], names[i])
  map_daten_analyse <- cbind(map_daten_analyse, map_daten_auc)
}

# Define thresholds
schwellenwerte <- c(75, 70, 65, 60, 55, 50)

# Function to calculate AUC for a given threshold
berechne_auc <- function(zeitreihe, threshold) {
  # Restrict time series to values before the first NA
  zeitreihe <- zeitreihe[!is.na(zeitreihe)]
  
  # AUC where the value is below the threshold
  differenzen <- ifelse(zeitreihe < threshold, threshold - zeitreihe, 0)
  
  sum(differenzen)
}

# Function to calculate TWA for a dataframe
berechne_twa <- function(map_daten, datensatz_name) {
  # Calculate AUC for each threshold
  map_daten_auc <- sapply(schwellenwerte, function(threshold) apply(map_daten, 1, berechne_auc, threshold = threshold))
  
  # Compute TWA by dividing AUC by the number of non-NA values in each row
  map_daten_twa <- sweep(map_daten_auc, 1, rowSums(!is.na(map_daten)), "/")
  
  # Round TWA values to two decimals
  map_daten_twa <- round(map_daten_twa, 2)
  
  # Set column names
  colnames(map_daten_twa) <- paste0("TWA of MAP under ", schwellenwerte, ": ", datensatz_name)
  
  return(as.data.frame(map_daten_twa))
}

# Compute TWA for each dataframe and append to results
for (i in seq_along(dataframes)) {
  map_daten_twa <- berechne_twa(dataframes[[i]], names[i])
  map_daten_analyse <- cbind(map_daten_analyse, map_daten_twa)
}

# Compute new time series feature statistics following 10.1001/jamanetworkopen.2021.2240 ####
# Implemented by Jonas Leins

# Function to calculate entropy
entropy <- function(time_series) {
  time_series <- time_series[!is.na(time_series)]  # Remove NA values
  individual_values <- unique(time_series)         # Unique values in the time series
  N <- length(time_series)                         # Length of the time series
  p_s <- 0
  
  for (x in individual_values) {
    n_x <- length(time_series[time_series == x])   # Count of value x
    p_x <- n_x / N                                 # Probability of observing x
    p_s <- p_s + p_x * log(p_x)                    # Sum of probabilities multiplied by their logarithm
  }
  
  ntrop <- -p_s                                     # Negative sum for entropy
  scaled_ntrop <- ntrop / log(length(individual_values))  # Scaled entropy
  return(scaled_ntrop)
}

# Function to compute statistical features
calculate_features <- function(time_series) {
  time_series <- time_series[!is.na(time_series)]    # Remove NA values
  mean_val <- mean(time_series, na.rm = TRUE)        # Mean
  std_val <- sd(time_series, na.rm = TRUE)           # Standard deviation
  entropy_val <- entropy(time_series)                # Entropy
  trend_val <- if (length(time_series) >= 3) mk.test(time_series)$estimates[["tau"]] else NA  # Trend only if ≥ 3 points
  kurtosis_val <- kurtosis(time_series, na.rm = TRUE)  # Kurtosis
  skewness_val <- skewness(time_series, na.rm = TRUE)  # Skewness
  
  return(c(mean_val, std_val, entropy_val, trend_val, kurtosis_val, skewness_val))
}

# Compute features for each dataframe and append to the existing dataframe
for (i in seq_along(dataframes)) {
  features <- t(apply(dataframes[[i]], 1, calculate_features))
  colnames(features) <- c(
    paste0("Mean_", names[i]),
    paste0("STD_", names[i]),
    paste0("Entropy_", names[i]),
    paste0("Trend_", names[i]),
    paste0("Kurtosis_", names[i]),
    paste0("Skewness_", names[i])
  )
  map_daten_analyse <- cbind(map_daten_analyse, features)
}

# Count number of rows in map_daten_analyse containing NA values
num_rows_with_na <- sum(apply(map_daten_analyse, 1, function(row) any(is.na(row))))

# Output number of rows with NA values
print(num_rows_with_na)

# Count number of columns containing NA values
na_columns <- colnames(map_daten_analyse)[apply(map_daten_analyse, 2, function(col) any(is.na(col)))]
num_cols_with_na <- length(na_columns)

# Output number of columns with NA values
print(num_cols_with_na)

# Output column names containing NA values
print(na_columns)  # 28 columns; likely occurs when intervals are too short (some inductions only last 1 minute)

print(sort(map_daten_analyse$einlDauer)[1:10])
print(sort(map_daten_analyse$opDauer)[1:10])


### Calculation of % features relative to a baseline MAP ####
# Problem: we do not have a true baseline MAP, only the first MAP value at induction.
# Previous studies showed no meaningful difference between absolute and percentage-based thresholds.

# Determining a baseline MAP is difficult because the MAP time series begins only at anaesthesia start.
# Therefore, the first MAP value is used as the pre-induction baseline MAP.

# The value is the first non-NA element of each row in map_daten
map_daten_analyse$Baseline_MAP <- map_daten[, 1]
daten$Baseline_MAP <- map_daten[, 1]

baseline_MAP <- map_daten_analyse$Baseline_MAP

# lowest % MAP decrease from baseline (%) for cumulative minutes (1, 3, 5, 10)

# Function to calculate the percentage deviation from baseline
berechne_prozent_abweichung <- function(baseline_MAP, niedrigster_MAP) {
  if (!is.na(baseline_MAP) && baseline_MAP != 0) {
    return(round((1 - niedrigster_MAP / baseline_MAP) * 100, 2))
  } else {
    return(NA)
  }
}

# Find all column names starting with "Lowest MAP for cumulative"
spaltennamen <- grep("Lowest MAP for cumulative", names(map_daten_analyse), value = TRUE)

# Compute percentage deviations for each relevant column
for (spaltenname in spaltennamen) {
  niedrigster_MAP <- map_daten_analyse[, spaltenname]
  
  # Compute percentage deviation and append to dataframe
  prozent_abweichung_spaltenname <- paste0(spaltenname, "- % decrease from baseline")
  map_daten_analyse[, prozent_abweichung_spaltenname] <- mapply(berechne_prozent_abweichung, baseline_MAP, niedrigster_MAP)
}

## lowest % MAP decrease from baseline (%) for sustained minutes (1, 3, 5, 10)

baseline_MAP <- map_daten_analyse$Baseline_MAP

# Function to compute percentage deviation (same as above)
berechne_prozent_abweichung <- function(baseline_MAP, niedrigster_MAP) {
  if (!is.na(baseline_MAP) && baseline_MAP != 0) {
    return(round((1 - niedrigster_MAP / baseline_MAP) * 100, 2))
  } else {
    return(NA)
  }
}

# All column names starting with "Lowest MAP for sustained"
spaltennamen <- grep("Lowest MAP for sustained", names(map_daten_analyse), value = TRUE)

# Compute percentage deviations
for (spaltenname in spaltennamen) {
  niedrigster_MAP <- map_daten_analyse[, spaltenname]
  
  prozent_abweichung_spaltenname <- paste0(spaltenname, " - % decrease from baseline")
  map_daten_analyse[, prozent_abweichung_spaltenname] <- mapply(berechne_prozent_abweichung, baseline_MAP, niedrigster_MAP)
}

# minutes of % MAP decrease under relative threshold <20, <30, <40, <50

# Define thresholds for the percentage drop
schwellen_prozent_abfall <- c(20, 30, 40, 50)

# Extract baseline MAP
baseline_MAP <- map_daten_analyse$Baseline_MAP

# Function to compute minutes below a percentage threshold
berechne_minuten_unter_schwelle <- function(map_daten, baseline_MAP, schwellen_prozent) {
  prozent_schwelle <- baseline_MAP * (1 - schwellen_prozent / 100)
  return(rowSums(map_daten < prozent_schwelle, na.rm = TRUE))
}

# Function to compute and store results for a dataframe
berechne_minuten_unter_schwelle_prozent_df <- function(map_daten, baseline_MAP, schwellen_prozent_abfall, datensatz_name) {
  minutes_under_prozent_threshold <- sapply(
    schwellen_prozent_abfall,
    function(schwellen_prozent) berechne_minuten_unter_schwelle(map_daten, baseline_MAP, schwellen_prozent)
  )
  
  df <- as.data.frame(minutes_under_prozent_threshold)
  colnames(df) <- paste0("Minutes of % MAP decrease under relative threshold: ",
                         schwellen_prozent_abfall, " (", datensatz_name, ")")
  return(df)
}

# Compute results for each dataframe and append to the main analysis dataframe
for (i in seq_along(dataframes)) {
  map_daten_minuten_unter_schwelle_prozent <- berechne_minuten_unter_schwelle_prozent_df(
    dataframes[[i]], baseline_MAP, schwellen_prozent_abfall, names[i]
  )
  map_daten_analyse <- cbind(map_daten_analyse, map_daten_minuten_unter_schwelle_prozent)
}

# AUC and TWA of % MAP decrease from baseline <10, <15, <20, <25, <30

# Define percentage thresholds
schwellen_prozent <- c(10, 15, 20, 25, 30)

# Function to compute threshold values for each row in map_daten
berechne_schwellenwerte <- function(baseline_MAP, schwellen_prozent) {
  schwellenwerte_df <- sapply(schwellen_prozent, function(prozent) {
    baseline_MAP * (1 - prozent / 100)
  })
  return(as.data.frame(schwellenwerte_df))
}

# Function to compute AUC and TWA for each row and each threshold
berechne_auc_und_twa <- function(map_daten, schwellenwerte, schwellen_prozent) {
  n <- nrow(map_daten)
  m <- length(schwellen_prozent)
  auc_matrix <- matrix(0, nrow = n, ncol = m)
  twa_matrix <- matrix(0, nrow = n, ncol = m)
  
  for (i in 1:n) {
    zeitreihe <- as.numeric(map_daten[i, ])
    zeitreihe <- zeitreihe[!is.na(zeitreihe)]  # Ignore NA values
    anzahl_werte <- length(zeitreihe)
    
    for (j in 1:m) {
      schwellenwert <- schwellenwerte[i, j]
      auc_matrix[i, j] <- sum(pmax(schwellenwert - zeitreihe[zeitreihe < schwellenwert], 0))
      twa_matrix[i, j] <- auc_matrix[i, j] / anzahl_werte
    }
  }
  
  list(auc = as.data.frame(auc_matrix), twa = as.data.frame(twa_matrix))
}

# List of dataframes to analyse and their names
dataframes <- list(
  map_daten_Narkose, map_daten_einl, map_daten_einlplus30, map_daten_einlplus60,
  map_daten_einlplus90, map_daten_einlplus120, map_daten_einlplus150, map_daten_OP
)
names <- c("Narkose", "einl", "einlplus30", "einlplus60", "einlplus90", "einlplus120", "einlplus150", "OP")

# Perform calculations for each dataframe and append results
baseline_MAP <- map_daten_analyse$Baseline_MAP

for (k in seq_along(dataframes)) {
  datensatz_name <- names[k]
  map_daten <- dataframes[[k]]
  
  schwellenwerte_df <- berechne_schwellenwerte(baseline_MAP, schwellen_prozent)
  
  ergebnisse <- berechne_auc_und_twa(map_daten, schwellenwerte_df, schwellen_prozent)
  auc_df <- ergebnisse$auc
  twa_df <- ergebnisse$twa
  
  colnames(auc_df) <- paste0("AUC of % MAP decrease from baseline <", schwellen_prozent, "%: ", datensatz_name)
  colnames(twa_df) <- paste0("TWA of % MAP decrease from baseline <", schwellen_prozent, "%: ", datensatz_name)
  
  map_daten_analyse <- cbind(map_daten_analyse, auc_df, twa_df)
}

map_daten_analyse <- round(map_daten_analyse, 2)

##### Write output files ####
write.csv(map_daten, file = "/your/path/MAP_Daten.csv", row.names = TRUE)
write.csv(map_daten_analyse, file = "/your/path/MAP_Features.csv", row.names = TRUE)
write.csv(daten, file = "/your/path/Datensatz_250126_MAPcleaned.csv", row.names = TRUE)
