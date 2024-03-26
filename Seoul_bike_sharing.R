#************************************************
#Analysis on Rented Bikes in Seoul
#************************************************

#Version: 1.0    Initial Version
#Version: 1.1    Last updated 12-11-2023
#Version: 1.2    Last updated 29-11-2023
#Version: 1.3    Last updated 10-12-2023
#Version: 1.4    Last updated 13-12-2023
#Version: 1.5    Last updated 17-12-2023
#Clear global environment
rm(list=ls())
#
#
#Global Environment variables - i.e. available to all functions
DATASET_NAME      <- "SeoulBikeData.csv"  #Name of the dataset to load
TARGET_COLUMN     <- "Rented_Bike_Count"  #Name of the target column
UNIQUE_NORM_ORD   <- 31                   #How may unique values is the treshold to encode an Ordinal value
THRESHOLD_ORD     <-  8                   #The maximum number of bins allowed for a column to be considered ORDINAL
TRAIN_SIZE        <- 0.8                  #Size of the training set(MUST be between 0 and 1), we decided to go with 0.8, so 80%
num_folds         <- 4                    #Number of folds for the K-fold Cross validation

#OTHER VARIABLES GO HERE
#
#
#Libraries used in this project
#Library version checked with function 'packageVersion("name_of_the_library")'
#===============================================================================

#Name of CRAN Library     Version
#   pacman                0.5.1
#   dplyr                 1.1.3
#   ggplot2               3.4.4
#   corrplot              0.92
#   outliers              0.15
#   naniar                1.0.0
#   caret                 6.0.94
#   rpart                 4.1.21
#   lubridate             1.9.3
#   formattable           0.2.1
#   randomForest          4.7.1.1
#   rnn                   1.9.0
#   h2o                   3.42.0.2
#   gridExtra             2.3
#   tidyverse             2.0.0
#   hrbrthemes            0.8.0

MYLIBRARIES<-c("dplyr",
               "ggplot2",
               "corrplot",
               "outliers",
               "naniar",
               "caret",
               "rpart",
               "lubridate",
               "formattable",
               "randomForest",
               "rnn",
               "h2o",
               "gridExtra",
               "tidyverse",
               "hrbrthemes"
)

# User defined functions are next
#################################################################################################################################################
##################################################################################################################################################
#==============================================================
#Function Name: check_missing
#Description: Checks and handles missing values in the dataset.
#
# variables inside : missing_values
#
#Input: data frame - datasett
#Output: data frame - dataset with missing values handled
#==============================================================
check_missing <- function(dataset) {
  #Check on the dataset in input
  if (!is.dataframe(dataset)) {
    stop("Input must be a dataset")
  }
  
  # Check for missing values in each column and store their indices
  missing_values <- sapply(dataset, function(x) which(is.na(x)))
  
  # Handle missing values if found
  if (any(sapply(missing_values, length) > 0)) {
    message("Missing values detected.")
    
    #Mean value insertion
    for (col in names(missing_values)) {
      if (length(missing_values[[col]]) > 0) {
        #Get mean without na values
        mean_val <- mean(dataset[[col]], na.rm = TRUE)
        
        #Replace with mean
        dataset[[col]][missing_values[[col]]] <- mean_val
      }
    }
  } else {
    message("0 missing values detected in the dataset")
  }
  
  return(dataset)
}


#==============================================================
#Function Name: remove_duplicates
#Description: Handles duplicated values in the dataset.
#
#
#Input: dataset
#Output: dataset with duplicated values handled
#==============================================================
remove_duplicates <- function(dataset) {
  unique_data <- dataset[!duplicated(dataset), ]
  return(unique_data)
}

#==============================================================
#Function Name: plot_outliers
#Description: Plots the column as boxplots to identify outliers
#
#Function inspired by the tutorials on https://r-graph-gallery.com
#Source: https://r-graph-gallery.com/262-basic-boxplot-with-ggplot2.html
#
#Input: dataset
#Output: Nothing
#==============================================================
plot_outliers <- function(dataset) {
  if (is.data.frame(dataset)) {
    #Only consider numeric columns
    numeric_columns <- sapply(dataset, is.numeric)
    
    if (any(numeric_columns)) {
      for (col_name in names(dataset)[numeric_columns]) {
        #create a boxplot for the current column
        boxplot_graph <- ggplot(dataset, aes(x = factor(1), y = .data[[col_name]])) +
          geom_boxplot(fill = "slateblue", alpha = 0.3) +
          labs(title = paste("Boxplot of", col_name), x = "", y = col_name) +
          theme_minimal()
        #print the boxplot
        print(boxplot_graph)
      }
      #Added some error handling to increase function's robustness
    } else {
      message("There are no numeric columns into the dataset")
    }
  } else {
    stop("No dataset detected")
  }
}


#==============================================================
#Function Name: remove_outliers
#Description: removes the rows with outliers (it's a drastic approach, we are not using it)
#This is only one way of handling outliers, it's left there to show that we can handle them.
#
#
#Input: dataset
#Output: Nothing
#==============================================================
remove_outliers <- function(data, lowerBound = 0.05, upperBound = 0.95) {
  cat("Removing outliers from the data \n")
  #determine what we consider an outlier
  cat("Defining outliers as values outside the", lowerBound * 100, "% to", upperBound * 100, "% range \n")
  #calculate the actual values at these percentiles
  lower_bound <- quantile(data, lowerBound, na.rm = TRUE)
  upper_bound <- quantile(data, upperBound, na.rm = TRUE)
  cat("All the values below", lower_bound, "and above", upper_bound, "will be considered outliers\n")
  
  #Remove the outliers
  no_outlier <- data[data > lower_bound & data < upper_bound]
  cat("Outliers removed \n")
  return(no_outlier)
}


#=========================================================================
#Function Name: plot_data
#Description: Plots 2 variables in a 2D Plot, can set plot type and line type.
#
#
#Input: dataset, column_x, column_y, plot_type, line_type
#Output: 2D Plot
#=======================================================================
plot_data <- function(nome_dataset, column_x, column_y, plot_type = "line", line_type = "lm") {
  if (!(column_x %in% names(nome_dataset))) {
    stop(paste("The value", column_x, "is not present in the dataset."))
  }
  
  if (!(column_y %in% names(nome_dataset))) {
    stop(paste("The value", column_y, "is not present in the dataset."))
  }
  
  ######################################################################################################ADD COMMENT TO EXPLAIN FUNCTION
  method_color <- switch(
    line_type,
    "lm" = "blue",
    "loess" = "green",
    "default_color"
  )
  
  if (method_color == "default_color" && plot_type %in% c("line", "smooth")) {
    stop("Error! Use either 'lm' or 'loess'")
  }
  
  gg <- ggplot(nome_dataset, aes(x = !!sym(column_x), y = !!sym(column_y)))
  
  if (plot_type == "line" || plot_type == "smooth") {
    gg <- gg +
      geom_smooth (method = line_type, se=FALSE, color=method_color)+
      geom_point()
    
  } else if (plot_type == "bar") {
    gg <- gg +
      geom_bar(stat = "identity", fill = "blue")
  } else {
    stop("Error! Plot type can be 'line' or 'bar' ")
  }
  
  gg +
    labs(
      x = column_x,
      y = column_y,
      title = paste("Plot of", column_y, "vs", column_x)
    ) +
    #This section of code is inspired by the tutorials on https://r-graph-gallery.com/histogram.html
    theme_minimal() +
    theme(
      plot.title = element_text(size = 16, face ="bold"),
      axis.title = element_text(size = 12),
      axis.text = element_text(size = 10),
      legend.title = element_text(size = 12),
      legend.text = element_text(size = 10)
    ) +
    scale_y_continuous(labels = scales::number_format(accuracy = 1))
}


#==============================================================
#Function Name: check_unique_values
#Description: Checks and prints unique values of a column.
#
#
#Input: dataset, columnName
#Output: Prints dataframe with unique value of the columns and their count
#==============================================================
check_unique_values<- function(dataset, columnName) {
  #We check if the column exixts first
  if(!columnName %in% names(dataset)) {
    stop("Column not found in the dataset")
  }
  
  #Count unique values
  valueCounts <- table(dataset[[columnName]])
  #Dataframe conversion
  countsDataFrame <- as.data.frame(valueCounts)
  #Renaming the display columns
  colnames(countsDataFrame) <- c("Value", "Count")
  print(countsDataFrame)
}


#==============================================================
#Function Name: determine_column_type
#Description: Sets column types, Numeric, Categorical or Other.
#
#
#Input: dataset
#Output: Vector called "columnTypes"
#==============================================================
determine_column_type <-function(dataFrame) {
  #Check to see if the input is a dataset
  if (!is.data.frame(dataFrame)) {
    stop("Error! The provided input is not a dataframe.")
  }
  
  #Vector to store the data types
  columnTypes <- rep(NA, ncol(dataFrame))
  
  #For each column into the dataset
  for (colIdx in seq_len(ncol(dataFrame))) {
    #Type assignement
    if (is.numeric(dataFrame[[colIdx]])) {
      columnTypes[colIdx] <- 'Numeric'
    } else if (is.character(dataFrame[[colIdx]])) {
      columnTypes[colIdx] <- 'Categorical'
    } else {
      columnTypes[colIdx] <- 'Other'
    }
  }
  
  return(columnTypes)
}

#==============================================================
#Function Name: normalizeDiscreteColumn
#Description: Normalizes discrete columns.
#
#
#Input: dataset, columnTypes
#Output: dataset normalized
#==============================================================

normalizeDiscreteColumn <- function(data, colIndex) {
  column <- data[[colIndex]]
  #We apply the min-max normalization for discrete numerical values
  min <- min(column)
  max <- max(column)
  data[[colIndex]] <- (column - min) / (max - min)
  return(data)
}


#==============================================================
#Function Name: normalizeOrdinalColumn
#Description: Normalizes ordinal columns.
#
#
#Input: dataset, columnTypes
#Output: dataset normalized
#==============================================================
normalizeOrdinalColumn <- function(data, colIndex) {
  column <- data[[colIndex]]
  
  #We apply the rank normalization
  data[[colIndex]] <- rank(column) / sum(!is.na(column))
  return(data)
}



#==================================================================
#Function Name: classifyAndNormalizeNumericFields
#Description: Classifies and normalizes all the numerical fields
#
#Input: dataset, columnTypes, treshold
#Output: dataset normalized
#==================================================================
#Impostare treshold come variabile globale
classifyAndNormalizeNumericFields <- function(data, colTypes, threshold = THRESHOLD_ORD) {
  NUMERIC <- "Numeric"
  DISCRETE <- "Discrete"
  ORDINAL <- "Ordinal"
  
  #We determine each field as Discrete or Ordinal
  for (colIndex in seq_along(colTypes)) {
    if (colTypes[colIndex] == NUMERIC) {
      distributionAnalysis <- hist(data[[colIndex]], breaks = 10, plot = FALSE)
      lowPopBins <- distributionAnalysis$counts / nrow(data) * 100
      
      colTypes[colIndex] <- if (sum(lowPopBins < 1) > threshold) DISCRETE else ORDINAL
    }
  }
  
  #Copy of the dataset to work on
  modifiedData <- data
  
  #Normalization skipping 'Rented_Bike_Count'
  for (colIndex in seq_along(colTypes)) {
    columnName <- names(modifiedData)[colIndex]
    
    if (columnName != TARGET_COLUMN) {
      print(columnName)
      if (colTypes[colIndex] == DISCRETE) {
        modifiedData <- normalizeDiscreteColumn(modifiedData, colIndex)
      } else if (colTypes[colIndex] == ORDINAL) {
        #Check the unique values to be 31
        #31 because we have 31 differen values for the day
        #and we do not want to encode the day since later we need it for Feature Transformations
        if (length(unique(modifiedData[[colIndex]])) > UNIQUE_NORM_ORD) {
          modifiedData <- normalizeOrdinalColumn(modifiedData, colIndex)
        }
      }
    }
  }
  
  return(modifiedData)
}


#==============================================================
#Function Name: encodeCategoricalColumns
#Description: Encodes categorical columns.
#
#
#Input: dataset, columnTypes
#Output: dataset encoded
#==============================================================
encodeCategoricalColumns <- function(data, colTypes) {
  CATEGORICAL <- "Categorical"
  #Check colTypes lenght is correct
  if (length(colTypes) != ncol(data)) {
    cat("Error! Wrong lenght of colTypes vector \n")
    return(NULL)
  }
  
  for (colName in names(data)[colTypes == CATEGORICAL]) {
    data[[colName]] <- as.factor(data[[colName]])
    #This is a simpler version of one-hot encoding
    #We were running in troubles using the model.Matrix() functions like in the labs
    #so the function assigns integers to all factors in the column
    #Since the index of factors starts from 1, we subtracts 1 from all the index values.
    data[[colName]] <- as.integer(data[[colName]]) - 1
  }
  
  return(data)
}

#==============================================================
#Function Name: plot_distributions_in_batches
#Description: Plots distributions in batches
#
#Inspired by the ggplot2 r-graph gallery tutorials
#Source:  https://r-graph-gallery.com/223-faceting-with-ggplot2.html
#
#Input: dataset, batch_size
#Output: Distribution plots
#==============================================================
plot_distributions_in_batches <- function(dataset, batch_size = 2) {
  #Selection of numeric columns and their names
  numeric_data <- dataset %>% select_if(is.numeric)
  numeric_columns <- names(numeric_data)
  #Calculate the number of batches needed
  #We use batches to perform a readable print, since 14 plots would have been difficult to analyze
  num_features <- length(numeric_columns)
  num_batches <- ceiling(num_features / batch_size)
  
  #For each batch
  for (batch_number in 1:num_batches) {
    # Calculating indexes for the current batch
    start_index <- (batch_number - 1) * batch_size + 1
    end_index <- min(batch_number * batch_size, num_features)
    
    #Current plot 
    plots <- lapply(numeric_columns[start_index:end_index], function(column) {
      #Calculae also mean and median
      mean_value <- mean(dataset[[column]])
      median_value <- median(dataset[[column]])
      
      ggplot(numeric_data, aes_string(x = column)) +
        geom_histogram(aes(y = ..density..), bins = 100, fill = "blue", color = "blue") +
        geom_density(color = "red", size = 1) +
        geom_vline(xintercept = mean_value, color = "green", linetype = "dashed", size = 1.5) +
        geom_vline(xintercept = median_value, color = "yellow", linetype = "dashed", size = 1.5) +
        theme_minimal() +
        ggtitle(paste("Distribution of", column))
    })
    
    #Check if plots are present
    if (length(plots) > 0) {
      #Print the batches execution on console
      print(paste("Batch", batch_number, "of", num_batches))
      gridExtra::grid.arrange(grobs = plots, ncol = 2)
    }
  }
}


#==============================================================
#Function Name: plot_correlation_matrix
#Description: plots the correlation matrix.
#
#
#Input: dataset
#Output: print Correlation matrix
#==============================================================
plot_correlation_matrix <- function(dataset) {
  if(!is.data.frame(dataset)) {
    stop("The input must be a dataset")
  }
  
  #Calculate correlation matrix
  correlation_matrix <- cor(dataset, use = "complete.obs")  #Suggestion from: https://stackoverflow.com/questions/18892051/complete-obs-of-cor-function 
  #Cyan for minimun correlation values, orange for maximum correation values
  col <- colorRampPalette(c("cyan", "orange"))(200)#200 are the shades between cyan and orange
  #or the levels of colour
  
  #Correlation matrix plotting
  corrplot(correlation_matrix, method = "shade", 
           number.cex = 0.7, addCoef.col = "black", 
           col = col, 
           tl.srt = 35,
           tl.col = "black")
}

#==============================================================
#Function Name: split_dataset
#Description: splits the dataset in input into two datasets: train and test.
#
#
#Input: dataset, TRAIN_SIZE
#Output: list(train_set, test_set)
#==============================================================
split_dataset <- function(data, TRAIN_SIZE) {
  #Size calculation
  index <- sample(seq_len(nrow(data)), size = floor(nrow(data) * TRAIN_SIZE))
  #Data split
  train_set <- data[index, ]
  test_set <- data[-index, ]
  
  return(list(train = train_set, test = test_set))
}

#==============================================================
#Function Name: split_datasets_list
#Description: applies the split_dataset function by returning a split_data object
#
#
#Input: dataset, TRAIN_SIZE
#Output: split_data
#==============================================================
split_datasets_list <- function(dataset, train_size) {
  split_data <- split_dataset(dataset, train_size)
  return(split_data)
}


#==============================================================
#Function Name: evaluate_model
#Description: function to evaluate the machine learning model.
#
#
#Input: model, test, TARGET_COLUMN
#Output: Print the metrics and returns metric values
#==============================================================
evaluate_model <- function(model, test_set, target) {
  #Make predictions
  predictions <- predict(model, test_set)
  predicted_values <- predict(model, test_set)
  
  #actual values
  actuals <- test_set[[target]]
  #MSE calculation
  mse <- mean((actuals - predictions)^2)
  #calculate RMSE
  rmse <- sqrt(mse)
  #calculate MAE
  mae <- mean(abs(actuals - predictions))
  #calculate R-squared (or R2)
  ss_total <- sum((actuals - mean(actuals))^2)
  ss_residuals <- sum((actuals - predictions)^2)
  r_squared <- 1 - (ss_residuals / ss_total)
  
  #Printing metrics
  cat("MSE:", mse, "\n")
  cat("RMSE:", rmse, "\n")
  cat("MAE:", mae, "\n")
  cat("R-squared:", r_squared, "\n")
  
  return(list(MSE = mse, RMSE = rmse, MAE = mae, R_squared = r_squared, actuals = actuals, predictions = predictions))
}


#==============================================================
#Function Name: plot_actual_vs_predicted
#Description: plot the actual and the predicted values in the same plot.
#
#
#Input: actual values, predicted values
#Output: Plot 150 values, 600 to 750 of actual vs predicted.
#==============================================================
plot_actual_vs_predicted <- function(actual, predicted) {
  data <- data.frame(Index = 1:length(actual), Actual = actual, Predicted = predicted)
  #select rows from 600 to 800
  data_subset <- data %>% slice(600:750)
  
  #Create the plot
  ggplot(data_subset, aes(x = Index)) +
    geom_line(aes(y = Actual, color = "Actual")) +
    geom_line(aes(y = Predicted, color = "Predicted")) +
    labs(title = "Actual vs Predicted", 
         x = "Index", 
         y = "Value") +
    scale_color_manual(values = c("Actual" = "blue", "Predicted" = "red")) +
    theme_minimal()
}

##############################################################################################################################
#Start of the main Function
main<-function(){
  
  #Dataset Loading
  original_SeoulDataset <- read.csv(DATASET_NAME,check.names = F)
  NonClean_dataset <- original_SeoulDataset
  
  
  #==================================================
  #       SECTION 1 - DATA PREPARATION 
  #==================================================
  
  #**************************************************
  #       SUB-SECTION 1.1 - Data Cleaning
  #**************************************************
  
  #Set new names for the columns
  #This is done to avoid potential problems from special characters included
  #Into the column names, like "Temperature C°"
  NonClean_dataset <- setNames(NonClean_dataset, c("Date","Rented_Bike_Count","Hour","Temperature","Humidity",
                                                   "Wind_Speed","Visibility","Dew_Point_Temperature","Solar_Radiation",
                                                   "Rainfall","Snowfall","Seasons","Holiday","Functioning_Day"))
  
  #We set Rented_Bike_Count as the target variable
  
  #Display the first 5 Rows of the dataset
  head(NonClean_dataset, n=5)
  
  
  #DATA DICTIONARY
  #********************************************************************************************
  # Column                  | Explanation of the content of the Column
  
  # Date                      Date in format gg/mm/YYYY.
  # Rented_Bike_Count         Total number of rented bikes in that hour
  # Hour                      Hour of the day.
  # Temperature               Weather Temperature in Celsius degrees.
  # Humidity                  Humidity measured in percentage.
  # Wind_Speed                Wind speed in meters/second.
  # Visibility                Atmospheric Visibility within 10 meters.
  # Dew_Point_Temperature     Dew Point Temperature in Celsius degrees.
  # Solar_Radiation           Solar Radiations measured in MJ/m2.
  # Rainfall                  Rainfall in millimeters.
  # Snowfall                  Snowfall in centimeters.
  # Seasons                   Season names.
  # Holiday                   If the day is an Holiday or not.
  # Functioning_Day           If the bike station is functioning or not.
  #*********************************************************************************************
  
  #Count the number of missing values into our dataset.
  missing_values <- sum(is.na(NonClean_dataset))
  cat("Number of missing values into the dataset: ", missing_values, "\n")
  #Since there are no missing values, we can avoid dealing with missing values
  #Hence we do not use our function: check_missing
  
  #Count the numbers of duplicated rows into the analysed dataset
  #We should have 0 duplication since we are using a dataset dealing
  #with Dates and Times, so each Datetime only occours once.
  cat("Duplicate entry in data:", sum(duplicated(NonClean_dataset)), "\n")
  #Since there are no duplicated rows, we can avoid dropping rows 
  #to remove duplicated values
  #Hence we do not use our function remove_duplicates
  
  
  #Print the shape of the dataset
  cat("Total number of Rows:", nrow(NonClean_dataset), "\n")
  cat("Total number of Features:", ncol(NonClean_dataset), "\n")
  
  #Checking for Outliers
  library(ggplot2)
  plot_outliers(NonClean_dataset)
  #We decided to keep the outliers since we are working with real-world data
  #We want our models to be able to learn even from extreme cases
  #In order to be prepared even for these occasions and generalise better
  #Hence we do not use the function "remove_outliers"
  #The function remove_outliers is purposefully inefficient to deal with this kind of outliers
  #We just made it to show one way of handling the outliers
  
  clean_dataset <- NonClean_dataset
  
  
  #*************************************************
  #   SUB-SECTION 1.2 - Data Transformation
  #**************************************************
  
  #Converting Date as a Datetime object to split it into new columns
  clean_dataset$Date <- as.Date(clean_dataset$Date, format = "%d/%m/%Y")
  clean_dataset$Day <- as.integer(format(clean_dataset$Date, "%d"))
  clean_dataset$Month <- as.integer(format(clean_dataset$Date, "%m"))
  clean_dataset$Year <- as.integer(format(clean_dataset$Date, "%Y"))

  #We do this here because it is simpler to encode afterwards
  clean_dataset$Day_of_week <- weekdays(clean_dataset$Date)
  clean_dataset$isWeekend <- ifelse(clean_dataset$Day_of_week %in% c('Saturday', 'Sunday'), "Yes", "No")
  
  
  #Drop Date column
  clean_dataset <- clean_dataset[, -1]
  head(clean_dataset, 5)
  
  #Use functions to split data types: Numerical and Categorical
  fieldTypes <- determine_column_type(clean_dataset)
  print(fieldTypes)
  
  #Normalization
  #everything except the target variable
  normalized_dataset <- classifyAndNormalizeNumericFields(clean_dataset, fieldTypes)
  
  #encoding categorical in a new dataset 
  #we need the old one 
  normalized_dataset_old <- normalized_dataset
  
  encoded_dataset <- encodeCategoricalColumns(normalized_dataset, fieldTypes)
  
  check_unique_values(encoded_dataset, "Month")
  
  #since data transform will be used again with log transform , we will be seeing it again
  
  
  #==================================================
  #       SECTION 2 - EXPLORATORY DATA ANALYSIS (EDA)
  #==================================================
  
  #**************************************************
  #   SUB-SECTION 2.1 - Insights and Visualization
  #**************************************************
  
  
  library(ggplot2) #leave the function call here
  #Plotting using my function
  plot_data(clean_dataset, "Hour", TARGET_COLUMN, plot_type = "line", line_type = "loess")
  plot_data(clean_dataset, "Seasons", TARGET_COLUMN, plot_type = "bar")
  plot_data(clean_dataset, "Holiday", TARGET_COLUMN, plot_type = "bar")
  
  # #Pie chart with seasons
  library(dplyr)
  # Summarize the data
  summary_data <- NonClean_dataset %>%
    group_by(Seasons) %>%
    dplyr::summarize(Total_Rented_Bikes = sum(Rented_Bike_Count))
  ggplot(summary_data, aes(x="", y=Total_Rented_Bikes, fill=Seasons)) +
    geom_bar(stat="identity", width=1) +
    coord_polar("y", start=0) +
    scale_fill_manual(values=c("Winter"="blue", "Summer"="orange", "Autumn"="brown", "Spring"="green")) +
    theme_void()

  #==================================================
  #       SECTION 2 - EXPLORATORY DATA ANALYSIS (EDA) VEDAT
  #==================================================
  # install.packages("Hmisc")
  library(Hmisc)
  
  describe(clean_dataset)
  
  ###############TODO: add other plots here
  
  #**************************************************
  #       SUB-SECTION 2.2 - Statistical Analysis
  #**************************************************
  
  #Some statistical insights
  summary(encoded_dataset)
  
  
  #Plotting the numerical values' distributions  
  library(gridExtra)
  plot_distributions_in_batches(encoded_dataset)
  
  #Corr matrix
  library(corrplot)
  plot_correlation_matrix (encoded_dataset)
  
  #We will address some insights found in the correlation matrix in the Feature engineering part
  
  
  #==================================================
  #       SECTION 3 - TIME SERIES ANALYSIS 
  #==================================================
  
  #For this analysis we will use the NonClean_dataset and make a copy of it
  #This is done in order to ease the process of analysis by having access to the "Date" column
  #"TS" stays for "Time Series" and "ts" stays for the specific Time series object

  library(forecast)
  library(tseries)
  library(lubridate)
  library(zoo)
  library(scales)
  library(viridis)

  TS_dataset <- NonClean_dataset

  TS_dataset$DateTime <- as.POSIXct(paste(TS_dataset$Date, TS_dataset$Hour), format = "%d/%m/%Y %H")
  TS_dataset <- TS_dataset %>% select(-Date, -Hour) %>% arrange(DateTime)

  #Hourly Bike rental patterns
  TS_dataset$HourOfDay <- hour(TS_dataset$DateTime)
  TS_dataset_hourly_avg <- TS_dataset %>%
    group_by(HourOfDay) %>%
    summarise(Avg_Hourly_Bike_Count = mean(Rented_Bike_Count))

  #Daily patterns
  TS_dataset_daily <- TS_dataset %>%
    mutate(Date = as.Date(DateTime)) %>%
    group_by(Date) %>%
    summarise(Daily_Bike_Count = sum(Rented_Bike_Count))

  #Weekly patterns
  TS_dataset$Week <- floor_date(TS_dataset$Date, unit = "week")
  TS_dataset_weekly <- TS_dataset %>%
    group_by(Week) %>%
    summarise(Weekly_Bike_Count = sum(Rented_Bike_Count))


  # Rolling Mean Calculation (24-hour rolling mean)
  TS_dataset$Rolling_Mean <- rollmean(TS_dataset$Rented_Bike_Count, k = 24, fill = NA, align = 'center')

  # #hourly average bike rentals
  ggplot(TS_dataset_hourly_avg, aes(x = HourOfDay, y = Avg_Hourly_Bike_Count)) +
    geom_line() +
    labs(x = "Hour of Day", y = "Average Hourly Bike Count", title = "Average Bike Rentals by Hour of Day") +
    theme_minimal()

  #daily bike rental over time
  ggplot(TS_dataset_daily, aes(x = Date, y = Daily_Bike_Count)) +
    geom_line() +
    labs(x = "Date", y = "Daily Bike Count", title = "Daily Bike Rentals Over Time") +
    theme_minimal()

  #rolling mean with trend line
  ggplot(TS_dataset, aes(x = DateTime, y = Rolling_Mean)) +
    geom_line(color = "blue") +
    geom_smooth(color = "red", se = FALSE) +
    scale_x_datetime(labels = date_format("%b"), breaks = date_breaks("1 month")) +
    labs(x = "Date", y = "24-Hour Rolling Mean of Bike Rentals", title = "Trend of Bike Rentals Over Time") +
    theme_minimal(base_size = 12) +
    theme(axis.text.x = element_text(angle = 45, hjust = 1), plot.title = element_text(hjust = 0.5))



  #Testing stationarity with the Dickey-Fuller test
  #converting the TS_dataset to a time series object
  ts_TS_dataset <- ts(TS_dataset$Rented_Bike_Count, frequency = 24*365)
  adf_test_result <- adf.test(ts_TS_dataset)
  #p-value to determine stationarity
  print(adf_test_result$p.value)
  #p-value of 0.01 indicates stationarity

  #Autocorrelation and Partial Autocorrelation Analysis
  acf(ts_TS_dataset)
  pacf(ts_TS_dataset)


  #Visualizing the trend overtime
  #simple Linear Trend Model
  TS_dataset$DateTimeNumeric <- as.numeric(as.POSIXct(TS_dataset$DateTime))
  trend_model <- lm(Rented_Bike_Count ~ DateTimeNumeric, data = TS_dataset)
  #preparing TS_dataset for trend line prediction
  TS_dataset$Date <- as.Date(TS_dataset$DateTime)
  valid_dates <- na.omit(TS_dataset$Date) #check NA values
  #use seq.Date() to create date sequences and convert to numeric
  trend_TS_dataset <- data.frame(Date = seq(from = min(valid_dates), to = max(valid_dates), by = "day"))
  trend_TS_dataset$DateTimeNumeric <- as.numeric(as.POSIXct(trend_TS_dataset$Date))

  #using the model
  trend_TS_dataset$Trend <- predict(trend_model, newdata = trend_TS_dataset)

  #Plot the results
  ggplot(TS_dataset, aes(x = DateTime, y = Rented_Bike_Count)) +
    geom_line(alpha = 0.3) +
    geom_line(data = trend_TS_dataset, aes(x = as.POSIXct(DateTimeNumeric, origin = "1970-01-01"), y = Trend), color = 'red') +
    labs(x = "Date", y = "Bike Rentals", title = "Bike Rentals with Linear Trend Line") +
    theme_minimal()

  # Creating a Heatmap of Hourly Bike Rentals
  # Source: https://r-graph-gallery.com/283-the-hourly-heatmap.html

  TS_dataset <- TS_dataset %>%
    mutate(
      Hour = hour(DateTime),
      Day = day(DateTime),
      Month = month(DateTime, label = TRUE),
      Year = year(DateTime)
    ) %>%
    filter(!is.na(Hour) & !is.na(Day) & !is.na(Month) & !is.na(Year))

  #creating an aggregate
  heatmap_TS_dataset <- TS_dataset %>%
    group_by(Year, Month, Day, Hour) %>%
    summarise(Avg_Rented_Bike_Count = mean(Rented_Bike_Count, na.rm = TRUE)) %>%
    ungroup()

  #Plotting the heatmap
  ggplot(heatmap_TS_dataset, aes(x = Day, y = Hour, fill = Avg_Rented_Bike_Count)) +
    geom_tile(color = "white", linewidth = 0.1) +
    scale_fill_viridis(name = "Avg Bike Rentals", option = "C") +
    facet_grid(Year ~ Month) +
    scale_y_continuous(trans = "reverse", breaks = unique(heatmap_TS_dataset$Hour)) +
    scale_x_continuous(breaks = c(1, 10, 20, 31)) +
    theme_minimal(base_size = 8) +
    labs(title = "Hourly Bike Rentals", x = "Day", y = "Hour") +
    theme(
      legend.position = "bottom",
      plot.title = element_text(size = 14),
      axis.text.y = element_text(size = 6),
      strip.background = element_rect(colour = "white"),
      axis.ticks = element_blank(),
      axis.text = element_text(size = 7),
      legend.title = element_text(size = 8),
      legend.text = element_text(size = 6)
    )

  #==================================================
  #       SECTION 4 - FEATURE ENGINEERING 
  #==================================================
  
  #With the time series analysis we gained a lot of insights 
  #that we are going to use to implement new features.
  
  #Create new column "Peak_hour"
  encoded_dataset$Peak_hour <- ifelse(encoded_dataset$Hour >= 7 & encoded_dataset$Hour <= 21, 1, 0)
  
  
  
  #Create new column "Hour_bins"
  #the bins represent: Morining, rush hour, mid day, evening rush hour and night
  encoded_dataset$Hour_bins <- ifelse(encoded_dataset$Hour >= 5 & encoded_dataset$Hour <= 7, 0,
                                      ifelse(encoded_dataset$Hour >= 8 & encoded_dataset$Hour <= 10, 1,
                                             ifelse(encoded_dataset$Hour >= 11 & encoded_dataset$Hour <= 16, 2,
                                                    ifelse(encoded_dataset$Hour >= 17 & encoded_dataset$Hour <= 22, 3,
                                                           ifelse(encoded_dataset$Hour >= 23 | encoded_dataset$Hour <= 4, 4, NA)))))
  
  
  #The time series analysis indicates stationarity
  
  
  #Analyze the Functioning_Day variable
  plot_data(clean_dataset, "Functioning_Day", TARGET_COLUMN, plot_type = "bar")
  check_unique_values(encoded_dataset, "Functioning_Day")
  #From this analysis we can see that when "Functioning_Day" is "No" (encoded with 0)
  #The rented_bike_counts are 0
  #We proceed to create another dataset to conduct further analysis without Functioning_Day
  
  No_FD_dataset <- encoded_dataset %>%
    filter(Functioning_Day != 0) %>%
    select(-Functioning_Day)
  
  head(No_FD_dataset)
  
  #We proceed to add new holydays to the Holyday Column (0 is Yes)
  check_unique_values(encoded_dataset, "Holiday")
  
  #Creating a new dataset with new special days
  SD_No_FD_dataset <- encoded_dataset
  # Your specific dates list in ddmmYYYY format
  special_dates <- c(
    "25122017", "07012018", "14022018","15022018", "16022018", "17022018", "01032018",
    "08032018", "01042018", "07052018", "22052018", "06062018", "13062018", "15082018", "24092018",
    "25092018", "26092018", "03102018", "09102018","31102018", "25082018", "10112018", "11112018"
  )
  #We added special dates like: Valentine's Day, Blackpink Concert 2018, Woman's Day, April Fools Day
  #And many others
  #Source for Holidays: https://www.timeanddate.com/holidays/south-korea/2018
  #Source for concert special days: https://kpoppersguide.wordpress.com/2019/02/02/2018-kpop-calendar/
  
  
  # Function to convert dates from ddmmyyyy to yyyymmdd
  convert_date <- function(date) {
    year <- substr(date, 5, 8)
    month <- substr(date, 3, 4)
    day <- substr(date, 1, 2)
    return(paste(year, month, day, sep = ""))
  }
  # Convert the specific dates to yyyymmdd format for comparison
  specific_dates_converted <- sapply(special_dates, convert_date)
  # Convert "Day", "Month", "Year" to a single date string in the new dataset
  SD_No_FD_dataset$CombinedDate <- with(SD_No_FD_dataset, paste(Year, Month, Day, sep = ""))
  # Update "Holiday" column in the new dataset
  SD_No_FD_dataset$Holiday <- ifelse(SD_No_FD_dataset$CombinedDate %in% specific_dates_converted, 0, SD_No_FD_dataset$Holiday)
  # Optionally, remove the "CombinedDate" column if it's no longer needed
  SD_No_FD_dataset$CombinedDate <- NULL
  
  check_unique_values(No_FD_dataset, "Holiday")
  check_unique_values(SD_No_FD_dataset, "Holiday")
  
  
  #Dataset without Dew Point temperature, due to its high correlation with Temperature
  #This is done to reduce the dimensionality and avoid multicollinearity issues
  library(dplyr)
  No_DEW_dataset <- select(encoded_dataset, -Dew_Point_Temperature)
  
  
  #Create a separate dataset without weakly correlated features, as shown from the Correlation Matrix
  #Dropping columns with less that 0.1 correlation (both positive and negative)
  No_Weak_dataset <- select(encoded_dataset, -Day_of_week, -Day, -Holiday, -isWeekend, -Hour_bins )
  
  #Create another datasets that combines No_DEW_dataset with No_FD_dataset
  No_FD_DEW_dataset <- select(No_FD_dataset, -Dew_Point_Temperature)
  
  # List of datasets to be split
  datasets_list <- list(encoded_dataset, No_FD_dataset, SD_No_FD_dataset, No_DEW_dataset, No_Weak_dataset, No_FD_DEW_dataset)
  
  # Split each dataset and store train and test sets
  splits <- lapply(datasets_list, function(ds) split_datasets_list(ds, TRAIN_SIZE))
  # Now you can access the train and test sets for each dataset as follows:
  # For the first dataset (encoded_dataset):
  encode_train <- splits[[1]]$train
  encode_test <- splits[[1]]$test
  
  # For the second dataset (No_FD_dataset):
  No_FD_train <- splits[[2]]$train
  No_FD_test <- splits[[2]]$test
  
  # # For the third dataset (SD_No_FD_dataset):
  SD_No_FD_train <- splits[[3]]$train
  SD_No_FD_test <- splits[[3]]$test

  # For the fourth dataset (No_DEW_dataset):
  No_DEW_train <- splits[[4]]$train
  No_DEW_test <- splits[[4]]$test

  # For the fifth dataset (No_Weak_dataset):
  No_Weak_train <- splits[[5]]$train
  No_Weak_test <- splits[[5]]$test

  # For the sixth dataset (No_FD_DEW_dataset):
  No_FD_DEW_train <- splits[[6]]$train
  No_FD_DEW_test <- splits[[6]]$test
  
  
  #==================================================
  #       SECTION 5 - MODELLING 
  #==================================================
  
  
  #******************************************************
  #       SUB-SECTION 5.1 - Model Training and Evaluation
  #******************************************************

  #Source for models
  source("MODELS_M.R")

  #Conditional Inference Tree

  MAX_CITree_results_encoded_dataset <- optimize_ci_tree(encoded_dataset, "encoded_dataset", TARGET_COLUMN)
  print(MAX_CITree_results_encoded_dataset$Performance$Rsquared)

  MAX_CITree_results_No_FD_dataset <- optimize_ci_tree(No_FD_dataset, "No_FD_dataset", TARGET_COLUMN)
  print(MAX_CITree_results_No_FD_dataset$Performance$Rsquared)

  MAX_CITree_results_SD_No_FD_dataset <- optimize_ci_tree(SD_No_FD_dataset, "SD_No_FD_dataset", TARGET_COLUMN)
  print(MAX_CITree_results_SD_No_FD_dataset$Performance$Rsquared)

  MAX_CITree_results_No_DEW_dataset <- optimize_ci_tree(No_DEW_dataset, "No_DEW_dataset", TARGET_COLUMN)
  print(MAX_CITree_results_No_DEW_dataset$Performance$Rsquared)

  MAX_CITree_results_No_Weak_dataset <- optimize_ci_tree(No_Weak_dataset, "No_Weak_dataset", TARGET_COLUMN)
  print(MAX_CITree_results_No_Weak_dataset$Performance$Rsquared)

  MAX_CITree_results_No_FD_DEW_dataset <- optimize_ci_tree(No_FD_DEW_dataset, "No_FD_DEW_dataset", TARGET_COLUMN)
  print(MAX_CITree_results_No_FD_DEW_dataset$Performance$Rsquared)


  # ##########################################################################################################################
  # 
  #XGBoost Regressor
  #a deprecated function is throwing a warning, I couldn't solve the problem but it should not
  #impact the results.

  MAX_XGBoost_results_encoded_dataset <- optimize_xgboost(encoded_dataset, "encoded_dataset", TARGET_COLUMN)
  print(MAX_XGBoost_results_encoded_dataset$Performance$Rsquared)

  MAX_XGBoost_results_No_FD_dataset <- optimize_xgboost(No_FD_dataset, "No_FD_dataset", TARGET_COLUMN)
  print(MAX_XGBoost_results_No_FD_dataset$Performance$Rsquared)

  MAX_XGBoost_results_SD_No_FD_dataset <- optimize_xgboost(SD_No_FD_dataset, "SD_No_FD_dataset", TARGET_COLUMN)
  print(MAX_XGBoost_results_SD_No_FD_dataset$Performance$Rsquared)

  MAX_XGBoost_results_No_DEW_dataset <- optimize_xgboost(No_DEW_dataset, "No_DEW_dataset", TARGET_COLUMN)
  print(MAX_XGBoost_results_No_DEW_dataset$Performance$Rsquared)

  MAX_XGBoost_results_No_Weak_dataset <- optimize_xgboost(No_Weak_dataset, "No_Weak_dataset", TARGET_COLUMN)
  print(MAX_XGBoost_results_No_Weak_dataset$Performance$Rsquared)

  MAX_XGBoost_results_No_FD_DEW_dataset <- optimize_xgboost(No_FD_DEW_dataset, "No_FD_DEW_dataset", TARGET_COLUMN)
  print(MAX_XGBoost_results_No_FD_DEW_dataset$Performance$Rsquared)


  
  print("end of main")
} #endof main()

# =================================================
# Start of R execution
#==================================================


# This section is organized following the Practical Business Analytics lab sheets guidelines
# Adapted from Prof. Alaa Marshan's code on Practical Business Analytics lab sheets
# Department of Computer Science
# University of Surrey
# GUILDFORD
# Surrey GU2 7XH

gc() # garbage collection to automatically release memory

# clear plots and other graphics
if(!is.null(dev.list())) dev.off()
graphics.off()

# Set Time Zone to Europe/London
Sys.setenv(TZ="Europe/London")

# clears the console area
cat("\014")

#Load the libraries used in this project

library(pacman)
pacman::p_load(char=MYLIBRARIES,install=TRUE,character.only=TRUE)

set.seed(123)

#Call the main function
main()

print("End of execution")