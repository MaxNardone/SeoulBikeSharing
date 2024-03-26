#This is the file with the models

set.seed(123)

#========================================
# FIRST MODEL: XGBoost   
#========================================
#This is the function to calculate the metrics (almost the same as the Xgboost one)
calculate_metrics1 <- function(actual, predicted) {
  # Apply the exponential transformation and subtract 1
  #actual_exp <- exp(actual) - 1
  #predicted_exp <- exp(predicted) - 1
  actual_exp <- actual
  predicted_exp <- predicted
  
  #calculation of the metrics
  RMSE <- sqrt(mean((predicted_exp - actual_exp)^2))
  MAE <- mean(abs(predicted_exp - actual_exp))
  MSE <- mean((predicted_exp - actual_exp)^2)
  Rsquared <- cor(actual_exp, predicted_exp)^2
  
  return(data.frame(RMSE = RMSE, MAE = MAE, MSE = MSE, Rsquared = Rsquared))
}

#this is the big function to assemble all together the Grid search, while running the Cross Validation
optimize_xgboost <- function(dataset, dataset_name, target_variable) {
  control <- trainControl(method = "cv", number = 5)
  
  #Cross validation for XGBoost
  initial_model <- train(as.formula(paste(target_variable, "~ .")), 
                         data = dataset, 
                         method = "xgbTree", 
                         trControl = control)
  initial_predictions <- predict(initial_model, dataset)
  initial_metrics <- calculate_metrics1(dataset[[target_variable]], initial_predictions)
  print("Initial model metrics:")
  print(initial_metrics)
  
  #Grid search parameters 
  grid <- expand.grid(
    eta = c(0.1, 0.15, 0.2, 0.25, 0.3),
    max_depth = c(6, 9, 10),    
    min_child_weight = c(1, 3, 4),            
    nrounds = 100,                    
    gamma = 0,                      
    colsample_bytree = 1,
    subsample = 1                         
  )
  
  
  tuned_model <- train(as.formula(paste(target_variable, "~ .")), 
                       data = dataset, 
                       method = "xgbTree", 
                       tuneGrid = grid, 
                       trControl = control)
  tuned_predictions <- predict(tuned_model, dataset)
  tuned_metrics <- calculate_metrics1(dataset[[target_variable]], tuned_predictions)
  best_hyperparameter_grid <- tuned_model$bestTune
  print("Best hyperparameter from grid search:")
  print(best_hyperparameter_grid)

  
  #The best models is selected using the R-squared metric
  if (tuned_metrics$Rsquared > initial_metrics$Rsquared) {
    final_metrics <- tuned_metrics
  } else {
    final_metrics <- initial_metrics
  }
  
  #Printing the difference in metrics
  print("Difference in metrics (Tuned - Initial):")
  print(tuned_metrics - initial_metrics)
  
  return(list("Performance" = final_metrics, "Dataset" = dataset, "DatasetName" = dataset_name))
}


#========================================
# SECOND MODEL: CONDITIONAL INFERENCE TREE
#========================================

library(party)
library(dplyr)
library(caret)

#CI Tree base function 
ci_tree_model1 <- function(train_data, target_variable) {
  formula <- as.formula(paste(target_variable, "~ ."))
  model <- ctree(formula, data = train_data)
  return(model)
}
###########################################################à

library(rBayesianOptimization)

#Function to calculate evaluation metrics
calculate_metrics <- function(actual, predicted) {
  # Apply the exponential transformation and subtract 1
  #actual_exp <- exp(actual) - 1
  #predicted_exp <- exp(predicted) - 1
  actual_exp <- actual
  predicted_exp <- predicted
  
  #Metrics calculation
  RMSE <- sqrt(mean((predicted_exp - actual_exp)^2))
  MAE <- mean(abs(predicted_exp - actual_exp))
  MSE <- mean((predicted_exp - actual_exp)^2)
  Rsquared <- cor(actual_exp, predicted_exp)^2
  
  return(data.frame(RMSE = RMSE, MAE = MAE, MSE = MSE, Rsquared = Rsquared))
}

#Grid Search + Bayesian Search + Cross Validation
optimize_ci_tree <- function(dataset, dataset_name, target_variable) {
  control <- trainControl(method = "cv", number = 5)
  
  #This is used to get the initial results to make a comparison
  initial_model <- train(as.formula(paste(target_variable, "~ .")), 
                         data = dataset, 
                         method = "ctree", 
                         trControl = control)
  initial_predictions <- predict(initial_model, dataset)
  initial_metrics <- calculate_metrics(dataset[[target_variable]], initial_predictions)
  print("Initial model metrics:")
  print(initial_metrics)
  
  #Grid search
  grid <- expand.grid(mincriterion = c(0.80, 0.85, 0.90, 0.95))
  tuned_model <- train(as.formula(paste(target_variable, "~ .")), 
                       data = dataset, 
                       method = "ctree", 
                       tuneGrid = grid, 
                       trControl = control)
  tuned_predictions <- predict(tuned_model, dataset)
  tuned_metrics <- calculate_metrics(dataset[[target_variable]], tuned_predictions)
  best_hyperparameter_grid <- tuned_model$bestTune
  print("Best hyperparameter from grid search: ")
  print(best_hyperparameter_grid)
  print("Best R-Squared from Grid search: ")
  printed_tuned_predictions <- tuned_metrics
  print(printed_tuned_predictions)
  
  #Bayesian optimization
  objective_function <- function(mincriterion) {
    model <- train(as.formula(paste(target_variable, "~ .")), 
                   data = dataset, 
                   method = "ctree", 
                   trControl = control,
                   tuneGrid = data.frame(mincriterion = mincriterion))
    score <- max(model$results$Rsquared, na.rm = TRUE)
    return(list(Score = score))
  }
  bayes_model <- BayesianOptimization(FUN = objective_function,
                                      bounds = list(mincriterion = c(0.80, 0.96)),
                                      init_points = 5, 
                                      n_iter = 20,
                                      acq = "ei", 
                                      verbose = TRUE)
  best_hyperparameter_bayes <- bayes_model$Best_Par
  print("Best hyperparameter from Bayesian search: ")
  print(best_hyperparameter_bayes)
  
  #Selecting the best model using R-squared
  if (tuned_metrics$Rsquared > initial_metrics$Rsquared) {
    final_metrics <- tuned_metrics
  } else {
    final_metrics <- initial_metrics
  }
  
  
  #Printing the differences
  print("Difference in metrics (Tuned - Initial):")
  print(tuned_metrics - initial_metrics)
  
  return(list("Performance" = final_metrics, "Dataset" = dataset, "DatasetName" = dataset_name))
}



