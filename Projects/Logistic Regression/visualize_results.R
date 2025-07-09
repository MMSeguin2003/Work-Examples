###########################
### Import Cleaned Data ###
###########################

train_data <- read.csv("Train.csv")
test_data <- read.csv("Test.csv")

########################
### Import Libraries ###
########################

library(dplyr)
library(ggplot2)
library(ROCR)

##########################
### Defining Functions ###
##########################

compute_performance_metrics <- function(actual, predicted) {
  cm <- table(Predicted = predicted, Actual = actual)
  accuracy <- sum(diag(cm)) / sum(cm)
  
  TP <- cm[2, 2]
  FN <- cm[1, 2]
  FP <- cm[2, 1]
  TN <- cm[1, 1]
  
  sensitivity <- TP/(TP + FN)
  specificity <- TN/(TN + FP)
  precision <- TP/(TP + FP)
  f1 <- 2*(precision*sensitivity)/(precision + sensitivity)
    
  return(list(
      ConfusionMatrix = cm,
      Accuracy = accuracy,
      Sensitivity = sensitivity,
      Specificity = specificity,
      Precision = precision,
      F1 = f1
  ))
}

visualize_for <- function(model, type, quiet = FALSE){
  if (model == "Rap"){
    pred_filename = "RapPredictions.csv"
    prob_filename = "RapProbabilities.csv"
    coef_filename = "RapCoefficients.csv"
  } else if (model == "Bliss") {
    pred_filename = "BlissPredictions.csv"
    prob_filename = "BlissProbabilities.csv"
    coef_filename = "BlissCoefficients.csv"
  } else {
    pred_filename = "BlendPredictions.csv"
    prob_filename = "BlendProbabilities.csv"
    coef_filename = "BlendCoefficients.csv"
  }
  
  Predictions <- read.csv(pred_filename)
  Probabilities <- read.csv(prob_filename)
  Coefficients <- read.csv(coef_filename)
  if (quiet == FALSE){
    print(Coefficients)
  }
  
  predictions <- Predictions[[paste0(model, ".Predictions")]]
  probs <- Probabilities[[paste0(model, ".Probabilities")]]
  Coefficients <- Coefficients %>%
    mutate(fill = ifelse(Coefficient < 0, "red", "blue"))
  Coefficients$Feature <- gsub("_", "\n", gsub("TimeSignature", "Time\nSignature", Coefficients$Feature))
  
  y_val <- test_data[[paste0("in_", model)]]
  
  metrics <- compute_performance_metrics(y_val, predictions)
  cm_df <- as.data.frame(metrics$ConfusionMatrix)
  pred <- prediction(probs, y_val)
  
  # Confusion Matrix
  if (type == "ConfusionMatrix"){
    cm_df %>%
      ggplot(aes(x = Actual, y = Predicted, fill = Freq)) +
      geom_tile() +
      geom_text(aes(label = Freq), color = "black", size = 5) +
      labs(title = paste0("Confusion Matrix (", model, ")"), x = "Actual", y = "Predicted", fill = "Frequency") +
      scale_fill_gradient2(low = "red", mid = "white", high = "#badb33") +
      theme_minimal() +
      theme(plot.title = element_text(hjust = 0.5))
  } else if (type == "Coefficient"){
    # Coefficient Bar Chart
    # So it displays in descnding order of magnitude
    Coefficients$Feature <- factor(
      Coefficients$Feature,
      levels = Coefficients$Feature[order(-abs(Coefficients$Coefficient))]
    )
    Coefficients %>%
      ggplot(aes(x = Feature, y = Coefficient)) +
      geom_col(aes(fill = fill), width = 0.8) +
      scale_fill_identity() +
      labs(
        title = paste0("Feature Coefficients (", model, ")\n"),
        x = "\nFeature",
        y = "Coefficient"
      ) +
      theme_minimal() +
      theme(plot.title = element_text(hjust = 0.5),
            axis.line = element_line(color = "black"),
            axis.ticks = element_line(color = "black"))
  } else if (type == "Lift"){
    # Lift Chart
    # Rate of positive predictions lift over random guessing
    perf_lift <- performance(pred, "lift", "rpp")
    
    lift_df <- data.frame(
      rpp = perf_lift@x.values[[1]] * 100,
      lift = perf_lift@y.values[[1]]
    )
    
    ggplot(lift_df, aes(x = rpp, y = lift)) +
      geom_line(col = "blue", linewidth = 0.5) +
      labs(
        title = paste0("Lift Chart (", model, ")"),
        x = "% of Sample (Ranked by Score)",
        y = "Lift"
      ) +
      theme_minimal() +
      theme(plot.title = element_text(hjust = 0.5),
            axis.line = element_line(color = "black"),
            axis.ticks = element_line(color = "black"))
  } else if (type == "ROC"){
    # ROC plot
    perf <- performance(pred, "tpr", "fpr")  # true pos rate vs false pos rate
    
    # Plot ROC curve
    plot(perf, col = "blue", lwd = 2, main = paste0("ROC Curve (", model, ")"))
    abline(a = 0, b = 1, lty = 2, col = "gray")
    # AUC
    auc <- performance(pred, measure = "auc")
    return(auc@y.values[[1]])
  } else {
    return(NA)
  }
}

########################
### Results for Rap  ###
########################

visualize_for("Rap", "ConfusionMatrix")
visualize_for("Rap", "Coefficient")
visualize_for("Rap", "Lift")
visualize_for("Rap", "ROC")

#########################
### Results for Bliss ###
#########################

visualize_for("Bliss", "ConfusionMatrix")
visualize_for("Bliss", "Coefficient")
visualize_for("Bliss", "Lift")
visualize_for("Bliss", "ROC")

#########################
### Results for Blend ###
#########################

visualize_for("Blend", "ConfusionMatrix")
visualize_for("Blend", "Coefficient")
visualize_for("Blend", "Lift")
visualize_for("Blend", "ROC")