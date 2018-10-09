################################# Stacking #############################################
library(e1071)
library(kernlab)

## Getting all the predictions on the train data into a dataframe

train_preds_df <- data.frame(logistic = log_train_class, c50 = c50_train_class,
                             rpart = rpart_train_class, rf = rf_train_class, 
                             rf_tuned = rf_tuned_train_class,
                             ada = ada_train_pred,
                             xgboost = xgb_train_pred, target1 = train_data$target1)

## Getting all the predictions from the validation data into a dataframe

val_preds_df <- data.frame(logistic = log_val_class, c50 = c50_val_class,
                           rpart = rpart_val_class, rf = rf_val_class, 
                           rf_tuned = rf_tuned_val_class,
                           ada = ada_val_pred,
                           xgboost = xgb_val_pred, target1 = val_data$target1)


stack_df <- rbind(train_preds_df, val_preds_df)
stack_df$target1 <- as.factor(stack_df$target1)

numeric_stack_df <- sapply(stack_df[, !(names(stack_df) %in% "target1")], 
                        function(x) as.numeric(as.character(x)))
head(numeric_stack_df)

### Now, since the outputs of the various models are extremely correlated let's use PCA 
### to reduce the dimensionality of the dataset

pca_stack <- prcomp(numeric_stack_df, scale = F)
pca_stack$sdev
predicted_stack <- as.data.frame(predict(pca_stack, numeric_stack_df))[1:3]
head(predicted_stack)

stacked_df <- data.frame(predicted_stack, target1 = stack_df$target1)

stacked_model <- ksvm(target1 ~ . , stacked_df, kernel = "anovadot")
stacked_model

