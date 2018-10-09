############################ Boosting Algorithms ############################################
library(xgboost)
########################## Setting up right data ########################################

## If non-standardized data to be used
train_data = train_dummified
val_data = val_dummified

#If Standardized data to be used
train_data = train_dummy
val_data = val_dummy

########################## Model #######################################################

train_data$target1 = as.numeric(train_data$target1) - 1
val_data$target1 = as.numeric(val_data$target1) -1

train_data$target1 = as.factor(train_data$target1)
val_data$target1 = as.factor(val_data$target1)

train_matrix = xgb.DMatrix(data = as.matrix(train_data[, !colnames(train_data) %in% c("target1")]),
                           label = as.matrix(train_data[, colnames(train_data) %in% c('target1')]))

val_matrix = xgb.DMatrix(data = as.matrix(val_data[, !colnames(val_data) %in% c("target1")]), 
                         label = as.matrix(val_data[, colnames(val_data) %in% c("target1")]))

params_list <- list("objective" = "binary:logistic",
                    "eta" = 0.1,
                    "early_stopping_rounds" = 10,
                    "max_depth" = 6,
                    "gamma" = 0.5,
                    "colsample_bytree" = 0.6,
                    "subsample" = 0.65,
                    "eval_metric" = "auc",
                    "silent" = 1)

xgb_model_with_params <- xgboost(data = train_matrix, params = params_list, nrounds = 500
                                 , early_stopping_rounds = 20)


###train. Decide on cutoff from AUC curve
xgb_train_proba <- predict(xgb_model_with_params, train_matrix, type="response", norm.votes=TRUE) 

#check auc 

xgb_train_pred_obj <- prediction(xgb_train_proba, train_data$target1)
perf_train <- ROCR::performance(xgb_train_pred_obj, measure = "tpr", x.measure ="fpr")

plot(perf_train,col = rainbow(10),colorize = T,print.cutoffs.at= seq(0,1,0.25))
auc_train <- ROCR::performance(xgb_train_pred_obj,measure = "auc")
auc_train@y.values[[1]]
xgb_train_pred <- ifelse(xgb_train_proba > 0.16 , 1 , 0)

caret::confusionMatrix(as.factor(xgb_train_pred), train_data$target1, positive="1")
#Accuracy 0.6592 Sens = 0.9817

#getF1Stat(as.factor(xgb_train_pred), train_data$target1)
#0.999


## Validation Data
xgb_val_proba <- predict(xgb_model_with_params, val_matrix, type="response", norm.votes=TRUE) 

#check auc 

xgb_val_pred_obj <- prediction(xgb_val_proba, val_data$target1)
perf <- performance(xgb_val_pred_obj, measure = "tpr", x.measure ="fpr")

plot(perf,col = rainbow(10),colorize = T,print.cutoffs.at= seq(0,1,0.05))
auc <- performance(xgb_val_pred_obj,measure = "auc")
auc@y.values[[1]]
xgb_val_pred <- ifelse(xgb_val_proba > 0.2 , 1 , 0)

caret::confusionMatrix(as.factor(xgb_val_pred), val_data$target1, positive="1")
#Accuracy = 0.6036 Sensi = 0.9171


##################### Working on important variables ###############################################
xgb_varimp_matrix <- xgb.importance(feature_names = colnames(train_matrix), 
                                                 model = xgb_model_with_params)

xgb.plot.importance(xgb_varimp_matrix)
xgb_imp_attr_30 <- data.frame(xgb_varimp_matrix[1:20,1])[, 'Feature']
xgb_imp_attr_30 =  append(xgb_imp_attr_30, 'target1')
xgb_imp_attr_30

train_data_xgb_impattr = train_data[, xgb_imp_attr_30]
val_data_xgb_impattr = val_data[, xgb_imp_attr_30]

train_matrix_impattr = xgb.DMatrix(data = as.matrix(train_data_xgb_impattr[, !colnames(train_data_xgb_impattr) %in% c("target1")]),
                           label = as.matrix(train_data_xgb_impattr[, colnames(train_data_xgb_impattr) %in% c('target1')]))

val_matrix_impattr = xgb.DMatrix(data = as.matrix(val_data_xgb_impattr[, !colnames(val_data_xgb_impattr) %in% c("target1")]), 
                         label = as.matrix(val_data_xgb_impattr[, colnames(val_data_xgb_impattr) %in% c("target1")]))


params_list <- list("objective" = "binary:logistic",
                    "eta" = 0.1,
                    "early_stopping_rounds" = 10,
                    "max_depth" = 6,
                    "gamma" = 0.5,
                    "colsample_bytree" = 0.6,
                    "subsample" = 0.65,
                    "eval_metric" = "auc",
                    "silent" = 1)

xgb_model_with_params_impattr <- xgboost(data = train_matrix_impattr, params = params_list, nrounds = 1500
                                 , early_stopping_rounds = 20)


###train. Decide on cutoff from AUC curve
xgb_train_proba <- predict(xgb_model_with_params_impattr, train_matrix_impattr, 
                           type="response", norm.votes=TRUE)

#check auc 

xgb_train_pred_obj <- prediction(xgb_train_proba, train_data_xgb_impattr$target1)
perf_train <- ROCR::performance(xgb_train_pred_obj, measure = "tpr", x.measure ="fpr")

plot(perf_train,col = rainbow(10),colorize = T,print.cutoffs.at= seq(0,1,0.25))
auc_train <- ROCR::performance(xgb_train_pred_obj,measure = "auc")
auc_train@y.values[[1]]
#.9563
xgb_train_pred <- ifelse(xgb_train_proba > 0.2 , "1" , "0")

confusionMatrix(as.factor(xgb_train_pred), as.factor(train_data_xgb_impattr$target), positive = "1")
#0.9986
#getF1Stat(as.factor(xgb_train_pred), train_data_xgb_impattr$target1)
#0.99952


## Validation Data
xgb_val_proba <- predict(xgb_model_with_params_impattr, val_matrix_impattr, 
                         type="response", norm.votes=TRUE) 

#check auc 

xgb_val_pred_obj <- prediction(xgb_val_proba, val_data_xgb_impattr$target1)
perf <- ROCR::performance(xgb_val_pred_obj, measure = "tpr", x.measure ="fpr")

plot(perf,col = rainbow(10),colorize = T,print.cutoffs.at= seq(0,1,0.05))
auc <- ROCR::performance(xgb_val_pred_obj,measure = "auc")
auc@y.values[[1]]
xgb_val_pred <- ifelse(xgb_val_proba > 0.25 , 1 , 0)

confusionMatrix(as.factor(xgb_val_pred), val_data_xgb_impattr$target1, positive = "1")
#0.9572


##################### tune the xgboost ################################################
train_data$target1 = as.factor(as.numeric(train_data$target1) - 1)
val_data$target1 = as.factor(as.numeric(val_data$target1) -1)

sampling_strategy = trainControl(method = 'repeatedcv', number = 8, repeats = 4, 
                                 verboseIter = T)

param_grid <- expand.grid(.nrounds = 30, .max_depth = c(4, 6, 8), .eta = c(0.1, 0.3),
                          .gamma = c(0.6, 0.5, 0.4), .colsample_bytree = c(0.6, 0.4),
                          .min_child_weight = 1, .subsample = c(0.6, 0.5 ,0.9))

xgb_tuned_model <- caret::train(x = train_data[ , !(names(train_data) %in% c("target1"))], 
                         y = train_data[ , names(train_data) %in% c("target1")], 
                         method = "xgbTree",
                         trControl = sampling_strategy,
                         tuneGrid = param_grid)



xgb_train_tuned_proba <- predict(xgb_tuned_model, train_matrix, type="response", norm.votes=TRUE) 

#check auc 

xgb_train_tuned_pred_obj <- prediction(xgb_train_tuned_proba, train_data$target1)
perf_train_tuned <- ROCR::performance(xgb_train_tuned_pred_obj, measure = "tpr", x.measure ="fpr")

plot(perf_train_tuned,col = rainbow(10),colorize = T,print.cutoffs.at= seq(0,1,0.25))
auc_train <- ROCR::performance(xgb_train_tuned_pred_obj,measure = "auc")
auc_train@y.values[[1]]
xgb_train_tune_pred <- ifelse(xgb_train_tuned_proba > 0.16 , 1 , 0)

caret::confusionMatrix(as.factor(xgb_train_tune_pred), train_data$target1, positive="1")
#Accuracy 0.6592 Sens = 0.9817

#getF1Stat(as.factor(xgb_train_pred), train_data$target1)
#0.999


## Validation Data
xgb_val_tuned_proba <- predict(xgb_tuned_model, val_matrix, type="response", norm.votes=TRUE) 

#check auc 

xgb_val_tuned_pred_obj <- prediction(xgb_val_tuned_proba, val_data$target1)
perf_tuned <- ROCR::performance(xgb_val_tuned_pred_obj, measure = "tpr", x.measure ="fpr")

plot(perf_tuned,col = rainbow(10),colorize = T,print.cutoffs.at= seq(0,1,0.05))
auc <- performance(xgb_val_tuned_pred_obj,measure = "auc")
auc@y.values[[1]]
xgb_val__tuned_pred <- ifelse(xgb_val_tuned_pred_obj > 0.2 , 1 , 0)

caret::confusionMatrix(as.factor(xgb_val__tuned_pred), val_data$target1, positive="1")
#Accuracy = 0.6036 Sensi = 0.9171


######################## MLR package with XGboost #########################################




######################## Adaboost ############################################################

library(ada)

## If non-standardized data to be used
train_data = train_dummy
val_data = val_dummy

#If Standardized data to be used
train_data = train_std
val_data = val_std


ada_model = ada(train_data[, !colnames(train_data) %in% c("target1")], y = train_data$target1, 
                loss = 'exponential', type = 'discrete', iter = 100, nu = 0.4, bag.frac = 0.4, 
                verbose = T)
ada_model

ada_train_pred = predict(object = ada_model, newdata = train_data[, !colnames(train_data) 
                                                                %in% c('target1')])
ada_val_pred = predict(object = ada_model, newdata = val_data[, !colnames(val_data) 
                                                              %in% c('target1')])
confusionMatrix(ada_val_pred, val_data$target1)
#0.9563
getF1Stat(ada_val_pred, val_data$target1)
#0.9774










