######################## Logistic Regression ##############################################

######## Decide on data to be used ##################
## If non-standardized data to be used
test_data = test_dummy

##If Standardized data to be used
test_data = test_std

log_test_proba = predict(object = log_reg, newdata = test_data, type = 'response')
log_test_class = ifelse(log_test_proba > 0.44, "Yes", "No")
table(log_test_class)
test_result = cbind(ID, log_test_class)
colnames(test_result) = c("RowID", "ExtraTime")
write.csv(test_result, file = "..\\submission\\simplelogistic.csv", row.names = F)

###################### C50 Decision trees ##########################################
## If non-standardized data to be used
test_data = test_dummy

##If Standardized data to be used
test_data = test_std

c50_test_class = predict(object = c50_model, newdata = test_data)
table(c50_test_class)
test_result = cbind(ID, c50_test_class)
colnames(test_result) = c("RowID", "ExtraTime")
write.csv(test_result, file = '..\\submission\\c50.csv', , row.names = F)

##################### Rpart Decision tress ########################################
## If non-standardized data to be used
test_data = test_dummy

##If Standardized data to be used
test_data = test_std

rpart_test_class = predict(object = rpart_model, newdata = test_data, type = 'class')
table(rpart_test_class)
test_result = data.frame(cbind(ID, rpart_test_class))
colnames(test_result) = c("RowID", "ExtraTime")
test_result$ExtraTime = ifelse(test_result$ExtraTime == 1, "No", "Yes")
write.csv(test_result, file = "..\\submission\\rpart.csv", row.names = F)

#################### Random forests and tuning #############################################
## If non-standardized data to be used
test_data = test_dummy

##If Standardized data to be used
test_data = test_std

rf_test_class = predict(object = rf_model, newdata = test_data, type='response', norm.votes = T)
table(rf_test_class)
test_result = data.frame(cbind(ID, rf_test_class))
colnames(test_result) = c("RowID", "ExtraTime")
test_result$ExtraTime = ifelse(test_result$ExtraTime == 1, "No", "Yes")
write.csv(test_result, file = '..\\submission\\rf.csv', row.names = F)

####Tuned RF model
rf_tuned_test_class = predict(object = rf_besttune_model, newdata = test_data, type = 'response', norm.votes = T)
table(rf_tuned_test_class)
test_result = data.frame(cbind(ID, rf_tuned_test_class))
colnames(test_result) = c("RowID", "ExtraTime")
test_result$ExtraTime = ifelse(test_result$ExtraTime == 1, "No", "Yes")
write.csv(test_result, file="..\\submission\\rf_tuned.csv", row.names = F)

#####Tuned Rf important variables
rf_tuned_topattr_test_class = predict(object = rf_besttune_topattr_model, newdata = test_data, 
                                      type = 'response', norm.votes = T)

table(rf_tuned_topattr_test_class)
test_result = data.frame(cbind(ID, rf_tuned_topattr_test_class))
colnames(test_result) = c("RowID", "ExtraTime")
write.csv(test_result, file="..\\submission\\rf_tuned_impattr.csv", row.names = F)

################################# Boosting #################################################
## If non-standardized data to be used
test_data = test_dummified

##If Standardized data to be used
test_data = test_std

xg_params_test_proba = predict(object = xgb_model_with_params, newdata = as.matrix(test_data), 
                               type='response', norm.votes = T)


xg_params_test_class = ifelse(xg_params_test_proba > 0.2, "1", "0")
table(xg_params_test_class)
test_result = data.frame(cbind(ID, xg_params_test_class))
colnames(test_result) = c("RowID", "ExtraTime")
test_result$ExtraTime = ifelse(test_result$ExtraTime == 1, "Yes", "No")
write.csv(test_result, file="..\\submission\\xg_params_new.csv", row.names = F)

#### Imp attr

test_data_impattr = test_data[, xgb_imp_attr_30[-length(xgb_imp_attr_30)]]
xg_params_varimp_test_proba = predict(object = xgb_model_with_params_impattr, 
                                      newdata = as.matrix(test_data_impattr), 
                               type='response', norm.votes = T)


xg_params_test_class = ifelse(xg_params_varimp_test_proba > 0.2, "1", "0")
table(xg_params_test_class)
test_result = data.frame(cbind(ID, xg_params_test_class))
colnames(test_result) = c("RowID", "ExtraTime")
test_result$ExtraTime = ifelse(test_result$ExtraTime == 1, "Yes", "No")
write.csv(test_result, file="..\\submission\\xg_params_varimp.csv", row.names = F)

#################### Adaboost #####################################################
## If non-standardized data to be used
test_data = test_dummy

##If Standardized data to be used
test_data = test_std

ada_test_class = predict(object = ada_model, newdata = test_data)
table(ada_test_pred)
test_result = cbind(ID, xg_params_test_class)
colnames(test_result) = c("RowID", "ExtraTime")
write.csv(test_result, file="..\\submission\\adaboost.csv")

############################### Stacking ##########################################

stack_df_test <- data.frame(logistic = log_test_class, c50 = c50_test_class,
                             rpart = rpart_test_class, rf = rf_test_class, 
                             rf_tuned = rf_tuned_test_class,
                             ada = ada_test_class,
                             xgboost = xg_params_test_class)

numeric_st_df_test <- sapply(stack_df_test, function(x) as.numeric(as.character(x)))
predicted_stack_test <- as.data.frame(predict(pca_stack, numeric_st_df_test))[1:3]
stacked_test_class <-  predict(stacked_model, predicted_stack_test)
table(stacked_test_class)
test_result = cbind(ID, stacked_test_class)
colnames(test_result) = c("RowID", "ExtraTime")
write.csv(test_result, file="..\\submission\\stacking.csv")






