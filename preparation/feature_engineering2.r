################################# Feature Engineering #########################################
train_dummy = train_imputed

library(corrplot)
library(caret)

######################### Numerical Attr ##############################################
##### Finding correlation matrix among numeric attr
cor_mat = cor(train_dummy[, !colnames(train_dummy) %in% cat_attr], use = "complete.obs")
corrplot(cor_mat, type = "lower", order = "hclust", 
         tl.col = "black", tl.srt = 45)
# The following corelations are observed
#1) attr18,7,14,11,22,48 +1
#2) attr35,11,22,48,9,36,34,33,63 +1
#3) attr34,33,63 +1
#4) attr 32,47
#5) attr 2,3,6 -1 2and 1 +1
#6) attr51,3,6 -1
#7) attr13,31,19,23,42,43,44,20,58,30,62  49,56 -1
#8) attr24,18,7,14,35,11,22,48,9,36 +1
#9) attr49,56,43,44,20,58,30,62 -1
#10) attr12,16,26 +1
#11) attr50,attr4,attr46

train_dummy$intr1 = train_dummy$Attr18 * train_dummy$Attr7 * train_dummy$Attr14 *
        train_dummy$Attr11 * train_dummy$Attr22 * train_dummy$Attr48

train_dummy$intr2 = train_dummy$Attr35 * train_dummy$Attr11 * train_dummy$Attr22 *
  train_dummy$Attr48 * train_dummy$Attr9 * train_dummy$Attr36 * train_dummy$Attr34 *
  train_dummy$Attr33 * train_dummy$Attr63

train_dummy$intr3 = train_dummy$Attr34 * train_dummy$Attr33 * train_dummy$Attr63

train_dummy$intr4 = train_dummy$Attr32 * train_dummy$Attr47

train_dummy$intr5 = train_dummy$Attr2 * train_dummy$Attr3 * train_dummy$Attr6 * train_dummy$Attr51

train_dummy$intr6 = train_dummy$Attr51 * train_dummy$Attr3 * train_dummy$Attr6

train_dummy$intr7 = train_dummy$Attr13 * train_dummy$Attr31 * train_dummy$Attr19 *
  train_dummy$Attr23 * train_dummy$Attr42 * train_dummy$Attr43 * train_dummy$Attr44 *
  train_dummy$Attr20 * train_dummy$Attr58 * train_dummy$Attr30 * train_dummy$Attr62

train_dummy$intr8 = train_dummy$Attr24 * train_dummy$Attr18 * train_dummy$Attr7 *
  train_dummy$Attr14 * train_dummy$Attr35 * train_dummy$Attr11 * train_dummy$Attr22 *
  train_dummy$Attr48 * train_dummy$Attr9 * train_dummy$Attr36

train_dummy$intr9 = train_dummy$Attr56 * train_dummy$Attr49 * train_dummy$Attr43 * train_dummy$Attr44 *
  train_dummy$Attr20 * train_dummy$Attr58 * train_dummy$Attr30 * train_dummy$Attr62


train_dummy$intr10 = train_dummy$Attr12 * train_dummy$Attr16 * train_dummy$Attr26


#train_dummy = train_dummy[, !colnames(train_dummy) %in% c("Attr18", "Attr7" ,"Attr14" ,"Attr11", 
#                                                          "Attr22", "Attr48",
#                                                          "Attr35", "Attr9" ,"Attr36" ,
#                                                          "Attr34" ,"Attr33", "Attr63", "Attr32", 
#                                                          "Attr47" ,"Attr2", "Attr3", "Attr6", 
#                                                          "Attr51" ,"Attr13", "Attr31", "Attr19",
#                                                          "Attr23" ,"Attr42", "Attr43" ,
#                                                          "Attr44", "Attr20" ,"Attr58" ,"Attr30",
#                                                          "Attr62", "Attr24" ,"Attr35" ,"Attr56",
#                                                          "Attr49", "Attr62")]


train_dummy$target1 = train_dummy$target
train_dummy$target = NULL
str(train_dummy)

############################ Splitting train data into train and test #########################
set.seed(143)
train_rows = createDataPartition(y = train_dummy$target1, p = 0.7, list = F)
train_data = train_dummy[train_rows, ]
val_data = train_dummy[-train_rows, ]

rm(train_rows)

########################### Standardizing the numerical attrs #################################

std_obj = preProcess(train_data[, !colnames(train_data) %in% c("target1")], method = c('center', 'scale'))

train_data = predict(object = std_obj, newdata = train_data)
val_data = predict(object = std_obj, newdata = val_data)

summary(train_data)
summary(val_data)

#Use the same object for prediction on test data either
#rm(std_obj)