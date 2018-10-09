################################# Feature Engineering #########################################
test_dummy = test_imputed

library(corrplot)
library(caret)

######################### Numerical Attr ##############################################
##### Finding correlation matrix among numeric attr
test_dummy$intr1 = test_dummy$Attr18 * test_dummy$Attr7 * test_dummy$Attr14 *
  test_dummy$Attr11 * test_dummy$Attr22 * test_dummy$Attr48

test_dummy$intr2 = test_dummy$Attr35 * test_dummy$Attr11 * test_dummy$Attr22 *
  test_dummy$Attr48 * test_dummy$Attr9 * test_dummy$Attr36 * test_dummy$Attr34 *
  test_dummy$Attr33 * test_dummy$Attr63

test_dummy$intr3 = test_dummy$Attr34 * test_dummy$Attr33 * test_dummy$Attr63

test_dummy$intr4 = test_dummy$Attr32 * test_dummy$Attr47

test_dummy$intr5 = test_dummy$Attr2 * test_dummy$Attr3 * test_dummy$Attr6 * test_dummy$Attr51

test_dummy$intr6 = test_dummy$Attr51 * test_dummy$Attr3 * test_dummy$Attr6

test_dummy$intr7 = test_dummy$Attr13 * test_dummy$Attr31 * test_dummy$Attr19 *
  test_dummy$Attr23 * test_dummy$Attr42 * test_dummy$Attr43 * test_dummy$Attr44 *
  test_dummy$Attr20 * test_dummy$Attr58 * test_dummy$Attr30 * test_dummy$Attr62

test_dummy$intr8 = test_dummy$Attr24 * test_dummy$Attr18 * test_dummy$Attr7 *
  test_dummy$Attr14 * test_dummy$Attr35 * test_dummy$Attr11 * test_dummy$Attr22 *
  test_dummy$Attr48 * test_dummy$Attr9 * test_dummy$Attr36

test_dummy$intr9 = test_dummy$Attr56 * test_dummy$Attr49 * test_dummy$Attr43 * test_dummy$Attr44 *
  test_dummy$Attr20 * test_dummy$Attr58 * test_dummy$Attr30 * test_dummy$Attr62


test_dummy$intr10 = test_dummy$Attr12 * test_dummy$Attr16 * test_dummy$Attr26


#test_dummy = test_dummy[, !colnames(test_dummy) %in% c("Attr18", "Attr7" ,"Attr14" ,"Attr11", 
#                                                          "Attr22", "Attr48",
#                                                          "Attr35", "Attr9" ,"Attr36" ,
#                                                          "Attr34" ,"Attr33", "Attr63", "Attr32", 
#                                                          "Attr47" ,"Attr2", "Attr3", "Attr6", 
#                                                          "Attr51" ,"Attr13", "Attr31", "Attr19",
#                                                          "Attr23" ,"Attr42", "Attr43" ,
#                                                          "Attr44", "Attr20" ,"Attr58" ,"Attr30",
#                                                          "Attr62", "Attr24" ,"Attr35" ,"Attr56",
#                                                          "Attr49", "Attr62")]


#test_dummy$target1 = test_dummy$target
#test_dummy$target = NULL

########################### Standardizing the numerical attrs #################################

std_obj = preProcess(test_dummy, method = c('center', 'scale'))

test_data = predict(object = std_obj, newdata = test_dummy)
summary(test_data)
rm(std_obj)