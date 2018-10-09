################################# Feature Engineering #########################################

##If imputed
#test_dummy = test_imputed

##If not imputed
test_dummy = test_pure


library(corrplot)
library(caret)

######################### Numerical Attr ##############################################
##### Finding correlation matrix among numeric attr

cor_mat = cor(test_dummy[, !colnames(test_dummy) %in% c("target")], use = "complete.obs")
#cor_mat
cor_mat = round(cor_mat, 2)
cor_mat[abs(cor_mat) < 0.9] = NA

cor_mat[lower.tri(cor_mat,diag = TRUE)] = NA  #Prepare to drop duplicates and meaningless information
cor_mat = as.data.frame(as.table(cor_mat))  #Turn into a 3-column table
cor_mat = na.omit(cor_mat)  #Get rid of the junk we flagged above
cor_mat = cor_mat[order(-abs(cor_mat$Freq)), ]    #Sort by highest correlation (whether +ve or -ve)
cor_mat = cor_mat %>% mutate(id = row_number())
cor_mat
which(cor_mat$Var1 == 'Attr7' | cor_mat$Var2 == 'Attr7'| cor_mat$Var1 == 'Attr14' | cor_mat$Var2 == 'Attr14')
test_dummy$Attr0714 = test_dummy$Attr7 * test_dummy$Attr14 * test_dummy$Attr1 * 
  test_dummy$Attr11 * test_dummy$Attr22

which(cor_mat$Var1 == 'Attr8' | cor_mat$Var2 == 'Attr8'| cor_mat$Var1 == 'Attr17' | cor_mat$Var2 == 'Attr17')
test_dummy$Attr0817 = test_dummy$Attr8 * test_dummy$Attr17

which(cor_mat$Var1 == 'Attr19' | cor_mat$Var2 == 'Attr19'| cor_mat$Var1 == 'Attr23' | cor_mat$Var2 == 'Attr23')
test_dummy$Attr1923 = test_dummy$Attr19 * test_dummy$Attr23 * test_dummy$Attr31 * test_dummy$Attr42 *
  test_dummy$Attr39 *test_dummy$Attr49

which(cor_mat$Var1 == 'Attr16' | cor_mat$Var2 == 'Attr16'| cor_mat$Var1 == 'Attr26' | cor_mat$Var2 == 'Attr26')
test_dummy$Attr1626 = test_dummy$Attr16 * test_dummy$Attr26

which(cor_mat$Var1 == 'Attr28' | cor_mat$Var2 == 'Attr28'| cor_mat$Var1 == 'Attr54' | cor_mat$Var2 == 'Attr54')
test_dummy$Attr2854 = test_dummy$Attr28 * test_dummy$Attr54 * test_dummy$Attr53 * test_dummy$Attr64

which(cor_mat$Var1 == 'Attr56' | cor_mat$Var2 == 'Attr56'| cor_mat$Var1 == 'Attr58' | cor_mat$Var2 == 'Attr58')
test_dummy$Attr5658 = test_dummy$Attr56 * test_dummy$Attr58

which(cor_mat$Var1 == 'Attr43' | cor_mat$Var2 == 'Attr43'| cor_mat$Var1 == 'Attr44' | cor_mat$Var2 == 'Attr44')
test_dummy$Attr4344 = test_dummy$Attr43 * test_dummy$Attr44* test_dummy$Attr30 * test_dummy$Attr62

which(cor_mat$Var1 == 'Attr02' | cor_mat$Var2 == 'Attr02'| cor_mat$Var1 == 'Attr10' | cor_mat$Var2 == 'Attr10')
test_dummy$Attr0210 = test_dummy$Attr2 * test_dummy$Attr10

which(cor_mat$Var1 == 'Attr40' | cor_mat$Var2 == 'Attr40'| cor_mat$Var1 == 'Attr46' | cor_mat$Var2 == 'Attr46')
test_dummy$Attr4046 = test_dummy$Attr40 * test_dummy$Attr46 * test_dummy$Attr4

test_dummy = test_dummy[, !colnames(test_dummy) %in% c("Attr7","Attr14","Attr11","Attr22",
                                                       "Attr1","Attr8","Attr19","Attr23",
                                                       "Attr31","Attr42","Attr39","Attr49",
                                                       "Attr16","Attr26","Attr28","Attr53",
                                                       "Attr64","Attr54",
                                                       "Attr53","Attr33","Attr63","Attr34",
                                                       "Attr56","Attr58", "Attr43",
                                                       "Attr44","Attr30","Attr62","Attr2","Attr10",
                                                       "Attr40","Attr46","Attr4")]

str(test_dummy)
#test_dummy$target1 = test_dummy$target
#test_dummy$target = NULL


############# Grouping of levels under specific categorical attr and Dummification ############
## Sample code. Replace values accordingly
#Lets group/categorize the qualification variable. 
#First we categorize then we shall try to make ordinal

test_dummy$qualification = gsub(pattern = "^10th", replacement = "Low", x = test_dummy$qualification)
test_dummy$qualification = gsub(pattern = "^11th", replacement = "Low", x = test_dummy$qualification)
test_dummy$qualification = gsub(pattern = "^12th", replacement = "Low", x = test_dummy$qualification)
test_dummy$qualification = gsub(pattern = "^1st-4th", replacement = "Low", x = test_dummy$qualification)
test_dummy$qualification = gsub(pattern = "^5th-6th", replacement = "Low", x = test_dummy$qualification)
test_dummy$qualification = gsub(pattern = "^7th-8th", replacement = "Low", x = test_dummy$qualification)
test_dummy$qualification = gsub(pattern = "^9th", replacement = "Low", x = test_dummy$qualification)
test_dummy$qualification = gsub(pattern = "^Preschool", replacement = "Low", x = test_dummy$qualification)

##Now create medium education group with assoc and bachelors
test_dummy$qualification = gsub(pattern = "^Assoc", replacement = "Medium", x = test_dummy$qualification)
test_dummy$qualification = gsub(pattern = "^Bachelors", replacement = "Medium", x = test_dummy$qualification)

##Matching part of string will replace that part and create a new column like Medium-acdm
test_dummy$qualification = gsub(pattern = "^Medium-acdm", replacement = "Medium", x = test_dummy$qualification)
test_dummy$qualification = gsub(pattern = "^Medium-voc", replacement = "Medium", x = test_dummy$qualification)

##Now create High education group with masters and HS-grad and prof-school
test_dummy$qualification = gsub(pattern = "^Prof-school", replacement = "High", x = test_dummy$qualification)
test_dummy$qualification = gsub(pattern = "^HS-grad", replacement = "High", x = test_dummy$qualification)
test_dummy$qualification = gsub(pattern = "^Masters", replacement = "High", x = test_dummy$qualification)

##Create VeryHigh group with doctorate
test_dummy$qualification = gsub(pattern = "^Doctorate", replacement = "VeryHigh", x = test_dummy$qualification)

##We shall add some-college people in existing proportion to low, high and medium classes
prop.table(table(test_dummy$qualification))

############# Ordinalizing qualification levels. Feature becomes numeric attr ###############
##Now lets dummify this qualification varibale with ordinality low as 1, medium as 2, high as 3, veryhigh as 4 

test_dummy$qualification = ifelse(test_dummy$qualification == 'Low', 1, 
                                   ifelse(test_dummy$qualification == 'Medium', 2, ifelse(test_dummy$qualification == 'High', 3, 4)))

str(test_dummy)
table(test_dummy$occupation)

## Or
## Just dummify without ordinalizing

dm_marital = dummyVars(formula = "~ marital_status", data = test_dummy)
df_marital = predict(object = dm_marital, newdata = test_dummy)
test_dummy = cbind(df_marital, test_dummy)

## Remove unnecessary stuff
rm(df_marital)
test_dummy$marital_status = NULL

########################### Imputation ################################################
##1) Decide on when to do imputation. If it is only numeric data and if you feel that attributes
## are highly correlated, do imputation post correlation
##2) If the attr is a mix bag of numeric and cat, please split into train and validation data
## first and do the imputation
##3) IMP: Always do split the data into train and validation and do the imputation


##Decide on central or knn imputation
test_dummy = centralImputation(data = test_dummy)
##test_dummy = knnImputation(data = test_dummy, k = 8, scale = T, meth = 'weighAvg')
sum(is.na(test_dummy))

##Check for integrity of data
str(test_dummy)

## Remove any unnecessary variables
#rm(train_pure)
#rm(val_pure)


########################### Standardizing the numerical attrs #################################

##Use train data's std obj
#std_obj = preProcess(test_dummy, method = c('center', 'scale'))

test_std = predict(object = std_obj, newdata = test_dummy)
summary(test_std)
rm(std_obj)



