library(readr)
dataA <- read_csv("0LUCIANO/PCA/Binary_DF.csv")
View(dataA)
head(dataA)

table(dataA$retention) 

# Perform PCA
data_PCA <- princomp(dataA, cor =T)
summary(data_PCA, loading = T)

# Plot Barchart to vizualize PCA cumulative percent:
install.packages("factoextra")
library(factoextra)

fviz_screeplot(data_PCA, main="Barchart of Cumulative Porportion",ncp=50) # add labels: addlabels = TRUE

# Plot biplot - Two Principal Components:
fviz_pca_biplot(data_PCA,col.var="contrib", invisible = "ind", habillage ="none", geom = "text", labelsize=4) + theme_minimal()


# Contributions of variables to PC1
fviz_contrib(data_PCA, choice = "var", axes = 1, top = 10)
# Contributions of variables to PC2
fviz_contrib(data_PCA, choice = "var", axes = 2, top = 10)


# http://www.sthda.com/english/wiki/factoextra-r-package-easy-multivariate-data-analyses-and-elegant-visualization



##########
### LOGIT
logitRet <- glm(retention ~ school_MS + sex_M + age + address_U + famsize_LE3 + Pstatus_T +
                  Medu + Fedu + internet_yes + schoolsup_yes + famsup_yes + paid_yes +
                  activities_yes + romantic_yes + traveltime + studytime + failures + famrel +
                  freetime + goout + Dalc + Walc + health + absences + G1 + G2 + G3,
                data=dataA, family=binomial(link="logit"))
summary(logitRet)

# VIF Library, if needed:
library("car")

### Perform VIF Analysis, to exclude variables with high Collinearity:
vif(logitRet)

# Plot barchart for VIF:
# create vector of VIF values
vif_values <- vif(logitRet)
#create horizontal bar chart to display each VIF value
barplot(vif_values, main = "VIF Values - Collinearity", horiz = FALSE,
        col = "steelblue", cex.names = 0.8)
#add vertical line at 5
abline(h = 5, lwd = 2, lty = 2, col='red')

# Create New df w/o collinearity (manually excluding the variables):
dataC2 <- dataA[,c("retention", "school_MS", "sex_M","age", "address_U","famsize_LE3", "Pstatus_T",
                   "Medu", "internet_yes", "schoolsup_yes", "famsup_yes", "paid_yes", 
                   "activities_yes", "romantic_yes", "traveltime", "studytime", "failures", "famrel",
                   "freetime", "goout", "Dalc", "health", "absences", "G1")]

# Run new Logit model w/o collinearity:
logitRet_C2 <- glm(retention ~ school_MS + sex_M + age + address_U + famsize_LE3 + Pstatus_T +
                  Medu + internet_yes + schoolsup_yes + famsup_yes + paid_yes +
                  activities_yes + romantic_yes + traveltime + studytime + failures + famrel +
                  freetime + goout + Dalc + health + absences + G1,
                data=dataC2, family=binomial(link="logit"))
summary(logitRet_C2)

# Get new VIF output:
vif(logitRet_C2)

# create vector of VIF values
vif_values <- vif(logitRet_C2)
#create horizontal bar chart to display each VIF value
barplot(vif_values, main = "VIF Values - No Collinearity", horiz = FALSE,
        col = "steelblue", cex.names = 0.8)
#add vertical line at 5
abline(h = 1.43, lwd = 2, lty = 2, col='red')




#################
### Create a Reference Model (Imbalanced) and a Balanced Model for comparison:

library(caret)
set.seed(649)
splitIndex <- createDataPartition(dataC$retention, p = .85,
                                  list = FALSE,
                                  times = 1)
trainSplit <- dataC[ splitIndex,] # 553 students
nrow(trainSplit)
table(trainSplit$retention) # 1= 458, 0=95
# check proportions:
prop.table(table(trainSplit$retention))

testSplit <- dataC[-splitIndex,] # 96 students
nrow(testSplit)

ctrl <- trainControl(method = "cv", number = 5)
tbmodel <- train(retention ~ ., data = trainSplit, method = "treebag",
                 trControl = ctrl)

predictors <- names(trainSplit)[names(trainSplit) != 'retention']

pred <- predict(tbmodel$finalModel, testSplit[,predictors])
table(pred)

library(pROC)
pred <- sapply(pred, as.numeric)
auc <- roc(testSplit$retention, pred)
print(auc)

plot(auc, ylim=c(0,1), print.thres=TRUE, main=paste('AUC:',round(auc$auc[[1]],2)))
abline(h=1,col='blue',lwd=2)
abline(h=0,col='red',lwd=2)


# Library to Balance the classifier:
install.packages("ROSE")
library(ROSE)

# Check Performance of Imbalanced Model:
library(rpart)

str(trainSplit)
treeimb <- rpart(retention ~ ., data = trainSplit)
pred.treeimb <- predict(treeimb, newdata = testSplit)
head(pred.treeimb)

accuracy.meas(testSplit$retention, pred.treeimb[,2])

roc.curve(testSplit$retention, pred.treeimb[,2], plotit = FALSE)

### Balancing the data:
# Resampling Option 1 (over):
data.bal.ov <- ovun.sample(retention ~ ., data = trainSplit, method = "over",
                             p=0.5, seed = 1)$data
table(data.bal.ov$retention)

# Resamplint Option 2 (under):
data.bal.un <- ovun.sample(retention ~ ., data = trainSplit, method = "under",
                           p = 0.5, seed = 1)$data
table(data.bal.un$retention)

# Resampling Option 3 (both):
data.bal.ou <- ovun.sample(retention ~ ., data = trainSplit, method = "both",
                           N = 553, p = 0.5, seed = 1)$data
table(data.bal.ou$retention)

# Resampling Option 4 (ROSE):
data.rose <- ROSE(retention ~ ., data = trainSplit, seed = 1)$data

table(data.rose$retention)

# Training the Classifiers and run test set:
tree.ov <- rpart(retention ~ ., data = data.bal.ov)
tree.un <- rpart(retention ~ ., data = data.bal.un)
tree.ou <- rpart(retention ~ ., data = data.bal.ou)
tree.rose <- rpart(retention ~ ., data = data.rose)


# Predict in the new data (test):
pred.tree.ov <- predict(tree.ov, newdata = testSplit)
pred.tree.un <- predict(tree.un, newdata = testSplit)
pred.tree.ou <- predict(tree.un, newdata = testSplit)
pred.tree.rose <- predict(tree.rose, newdata = testSplit)


# Model Evaluation - ROC Curve:
roc.curve(testSplit$retention, pred.tree.ov[,2], add.roc = TRUE, col = 12, lty = 2)
#roc.curve(testSplit$retention, pred.tree.un[,2], add.roc = TRUE, col = 3, lty = 3)
#roc.curve(testSplit$retention, pred.tree.ou[,2], add.roc = TRUE, col = 4, lty = 4)
roc.curve(testSplit$retention, pred.tree.rose[,2], col = 0, main= "AUC: 0.66")
# Imbalanced ROC
roc.curve(testSplit$retention, pred.treeimb[,1], add.roc = TRUE, col = 2, lty = 2) # imbalanced



#############
### Logit Regression Machine Learning Modeling:

# Create balanced training data for LR:
LR_train <- ovun.sample(retention ~ ., data = trainSplit, method = "both",
                        N = 1106, seed =1)$data # 1106 (553x2)
table(LR_train$retention)

# Create balanced test data for LR:
LR_test <- ovun.sample(retention ~ ., data = testSplit, method = "both",
                        N = 192, seed = 1)$data # 192 (96x2)
table(LR_test$retention)


# Run LR:
model1 <- glm(retention ~ school_MS + sex_M + age + address_U + famsize_LE3 + Pstatus_T +
              Medu + internet_yes + schoolsup_yes + famsup_yes + paid_yes +
              activities_yes + romantic_yes + traveltime + studytime + failures + famrel +
              freetime + goout + Dalc + health + absences + G3,
              family = "binomial", 
              data = LR_train)
summary(model1)

install.packages("vip")
library(vip)
library(caret)

# Interpreting coeficients (odds):
exp(coef(model1)) #ex: the odds of a student retaining increases
# multiplicativelly by 2.4040806 for every extra educational support (schoolsup_yes).

# Get Confident Intervals:
confint(model1)

# CI for Odds:
exp(confint(model1))


# Model Accuracy Assessment (10-fold Cross Validation):
cv_model1 <- train(retention ~ ., data = LR_train, method = "glm",
  family = "binomial", trControl = trainControl(method = "cv", number = 10))

# extract out of sample performance measures
summary(resamples(list(model1 = cv_model1, model2 = cv_model1)))$statistics$Accuracy


# Confusion Matrix:
# predict class
pred_class <- predict(cv_model1, LR_train)

# create confusion matrix
confusionMatrix(data = relevel(pred_class, ref = "Yes"), reference = relevel(LR_train$retention, ref = "Yes"))




                  