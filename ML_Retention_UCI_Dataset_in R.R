##### Assumptions:
## 1 - Missing values
## 2 - Imbalanced Classifier
## 3 - Random Classifier
## 4 - Overfitting
## 5 - Underfitting
## 5 - Multicollinearity

####################################################
### PHASE 1 - Feature Engineering / Data Munging ###
####################################################

### Import the two original csv files:
d1=read.table("student-mat.csv",sep=";",header=TRUE)
d2=read.table("student-por.csv",sep=";",header=TRUE)
print(nrow(d2)) # 649 students

### Merge csv files into a new df:
d3=merge(d1,d2,by=c("school","sex","age","address","famsize","Pstatus","Medu","Fedu","Mjob","Fjob","reason","nursery","internet"))
print(nrow(d3)) # 382 students


### Shapping the dataset:
# Create a new binary column called "retention" (1= retained; 0= not retained):
d2$retention <- set.seed(649) # create the new column and set seed to ensure we will get the same distribution when getting the random binary numbers (1 and 0)
d2$retention <- sample(1:0, 649, replace=T, prob=c(retention = 0.85, 0.15)) # setting the counts of 1 and 0 that we want to get
d2 <- data.frame(d2)
# Count Binary elements (double-checking):
table(d2$retention) # we shoud get 0= 111, and 1= 538
View(d2)

### Create Dummy (Binary) codes for the existing variables:
install.packages("fastDummies")
library('fastDummies')

dataDM <- dummy_cols(d2, select_columns = c('school', 'sex', 'address',
                                            'famsize','Mjob', 'Pstatus', 'internet', 'schoolsup',
                                            'famsup', 'paid', 'activities', 'romantic',
                                            'schoolsup', 'famsup','paid', 'activities', 'romantic'),
                     remove_first_dummy = TRUE)
head(dataDM)

str(dataDM)

# Create new data frame just with all Numeric and Dummies variables we are going to study:

dataA <- dataDM[,c("retention", "school_MS", "sex_M", "age", "address_U",
                    "famsize_LE3", "Mjob_health", "Mjob_other", "Mjob_services",
                    "Mjob_teacher", "Pstatus_T", "Medu", "Fedu", "internet_yes",
                    "schoolsup_yes", "famsup_yes", "paid_yes", "activities_yes",
                    "romantic_yes", "traveltime", "studytime", "failures", "famrel",
                    "freetime", "goout", "Dalc", "Walc", "health", "absences", "G1", "G2", "G3")]
str(dataA)

# * it would be probably more efficient to just drop the columns we didn't want in the df *
head(dataA)
# Check Columns's Names:
ls(dataA) # or 'names(data)'
# Check Dimensions (number of rolls and columns):
dim(dataA) # or 'nrow(data)' and 'ncol(data)'
# Check Class:
class(dataA)

#### Check Missing Values:
sum(is.na(dataA)) # we have zero missing data.
any(is.na(dataA)) # it returns FALSE because we don't have missing data.
# Plot Missing Value Matrix:
install.packages("Amelia") # you may be prompted with a question.
library(Amelia)

# Run this all togehter to plot the map:
missmap(dataA, col = c("indianred", "lightgreen"),
        main = "Missing Values vs Observed")


# Export merged/binary data into csv file:
write.csv(dataA,'Binary_DF.csv')



#######################################
### PHASE 2 - EXPLORATORY ANALYSIS ###
#######################################

# grouped bar plot
install.packages("tidyverse")
library(ggplot2)

#Barchart Retention by Guardian type:
RT <-ggplot(dataDM, aes(x = retention)) + geom_bar(position = "dodge")
RT

RG1 <-ggplot(dataDM, aes(x = retention, fill = guardian)) + geom_bar(position = "dodge")
RG1


# scatterplot with linear fit line
ggplot(dataDM,
       aes(x = G1, 
           y = G2)) +
  geom_point(color= "steelblue") +
  geom_smooth(method = "lm")



#library(reshape2)
#sp <- ggplot(dataDM, aes(x=G3, y=age, colour = factor(sex))) + geom_point(shape=7)
#sp + facet_grid(sex ~ address)

install.packages("GGally")
library(GGally)
install.packages("hrbrthemes")
library(hrbrthemes)
install.packages("viridis")
library(viridis)

# Plot
dataplot <- dataDM[,c("studytime","freetime", "G1", "sex")]
ggparcoord(dataplot,
           columns = 1:3, groupColumn =4, order = "anyClass",
           scale="uniminmax", #Standardize to Min = 0 and Max = 1
           showPoints = TRUE, 
           alphaLines = 0.6
) + 
  scale_color_viridis(discrete=TRUE) +
  theme_ipsum()+
  theme(
    plot.title = element_text(size=10)
)


################################################################
### PHASE 3 - Dimensionality Reduction / Innitial Assessment ###
################################################################

### Outliers Detection:
boxplot(dataA, col="steelblue")#, boxwex = TRUE)

boxplot(dataA$absences, col="steelblue")

# Plot Bivariate Plot for G1 and G2 variables:
library(MVA)
# extract "G1" and "G2" variables from data frame 
mydata <- dataA[, c("G1", "G2")] 
# Run these two lines together:
bvbox(mydata, xlab="G1", ylab="G2", type="n")
text(mydata$G1, mydata$G2, cex = 0.6, col="steelblue")

### Histogram w boxplot for variable "G3":
mydata2 <-dataA[,c("G1")]
# Layout to split the screen
layout(mat = matrix(c(1,2),2,1, byrow=TRUE),  height = c(1,8))
# Draw the boxplot and the histogram 
par(mar=c(0, 3.1, 1.1, 2.1))
boxplot(mydata2 , horizontal=TRUE , ylim=c(0,20), xaxt="n" , 
        col="steelblue", frame=F)
par(mar=c(4, 3.1, 1.1, 2.1))
hist(mydata2 , breaks=40 , col="steelblue", border=T , 
     main="" , xlim=c(0,20))




### Skewness and Kurtosis: for variable G1:
install.packages("moments")
library(moments)
skewness(dataA)
kurtosis(dataA)


### Dendogram/Correlation Matrix:
install.packages("gplots")
library("gplots")

# Create Matrix
dataM <- data.matrix(dataA)
# Correlation Matrix:
data_cor <- cor(dataM)
# Dendogram Correlation Matrix:
heatmap.2(data_cor, scale="none", col=redblue(100),
          trace="none", density.info="none")

# Correlation Matrix:
install.packages("corrplot")
library(corrplot)
correlations <- cor(data_cor)
corrplot(correlations, tl.col = "gray", method="circle", order = "hclust") # addrect = 3


### PCA:
data_PCA <- princomp(dataA, cor=T)
summary(data_PCA, loading = T)
score <- data_PCA$scores
head(score)
# optional biplot:
# biplot(data_PCA, col=c("black", "red"), cex = 0.6) 

# PCA Plots:
library(devtools)
install_github("ggbiplot")
library(ggbiplot)

ggbiplot(data_PCA, pc.biplot = TRUE, alpha = 0, varname.adjust = 2, varname.size = 3.5)


# Plot Barchart to vizualize PCA cumulative percent:
install.packages("factoextra") # won't work on MAC
library(factoextra)

# Scree PLot of Dimmentions
fviz_screeplot(data_PCA, main=" ",ncp=50) # add labels: addlabels = TRUE

# Biplot - Two Principal Components:
fviz_pca_biplot(data_PCA,col.var="contrib", invisible = "ind", habillage ="none", geom = "text", labelsize=4) + theme_minimal()

# Contributions of variables to PC1
fviz_contrib(data_PCA, choice = "var", axes = 1, top = 10)
# Contributions of variables to PC2
fviz_contrib(data_PCA, choice = "var", axes = 2, top = 10)

### VIF Library, if needed:
library("car") # may not work on Mac

### Build Logit model to test multicolinearity:
# Perform Logit Regression to deal with Collinearity:
logitRet <- glm(retention ~ school_MS + sex_M + age + address_U + famsize_LE3 + Pstatus_T +
                  Mjob_health + Mjob_other + Mjob_services + Mjob_teacher +
                  Medu + Fedu + internet_yes + schoolsup_yes + famsup_yes + paid_yes +
                  activities_yes + romantic_yes + traveltime + studytime + failures + famrel +
                  freetime + goout + Dalc + Walc + health + absences + G1 + G2 + G3,
                data=dataA, family=binomial(link="logit"))
summary(logitRet)


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

# Create New df w/o collinearity (manually excluding the variable G2, Walc, and Fedu):
dataC1 <- dataA[,c("retention", "school_MS", "sex_M","age", "address_U","famsize_LE3", "Pstatus_T",
                  "Mjob_health", "Mjob_other", "Mjob_services", "Mjob_teacher",
                  "Medu", "internet_yes", "schoolsup_yes", "famsup_yes", "paid_yes", 
                  "activities_yes", "romantic_yes", "traveltime", "studytime", "failures", "famrel",
                  "freetime", "goout", "Dalc", "health", "absences", "G1", "Fedu", "Walc")]

# Run new Logit model w/o collinearity:
logitRet_C1 <- glm(retention ~ school_MS + sex_M + age + address_U + famsize_LE3 + Pstatus_T +
                    Mjob_health + Mjob_other + Mjob_services + Mjob_teacher +
                    Medu + internet_yes + schoolsup_yes + famsup_yes + paid_yes +
                    activities_yes + romantic_yes + traveltime + studytime + failures + famrel +
                    freetime + goout + Dalc + health + absences +
                    G1 + Fedu + Walc,
                  data=dataC1, family=binomial(link="logit"))
summary(logitRet_C1)

# Get new VIF output:
vif(logitRet_C1)

# create vector of VIF values
vif_values <- vif(logitRet_C1)
#create horizontal bar chart to display each VIF value
barplot(vif_values, main = "VIF Values - No Collinearity", horiz = FALSE,
        col = "steelblue", cex.names = 0.8)
#add vertical line at 5
abline(h = 2.5, lwd = 2, lty = 2, col='red')


# Run new Logit model just with Statistical Significant variables:
dataC <- dataC1[,c("retention", "Dalc", "famsup_yes", "Medu", "Fedu", "schoolsup_yes")]
                  
logitRet_C <- glm(retention ~ Dalc + famsup_yes + Medu + Fedu + schoolsup_yes,
                  data=dataC, family=binomial(link="logit"))
summary(logitRet_C)


# Run new Logit model just with Statistical Significant variables at "0.05":
dataC <- dataC1[,c("retention", "Dalc", "famsup_yes", "Medu")]

logitRet_C <- glm(retention ~ Dalc + famsup_yes + Medu,
                  data=dataC, family=binomial(link="logit"))
summary(logitRet_C)

# Get new VIF output:
vif(logitRet_C)

# create vector of VIF values
vif_values <- vif(logitRet_C)
#create horizontal bar chart to display each VIF value
barplot(vif_values, main = "VIF Values - No Collinearity", horiz = FALSE,
        col = "steelblue", cex.names = 0.8)
#add vertical line at 5
abline(h = 0.99, lwd = 2, lty = 2, col='red')


# Create New Correlation Matrix:
dataM2 <- data.matrix(dataC)
# Correlation Matrix:
data_cor2 <- cor(dataM2)
correlations2 <- cor(data_cor2)
col <- colorRampPalette(c("#BB4444", "#EE9988", "#FFFFFF", "#77AADD", "#4477AA"))
corrplot(correlations2, method = "color", col = col(200),
         type = "upper", order = "hclust", number.cex = .6,
         addCoef.col = "black", # Add coefficient of correlation
         tl.col = "gray", tl.srt = 90, # Text label color and rotation
         # Combine with significance
         #p.mat = p.mat, sig.level = 0.01, insig = "blank", 
         # hide correlation coefficient on the principal diagonal
         diag = FALSE)

#Plot "kdepairs" Matrix: 
install.packages("ResourceSelection")
library(ResourceSelection) 
cor_dataA <- cor(dataA)
kdepairs(data_cor2)





##################################
### PHASE 4 - Machine Learning ###
##################################

ls(dataC) # check columns 
dim(dataC) # reduction on number of variables (from 28 to 24)
table(dataC$retention)

head(dataC)
# Make all variavles a factor:
dataC[sapply(dataC, is.numeric)] <- lapply(dataC[sapply(dataC, is.numeric)], as.factor)
                                           
str(dataC)
table(dataC$retention)

### Dealing with Imbalanced classification on the Regressor (retetion variable):
### Three options/techniques available: SMOTE, splitBalanced, and ROSE.

## Create a Reference Model (Imbalanced) for comparison with a set of Balanced Models:
library(caret)
set.seed(649)
splitIndex <- createDataPartition(dataC$retention, p = .85,
                                  list = FALSE,
                                  times = 1)
trainSplit <- dataC[splitIndex,]
nrow(trainSplit) # 553 students
table(trainSplit$retention) # 1= 458, 0=95
# check proportions:
prop.table(table(trainSplit$retention))

testSplit <- dataC[-splitIndex,] 
nrow(testSplit) # 96 students


# Check Performance of Imbalanced Model:
library(rpart)

treeimb <- rpart(retention ~ ., data = trainSplit)
pred.treeimb <- predict(treeimb, newdata = testSplit)
head(pred.treeimb)

accuracy.meas(testSplit$retention, pred.treeimb[,2])

roc.curve(testSplit$retention, pred.treeimb[,2], plotit = FALSE)


# Library to Balance the classifier:
install.packages("ROSE")
library(ROSE)

### Balancing the data:
# Resampling Option 1 (over):
data.bal.ov <- ovun.sample(retention ~ ., data = trainSplit, method = "over",
                           p=0.5, seed = 2)$data
table(data.bal.ov$retention)

# Resamplint Option 2 (under):
data.bal.un <- ovun.sample(retention ~ ., data = trainSplit, method = "under",
                           p = 0.5, seed = 1)$data
table(data.bal.un$retention)

# Resampling Option 3 (both):
data.bal.ou <- ovun.sample(retention ~ ., data = trainSplit, method = "both",
                           N = 553, p = 0.5, seed = 2)$data
table(data.bal.ou$retention)

# Resampling Option 4 (ROSE):
data.rose <- ROSE(retention ~ ., data = trainSplit, seed = 1)$data

table(data.rose$retention)

# Training the Classifiers and run test set using classification trees:
library(rpart)
tree.ov <- rpart(retention ~ ., data = data.bal.ov)
tree.un <- rpart(retention ~ ., data = data.bal.un)
tree.ou <- rpart(retention ~ ., data = data.bal.ou)
tree.rose <- rpart(retention ~ ., data = data.rose)

# Predict in the new data (test):
pred.tree.ov <- predict(tree.ov, newdata = testSplit)
pred.tree.un <- predict(tree.un, newdata = testSplit)
pred.tree.ou <- predict(tree.un, newdata = testSplit)
pred.tree.rose <- predict(tree.rose, newdata = testSplit)

# Plot ROC Curve - Model Evaluation:
roc.curve(testSplit$retention, pred.tree.rose[,2], col = 0, main= "AUC: 0.75", lty = 1)
roc.curve(testSplit$retention, pred.tree.ov[,2], add.roc = TRUE, col = 12, lty = 2) 
roc.curve(testSplit$retention, pred.tree.un[,2], add.roc = TRUE, col = 0, lty = 3) 
roc.curve(testSplit$retention, pred.tree.ou[,2], add.roc = TRUE, col = 0, lty = 4)



### Machine Learning Spot Checking:
dataC <- dataC1[,c("retention", "Dalc", "famsup_yes", "Medu")]
dataC[,'retention'] <- as.factor(as.character(dataC[,'retention']))

dim(dataC)

# Create balanced training data for LR:
dataC <- ovun.sample(retention ~ ., data = dataC, method = "both",
                        N = 1298, seed =2)$data # 1298 (649x2)
table(dataC$retention)
str(dataC)




#dataC[sapply(dataC, is.factor)] <- lapply(dataC[sapply(dataC, is.factor)], as.numeric)

str(dataC)

install.packages("mlbench")
library(mlbench)
library(caret)

# Set test options (10-fold cross validation with 3 repeats)
control <- trainControl(method="repeatedcv", number=10, repeats=3)
seed <- 7

# Set evaluation metric:
metric <- "Accuracy"

# Scale and Center the data:
preProcess=c("center", "scale")


# Run multiple ML models:
# Linear Discriminant Analysis
set.seed(seed)
fit.lda <- train(retention~., data=dataC, method="lda", metric=metric, preProc=c("center", "scale"), trControl=control)
# Logistic Regression
set.seed(seed)
fit.glm <- train(retention~., data=dataC, method="glm", metric=metric, trControl=control)
# GLMNET
set.seed(seed)
fit.glmnet <- train(retention~., data=dataC, method="glmnet", metric=metric, preProc=c("center", "scale"), trControl=control)
# SVM Radial
set.seed(seed)
fit.svmRadial <- train(retention~., data=dataC, method="svmRadial", metric=metric, preProc=c("center", "scale"), trControl=control, fit=FALSE)
# kNN
set.seed(seed)
fit.knn <- train(retention~., data=dataC, method="knn", metric=metric, preProc=c("center", "scale"), trControl=control)
# Naive Bayes
set.seed(seed)
fit.nb <- train(retention~., data=dataC, method="nb", metric=metric, trControl=control)
# CART
set.seed(seed)
fit.cart <- train(retention~., data=dataC, method="rpart", metric=metric, trControl=control)
# C5.0
set.seed(seed)
fit.c50 <- train(retention~., data=dataC, method="C5.0", metric=metric, trControl=control)
# Bagged CART
set.seed(seed)
fit.treebag <- train(retention~., data=dataC, method="treebag", metric=metric, trControl=control)
# Random Forest
set.seed(seed)
fit.rf <- train(retention~., data=dataC, method="rf", metric=metric, trControl=control)
# Stochastic Gradient Boosting (Generalized Boosted Modeling)
set.seed(seed)
fit.gbm <- train(retention~., data=dataC, method="gbm", metric=metric, trControl=control, verbose=FALSE)


# Select Model:
results <- resamples(list(lda=fit.lda, logistic=fit.glm, glmnet=fit.glmnet,
                          svm=fit.svmRadial, knn=fit.knn, nb=fit.nb, cart=fit.cart, c50=fit.c50,
                          bagging=fit.treebag, rf=fit.rf, gbm=fit.gbm))
# Table comparison
summary(results)



# boxplot comparison
bwplot(results)
# Dot-plot comparison
dotplot(results)










#### RESOURCES:
# source: https://archive.ics.uci.edu/ml/datasets/student+performance
# source: https://www.kaggle.com/rafjaa/dealing-with-very-small-datasets
# source: https://www.rdocumentation.org/packages/DMwR/versions/0.4.1/topics/SMOTE
# source: https://rpubs.com/abhaypadda/smote-for-imbalanced-data
# source: https://cran.r-project.org/web/packages/corrplot/vignettes/corrplot-intro.html
# source: https://www.datacamp.com/community/tutorials/pca-analysis-r
# source: http://www.sthda.com/english/wiki/factoextra-r-package-easy-multivariate-data-analyses-and-elegant-visualization
# source: https://www.statology.org/how-to-easily-calculate-variance-inflation-factor-vif-in-r/
# source: https://www.rdocumentation.org/packages/DMwR/versions/0.4.1/topics/SMOTE
# source: https://rpubs.com/abhaypadda/smote-for-imbalanced-data
# source: https://journal.r-project.org/archive/2014-1/menardi-lunardon-torelli.pdf
# source: http://www.cookbook-r.com/Graphs/Facets_(ggplot2)/
# source: https://rkabacoff.github.io/datavis/Bivariate.html
# source: https://bradleyboehmke.github.io/HOML/logistic-regression.html
# source: https://pdfs.semanticscholar.org/ff48/e76d3fd0adcd16d123465d273eaed0ea58fe.pdf
# source: https://machinelearningmastery.com/evaluate-machine-learning-algorithms-with-r/
