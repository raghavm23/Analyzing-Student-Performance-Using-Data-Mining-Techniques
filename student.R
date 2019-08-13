
#1 school - student's school (binary: "GP" - Gabriel Pereira or "MS" - Mousinho da Silveira)
#2 sex - student's sex (binary: "F" - female or "M" - male)
#3 age - student's age (numeric: from 15 to 22)
#4 address - student's home address type (binary: "U" - urban or "R" - rural)
#5 famsize - family size (binary: "LE3" - less or equal to 3 or "GT3" - greater than 3)
#6 Pstatus - parent's cohabitation status (binary: "T" - living together or "A" - apart)
#7 Medu - mother's education (numeric: 0 - none,  1 - primary education (4th grade), 2 ??? 5th to 9th grade, 3 ??? secondary education or 4 ??? higher education)
#8 Fedu - father's education (numeric: 0 - none,  1 - primary education (4th grade), 2 ??? 5th to 9th grade, 3 ??? secondary education or 4 ??? higher education)
#9 Mjob - mother's job (nominal: "teacher", "health" care related, civil "services" (e.g. administrative or police), "at_home" or "other")
#10 Fjob - father's job (nominal: "teacher", "health" care related, civil "services" (e.g. administrative or police), "at_home" or "other")
#11 reason - reason to choose this school (nominal: close to "home", school "reputation", "course" preference or "other")
#12 guardian - student's guardian (nominal: "mother", "father" or "other")
#13 traveltime - home to school travel time (numeric: 1 - <15 min., 2 - 15 to 30 min., 3 - 30 min. to 1 hour, or 4 - >1 hour)
#14 studytime - weekly study time (numeric: 1 - <2 hours, 2 - 2 to 5 hours, 3 - 5 to 10 hours, or 4 - >10 hours)
#15 failures - number of past class failures (numeric: n if 1<=n<3, else 4)
#16 schoolsup - extra educational support (binary: yes or no)
#17 famsup - family educational support (binary: yes or no)
#18 paid - extra paid classes within the course subject (Math or Portuguese) (binary: yes or no)
#19 activities - extra-curricular activities (binary: yes or no)
#20 nursery - attended nursery school (binary: yes or no)
#21 higher - wants to take higher education (binary: yes or no)
#22 internet - Internet access at home (binary: yes or no)
#23 romantic - with a romantic relationship (binary: yes or no)
#24 famrel - quality of family relationships (numeric: from 1 - very bad to 5 - excellent)
#25 freetime - free time after school (numeric: from 1 - very low to 5 - very high)
#26 goout - going out with friends (numeric: from 1 - very low to 5 - very high)
#27 Dalc - workday alcohol consumption (numeric: from 1 - very low to 5 - very high)
#28 Walc - weekend alcohol consumption (numeric: from 1 - very low to 5 - very high)
#29 health - current health status (numeric: from 1 - very bad to 5 - very good)
#30 absences - number of school absences (numeric: from 0 to 93)

#######################################################
# EDA
#######################################################

rm(list=ls())

library(mlbench)
library(caret)

setwd('/Users/Raghav/Downloads/student')
data <- read.csv("student.csv", sep=",")

#converting integer to factor
data$Fedu = factor(data$Fedu)
data$traveltime = factor(data$traveltime)
data$Medu = factor(data$Medu)
data$Walc = factor(data$Walc)
data$studytime = factor(data$studytime)
data$failures = factor(data$failures)
data$freetime = factor(data$freetime)
data$famrel = factor(data$famrel)
data$goout = factor(data$goout)
data$Dalc = factor(data$Dalc)
data$health = factor(data$health)
str(data)

#removing G3
data$G3 = NULL

#ordering the DV levels
levels(data$G3.grade)
data$G3.grade<-factor(data$G3.grade,levels(data$G3.grade)[c(3,1,4,2)])
levels(data$G3.grade)


#######################################################
# FEATURE SELECTION USING RECURSIVE FEATURE ELIMINATION
#######################################################
set.seed(2018)

# define  control using a RF selection function
control <- rfeControl(functions=rfFuncs, method="cv", number=10)
# run RFE algorithm
results <- rfe(data[,1:33], data[,34], sizes=c(1:33), rfeControl=control)
# summarize the results
print(results)
# list the chosen features
predictors(results)
top = c("G2","failures","higher","course","absences","G3.grade")
# plot the results
plot(results, type=c("g", "o"))


#######################################################
# CORRELATION AMONGST CONTINUOUS VARIABLES
#######################################################

# calculate correlation matrix
numeric.var <- sapply(data, is.numeric)
abc = cor(data[,numeric.var])
corrplot(abc, main="\n\nCorrelation Plot for Independent Variables", method="number")
print(abc)
# removing the correlated variables
data$Medu = NULL
data$Walc = NULL
str(data)

#######################################################
# RANK OF IMPORTANT FEATURES
#######################################################

# prepare training scheme
control <- trainControl(method="cv", number=10, repeats=5)
# train the model
model <- train(G3.grade~.,data=data, method="lvq", preProcess="scale", trControl=control)
# estimate variable importance
importance <- varImp(model, scale=FALSE)
# summarize importance
print(importance)
# plot importance
plot(importance)
str(data)

#######################################################
# C5.0 MODEL
#######################################################

library(C50)

###### #model 1 - both G1 & G2
index <- sample(1:dim(data)[1], dim(data)[1] * .75, replace=FALSE)
training <- data[index, ]
testing <- data[-index, ]

cFiftyModel = C5.0(G3.grade ~ ., data=training, trials=1)
c <- predict(cFiftyModel, testing[,-32])

caret::confusionMatrix(c, testing$G3.grade, positive="Fail")
  (cFiftyModelAccuracy = (1-mean(c != testing$G3.grade)))

#winnowing but does not improve performance
cFiftyWinnow <- C5.0(G3.grade ~ ., data = training, control = C5.0Control(winnow = TRUE))
c <- predict(cFiftyWinnow, testing[,-32])
caret::confusionMatrix(c, testing$G3.grade, positive="Fail")
(cFiftyModelAccuracy = (1-mean(c != testing$G3.grade)))

#Brute force optimization on the basis of Kappa
control <- trainControl(method="repeatedcv", number=10, repeats=5) #5 x 10-fold cv
metric <- "Kappa"
train(G3.grade~., data=data, method="C5.0", metric=metric, trControl=control)

###### model 2 - only G2
data$G1 = NULL
index <- sample(1:dim(data)[1], dim(data)[1] * .75, replace=FALSE)
training <- data[index, ]
testing <- data[-index, ]


cFiftyModel = C5.0(G3.grade ~ ., data=training, trials=10)
c <- predict(cFiftyModel, testing[,-31])

caret::confusionMatrix(c, testing$G3.grade, positive="Fail")
(cFiftyModelAccuracy = (1-mean(c != testing$G3.grade)))

###### model 3 - top 5
data = data[,top]
index <- sample(1:dim(data)[1], dim(data)[1] * .75, replace=FALSE)
training <- data[index, ]
testing <- data[-index, ]


cFiftyModel = C5.0(G3.grade ~ ., data=training, trials=10)
c <- predict(cFiftyModel, testing[,-6])

caret::confusionMatrix(c, testing$G3.grade, positive="Fail")
(cFiftyModelAccuracy = (1-mean(c != testing$G3.grade)))

#######################################################
# DECISION TREE
#######################################################

library(rpart)
library(rpart.plot)
library(RColorBrewer)
regressionTree <- rpart(G3.grade ~ ., data=training, method="class")
plot(regressionTree)
text(regressionTree)
library(rattle)
fancyRpartPlot(regressionTree)
newRpart <- rpart(G3.grade ~ ., data=training, method="class", control=rpart.control(minsplit=2, cp=0))
fancyRpartPlot(newRpart)

rpartPrediction <- predict(regressionTree, training, type = "class")
confusionMatrix(rpartPrediction, training$G3.grade, positive = "Fail")
rpartPrediction <- predict(newRpart, training, type = "class")
confusionMatrix(rpartPrediction, training$G3.grade, positive = "Fail")

#######################################################
# RANDOM FOREST
#######################################################

library(randomForest)
forest <- randomForest(G3.grade ~ ., data=training, importance=TRUE, ntree=2000)
varImpPlot(forest)
rf <- predict(forest, testing, type = "class")
confusionMatrix(rf, testing$G3.grade, positive = "Yes")

library(partykit)
cTree <- ctree(G3.grade ~., data=training)
print(cTree)
plot(cTree, type="simple")

#######################################################
# PLOTTING ##check source
#######################################################

library(dplyr)
library(ggplot2)
library(viridis)
df_count<-data[,sapply(data,is.integer)]
df_factor<-data[,sapply(data,is.factor)]
glimpse(df_count)
glimpse(df_factor)


plothist<-function(data_a,i){
  data2<-data.frame(x=data_a[[i]])
  a<-ggplot(data2,aes(x))+
    geom_histogram(col="black",fill="white",binwidth = 5)+
    labs(x=names(data_a)[i])+
    theme(axis.text.x =  element_text(angle = 90,hjust = 1,vjust = .5))
  return(a)
}

plotbar<-function(data_a,i){
  data2<-data.frame(x=data_a[[i]])
  a<-ggplot(data2,aes(x,fill=x))+
    geom_bar()+
    labs(x=names(data_a)[i])+
    theme(axis.text.x =  element_text(angle = 90,hjust = 1,vjust = .5),
          legend.position = "none")
  return(a)
}

plotbox<-function(data_a,i){
  data2<-data.frame(x=data_a[[i]])
  a<-ggplot(data2,aes(df$class,x))+
    geom_boxplot(col="black",fill="white",binwidth = 5)+
    labs(x="class",y=names(data_a)[i])+
    coord_flip()
  return(a)
}

plotmosaic<-function(data_a,i){
  data2<-df%>%
    group_by(x=data_a[[i]],class)%>%
    summarise(n=n())%>%
    mutate(prob=signif(n/sum(n),3))
  
  
  a<-ggplot(data2,aes(class,x,fill=n))+
    geom_tile(col="white")+
    scale_fill_viridis()+
    geom_text(aes(label=n),col="white")+
    labs(y=names(df_factor)[i])+
    theme(legend.position = "none")
  return(a)
}

plotmosaic2<-function(data_a,i){
  data2<-df%>%
    group_by(x=data_a[[i]],class)%>%
    summarise(n=n())%>%
    mutate(prob=signif(n/sum(n),3))
  
  
  a<-ggplot(data2,aes(class,x,fill=prob))+
    geom_tile(col="white")+
    scale_fill_viridis()+
    geom_text(aes(label=prob),col="white")+
    labs(y=names(df_factor)[i])+
    theme(legend.position = "none")
  return(a)
}

doPlots <- function(data_a, fun, ii, ncol=3) {
  pp <- list()
  for (i in ii) {
    p <- fun(data_a=data_a, i=i)
    pp <- c(pp, list(p))
  }
  do.call("grid.arrange", c(pp, ncol=ncol))
}


doPlots2 <- function(data_a, fun,fun2, ii, ncol=3) {
  pp <- list()
  for (i in ii) {
    p <- fun(data_a=data_a,i=i)
    p1<-fun2(data_a=data_a,i=i)
    pp <- c(pp, list(p),list(p1))
  }
  do.call("grid.arrange", c(pp, ncol=ncol))
}

doPlots(df_count,fun = plothist,ii = 1:3,ncol = 2)
doPlots(df_factor,fun=plotbar,ii=1:4,ncol = 2)
doPlots(df_factor,fun=plotbar,ii=5:8,ncol = 2)
doPlots(df_factor,fun=plotbar,ii=9:12,ncol = 2)
doPlots(df_factor,fun=plotbar,ii=13:16,ncol = 2)
doPlots(df_factor,fun=plotbar,ii=17:21,ncol = 2)
doPlots(df_factor,fun=plotbar,ii=21:25,ncol = 2)
doPlots(df_factor,fun=plotbar,ii=25:28,ncol = 2)



#########################################################################################################

