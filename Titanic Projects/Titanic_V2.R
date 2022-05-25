library(gridExtra)
library(ggplot2)
library(tidyverse)
library(data.table)
library(xgboost)
library(pROC)
set.seed(1)

# Loading the training set
train <- fread('train.csv', header = T, sep = ',')

#### Plot ####

# plot the male and female survival
plot1 <- train %>% group_by(Sex, Survived) %>% summarise(Yes = sum(Survived==1)) %>% ggplot() + 
  geom_col(aes(Sex, Yes, fill = Yes))
plot2 <- train %>% group_by(Sex, Survived) %>% summarise(No = sum(Survived==0)) %>% ggplot() + 
  geom_col(aes(Sex, No, fill = No))
grid.arrange(plot1, plot2)

# Plot the Age against survival
plot1 <- train %>% group_by(Age, Survived) %>% summarise(Yes = sum(Survived==1)) %>% ggplot() + 
  geom_col(aes(Age, Yes, fill = Yes))
plot2 <- train %>% group_by(Age, Survived) %>% summarise(No = sum(Survived==0)) %>% ggplot() + 
  geom_col(aes(Age, No, fill = No))
grid.arrange(plot1, plot2)

# Plot the Age aginst survival
plot1 <- train %>% group_by(Pclass, Survived) %>% summarise(Yes = sum(Survived==1)) %>% ggplot() + 
  geom_col(aes(Pclass, Yes, fill = Yes))
plot2 <- train %>% group_by(Pclass, Survived) %>% summarise(No = sum(Survived==0)) %>% ggplot() + 
  geom_col(aes(Pclass, No, fill = No))
grid.arrange(plot1, plot2)


# Plot the Fare against Pclass
train %>% group_by(Pclass, Fare) %>% summarise(sum = sum(Pclass)) %>% ggplot() + 
  geom_bar(stat = 'identity', aes(Pclass, sum, fill = Fare))

# Compare different gender with # of Parch
ggplot(train, aes(Sex, Parch, fill = Parch)) + 
  geom_bar(stat = 'identity', position = 'dodge')




# Creates a function that normalizes with min-max
normalization <- function(x){
  
  (x-min(x))/(max(x)-min(x))
  
}





#### Pre-Processing for Logistic Regression ####
train <- fread('train.csv', header = T, sep = ',')

# Remove cabin, too many missing value
# Remove ticket as well
# Remove Passenger-ID
# Remove Fare
train <- train[,-c(1, 4, 9:11)]

# Replace missing age number with some arbitrary big value
train[is.na(train)] <- 100 # Odds of a person that is 200 years old is very very low

# Applies normalization to the training set
train <- train %>% mutate_at(c(2, 4:6), list(normalization))

# Setting Pclass, gender, SibSP, Parch, Embarked as factor
train$Pclass <- factor(train$Pclass, ordered = T)
train$SibSp <- factor(train$SibSp, ordered = T)
train$Parch <- factor(train$Parch, ordered = T)
train$Embarked <- factor(train$Embarked, labels = c(1, 2, 3, 4))
train$Sex <- factor(train$Sex, labels = c(0, 1))






sample <- sample.int(nrow(train), 5000, replace = T)
sample_train <- train[sample,]

# Split into 80/20 train/validate
sample <- sample.int(nrow(sample_train), 0.8*nrow(sample_train), replace = F)
new_train <- sample_train[sample,]
validate <- sample_train[-sample,]


# Building the logistic regression
logistic <- glm(new_train$Survived~., data = new_train, family = binomial(link = 'logit'))

pred_logit <- predict(logistic, validate)


# ROC for logit
roc(validate$Survived, pred_logit, plot = T, percent = T,
    xlab = "False Positive %", ylab = "True Positive %", col = rainbow(1), 
    lwd = 4, legacy.axes = T, print.auc = T)




#### Pre-Processing for ANN ####
train <- fread('train.csv', header = T, sep = ',')
# Check for NA in the age group
which(is.na(train), arr.ind = T)

# Find the column name for the 6th column
colnames(train[,6])

# Creates one hot encoding for Pclass
train <- train %>% mutate(value = 1) %>% spread(Pclass, value, fill = 0)

# Replace missing age number with some arbitrary big value
train[is.na(train)] <- 100 # Odds of a person that is 200 years old is very very low

# Creates binary labels for gender
train$Sex <- ifelse(train$Sex == 'male', 1, 0)

# Remove cabin, too many missing value
# Remove ticket as well
# Remove Passenger-ID
# Remove Fare
train <- train[,-c(1, 3, 8:11)]

# Applies normalization to the training set
train <- train %>% mutate_at(c(3:5), list(normalization))


#### XGBoost ####
sample <- sample.int(nrow(train), 5000, replace = T)
sample_train <- train[sample,]

# Split into 80/20 train/validate
sample <- sample.int(nrow(sample_train), 0.8*nrow(sample_train), replace = F)
new_train <- sample_train[sample,]
validate <- sample_train[-sample,]


x <- as.matrix(new_train[,-1])
y <- as.matrix(new_train[, 1])



# Building XGBoost model
xgb_model <- xgboost(data = x,
                     label = y,
                     nrounds = 100,
                     lambda = 100,
                     early_stopping_rounds = 3)


xgb_validate <- predict(xgb_model, as.matrix(validate))

# Plot the ROC curve of XGB model against neural net
plot.roc(validate$Survived, xgb_validate, percent = T,
         xlab = "False Positive %", ylab = "True Positive %", col = 'yellow', 
         lwd = 4, legacy.axes = T, print.auc = T, add = T, print.auc.y = 20)

1-mean(validate$Survived == ifelse(xgb_validate >= 0.5, 1, 0))





Test <- fread('test.csv', header = T, sep = ',')

passid <- Test$PassengerId

# Creates one hot encoding for Pclass
Test <- Test %>% mutate(value = 1) %>% spread(Pclass, value, fill = 0)

# Replace missing age number with some arbitrary big value
Test[is.na(Test)] <- 100 # Odds of a person that is 200 years old is very very low

# Creates binary labels for gender
Test$Sex <- ifelse(Test$Sex == 'male', 1, 0)

# Remove cabin, too many missing value
# Remove ticket as well
# Remove Passenger-ID
# Remove Fare
Test <- Test[,-c(1, 2, 7:10)]

# Applies normalization to the Testing set
Test <- Test %>% mutate_at(c(3:5), list(normalization))

pred <- predict(xgb_model, as.matrix(Test))

pred <- ifelse(pred >= 0.5, 1, 0)

submission <- cbind(passid, pred)

colnames(submission) <- c('PassengerId', 'Survived')

write.csv(submission, 'submission.csv', row.names = F)



