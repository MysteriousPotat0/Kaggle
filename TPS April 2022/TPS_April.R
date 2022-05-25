#### Loading Libraries and neural network functions####
library(keras)
library(tensorflow)
library(dplyr)
library(zoo)
library(data.table)
library(pROC)
library(moments)
library(xgboost)
library(nloptr)
source('backpropagation.R')
source('cost_function.R')
source('initializing_weight.R')
source('sigmoid_function.R')
source('prediction.R')
source('load_theta.R')
set.seed(1)

#### Reading the training data ####
training <- fread('train.csv', header = T, sep = ',')
train_label <- fread('train_labels.csv', header = T, sep = ',')


#### Feature selection functions ####
variance <- function(x){
  
  # Calculates the sample variance
  
  1/(dim(training)[1]-1) * (sum(x-mean(x)))^2
  
}

# Calculates the standard deviation
std_dev <- function(x){
  
  sqrt(variance(x))
  
}

# Calculates the quantile
quant <- function(x){

  quan <- quantile(x, probs = c(0.01, 0.25, 0.75, 0.9, 0.99))

  cbind(quan[1], quan[2], quan[3], quan[4], quan[5])

}

# Normalizes the data
normalization <- function(x){
  
  
  (x-min(x, na.rm = T))/(max(x, na.rm = T)-min(x, na.rm = T))
  
}




#### Aggregating Data ####
# Aggregate the data into by mean for each column
training <- training %>% group_by(sequence) %>% summarize_at(vars(names(training)[c(4:16)]),
                                                             list(min = min, 
                                                                  median = median, 
                                                                  mean = mean, 
                                                                  max = max, 
                                                                  var = variance, 
                                                                  std = std_dev,
                                                                  skewness = skewness,
                                                                  kurtosis = kurtosis,
                                                                  quantile = quant))

training <- as.data.table(training)

#### Scaling all the variables ####
training <- training %>% mutate_at(c(4:16), normalization)

# Adding the classification labels onto training data
training$state <- train_label$state

which(is.na(training), arr.ind = T) # Check for NA
training <- training[, -1] # Remove sequence
training <- na.omit(training) # Omit NA rows

# Divide the data into 70/20/10 train/validate set/test
sample <- sample.int(nrow(training), 0.7*nrow(training), replace = F)
sample_training <- training[sample, ] 

# Divide into validation and test set
new_train <- training[-sample, ]
sample <- sample.int(nrow(new_train), 0.9*nrow(new_train), replace = F)

# Validation set
validate <- as.matrix(new_train[sample, ])

validate_test <- as.matrix(validate[,1:(ncol(validate)-1)])

validate_test <- array(validate_test, dim = c(dim(validate_test)[1], 1, dim(validate_test)[2]))

# Test set
test <- new_train[-sample, ]


#### DIY Neural Net ###

# Input variables
x <- as.matrix(sample_training[,1:(ncol(sample_training)-1)])
m <- dim(x)[1]

# Setting up the output variable
y <- sample_training$state

# Setting up the input, hidden layers, and output
input <- dim(x)[2]
hidden_layer <- 250
hidden_layer2 <- 250
output <- 1
lambda = 1e-5

# Setting up the theta for the neural network
theta1 <- initial_weights(input, hidden_layer)
theta2 <- initial_weights(hidden_layer, hidden_layer2)
theta3 <- initial_weights(hidden_layer2, output)
Theta <- cbind(c(theta1, theta2, theta3))

# Size of the parameters
theta1_size <- dim(theta1)[1] * dim(theta1)[2] # Set the size of input layer weights
theta2_size <- dim(theta2)[1] * dim(theta2)[2] # Set the size of hidden layer weights
theta3_size <- dim(theta3)[1] * dim(theta3)[2] # Set the size of hidden layer weights

# Options for the nloptr function
options <- list('algorithm' = 'NLOPT_LD_LBFGS', 'xtol_rel' = 1e-8,
                'maxeval' = 1)

# Finds the minimum of theta using cost function and gradient descent
Theta <- nloptr(Theta, eval_f = cost, eval_grad_f = grad, opts = options)

# Update the weights of NN
Theta <- matrix(Theta$solution)


# save_theta(Theta)


# Theta <- load_theta('Theta.csv')

# Calculates the cost between the train and validation set
train_cv_error(Theta, x, validate[, 1:ncol(validate)-1], y, 
               validate[,ncol(validate)])


# Predicts the validation set using trained weight of neural network
p <- pred(Theta, validate[,1:ncol(validate)-1])


# Create an ROC plot for the neural net
roc(validate[,ncol(validate)], p, plot = T, percent = T,
    xlab = "False Positive %", ylab = "True Positive %", col = 'red', 
    lwd = 4, legacy.axes = T, print.auc = T)


# If probability greater than 50%, replace with 1, otherwise 0
p <- ifelse(p >= 0.5, 1, 0)


# Check the error rate of the neural net on the validation set
1-mean((validate[,ncol(validate)])==p)



#### XGB Model ####

# Fit the XGB model
xgb_model <- xgboost(data = x, label = y, nrounds = 250, max.depth = 10, lambda = 1,
                     early_stopping_rounds = 5, objective = 'binary:logistic')

# Predict the validation using the trained XGB model
xgb_pred <- predict(xgb_model, data.matrix(validate[,1:ncol(validate)-1]))

# Plot the ROC curve of XGB model against neural net
plot.roc(validate[,ncol(validate)], xgb_pred, percent = T,
         xlab = "False Positive %", ylab = "True Positive %", col = 'yellow', 
         lwd = 4, legacy.axes = T, print.auc = T, add = T, print.auc.y = 20)

# Measure the accuracy of the XGB model
1-mean(validate[,ncol(validate)] == ifelse(xgb_pred >= 0.5, 1, 0))

# Combine the XGB and neural net model
# Take the average of their result
model_validate <- cbind(validate[,ncol(validate)], xgb_pred, t(p))
model_validate <- cbind(model_validate, apply(model_validate[,2:3], 1, FUN = mean))

# Find the error rate of the combined result
1-mean(validate[,ncol(validate)] == ifelse(model_validate[,4] >= 0.5, 1, 0))


# Plotting feature importance
xgb_matrix <- xgb.DMatrix(data = x, label = y)
xgb.plot.importance(xgb.importance(colnames(xgb_matrix), model = xgb_model), 
                    col = rainbow(50))




#### Preparing to predict the test set ####
Test <- fread('test.csv', header = T, sep = ',')


# Aggregate the data into by mean for each column
Test <- Test %>% group_by(sequence) %>% summarize_at(vars(names(Test)[c(4:16)]),
                                                             list(min = min, 
                                                                  median=median, 
                                                                  mean=mean, 
                                                                  max=max, 
                                                                  var = variance, 
                                                                  std = std_dev,
                                                                  skewness = skewness,
                                                                  kurtosis = kurtosis))%>% as.data.table()


Test <- Test %>% mutate_at(c(4:16), normalization) 

# Remove steps and subjects
sequences <- Test$sequence
Test <- Test[, -1]

# Set all NA to 0
Test[is.na(Test)] <- 0


# Make the prediction 
nn_final <- pred(Theta, Test) # Using neural net
xgb_final <- predict(xgb_model, data.matrix(Test)) # Using XGB

# Combine the result of neural net and xgb model
combined_model <- cbind(t(nn_final), xgb_final)

# Taking the average of the model result
combined_model <- cbind(combined_model, apply(combined_model, 1, mean))
submission <- ifelse(combined_model[,1] >=  0.5, 1, 0)

# Writes the submission file in .csv
submission <- cbind(sequences, submission)
colnames(submission) <- c('sequence', 'state')
write.csv(submission, 'submission.csv', row.names = F)









