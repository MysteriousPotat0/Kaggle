# Kaggle TPS April with XGBoost, DIY Neural Network, Bi-directional LSTM
## Introdution
Kaggle April 2022 includes data from 13 bio-sensors, for each participant. Participant have 60 seconds of observation, and the task is to predict which of the two state a given sequence will take on.  
  
  
### Training set  
The training set contains the following variables:  
**Sensors**: 0-12 sensors  
**Subjects**: 671 subjects  
**Sequence**: 25967 sequence  
**Steps**: 0-59 seconds  
  
### Train labels  
The train labels set contains the following variables:  
**Sequence**: 25967 sequence  
**State**: 0 or 1  
  
### Test set 
The test set contains the following variables:   
**Sensors**: 0-12 sensors  
**Subjects**: 318 subjects  
**Sequence**: 12217 sequence  
**Steps**: 0-59 seconds  
  
Goal is to predict the state of each sequence in the test set

## Plots  
The following plots is plotting every sensor by their value
![Rplot001](https://user-images.githubusercontent.com/101752427/166068313-cb504e49-3bb9-4852-8b73-5839a1c12ba3.png)
![Rplot002](https://user-images.githubusercontent.com/101752427/166068343-bd7cdee1-b322-43b6-8e0f-d599010eb60e.png)
![Rplot003](https://user-images.githubusercontent.com/101752427/166068353-d246666c-4ed4-4aa6-8619-921b9aa2416a.png)
![Rplot004](https://user-images.githubusercontent.com/101752427/166068358-13db7278-6b06-4251-9d33-e57aeb5d4544.png)
![Rplot005](https://user-images.githubusercontent.com/101752427/166068392-b801576d-3c7f-403e-a6c1-87efd873ef9e.png)
  
## Preprocessing  
  
The preprocessing step will be different for the Neural Network + XGB model and the Bi-directional LSTM.  
  
### For XGB and Neural Network  
  
XGB and the neural network will use aggregated data to train. Since there are 13 sensors in total, each feature will generate 13 new columns.  
  
This function takes in input of x and calculates the sample variance.
```r
variance <- function(x){
  
  # Calculates the sample variance
  
  1/(dim(training)[1]-1) * (sum(x-mean(x)))^2
  
}
```  
This function takes in input of x and calculates the standard deviation.
```r
std_dev <- function(x){
  
  sqrt(variance(x))
  
}
```
This function takes in input of x and calculates the different quantiles.
```r
quant <- function(x){
  
  quan <- quantile(x, probs = c(0.01, 0.25, 0.75, 0.9, 0.99))
  
  cbind(quan[1], quan[2], quan[3], quan[4], quan[5])
  
}
```  
This function takes in input of x and normalizes the data between 0-1.
```r
normalization <- function(x){
  
  
  (x-min(x, na.rm = T))/(max(x, na.rm = T)-min(x, na.rm = T))
  
}
```
The training data is then aggregated into one row of each unique sequence using the following command.
```r
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
```
This results in a training data set with 170 column.  
  
### For Bi-directional LSTM  
  
The preprocessing method for Bi-directional LSTM takes a different approach. Instead of aggregating all time series data into a single row, we can take use the full 60 seconds time series data for the LSTM model.  
  
This command groups first groups everything by sequence number, then it applies the lag function that shifts every row up by 1 for the 13 sensor columns.
```r
group_by(sequence) %>% mutate_at(c(4:ncol(tf_test)), list(lag1 = lag)) 
```
This command groups everything by sequence, then it finds the different between current and the previous rows.
```r
tf_test <- tf_test %>% group_by(sequence) %>% mutate_at(c(4:16), funs(diff = . - lag(.)))
```
This command groups everything by sequence, then applies the rolling mean, rolling max, rolling median function for all the time series data points.
```r
tf_test <- tf_test %>% group_by(sequence) %>% mutate_at(c(4:16), list(roll_mean = roll_mean,
                                                                      roll_max = roll_max,
                                                                      roll_med = roll_med))
```
```r
tf_test <- tf_test %>% mutate_at(c(4:ncol(tf_test)), normalization) %>% as.data.table()
```
After all the previous feature engineering steps, min-max normalization is applied to make all value range between 0-1.  
  
## Modeling using XGB and DIY Neural Network  
  
First, the training set is divided into 70/20/10 for the train/validate/test set. 
```r
# Divide the data into 70/20/10 train/validate set/test
sample <- sample.int(nrow(training), 0.7*nrow(training), replace = F)
sample_training <- training[sample, ]

# Divide into validation and test set
new_train <- training[-sample, ]
sample <- sample.int(nrow(new_train), 0.9*nrow(new_train), replace = F)

# Validation set
validate <- new_train[sample, ]

# Test set
test <- new_train[-sample, ]
```
### DIY Neural Network  
  
The DIY neural network uses the function in the backpropagaion, and the cost files to calculate the cost and update the weight. The cost function work by doing a forward feed onto the neural network, and producing an ouput. This output it then used in a cost function to calculate the cost. 
```r
 # Implementing the cost function with regularization
  (1/m)*(sum((-y * log(h)) - ((1-y)*log(1-h)))) + (lambda/(2*m))*(sum(theta1[, 2:ncol(theta1)]^2) + sum(theta2[, 2:ncol(theta2)]^2) + 
                                                                    sum(theta3[,2:ncol(theta3)]^2))
```
This is the same formula for binary logistic function with regularization.  
The back propagation works by doing a forward feed, then backpropagates through the different layers to find the error. 
```r
  delta_4 <- h - y
  
  delta_3 <- (theta3[2:ncol(theta3)]) %*% delta_4 * sig_grad(z3)
  
  delta_2 <- (theta2[,2:ncol(theta2)]) %*% delta_3 * sig_grad(z2)
```
The delta's are used to update the weight of each layers every iteration.  
  
Now, it is time to set up the neural network layers and number of neural units.
```r
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
```
The way that the weight are initialized is by using random uniform function in R. This generates a random number with equal probability.
```r
# Initialized the weight each theta through random uniform distribution
initial_weights = function(in_layer, out_layer){
  
  eps = 0.1

  rand_num <- runif(out_layer*(in_layer+1))
  
  rand_num <- (rand_num * 2 * eps) - eps 
  
  result <- matrix(rand_num, out_layer, (in_layer+1))
}
```
Other than just randomly initializing some weight for neural network to begin, it also has a process of breaking symmtery of neural network. If all the weights are the same, then after each backprogation, weights does not change. This is aimed to inhibit, or reduce the probability of that happening.  
  
Neural network model cost function is optimized by using the package nloptr. This is a package that can take a cost, and graident function, and minimizes the cost function using the gradient function.  
The algorithm used is the Low-storage Broyden-Fletcher-Goldfarb-Shanno method. This is similar to BFGS method except it saves more memory by not saving the entire nxn matrix. The algorith is an iterative method for solving unconstraint nonlinear problems.  
The relative tolerance is set to 1x10^-8, so it means if the difference between cost is less than that, the iteration method stops.  
The number of times it evaluates function is set to 100.
```r
# Options for the nloptr function
options <- list('algorithm' = 'NLOPT_LD_LBFGS', 'xtol_rel' = 1e-8,
                'maxeval' = 100)

# Finds the minimum of theta using cost function and gradient descent
Theta <- nloptr(Theta, eval_f = cost, eval_grad_f = grad, opts = options)

# Update the weights of NN
Theta <- matrix(Theta$solution)
```
  
### XGBoost  
  
XGboost stands for extreme gradient boosting, it takes a bunch of weak learners (trees), and combine them to make a strong model. The weak models themselves have high bias and low variance, meaning they tend to underfit the data. Thus, combining their result, can result in a stronger model compared to single weak learner. Every iteration of boosting process, it is aimed to reduce the error of the previous tree.  
```r
# Fit the XGB model
xgb_model <- xgboost(data = x, label = y, nrounds = 250, max.depth = 10, lambda = 1,
                     early_stopping_rounds = 5, objective = 'binary:logistic')
```
Here, the model is trained to fit on x and use y as verification. This is basically supervised learning. The maximum number of times the model can iterate is 200 times. Each trees can have a max depth of 10 leaves. Lamdba is the regularization term that reduces/prevents overfitting. If there are no improvement to the cost in 5 iterations, the model stops.  
  
### Bi-directional LSTM  
  
The LSTM model is a type of Recurrent Neural Network. It works by having input of the features + the input from the previous nodes. Therefore, it is helpful in situation where the previous input matter like in time series. 
  
Bi-LSTM works similar way, it has a forward feed mechanism, but it also have a backpropagation method similar to MLP type neural network. Each layers have forward feed that makes and ouput, then it has a backward feed that also makes an output.  
The model used will have a structure like this
![Rplot](https://user-images.githubusercontent.com/101752427/166114544-a912b7b5-4942-474a-8739-da183c7f38fd.png)
  
Input is the (# of total rows, 60, # of features) array. The input is then fed into the first layer containing 128 neural units, but because it is Bi-directional, the layer actually consists of 128x2 = 256 units. 
```r
tf_model <- keras_model_sequential()
tf_model %>%
  bidirectional(layer_lstm(input_shape = dim(tf_array)[2:3], units = 128, return_sequences = T)) %>%
  bidirectional(layer_lstm(units = 64, return_sequences = T)) %>%
  bidirectional(layer_lstm(units = 64, return_sequences = T)) %>%
  bidirectional(layer_lstm(units = 32, return_sequences = T)) %>%
  layer_global_average_pooling_1d() %>%
  layer_dense(units = 32, activation = 'relu') %>%
  layer_dense(units = 1, activation = 'sigmoid')
```
The array gets averaged into a 1 Dimensional array that can be passed into a fully connected layer with 32 units. This layer uses an activation function called the rectified linear unit. It outputs a 1 if it is above a certain value, and 0 otherwise. Lastly, the array is being passed into an output layer with the activation function sigmoid.
