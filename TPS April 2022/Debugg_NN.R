# This finds the cost error from 5-80 neural units and plots it
# Using the NLOPT_LD_LBFGS optimization method from nloptr

diff_hidden_size <- seq(from = 5, 80, by = 3)

i = 1

error <- list()

# Options for the nloptr function
# Max evaluate 50 times each loop
options <- list('algorithm' = 'NLOPT_LD_LBFGS', 'xtol_rel' = 1e-4,
                'maxeval' = 50)

for (i in 1:length(diff_hidden_size)){
  
  print(paste('Currently evaluating', diff_hidden_size[i], 'neural units model'))
  
  hidden_layer <- diff_hidden_size[i]
  hidden_layer2 <- diff_hidden_size[i]
  
  theta1 <- initial_weights(input, hidden_layer)
  theta2 <- initial_weights(hidden_layer, hidden_layer2)
  theta3 <- initial_weights(hidden_layer2, output)
  Theta <- cbind(c(theta1, theta2, theta3))
  
  theta1_size <- dim(theta1)[1] * dim(theta1)[2]
  theta2_size <- dim(theta2)[1] * dim(theta2)[2]
  theta3_size <- dim(theta3)[1] * dim(theta3)[2]
  
  # Finds the minimum of theta using cost function and gradient descent
  Theta <- nloptr(Theta, eval_f = cost, eval_grad_f = grad, opts = options)
  
  # Update the weights of NN
  Theta <- matrix(Theta$solution)
  
  # Calculate the cost between training set and validation set
  current_cost <- train_cv_error(Theta, x, validate[, 1:ncol(validate)-1],
                                 y, validate[ncol(validate)])
  
  # Print the costs
  print(paste('Cost for training set:', current_cost[1]))
  print(paste('cost for validaton set:', current_cost[2]))
  
  # Store the cost of training and validation set after each loop
  error <- rbind(error, current_cost)
  
}

# Plots the cost for training and validation set
plot(1:dim(error)[1], error[,1], col = 'red', pch = 20)
lines(1:dim(error)[1], error[,1], col = 'red', pch = 20)
lines(1:dim(error)[1], error[,2], col = 'blue', pch = 20)

################################################################################


# This is a list containing some of the nloptr algo options that uses gradient function
algo <- list('NLOPT_LD_LBFGS_NOCEDAL', 
             'NLOPT_LD_MMA', 'NLOPT_LD_CCSAQ',
             'NLOPT_LD_LBFGS', 'NLOPT_LD_VAR1',
             'NLOPT_LD_VAR2', 'NLOPT_LD_TNEWTON',
             'NLOPT_LD_TNEWTON_RESTART',
             'NLOPT_LD_TNEWTON_PRECOND',
             'NLOPT_LD_TNEWTON_PRECOND_RESTART')

algo_cost <- list()

i = 1

for (i in 1:length(algo)){
  
  # Setting up the theta for the neural network
  theta1 <- initial_weights(input, hidden_layer) # Input layer
  theta2 <- initial_weights(hidden_layer, hidden_layer2) # First hidden layer
  theta3 <- initial_weights(hidden_layer2, output) # Second hidden layer
  Theta <- cbind(c(theta1, theta2, theta3)) # Output layer
  
  # Options for the nloptr function
  options <- list('algorithm' = unlist(algo)[i], 'xtol_rel' = 1e-4,
                  'maxeval' = 100)
  
  print(paste('Current selected algorithm:', algo[i]))
  
  # Finds the minimum of theta using cost function and gradient descent
  Theta <- nloptr(Theta, eval_f = cost, eval_grad_f = grad, opts = options)
  
  # Update the weights of NN
  Theta <- matrix(Theta$solution)
  
  # Calculates the cost for training and validation set
  current_cost <- train_cv_error(Theta, x, validate[, 1:ncol(validate)-1], 
                                 y, validate[ncol(validate)])
  
  print(paste('Cost for training set:', current_cost[1]))
  print(paste('cost for validaton set:', current_cost[2]))
  
  # Store the calculated cost into algo_cost
  algo_cost <- rbind(algo_cost, current_cost)
  
  
}

# Plot the training set vs the validation set for each optimization algorithm
plot(1:dim(algo_cost)[1], algo_cost[,1], col = 'red', pch = 20)
lines(1:dim(algo_cost)[1], algo_cost[,1], col = 'red', pch = 20)
lines(1:dim(algo_cost)[1], algo_cost[,2], col = 'blue', pch = 20)
################################################################################





















