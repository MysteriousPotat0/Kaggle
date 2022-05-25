# Cost function for the neural network, this function is only used for minimizing the Theta parameter

cost <- function(Theta){
  
  # Forward Feed
  theta1 <- matrix(Theta[1:theta1_size], nrow(theta1), ncol(theta1))
  theta2 <- matrix(Theta[theta1_size + 1:theta2_size], nrow(theta2), ncol(theta2))
  theta3 <- matrix(tail(Theta, theta3_size), nrow(theta3), ncol(theta3))
  
  m <- dim(x)[1]
  
  # Forward Feed
  # x <- x*matrix(rbinom(dim(x)[2], 1, 0.8), dim(x)[1], dim(x)[2], byrow = T)
  
  x <- cbind(rep(1, m), x) # Add a column of 1's to x
  
  z2 <- theta1 %*% t(x)
  
  a2 <- sig(z2)
  
  # a2 <- a2 * matrix(rbinom(dim(a2)[2], 1, 0.5), dim(a2)[1], dim(a2)[2], byrow = T)
  
  a2 <- cbind(c(rep(1, m)), t(a2))
  
  z3 <- theta2 %*% t(a2)
  
  a3 <- sig(z3)
  
  # a3 <- a3 * matrix(rbinom(dim(a3)[2], 1, 0.3), dim(a3)[1], dim(a3)[2], byrow = T)
  
  a3 <- cbind(c(rep(1, m)), t(a3))
  
  z4 <- theta3 %*% t(a3)
  
  h <- sig(z4)
  
  # Implementing the cost function with regularization
  (1/m)*(sum((-y * log(h)) - ((1-y)*log(1-h)))) + (lambda/(2*m))*(sum(theta1[, 2:ncol(theta1)]^2) + sum(theta2[, 2:ncol(theta2)]^2) + 
                                                                    sum(theta3[,2:ncol(theta3)]^2))
  
}

################################################################################


# Function that is used for calculating the error the fitted value

error_cost <- function(Theta, x, y){
  
  # Forward Feed
  theta1 <- matrix(Theta[1:theta1_size], nrow(theta1), ncol(theta1))
  theta2 <- matrix(Theta[theta1_size + 1:theta2_size], nrow(theta2), ncol(theta2))
  theta3 <- matrix(tail(Theta, theta3_size), nrow(theta3), ncol(theta3))
  
  m <- dim(x)[1]
  
  # Forward Feed
  # x <- x*matrix(rbinom(dim(x)[2], 1, 0.8), dim(x)[1], dim(x)[2], byrow = T)
  
  x <- cbind(rep(1, m), x) # Add a column of 1's to x
  
  z2 <- theta1 %*% t(x)
  
  a2 <- sig(z2)
  
  # a2 <- a2 * matrix(rbinom(dim(a2)[2], 1, 0.5), dim(a2)[1], dim(a2)[2], byrow = T)
  
  a2 <- cbind(c(rep(1, m)), t(a2))
  
  z3 <- theta2 %*% t(a2)
  
  a3 <- sig(z3)
  
  # a3 <- a3 * matrix(rbinom(dim(a3)[2], 1, 0.3), dim(a3)[1], dim(a3)[2], byrow = T)
  
  a3 <- cbind(c(rep(1, m)), t(a3))
  
  z4 <- theta3 %*% t(a3)
  
  h <- sig(z4)

  # Implementing the cost function with regularization
  (1/m)*(sum((-y * log(h)) - ((1-y)*log(1-h)))) + (lambda/(2*m))*
    (sum(theta1[, 2:ncol(theta1)]^2) + sum(theta2[, 2:ncol(theta2)]^2) +
       sum(theta3[2:ncol(theta3)]^2))

}

################################################################################


# Function used to calculate the error between two data set

train_cv_error <- function(Theta, set1, set2, y1, y2){
  
  
  cost1 <- error_cost(Theta, set1, y1)
  
  y <- validate[, ncol(validate)]
  
  cost2 <- error_cost(Theta, set2, y2)
  
  c(cost1, cost2)
  
  
}

################################################################################







