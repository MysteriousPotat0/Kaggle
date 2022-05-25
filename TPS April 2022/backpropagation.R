grad = function(Theta){
  
  # Forward Propagation
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
  
  # Back propagation
  delta_4 <- h - y
  
  # delta_4 <- (theta4[2:ncol(theta4)]) %*% delta_5 * sig_grad(z4)
  
  delta_3 <- t(theta3[,2:ncol(theta3), drop = F]) %*% delta_4 * sig_grad(z3)
  
  delta_2 <- t(theta2[,2:ncol(theta2), drop = F]) %*% delta_3 * sig_grad(z2)
  
  # 0th theta 1 gradient
  theta1_grad <- delta_2 %*% x
  theta1_grad[, 1] <- (1/m)*theta1_grad[, 1, drop = F]
  
  # When theta 1 gradient = j
  theta1_grad[, 2:ncol(theta1_grad)] <-(1/m)*(theta1_grad[2:ncol(theta1_grad)] + lambda*theta1[2:ncol(theta1)])
  
  # 0th theta 2 gradient
  theta2_grad <- delta_3 %*% a2
  theta2_grad[, 1] <- (1/m)*theta2_grad[, 1, drop = F]
  
  # When theta 2 gradient = j
  theta2_grad[,2:ncol(theta2_grad)] <- (1/m)*(theta2_grad[,2:ncol(theta2_grad), drop = F] + lambda*theta2[,2:ncol(theta2), drop = F])
  
  # 0th theta 3 gradient
  theta3_grad <- delta_4 %*% a3
  theta3_grad[, 1] <- (1/m)*theta3_grad[, 1, drop = F]
  
  # When theta 3 gradient = j
  theta3_grad[2:ncol(theta3_grad)] <- (1/m)*(theta3_grad[2:ncol(theta3_grad), drop = F] + lambda*theta3[2:ncol(theta3), drop = F])
  
  # # 0th theta 4 gradient
  # theta4_grad <- delta_5 %*% a4
  # theta4_grad[, 1] <- (1/m)*theta4_grad[, 1]
  # 
  # # When theta 4 gradient = j
  # theta4_grad[2:ncol(theta4_grad)] <- (1/m)*(theta4_grad[2:ncol(theta4_grad)] + lambda*theta4[2:ncol(theta4)])
  
  grad <- cbind(c(theta1_grad, theta2_grad, theta3_grad))
  

}
