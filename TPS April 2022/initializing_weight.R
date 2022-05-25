# Initialized the weight each theta through random uniform distribution
initial_weights = function(in_layer, out_layer){
  
  eps = 0.1

  rand_num <- runif(out_layer*(in_layer+1))
  
  rand_num <- (rand_num * 2 * eps) - eps 
  
  result <- matrix(rand_num, out_layer, (in_layer+1))
  
  
}