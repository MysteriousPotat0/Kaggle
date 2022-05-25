# Function that calculates the sigmoid values using 1/(1+exp^-x)
sig = function(z){
  
  1/(1+exp(-z))
  
}

# Calculates the sigmoid gradient

sig_grad = function(z){
  
  sig(z) * (1-sig(z))
  
}
