library(data.table)

# Load Theta parameters from a csv file

load_theta <- function(x){
  
  Theta <- fread(x)
  
  Theta <- as.matrix(Theta)
  
}


# Saving the Theta parameters to a csv file
save_theta <- function(x){
  
  write.csv(x, 'Theta.csv', row.names = F)
  
  
}