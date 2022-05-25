#### Loading Libraries ####
library(keras)
library(tensorflow)
library(dplyr)
library(zoo)
library(moments)
library(data.table)
library(pROC)
set.seed(1)


train_label <- fread('train_labels.csv', header = T, sep = ',')

tf_test <- fread('train.csv', header = T, sep = ',')


#### Feature selection functions ####
normalization <- function(x){
  
  
  (x-min(x, na.rm = T))/(max(x, na.rm = T)-min(x, na.rm = T))
  
}

roll_mean <- function(x){
  
  rollmean(x, k = 2, fill = 0, align = 'right')
  
}

roll_med <- function(x){
  
  rollmedian(x, k = 3, fill = 0, align = 'right')
  
}

roll_max<- function(x){
  
  rollmax(x, k = 2, fill = 0, align = 'right')
  
}


tf_test <- tf_test %>% group_by(sequence) %>% mutate_at(c(4:ncol(tf_test)), list(lag1 = lag)) 

tf_test <- tf_test %>% group_by(sequence) %>% mutate_at(c(4:16), funs(diff = . - lag(.)))

tf_test <- tf_test %>% group_by(sequence) %>% mutate_at(c(4:16), list(roll_mean = roll_mean,
                                                                      roll_max = roll_max,
                                                                      roll_med = roll_med))

tf_test <- tf_test %>% mutate_at(c(4:ncol(tf_test)), normalization) %>% as.data.table()



tf_test[is.na(tf_test)] <- 0

tf_test <- tf_test[, -c(1:3)]

y = train_label$state

tf_array <- reticulate::array_reshape(as.matrix(tf_test), dim = c(nrow(train_label), 60, 78))





tf_model <- keras_model_sequential()
tf_model %>%
  bidirectional(layer_lstm(input_shape = dim(tf_array)[2:3], units = 128, return_sequences = T)) %>%
  bidirectional(layer_lstm(units = 64, return_sequences = T)) %>%
  bidirectional(layer_lstm(units = 64, return_sequences = T)) %>%
  bidirectional(layer_lstm(units = 32, return_sequences = T)) %>%
  layer_global_average_pooling_1d() %>%
  layer_dense(units = 32, activation = 'relu') %>%
  layer_dense(units = 1, activation = 'sigmoid')


tf_model %>% compile(loss = 'binary_crossentropy', optimizer = optimizer_adam(learning_rate = 0.001, decay = 1e-4),
                     metrics = metric_auc())

tf_hist <- tf_model %>% fit(tf_array, y, epochs = 15, batch_size = 128,
                            validation_split = 0.2)


tf_model %>% save_model_tf('tf_model')

Test <- fread('test.csv', header = T, sep = ',')

Test <- Test %>% group_by(sequence) %>% mutate_at(c(4:ncol(Test)), list(lag1 = lag)) 

Test <- Test %>% group_by(sequence) %>% mutate_at(c(4:16), funs(diff = . - lag(.)))

Test <- Test %>% group_by(sequence) %>% mutate_at(c(4:16), list(roll_mean = roll_mean,
                                                                      roll_max = roll_max,
                                                                      roll_med = roll_med))



Test <- Test %>% mutate_at(c(4:ncol(Test)), normalization) %>% as.data.table()

Test[is.na(Test)] <- 0

sequences <- Test %>% group_by(sequence) %>% summarise(mean = mean(sensor_00))

sequences <- sequences[,1]

Test <- Test[, -c(1:3)]

Test_array <- reticulate::array_reshape(as.matrix(Test), dim = c(12218, 60, 78))

tf_predict <- predict(tf_model,Test_array)

tf_predict <- ifelse(tf_predict >= 0.5, 1, 0)

submission <- cbind(sequences, tf_predict)

colnames(submission) <- c('sequence', 'state')

write.csv(submission, 'submission.csv', row.names = F)


