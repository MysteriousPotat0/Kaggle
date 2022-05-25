library(tidyverse)
library(MASS)
library(lubridate)
library(ggplot2)
set.seed(1)



# Importing the train and test data from the csv
training <- read.csv("train.csv", header = T, sep = ",")
test<- read.csv("test.csv", header = T, sep = ",")


# Checking for outliers
mean(training$congestion)
sd(training$congestion)
training %>% group_by(congestion) %>% summarise("# of congestion outside of 95%" = 
                              sum((pnorm(congestion, mean(training$congestion), 
                             sd(training$congestion)) >= 0.975 | 
                         pnorm(congestion, mean(training$congestion), 
                               sd(training$congestion)) <= 0.025)))







# If the value is outside of 95%, then replace it with median
training$congestion[pnorm(training$congestion, mean(training$congestion), 
                                 sd(training$congestion)) >= 0.975 | 
                           pnorm(training$congestion, mean(training$congestion), 
                                 sd(training$congestion)) <= 0.025] <- median(training$congestion)
  





# Splitting the date and time into two column for train set
training <-
  tidyr::separate(training, time, c("Date", "Time"), sep = " ", remove = T)


# Transforming Date column for training into Year, Month, and Day
training$Date <- as.Date(training$Date)
training <- tidyr::separate(training, Date, c("Year", "Month", "Day"), 
                            sep = "-", remove = T)

# Remove the year column in training
training <- training[,c(-2)]

# Converting H:M:S into minutes
training$Time <- hms(training$Time)
training$Time <- as.numeric(hour(training$Time)) + 
  as.numeric(minute(training$Time))/60






# If time is between 5-9 AM is early morning
training$Time_range[training$Time >= 5 & 
                      training$Time <= 9] <- "Early Morning"

# If time is between 9-12 AM is morning
training$Time_range[training$Time > 9 & training$Time <= 12] <- "Morning"

# IF the time is between 12-3 PM mid afternoon
training$Time_range[training$Time > 12 & 
                      training$Time <= 15] <- "Early Afternoon"

# IF the time is between 3-5 PM early afternoon
training$Time_range[training$Time > 15 & training$Time <= 17] <- "Afternoon"

# If time is between 5-9 PM is evening
training$Time_range[training$Time > 17 & training$Time <= 21] <- "Evening"

# If time is between 9-5 AM is night
training$Time_range[training$Time < 5 |training$Time > 21 & 
                training$Time <= 24] <- "Night"


training$Time_range <- factor(training$Time_range, ordered = T, 
                              levels = c("Early Morning", "Morning", 
                                         "Early Afternoon", "Afternoon",
                                         "Evening", "Night"))



# Determining which day it 
training$weekday[as.numeric(training$Day) %% 7 == 1] <- "Monday"
training$weekday[as.numeric(training$Day) %% 7 == 2] <- "Tuesday"
training$weekday[as.numeric(training$Day) %% 7 == 3] <- "Wednesday"
training$weekday[as.numeric(training$Day) %% 7 == 4] <- "Thursday"
training$weekday[as.numeric(training$Day) %% 7 == 5] <- "Friday"
training$weekday[as.numeric(training$Day) %% 7 == 6] <- "Saturday"
training$weekday[as.numeric(training$Day) %% 7 == 0] <- "Sunday"





# Turning training Month, Day into a factor
training$Month <- factor(training$Month, ordered = T)
training$Day <- factor(training$Day, ordered = T)
training$weekday <- factor(training$weekday, ordered = T, levels = 
                             c("Monday", "Tuesday", "Wednesday", "Thursday",
                               "Friday", "Saturday", "Sunday"))

# Setting direction as a factor
training$direction <- factor(training$direction)

# Setting x as factor
training$x <- factor(training$x, ordered = T)

# Setting y as factor
training$y <- factor(training$y, order = T)


# Turning training Time into factor
training$Time <- factor(training$Time, ordered = T)






# Plot some sample of the training data
sample_data <- sample.int(nrow(training), size = 100000, replace = F)
sample_set <- training[sample_data, ]


ggplot(data = sample_set, aes(weekday, congestion)) + 
  geom_bar(stat = "identity")

plot(sample_set$Day, sample_set$congestion, col = colours(sample_set$Day))

ggplot(data = sample_set, aes(weekday, congestion)) + geom_point()

ggplot(data = sample_set, aes(direction, congestion)) + 
  geom_bar(stat = "identity")

ggplot(data = sample_set, aes(Time, congestion)) + 
  geom_bar(stat = "identity")



# Splitting the training set into train and test set 75/20
split_training <- sample.int(n = nrow(training), size = floor(0.75*nrow(training)), 
                                      replace = F)

new_training <- training[split_training, ]

validation_set <- training[-split_training, ]


# Re-sampling original data into new training data
split_training_1 <- sample.int(n = nrow(new_training), replace = T)

new_training_1 <- new_training[split_training_1, ]


# Re-sampling original data into new training data
split_training_2 <- sample.int(n = nrow(new_training),  replace = T)

new_training_2 <- new_training[split_training_2, ]


# Re-sampling original data into new training data
split_training_3 <- sample.int(n = nrow(new_training), replace = T)

new_training_3 <- new_training[split_training_3, ]

# Re-sampling original data into new training data
split_training_4 <- sample.int(n = nrow(new_training), replace = T)

new_training_4 <- new_training[split_training_4, ]




# Trying different data set with min, max, medium, and mean congestion
new_training_1 <- new_training_1 %>% group_by(Month ,Day, weekday, Time_range, x, y, direction) %>% summarise(as.integer(min(congestion)))
new_training_2 <- new_training_2 %>% group_by(Month ,Day, weekday, Time_range, x, y, direction) %>% summarise(as.integer(max(congestion)))
new_training_3 <- new_training_3 %>% group_by(Month ,Day, weekday, Time_range, x, y, direction) %>% summarise(as.integer(median(congestion)))
new_training_4 <- new_training_4 %>% group_by(Month ,Day, weekday, Time_range, x, y, direction) %>% summarise(as.integer(mean(congestion)))





# Renaming the columns 8 for new_training 1-4 to congestion
colnames(new_training_1)[c(8)] <- "congestion"
colnames(new_training_2)[c(8)] <- "congestion"
colnames(new_training_3)[c(8)] <- "congestion"
colnames(new_training_4)[c(8)] <- "congestion"
colnames(validation_set)[c(8)] <- "congestion"





# Fitting different linear model with min, max, median, mean
congestion_lm <- lm(congestion ~ weekday + x + y + direction + Time_range +
  Month+ x:y:direction, data = new_training_1)


congestion_lm_2 <- lm(congestion ~ weekday + x + y + direction + Time_range +
                        Month+x:y:direction,data = new_training_2)


congestion_lm_3 <- lm(congestion ~ weekday + x + y + direction + Time_range +
                        Month+x:y:direction, data = new_training_3)


congestion_lm_4 <- lm(congestion ~ weekday + x + y + direction + Time_range +
                        Month +x:y:direction, data = new_training_4)







summary(congestion_lm)
summary(congestion_lm_2)
summary(congestion_lm_3)
summary(congestion_lm_4)


par(mfrow = c(2,2))
plot(congestion_lm)
plot(congestion_lm_2)
plot(congestion_lm_3)
plot(congestion_lm_4)




# Using linear model on the validation data to predict
lm_test_prediciton1 <- predict(congestion_lm, validation_set)
lm_test_prediciton2 <- predict(congestion_lm_2, validation_set)
lm_test_prediciton3 <- predict(congestion_lm_3, validation_set)
lm_test_prediciton4 <- predict(congestion_lm_4, validation_set)


test_prediciton <- cbind(validation_set$congestion, lm_test_prediciton1,
                         lm_test_prediciton2, lm_test_prediciton3, 
                         lm_test_prediciton4,
                         (lm_test_prediciton1 + lm_test_prediciton2 + 
                           lm_test_prediciton3 + lm_test_prediciton4)/4)








# Splitting the date and time into two column for the test set
test <- tidyr::separate(test, time, c("Date", "Time"), sep = " ", remove = T)



# Transform Date column for test into Year, Month, and Day
test$Date <- as.Date(test$Date)
test <- tidyr::separate(test, Date, c("Year", "Month", "Day"), sep = "-",
                        remove = T) 


# Converting H:M:S into minutes
test$Time <- hms(test$Time)
test$Time <- as.numeric(hour(test$Time)) + 
  as.numeric(minute(test$Time))/60



# If time is between 5-9 AM is early morning
test$Time_range[test$Time >= 5 & 
                  test$Time <= 9] <- "Early Morning"

# If time is between 9-12 AM is morning
test$Time_range[test$Time > 9 & test$Time <= 12] <- "Morning"

# IF the time is between 12-3 PM mid afternoon
test$Time_range[test$Time > 12 & 
                  test$Time <= 15] <- "Early Afternoon"

# IF the time is between 3-5 PM early afternoon
test$Time_range[test$Time > 15 & test$Time <= 17] <- "Afternoon"

# If time is between 5-9 PM is evening
test$Time_range[test$Time > 17 & test$Time <= 21] <- "Evening"

# If time is between 9-5 AM is night
test$Time_range[test$Time < 5 |test$Time > 21 & 
                  test$Time <= 24] <- "Night"


test$Day <- as.numeric(test$Day)


# Determining which day it 
test$weekday[test$Day %% 7 == 1] <- "Monday"
test$weekday[test$Day %% 7 == 2] <- "Tuesday"
test$weekday[test$Day %% 7 == 3] <- "Wednesday"
test$weekday[test$Day %% 7 == 4] <- "Thursday"
test$weekday[test$Day %% 7 == 5] <- "Friday"
test$weekday[test$Day %% 7 == 6] <- "Saturday"
test$weekday[test$Day %% 7 == 0] <- "Sunday"


# Turning test Month, Day into a factor
test$Month <- as.factor(test$Month)
test$Day <- as.factor(test$Day)



# Turning test Time into factor
test$Time <- as.factor(test$Time)



# As factor for time range, weekday
test$Time_range <- as.factor(test$Time_range)
test$weekday <- as.factor(test$weekday)



# Turning test Year, Month, Day into factor
test$Month <- as.factor(test$Month)
test$Day <- as.factor(test$Day)



# Turning test time into factor
test$Time <- as.factor(test$Time)



# Turning test x as factor
test$x <- as.factor(test$x)



# Turning test y as factor
test$y <- as.factor(test$y)



# Turning test direction as factor
test$direction <- as.factor(test$direction)




# Prediction on the actual test data
lm_prediction1 <- predict(congestion_lm, test)
lm_prediction2 <- predict(congestion_lm_2, test)
lm_prediction3 <- predict(congestion_lm_3, test)
lm_prediction4 <- predict(congestion_lm_4, test)


predictions <- cbind(lm_prediction1, lm_prediction2, lm_prediction3, 
                     lm_prediction4, (lm_prediction1 + lm_prediction2 + 
                                        lm_prediction3 +lm_prediction4)/4)



result <- cbind(test$row_id, as.integer(predictions[,5]))
colnames(result)[c(1, 2)] <- c("row_id", "congestion")
write.csv(result, "Submission.csv", row.names = F)


