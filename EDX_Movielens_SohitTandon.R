################################
# Create edx set, validation set
################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding")
# if using R 3.5 or earlier, use `set.seed(1)` instead
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

knitr::opts_chunk$set(echo = TRUE,warning = FALSE,cache = TRUE, fig.align = 'center' )

glimpse(edx)
glimpse(validation)

## Checking for NAs in edx dataset
sapply(edx, {function(x) any(is.na(x))}) %>% knitr::kable()

## Checking for NAs in validation dataset
sapply(validation, {function(x) any(is.na(x))}) %>% knitr::kable()

######################################## 
########## Data Wrangling ############## 
######################################## 

## Converting timestamp column to a human readable value
## need to install lubridate package as need the function as_datetime
if(!require(lubridate)) install.packages("lubridate", repos = "http://cran.us.r-project.org")

## using mutate function to add a new column year_rated to the data set edx and 
##storing the result in a new data frame edx_ts
edx_ts <- mutate(edx, year_rated = year(as_datetime(timestamp)))

## using mutate function to add a new column year_release to the data set edx and 
##storing the result in the frame edx_ts
edx_ts <- mutate(edx_ts, year_release = as.numeric(str_sub(title,-5,-2)))
head (edx_ts)
## using mutate function to add a new column year_rated to the data set validate and 
##storing the result in a new data frame validation_ts
validation_ts <- mutate(validation, year_rated = year(as_datetime(timestamp)))

## using mutate function to add a new column year_release to the data set validate and 
##storing the result in the data frame validation_ts
validation_ts <- mutate(validation_ts, year_release = as.numeric(str_sub(title,-5,-2)))
head (validation_ts)
###### Visualizing the matrix to see how  dense or sparse it is. 
##Matrix of 1000 users and 1000 movies with yellow indicating a user/movie 
##combination with rating
users <- sample(unique(edx$userId), 1000)

edx %>% filter(userId %in% users) %>% 
  select(userId, movieId, rating) %>%
  mutate(rating = 1) %>%
  spread(movieId, rating) %>% select(sample(ncol(.), 1000)) %>% 
  as.matrix() %>% t(.) %>%
  image(1:1000, 1:1000,. , xlab="Movies", ylab="Users")

#### Visualizing number of ratings for each movie ##############################

edx %>% count(movieId) %>%
  ggplot(aes(n)) + 
  geom_histogram(bins = 30, color = "black") + 
  scale_x_log10() + 
  ggtitle("Movies")

#### Visualizing number of ratings for each user ##############################

edx %>% count(userId) %>%
  ggplot(aes(n)) + 
  geom_histogram(bins = 30, color = "black") + 
  scale_x_log10() + 
  ggtitle("Users")

#### Visualizing number of count for each rating ##############################
edx %>% ggplot(aes(rating)) + geom_histogram(binwidth = 0.5, color = "black") + 
  scale_x_discrete(limits = c(seq(0.5,5,0.5))) + 
  scale_y_continuous(breaks = c(seq(0, 3000000, 500000)))

#### Visualizing year of rating ###############################################
edx_ts %>% ggplot(aes(year_rated))+geom_histogram(bins = 30, color = "black")

#### Visualizing year of release ###############################################
edx_ts %>% ggplot(aes(year_release))+geom_histogram(bins = 30, color = "black")

### Top rated movie genres######
####This will take quite a while please be patient ###############
edx_ts %>% separate_rows(genres, sep = "\\|") %>%
  group_by(genres) %>%
  summarize(count = n()) %>%
  arrange(desc(count)) %>% 
  knitr::kable()

### Visualizing the ratings by different genres
####This will take quite a while please be patient ###############
edx_ts %>% separate_rows(genres, sep = "\\|") %>% group_by(genres) %>% 
  summarize(count = n()) %>%  ggplot(aes(x = genres, y = count)) + 
  theme_classic()+
  geom_col()+
  theme(axis.text.x = element_text(angle = 90, hjust = 1))


#######Visualizing the average rating for user u who have rated over 100 movies ########
edx %>% 
  group_by(userId) %>% 
  summarize(b_u = mean(rating)) %>% 
  filter(n()>=100) %>%
  ggplot(aes(b_u)) + 
  geom_histogram(bins = 30, color = "black")

###### RMSE Loss function definition
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

########## First Model ###########################################
######## Step 1: Calculate mean rating of the training set ########
mu_hat <- mean(edx_ts$rating)
mu_hat

########Step 2: Calculate the RMSE for Model1 on the validation data set#### 
naive_rmse <- RMSE(validation_ts$rating, mu_hat)
naive_rmse 

########Creating a results table to store results of different models ####
rmse_results <- data_frame(Method ="Just the average", RMSE = naive_rmse)
rmse_results %>% knitr::kable()

### Second Model #######################################
######## Step1: Calculate 'bias' b which refers to some movies which 
####### are generally rated higher than others ########
movie_avgs <- edx_ts %>% 
  group_by(movieId) %>% 
  summarize(b_i = mean(rating - mu_hat))

##############Step2: Plotting b_i to see the distribution of the estimates##############
movie_avgs %>% qplot(b_i, geom ="histogram", bins = 10, data = ., color = I("black"))

##############Step3: add the b_i to our Model1 predicted ratings to arrive 
#############        at the new predicted#####
predicted_ratings <- mu_hat + validation_ts %>% left_join(movie_avgs,by='movieId') %>% 
  pull(b_i)

######## Step4: Calculate the RMSE for Model2 on the validation data set################
model2_rmse <- RMSE(predicted_ratings,validation_ts$rating)
model2_rmse

########## Store the result set of Model2 to rmse_results data frame #################
rmse_results <- bind_rows(rmse_results,data_frame(Method = "Movie Effect Model", 
                                                  RMSE = model2_rmse))
rmse_results %>% knitr::kable()

##############Third Model################################
##########Step1: Calculate the affect b_u to counter the cranky user who rates  
#########        a great movie 3 rather than 5
user_avgs <- edx_ts %>% 
  left_join(movie_avgs, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu_hat - b_i))

############Step2: calculating the new predicted ratings ##############################
predicted_ratings <- validation_ts %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  mutate(pred = mu_hat + b_i + b_u) %>%
  pull(pred)

################Step3: Calculate the RMSE for Model3 on the validation data set#######
model3_rmse <- RMSE(predicted_ratings,validation_ts$rating)
model3_rmse
######### Store the result set of Model3 to rmse_results data frame #################
rmse_results <- bind_rows(rmse_results,data_frame(Method = "Movie + User Effect Model", 
                                                  RMSE = model3_rmse))
rmse_results %>%knitr::kable()

##########Regularization#############################################
########## Here we use cross-validation to pick a lambda #################
lambdas <- seq(0, 10, 0.25)

rmses <- sapply(lambdas, function(l){
  mu_hat <- mean(edx_ts$rating)
  
  b_i <- edx_ts %>%
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu_hat)/(n()+l))
  
  b_u <- edx_ts %>%
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu_hat)/(n()+l))
  
  predicted_ratings <-
    validation_ts %>%
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(pred = mu_hat + b_i + b_u) %>%
    pull(pred)
  
  return(RMSE(predicted_ratings, validation_ts$rating))
})

##########Plotting the RMSE against lambdas ####
qplot(lambdas, rmses)  

lambda <- lambdas[which.min(rmses)]
lambda
rmse_results <- bind_rows(rmse_results,
                          data_frame(Method="Regularized Movie + User Effect Model",  
                                     RMSE = min(rmses)))
rmse_results %>%knitr::kable()

