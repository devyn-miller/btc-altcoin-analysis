# QUESTION 3: Does including past Bitcoin prices improve predictions of current altcoin prices?



library(httr)
library(jsonlite)
library(tidyverse)
library(lubridate)
library(caret)
library(rstanarm)
library(brms)
library(boot)
library(forecast)
library(broom)
library(reshape2)
library(ggfortify)
library(tseries)
library(glmnet) 
library(forecast)
library(dotenv)

load_dotenv()
api_key <- Sys.getenv("API_KEY")


# Fetch Historical Cryptocurrency Data
fetch_crypto_data <- function(symbol, currency = "USD", limit = 2000) {
  base_url <- "https://min-api.cryptocompare.com/data/v2/histoday"
  response <- GET(base_url,
                  query = list(
                    fsym = symbol,
                    tsym = currency,
                    limit = limit,
                    api_key = api_key
                  ))
  if (status_code(response) == 200) {
    data <- content(response, as = "parsed", simplifyDataFrame = TRUE)$Data$Data
    return(as.data.frame(data))
  } else {
    stop(paste("Failed to fetch data for:", symbol))
  }
}

btc_data <- fetch_crypto_data("BTC")
eth_data <- fetch_crypto_data("ETH")
sol_data <- fetch_crypto_data("SOL")
bnb_data <- fetch_crypto_data("BNB")
xrp_data <- fetch_crypto_data("XRP")

# Data Cleaning
clean_crypto_data <- function(data) {
  data %>%
    mutate(
      date = as.Date(as.POSIXct(time, origin = "1970-01-01", tz = "UTC")),
      price = (high + low) / 2
    ) %>%
    select(date, price, volumefrom) %>%
    rename(volume = volumefrom)
}

btc_clean <- clean_crypto_data(btc_data)
eth_clean <- clean_crypto_data(eth_data)
sol_clean <- clean_crypto_data(sol_data)
bnb_clean <- clean_crypto_data(bnb_data)
xrp_clean <- clean_crypto_data(xrp_data)

# Combine Altcoin Data
altcoins_clean <- bind_rows(
  eth_clean %>% mutate(altcoin = "ETH"),
  sol_clean %>% mutate(altcoin = "SOL"),
  bnb_clean %>% mutate(altcoin = "BNB"),
  xrp_clean %>% mutate(altcoin = "XRP")
)

# Aggregate Altcoin Data
altcoins_grouped <- altcoins_clean %>%
  group_by(date) %>%
  summarise(
    altcoin_price = mean(price, na.rm = TRUE),
    altcoin_volatility = sd(price, na.rm = TRUE),
    altcoin_volume = sum(volume, na.rm = TRUE)
  )

# Merge with btc Data
merged_data <- left_join(btc_clean, altcoins_grouped, by = "date") %>%
  mutate(
    lag_7 = lag(price, 7),
    lag_14 = lag(price, 14),
    lag_30 = lag(price, 30)
  ) %>%
  drop_na() # Drop rows with missing data

# EDA
# Correlation Heatmap
corr_data <- merged_data %>%
  select(price, altcoin_price, lag_7, lag_14, lag_30, altcoin_volume, altcoin_volatility) %>%
  cor(use = "complete.obs")
corr_data_melt <- melt(corr_data)

ggplot(corr_data_melt, aes(x = Var1, y = Var2, fill = value)) +
  geom_tile() +
  scale_fill_gradient2(low = "blue", high = "red", mid = "white", midpoint = 0) +
  labs(title = "Correlation Heatmap", x = "", y = "") +
  theme_minimal()

# Density Plots
ggplot(merged_data, aes(x = price)) +
  geom_density(fill = "skyblue", alpha = 0.7) +
  labs(title = "Density Plot of Bitcoin Prices", x = "Price (USD)", y = "Density") +
  theme_minimal()

ggplot(altcoins_grouped, aes(x = altcoin_price)) +
  geom_density(fill = "lightgreen", alpha = 0.7) +
  labs(title = "Density Plot of Altcoin Prices", x = "Price (USD)", y = "Density") +
  theme_minimal()

# Time Series Decomposition for Bitcoin
btc_ts <- ts(merged_data$price, frequency = 365) # Create time series object
btc_decomp <- decompose(btc_ts)  # Decompose time series (additive decomposition)

# Decomposing Bitcoin prices and plotting
autoplot(btc_decomp) +
  labs(title = "Time Series Decomposition: Bitcoin Price", x = "Date", y = "Price (USD)")

# Time Series Decomposition for Altcoins
altcoin_ts <- ts(altcoins_grouped$altcoin_price, frequency = 365) # Altcoin price decomposition
altcoin_decomp <- decompose(altcoin_ts)  # Decompose time series

# Decomposing Altcoin prices and plotting
autoplot(altcoin_decomp) +
  labs(title = "Time Series Decomposition: Altcoin Price", x = "Date", y = "Price (USD)")

# Time Series Trends
ggplot(merged_data, aes(x = date)) +
  geom_line(aes(y = price, color = "Bitcoin"), size = 1) +
  geom_line(aes(y = altcoin_price, color = "Altcoins"), size = 1) +
  labs(
    title = "Price Trends of Bitcoin vs Altcoins",
    x = "Date",
    y = "Price (USD)",
    color = "Cryptocurrency"
  ) +
  theme_minimal()

# Frequentist Modeling
model_a <- lm(altcoin_price ~ price, data = merged_data)
model_b <- lm(altcoin_price ~ price + lag_7 + lag_14 + lag_30 + altcoin_volume + altcoin_volatility, data = merged_data)

# Model Comparison
aic_values <- AIC(model_a, model_b)
bic_values <- BIC(model_a, model_b)
anova_test <- anova(model_a, model_b)

# Time-Series Cross-Validation for Model B
set.seed(123)
ts_cv <- function(model_formula, data, h = 7) {
  train_indices <- seq_len(nrow(data) - h)
  train_data <- data[train_indices, ]
  test_data <- data[-train_indices, ]
  
  model <- lm(model_formula, data = train_data)
  predictions <- predict(model, test_data)
  actuals <- test_data$altcoin_price
  
  data.frame(
    RMSE = sqrt(mean((predictions - actuals)^2)),
    MAE = mean(abs(predictions - actuals))
  )
}

cv_results <- ts_cv(
  altcoin_price ~ price + lag_7 + lag_14 + lag_30 + altcoin_volume + altcoin_volatility,
  data = merged_data
)

# Bayesian Regression
bayesian_model <- brm(
  altcoin_price ~ price + lag_7 + lag_14 + lag_30 + altcoin_volume + altcoin_volatility,
  data = merged_data,
  prior = c(prior(normal(0, 5), class = "b")),
  iter = 4000,  # Increased iterations
  chains = 4,
  control = list(max_treedepth = 20, adapt_delta = 0.99)  # Increased adapt_delta
)

# Posterior Predictive Checks
pp_check(bayesian_model) +
  labs(title = "Posterior Predictive Check", x = "Price (USD)", y = "Density")

# Regularization (Ridge Regression)
ridge_model <- cv.glmnet(
  as.matrix(merged_data %>% select(price, lag_7, lag_14, lag_30, altcoin_volume, altcoin_volatility)),
  merged_data$altcoin_price,
  alpha = 0,  # Ridge regression
  nfolds = 10
)

# Summarize Regularization Results
print(paste("Best Lambda for Ridge:", ridge_model$lambda.min))

# Results Summary
metrics <- data.frame(
  Model = c("Model A", "Model B"),
  RMSE = c(sqrt(mean(residuals(model_a)^2)), sqrt(mean(residuals(model_b)^2))),
  MAE = c(mean(abs(residuals(model_a))), mean(abs(residuals(model_b))))
)

# Metrics
print(metrics)
print(cv_results)
print(aic_values)
print(bic_values)
print(anova_test)


# Model Comparison Results
# Get RMSE and MAE for both models
rmse_a <- sqrt(mean(residuals(model_a)^2))
mae_a <- mean(abs(residuals(model_a)))
rmse_b <- sqrt(mean(residuals(model_b)^2))
mae_b <- mean(abs(residuals(model_b)))

# Create a df with RMSE and MAE values for both models
comparison_results <- data.frame(
  Model = c("Model A (Current Bitcoin Price)", "Model B (Lagged Bitcoin Prices)"),
  RMSE = c(rmse_a, rmse_b),
  MAE = c(mae_a, mae_b)
)

# Print the comparison results
print(comparison_results)


# Get Adjusted R-squared for Model A
summary_model_a <- summary(model_a)
adjusted_r2_a <- summary_model_a$adj.r.squared

# Get Adjusted R-squared for Model B
summary_model_b <- summary(model_b)
adjusted_r2_b <- summary_model_b$adj.r.squared

# Print Adjusted R-squared values
print(paste("Adjusted R-squared for Model A:", adjusted_r2_a))
print(paste("Adjusted R-squared for Model B:", adjusted_r2_b))



# Time-Series Cross-Validation RMSE Comparison for Model B
# Plot the RMSE for both models
ggplot(data = comparison_results, aes(x = Model, y = RMSE, fill = Model)) +
  geom_bar(stat = "identity", show.legend = FALSE) +
  labs(title = "RMSE Comparison Between Models", x = "Model", y = "RMSE") +
  theme_minimal() +
  scale_fill_manual(values = c("skyblue", "lightgreen"))

# Model Predictions and Residuals Plot for Model A and Model B
# Compare the predictions and residuals visually
predictions_a <- predict(model_a, merged_data)
predictions_b <- predict(model_b, merged_data)

# Plot Residuals for Model A and Model B
ggplot() +
  geom_line(aes(x = merged_data$date, y = residuals(model_a), color = "Model A Residuals"), size = 1.2) +
  geom_line(aes(x = merged_data$date, y = residuals(model_b), color = "Model B Residuals"), size = 1.2) +
  labs(
    title = "Residuals of Model A and Model B", 
    x = "Date", 
    y = "Residuals"
  ) +
  scale_color_manual(values = c("Model A Residuals" = "darkblue", "Model B Residuals" = "darkorange")) +
  theme_minimal() +
  theme(legend.title = element_blank()) +
  theme(legend.position = "top") +
  scale_color_manual(name = "Model", values = c("darkblue", "darkorange"))

# Prediction vs Actual Values Plot (Compare Model A and Model B Predictions)
ggplot() +
  geom_line(aes(x = merged_data$date, y = predictions_a, color = "Model A Predictions"), size = 1.2) +
  geom_line(aes(x = merged_data$date, y = predictions_b, color = "Model B Predictions"), size = 1.2) +
  geom_line(aes(x = merged_data$date, y = merged_data$altcoin_price, color = "Actual Prices"), size = 1.2, linetype = "solid") +
  labs(
    title = "Predictions vs Actual Prices for Model A and Model B", 
    x = "Date", 
    y = "Price (USD)"
  ) +
  scale_color_manual(values = c("Model A Predictions" = "blue", "Model B Predictions" = "red", "Actual Prices" = "black")) +
  theme_minimal() +
  theme(legend.title = element_blank()) +
  theme(legend.position = "top") +
  scale_color_manual(name = "Legend", values = c("blue", "red", "black"))

# Bayesian Posterior Predictive Checks for Model B
pp_check(bayesian_model) +
  labs(title = "Posterior Predictive Check for Model B", x = "Price (USD)", y = "Density") +
  theme_minimal()

# 6. Additional Summary Metrics
# Get AIC and BIC values for both models to compare
aic_values <- AIC(model_a, model_b)
bic_values <- BIC(model_a, model_b)

# Print AIC and BIC results
print("AIC Values:")
print(aic_values)
print("BIC Values:")
print(bic_values)

# Hypothesis Test (Compare RMSEs using Diebold-Mariano Test)

dm_test <- dm.test(predictions_a, predictions_b, alternative = "less")
print(dm_test)

# Conclusion: Compare RMSE, MAE, and hypothesis testing results
if (rmse_b < rmse_a) {
  print("Including lagged Bitcoin prices improves prediction accuracy for altcoin prices (Model B is better).")
} else {
  print("Including lagged Bitcoin prices does not improve prediction accuracy for altcoin prices (Model A is better).")
}
