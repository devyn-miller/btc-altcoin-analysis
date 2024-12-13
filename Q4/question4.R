library(httr)
library(jsonlite)
library(dplyr)
library(lubridate)
library(mgcv)
library(ggplot2)
library(Metrics)   # For MAPE, MAE, MSE
library(zoo)       # For rolling calculations
library(dotenv)

# PARAMS
base_url       <- "https://min-api.cryptocompare.com/data/v2/histoday"
coins          <- c("BTC", "ETH", "SOL", "BNB", "XRP")
currency       <- "USD"
limit_days     <- 2000
load_dotenv()
api_key <- Sys.getenv("API_KEY")


# Function to Fetch Data
get_crypto_data <- function(coin, currency="USD", limit=2000, api_key) {
  url <- paste0(base_url, "?fsym=", coin, "&tsym=", currency, "&limit=", limit)
  response <- GET(url, if (!is.null(api_key)) add_headers('authorization' = paste("Apikey", api_key)))
  data_json <- content(response, as="text", encoding="UTF-8")
  data_list <- fromJSON(data_json)
  
  if (data_list$Response == "Success") {
    df <- data_list$Data$Data
    df <- df %>%
      mutate(
        time = as_datetime(time),
        coin = coin
      ) %>%
      rename(
        open = open, high = high, low = low, close = close,
        volume_from = volumefrom, volume_to = volumeto
      ) %>%
      select(time, coin, open, high, low, close, volume_from, volume_to)
    return(df)
  } else {
    stop(paste("Error in fetching data for", coin, ":", data_list$Message))
  }
}

# Fetch Data for All Coins
crypto_data_list <- lapply(coins, function(coin) {
  cat("Fetching data for:", coin, "\n")
  get_crypto_data(coin, currency=currency, limit=limit_days, api_key=api_key)
})
crypto_data <- bind_rows(crypto_data_list) %>% arrange(coin, time)

# Add Log Returns and Rolling Volatility
rolling_window <- 7
crypto_data <- crypto_data %>%
  group_by(coin) %>%
  mutate(
    log_return = log(close / lag(close)),
    volatility = rollapply(log_return, width=rolling_window, FUN=sd, align="right", fill=NA, na.rm=TRUE)
  ) %>%
  ungroup()

# Visualization: BTC Volatility
btc_data <- crypto_data %>% filter(coin == "BTC")
ggplot(btc_data, aes(x=time, y=volatility)) +
  geom_line(color="blue", size=0.8) +
  labs(title="BTC Daily Rolling Volatility (7-day window)",
       x="Date", y="Volatility (std. dev. of log returns)") +
  theme_minimal()

# Merge BTC Volatility for Altcoins
btc_vol <- btc_data %>% select(time, btc_volatility = volatility)
crypto_data_merged <- crypto_data %>%
  left_join(btc_vol, by="time") %>%
  filter(coin != "BTC", !is.na(log_return), !is.na(btc_volatility))


# Fit Models and Create Plots for Each Altcoin
results_list <- list()
for (alt in c("ETH", "SOL", "BNB", "XRP")) {
  cat("Building models for:", alt, "\n")
  
  # Subset for current altcoin
  alt_data <- crypto_data_merged %>% filter(coin == alt)
  
  # Baseline GLM Model
  glm_baseline <- glm(log_return ~ 1, data=alt_data)
  
  # GLM with BTC Volatility
  glm_model <- glm(log_return ~ btc_volatility, data=alt_data)
  
  # GAM with BTC Volatility as Smooth Term
  gam_model <- gam(log_return ~ s(btc_volatility, k=5), data=alt_data, method="REML")
  
  # Predictions
  alt_data <- alt_data %>%
    mutate(
      pred_glm_baseline = predict(glm_baseline, newdata=alt_data),
      pred_glm_model = predict(glm_model, newdata=alt_data),
      pred_gam_model = predict(gam_model, newdata=alt_data)
    )
  alt_data <- alt_data %>% filter(abs(log_return) > 1e-6)
  
  
  # Evaluate Models
  eval_metrics <- data.frame(
    Coin = alt,
    Model = c("GLM_Baseline", "GLM_withBTCVol", "GAM_withBTCVol"),
    MAE = c(
      mae(alt_data$log_return, alt_data$pred_glm_baseline),
      mae(alt_data$log_return, alt_data$pred_glm_model),
      mae(alt_data$log_return, alt_data$pred_gam_model)
    ),
    MSE = c(
      mse(alt_data$log_return, alt_data$pred_glm_baseline),
      mse(alt_data$log_return, alt_data$pred_glm_model),
      mse(alt_data$log_return, alt_data$pred_gam_model)
    ),
    MAPE = c(
      mape(alt_data$log_return, alt_data$pred_glm_baseline),
      mape(alt_data$log_return, alt_data$pred_glm_model),
      mape(alt_data$log_return, alt_data$pred_gam_model)
    )
  )
  
  # Save Results
  results_list[[alt]] <- list(data=alt_data, glm_model=glm_model, gam_model=gam_model, eval_metrics=eval_metrics)
  
  # Visualization: Actual vs. Predicted Log Returns for GLM and GAM
  ggplot(alt_data, aes(x=time)) +
    geom_line(aes(y=log_return), color="black", size=0.8) +
    geom_line(aes(y=pred_glm_model), color="blue", size=0.8, alpha=0.7) +
    geom_line(aes(y=pred_gam_model), color="red", size=0.8, alpha=0.7) +
    labs(title=paste(alt, ": Actual vs. Predicted Log Returns"),
         x="Date", y="Log Return") +
    theme_minimal()
  
  # Partial Effect Plot (BTC Volatility for GAM)
  plot(gam_model, shade=TRUE, shade.col="lightblue", se=TRUE,
       main=paste("Partial Effect of BTC Volatility on", alt, "(GAM)"),
       xlab="BTC Volatility", ylab=paste("Effect on", alt, "Log Returns"))
  
  # Partial Dependence Plot for GAM
  btc_vol_seq <- seq(min(alt_data$btc_volatility, na.rm=TRUE),
                     max(alt_data$btc_volatility, na.rm=TRUE),
                     length.out=100)
  pd_data <- data.frame(btc_volatility = btc_vol_seq)
  pd_data$pred <- predict(gam_model, newdata=pd_data, type="response")
  
  ggplot(pd_data, aes(x=btc_volatility, y=pred)) +
    geom_line(color="red", size=1) +
    labs(title=paste("GAM Partial Dependence: BTC Volatility →", alt, "Log Returns"),
         x="BTC Volatility (7-day rolling SD)",
         y="Predicted Log Return") +
    theme_minimal()
}


# Combine Evaluation Metrics
all_eval <- bind_rows(lapply(results_list, function(x) x$eval_metrics))
print(all_eval)
summary(gam_model)
