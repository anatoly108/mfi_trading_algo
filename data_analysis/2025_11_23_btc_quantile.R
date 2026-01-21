# Install once if needed:
# install.packages("quantreg")
# install.packages("ggplot2")

library(quantreg)
library(ggplot2)
library(data.table)

# ---------------------------
# 1. Load / prepare your data
# ---------------------------
# Expecting a data frame with at least:
#   date  - Date or character
#   close - BTC closing price (or any price you like)

# Example: read from CSV (adjust path / column names as needed)
btc <- fread("ml/btcusd_1-min_data.csv")
colnames(btc) <- tolower(colnames(btc))

# Collapse minute-level data to daily bars so the day index advances by 1 per row.
btc[, timestamp := as.POSIXct(timestamp, tz = "UTC")]
if ("days_since_start" %in% names(btc)) {
  btc[, days_since_start := NULL]
}
btc[, date := as.Date(timestamp)]
setorder(btc, timestamp)

btc_daily <- btc[, .(
  open   = first(open),
  high   = max(high, na.rm = TRUE),
  low    = min(low, na.rm = TRUE),
  close  = last(close),
  volume = sum(volume, na.rm = TRUE)
), by = date]

btc <- btc_daily

# Make sure date is Date and price is numeric
btc$date  <- as.Date(btc$date)
btc$close <- as.numeric(btc$close)

# Use "age" in days since first observation as time axis
btc$days_since_start <- as.numeric(btc$date - min(btc$date)) + 1

# Log-log transform (base 10; natural log is also fine, just be consistent)
btc$log_days   <- log10(btc$days_since_start)
btc$log_price  <- log10(btc$close)

# ----------------------------------
# 2. Fit quantile trend regressions
# ----------------------------------
# Choose the quantiles you want, e.g. 3%, 50%, 97%
taus <- c(0.03, 0.5, 0.97)

# Fit log_price ~ log_days for each quantile
fit_list <- lapply(taus, function(tau) {
  rq(log_price ~ log_days, tau = tau, data = btc)
})
names(fit_list) <- paste0("tau_", taus)

# --------------------------------
# 3. Create smooth prediction grid
# --------------------------------
pred_grid <- data.frame(
  log_days = seq(
    from = min(btc$log_days),
    to   = max(btc$log_days),
    length.out = 300
  )
)

# Predict log-price for each quantile model on the grid
pred_mat_log <- sapply(fit_list, predict, newdata = pred_grid)

# Convert back from log10 to price
pred_df <- cbind(pred_grid, as.data.frame(pred_mat_log))
pred_df$days_since_start <- 10^pred_df$log_days
pred_df$date <- min(btc$date) + (round(pred_df$days_since_start) - 1)

# Rename columns for convenience
colnames(pred_df)[colnames(pred_df) == "tau_0.03"] <- "q_low"
colnames(pred_df)[colnames(pred_df) == "tau_0.5"]  <- "q_med"
colnames(pred_df)[colnames(pred_df) == "tau_0.97"] <- "q_high"

# Back-transform to actual prices
pred_df$q_low_price  <- 10^pred_df$q_low
pred_df$q_med_price  <- 10^pred_df$q_med
pred_df$q_high_price <- 10^pred_df$q_high

pred_df <- as.data.table(pred_df)

# --------------------------
# 4. Plot the trend corridor
# --------------------------
ggplot() +
  # Actual BTC price
  geom_line(data = btc[date>"2020-01-01"],
            aes(x = date, y = close),
            linewidth = 0.4, alpha = 0.7) +
  # Quantile trend lines (bands)
  geom_line(data = pred_df,
            aes(x = date, y = q_low_price),
            colour = "red", linewidth = 0.7, linetype = "dashed") +
  geom_line(data = pred_df,
            aes(x = date, y = q_med_price),
            colour = "blue", linewidth = 0.8) +
  geom_line(data = pred_df,
            aes(x = date, y = q_high_price),
            colour = "red", linewidth = 0.7, linetype = "dashed") +
  scale_y_log10() +
  labs(
    title = "Bitcoin Quantile Trend Regression (log–log power-law style)",
    x     = "Date",
    y     = "BTC price (log scale)"
  ) +
  theme_minimal()

# --------------------------
# 5. (Optional) Inspect fits
# --------------------------
summary(fit_list$tau_0.03)
summary(fit_list$tau_0.5)
summary(fit_list$tau_0.97)
