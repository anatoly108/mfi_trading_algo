library(data.table)
library(openxlsx)

data <- data.table(read.xlsx("out/2024_09_07/analysis/2024_09_07_19_20_05/2024_09_07_19_20_05_crypto_mfi_analysis.xlsx"))
data <- data.table(read.xlsx("out/2024_09_07/analysis/2024_09_07_19_32_38/2024_09_07_19_32_38_crypto_mfi_analysis.xlsx"))

cor(data$total_profit, data$liquidity_score)
cor(data$total_profit, data$quoteVolume_raw)
cor(data$total_profit, data$trades_num)
cor(data$total_profit, data$asset_price_change)
max(data$liquidity_score)
min(data$liquidity_score)
data[symbol == "BTCUSDT"]

# binance: 70% of runs are positive, that's good
# mexc: 40% runs are pos
nrow(data[total_profit > 0]) / nrow(data)

# binance: although only 17% are positve after fees deduction :/
# mexc: 35% of runs are pos even after fees!
nrow(data[total_profit_minus_fees > 0]) / nrow(data)

boxplot(data$total_profit)
boxplot(data$total_profit_minus_fees)

# TODO: after many dry runs: calculate which combination of parameters gives most profitable result
# may be mexc is not so bad after all
