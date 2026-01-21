import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report, mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error, roc_auc_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
import joblib
import matplotlib.pyplot as plt
import cupy as cp
import optuna
import json
from optuna.integration import XGBoostPruningCallback

cp.cuda.Device(0).use()

def hist(data, bins=10, title="", x_label="", y_label="Frequency"):
    """
    Create a quick histogram using Matplotlib.

    Parameters:
    - data: List or array of data points.
    - bins: Number of bins for the histogram.
    - title: Title for the histogram.
    - x_label: Label for the x-axis.
    - y_label: Label for the y-axis.
    """
    plt.hist(data, bins=bins, edgecolor='black')
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()

def table(data):
    """
    Create a frequency table from categorical data.

    Parameters:
    - data: List or array of categorical data.

    Returns:
    - freq_table: A dictionary where keys are unique values and values are frequencies.
    """
    freq_table = {}
    for item in data:
        if item in freq_table:
            freq_table[item] += 1
        else:
            freq_table[item] = 1
    return freq_table

# Specify the directory containing the CSV files
# directory = '/Users/anatoly/Downloads/2024_09_15/2024_09_15/grand_analysis/2024_09_15_14_15_18_Binance'
directory = '/home/anatoly/do/t/out/binance_grand_analysis_6months_2hours_merged'

# List to store each dataframe
dataframes = []

def file_contains_non_empty_lines(filepath):
    with open(filepath, 'r') as file:
        for line in file:
            if line.strip():  # Check if the line is not empty
                return True
    return False

# Iterate over all files in the directory
for filename in os.listdir(directory):
    if filename.endswith('.csv') and 'results' not in filename and 'trades' not in filename:
        # print(filename)
        filepath = os.path.join(directory, filename)
        if not file_contains_non_empty_lines(filepath):
            continue

        # Read the CSV file into a DataFrame
        df = pd.read_csv(filepath)
        
        # Sort the DataFrame by the 'timepoint' column in ascending order
        df = df.sort_values(by='timepoint', ascending=True)
        
        # Create a new column 'total_profit_future_timeframe' with the next row's 'total_profit'
        # df['total_profit_future_timeframe'] = df['total_profit'].shift(-1)
        df['pnl_future_timeframe'] = df['pnl'].shift(-1)

        # df['total_profit_past_timeframe'] = df['total_profit'].shift(1)
        # df['pnl_past_timeframe'] = df['pnl'].shift(1)
        
        # Append the processed DataFrame to the list
        dataframes.append(df)

# Concatenate all dataframes into one
data = pd.concat(dataframes, ignore_index=True)
# data = data[data["total_profit_future_timeframe"].notna()]
# data = data[data["total_profit_past_timeframe"].notna()]
# data = data[data["pnl_past_timeframe"].notna()]
data = data[data["pnl_future_timeframe"].notna()]
# clean-up
data = data[data["empty_candles_fraction"] < 0.05].reset_index(drop=True)
data = data[data["quote_volume"] > 500e3].reset_index(drop=True)
print(data.shape[0])

# ----
# option 1: regression, predict pnl
# data["Y"] = data["pnl_future_timeframe"]

# option 2: classification, 5 classes
# data["Y"] = -1
# data.loc[data["pnl_future_timeframe"]<-2, "Y"] = 0
# data.loc[(data["pnl_future_timeframe"]>=-2) & (data["pnl_future_timeframe"]<0), "Y"] = 1
# data.loc[(data["pnl_future_timeframe"]==0), "Y"] = 2
# data.loc[(data["pnl_future_timeframe"]>0) & (data["pnl_future_timeframe"]<=2), "Y"] = 3
# data.loc[(data["pnl_future_timeframe"]>2), "Y"] = 4
# data["Y"] = data["Y"].astype(int)
# table(data["Y"])

# option 3: classification, 3 classes
data.loc[data["pnl_future_timeframe"]<0, "Y"] = 0
data.loc[(data["pnl_future_timeframe"]==0), "Y"] = 1
data.loc[(data["pnl_future_timeframe"]>0), "Y"] = 2
data["Y"] = data["Y"].astype(int)
table(data["Y"])

# ----

data.columns

# Handle missing values
data = data.sort_values(by='timepoint', ascending=True)
xgboost_columns = ['pnl', 'orders_num', 'trades_num', 
        'stop_loss_n', 'neg_profit_interrupted',
       'asset_price_change', 'range_bound_score',
       'volatility_score', 'quote_volume', 'win_rate', 'average_trade_profit',
       'max_drawdown', 'sharpe_ratio', 'average_holding_time',
       'time_in_market', 'atr', 'rsi', 'bb_width', 'roc', 'std_dev_returns',
       'kurtosis', 'vwap', 'average_daily_volume',
       'volume_volatility', 'macd_line', 'signal_line', 'macd_histogram',
       'empty_candles_fraction', 'ema100_start_normalized', 
       'ema100_latest_normalized', 'ema200_start_normalized',
       'ema200_latest_normalized', "Y"]

data = data[xgboost_columns]
data = data.dropna()
np.sum(data == np.Inf)
np.sum(data == np.nan)

# Define features and target
y = data['Y']
X = data.drop(['Y'], axis=1)

# Convert to numpy arrays
feature_names = X.columns.tolist()
X = X.values
y = y.values
table(y)

# Split data while maintaining temporal order
train_size = int(len(data) * 0.8)
X_train = X[:train_size]
y_train = y[:train_size]
X_test = X[train_size:]
y_test = y[train_size:]

# Standardize features
scaler = StandardScaler()
X_train_scaled_np = scaler.fit_transform(X_train)
X_test_scaled_np = scaler.transform(X_test)

# Convert to cupy arrays
X_train_scaled = cp.array(X_train_scaled_np)
X_test_scaled = cp.array(X_test_scaled_np)
y_train_cp = cp.array(y_train)
y_test_cp = cp.array(y_test)

# Set up TimeSeriesSplit
n_splits = 5
n_classes = len(np.unique(y))

# Define the objective function for Optuna
def objective(trial):
    param = {
        'objective': 'multi:softprob',
        'eval_metric': 'mlogloss',
        'device': 'cuda',
        'random_state': 42,
        'num_class': n_classes,
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 0.5),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
        'max_leaves': trial.suggest_int('max_leaves', 15, 127),
    }
    
    tscv = TimeSeriesSplit(n_splits=3)
    auc_scores = []
    
    for train_index, valid_index in tscv.split(X_train_scaled):
        X_tr = X_train_scaled[train_index]
        X_val = X_train_scaled[valid_index]
        y_tr = y_train_cp[train_index]
        y_val = y_train_cp[valid_index]
        
        # Create DMatrix
        dtrain = xgb.DMatrix(X_tr, label=y_tr)
        dval = xgb.DMatrix(X_val, label=y_val)
        evals = [(dval, 'validation')]
        
        # Set num_boost_round - max number of trees
        # actual number of trees will most likely be less 
        # because we set early_stopping_rounds=50
        # and training will stop if no progress is visible in 50 last rounds
        num_boost_round = 1000
        
        # Train the model
        xgb_model = xgb.train(
            params=param,
            dtrain=dtrain,
            num_boost_round=num_boost_round,
            evals=evals,
            early_stopping_rounds=50,
            verbose_eval=False
        )
        
        # Predict probabilities
        y_val_pred_proba_cp = xgb_model.predict(dval)
        
        # Convert cupy arrays to numpy arrays for scikit-learn metrics
        y_val_np = cp.asnumpy(y_val)
        y_val_pred_proba = cp.asnumpy(y_val_pred_proba_cp)
        
        y_val_binarized = label_binarize(y_val_np, classes=np.arange(n_classes))
        
        # Compute ROC AUC using micro-average
        auc_score = roc_auc_score(y_val_binarized, y_val_pred_proba, average='micro', multi_class='ovr')
        auc_scores.append(auc_score)
    
    # Return the average AUC over the folds
    return 1.0 - np.mean(auc_scores)  # Minimize 1 - AUC

# Create an Optuna study
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)

# Print the best parameters
print("Best trial:")
trial = study.best_trial

print("  Value (1 - AUC): {}".format(trial.value))
print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))

# Train the final model with the best parameters
best_params = trial.params
best_params.update({
    "objective": 'multi:softprob',
    "random_state": 42,
    "eval_metric": 'mlogloss',
    "device": "cuda",  # Use 'gpu_hist' for GPU acceleration, 'hist' for CPU
    "num_class": n_classes
})

# Convert training data to DMatrix
dtrain_full = xgb.DMatrix(X_train_scaled, label=y_train_cp)
dtest = xgb.DMatrix(X_test_scaled)

# Set num_boost_round
num_boost_round = 1000

# Train the final model
final_model = xgb.train(
    params=best_params,
    dtrain=dtrain_full,
    num_boost_round=num_boost_round,
    early_stopping_rounds=50,
    evals=[(dtrain_full, 'train')],
    verbose_eval=False
)

# Predict probabilities on the test set
y_pred_proba_cp = final_model.predict(dtest)
y_pred = np.argmax(cp.asnumpy(y_pred_proba_cp), axis=1)

# Convert cupy arrays to numpy arrays
y_test_np = cp.asnumpy(y_test_cp)
y_pred_proba = cp.asnumpy(y_pred_proba_cp)

# Binarize the labels for ROC computation
y_test_binarized = label_binarize(y_test_np, classes=np.arange(n_classes))

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_pred_proba[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test_binarized.ravel(), y_pred_proba.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Compute macro-average ROC curve and ROC area
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
mean_tpr = np.zeros_like(all_fpr)

for i in range(n_classes):
    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

mean_tpr /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

print(f"Test ROC AUC (micro-average): {roc_auc['micro']}")
print(f"Test ROC AUC (macro-average): {roc_auc['macro']}")

# Classification report
print("Classification Report:")
print(classification_report(y_test_np, y_pred))

# Confusion matrix
conf_matrix = confusion_matrix(y_test_np, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# Plot ROC Curve for each class
plt.figure()
lw = 2
for i in range(n_classes):
    plt.plot(fpr[i], tpr[i],
             lw=lw, label=f'ROC curve of class {i} (area = {roc_auc[i]:0.2f})')
plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Multiclass XGBoost Classifier')
plt.legend(loc="lower right")
plt.show()

# Plot feature importances
# Get feature importances from the model
importance = final_model.get_score(importance_type='gain')

# Map feature indices to feature names
importance = {feature_names[int(k[1:])]: v for k, v in importance.items()}  # k[1:] removes 'f' prefix

# Sort the feature importances
importance = dict(sorted(importance.items(), key=lambda item: item[1], reverse=True))

# Plot the feature importances
plt.figure(figsize=(10, 8))
plt.barh(list(importance.keys())[::-1], list(importance.values())[::-1])
plt.title('Feature Importances')
plt.xlabel('Gain')
plt.ylabel('Features')
plt.tight_layout()
plt.show()

# Save the model and scaler
joblib.dump(final_model, '2024_09_21_binance_6months_2hours_xgboost.pkl')
joblib.dump(scaler, '2024_09_21_binance_6months_2hours_scaler.pkl')



# ---------
# check if very high score class 3 is what I need
hist(y_pred_proba[:,2])
high_confidence = np.quantile(y_pred_proba[:,2], 0.9)
high_confidence_real_y = y_test_np[y_pred_proba[:,2] >= high_confidence]
high_confidence_real_y_table = table(high_confidence_real_y)
high_confidence_real_y_df = pd.DataFrame.from_dict(high_confidence_real_y_table, orient='index', columns=['Value']).reset_index()
high_confidence_real_y_df.columns = ['Class', 'Value']
high_confidence_real_y_df["Chance"] = high_confidence_real_y_df["Value"]/len(high_confidence_real_y)
high_confidence_real_y_df.sort_values(by="Class", inplace=True)
high_confidence_real_y_df


total_real_y = table(data["Y"])
total_real_y_df = pd.DataFrame.from_dict(total_real_y, orient='index', columns=['Value']).reset_index()
total_real_y_df.columns = ['Class', 'Value']
total_real_y_df["Chance"] = total_real_y_df["Value"]/len(data["Y"])
total_real_y_df.sort_values(by="Class", inplace=True)
total_real_y_df


total_real_y_high_pnl = table(data[data["pnl"] > 1]["Y"])
total_real_y_high_pnl_df = pd.DataFrame.from_dict(total_real_y_high_pnl, orient='index', columns=['Value']).reset_index()
total_real_y_high_pnl_df.columns = ['Class', 'Value']
total_real_y_high_pnl_df["Chance"] = total_real_y_high_pnl_df["Value"]/len(data[data["pnl"] > 1])
total_real_y_high_pnl_df["Chance"] = total_real_y_high_pnl_df["Value"]/len(data[data["pnl"] > 1])
total_real_y_high_pnl_df.sort_values(by="Class", inplace=True)
total_real_y_high_pnl_df


# -----------
model_values = {
    "high_confidence_score": high_confidence,
    "best_params": best_params,
    "xgboost_columns": xgboost_columns
}

model_values_string = json.dumps(model_values, indent=4)  # indent=4 for pretty printing

with open('2024_09_21_binance_6months_2hours_values.json', 'w') as f:
    f.write(model_values_string)

