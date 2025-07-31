import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import joblib
import tensorflow as tf
import keras_tuner as kt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

warnings.filterwarnings('ignore')

# --- Configuration ---
DATASET_PATH = 'Historical Product Demand.csv'
ARIMA_MODEL_PATH = 'saved_models/arima_model.joblib'
BEST_LSTM_MODEL_PATH = 'saved_models/best_lstm_model.keras'
LOOK_BACK = 6 # Use previous 12 months to predict the next

# --- Utility Functions ---
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    non_zero_mask = y_true != 0
    # Return 0 if all true values are zero
    if not np.any(non_zero_mask):
        return 0.0
    return np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100

# --- 1. Data Loading and Feature Engineering ---
print("--- 1. Loading Data & Feature Engineering ---")
os.makedirs('saved_models', exist_ok=True)

try:
    df = pd.read_csv(DATASET_PATH)
except FileNotFoundError:
    print(f"Error: '{DATASET_PATH}' not found.")
    exit()

df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df.dropna(subset=['Date'], inplace=True)
df['Order_Demand'] = df['Order_Demand'].str.replace(r'[()]', '', regex=True)
df['Order_Demand'] = pd.to_numeric(df['Order_Demand'], errors='coerce')
df.dropna(subset=['Order_Demand'], inplace=True)

top_product = df['Product_Code'].value_counts().idxmax()
print(f"Analyzing Product: {top_product}")

monthly_df = df[df['Product_Code'] == top_product].set_index('Date').resample('M')[['Order_Demand']].sum()
monthly_df['month'] = monthly_df.index.month
monthly_df['year'] = monthly_df.index.year

features = ['Order_Demand', 'month', 'year']
data_to_forecast = monthly_df[features].copy()

# Split data
train_size = int(len(data_to_forecast) * 0.8)
train_df, test_df = data_to_forecast[:train_size], data_to_forecast[train_size:]
print(f"\nTraining data size: {len(train_df)}, Testing data size: {len(test_df)}")

# --- ROBUSTNESS CHECK ---
if len(test_df) <= LOOK_BACK:
    print(f"\n❌ ERROR: Test data size ({len(test_df)}) is not greater than the LOOK_BACK period ({LOOK_BACK}).")
    print("Cannot create test sequences. Please choose a product with more data or reduce the LOOK_BACK value.")
    exit()

# --- 2. ARIMA Model (Univariate) ---
print("\n--- 2. Building/Loading ARIMA Model ---")
arima_train_data = train_df['Order_Demand']

if os.path.exists(ARIMA_MODEL_PATH):
    print("Loading saved ARIMA model...")
    arima_fit = joblib.load(ARIMA_MODEL_PATH)
else:
    print("Training new ARIMA model...")
    arima_model = ARIMA(arima_train_data, order=(5, 1, 5))
    arima_fit = arima_model.fit()
    joblib.dump(arima_fit, ARIMA_MODEL_PATH)
    print("ARIMA model trained and saved.")

arima_predictions = arima_fit.forecast(steps=len(test_df))

# --- 3. Multivariate LSTM Model with Hyperparameter Tuning ---
print("\n--- 3. Building/Loading LSTM Model with Keras Tuner ---")

scaler = MinMaxScaler(feature_range=(0, 1))
train_scaled = scaler.fit_transform(train_df)
test_scaled = scaler.transform(test_df)

def create_multivariate_dataset(dataset, target_col_index, look_back=1):
    X, Y = [], []
    for i in range(len(dataset) - look_back):
        X.append(dataset[i:(i + look_back), :])
        Y.append(dataset[i + look_back, target_col_index])
    return np.array(X), np.array(Y)

target_col_idx = features.index('Order_Demand')
X_train, y_train = create_multivariate_dataset(train_scaled, target_col_idx, LOOK_BACK)
X_test, y_test = create_multivariate_dataset(test_scaled, target_col_idx, LOOK_BACK)

def build_model(hp):
    model = tf.keras.Sequential()
    model.add(LSTM(
        units=hp.Int('units_1', min_value=50, max_value=150, step=25),
        return_sequences=True,
        input_shape=(X_train.shape[1], X_train.shape[2])
    ))
    model.add(Dropout(rate=hp.Float('dropout_1', min_value=0.1, max_value=0.5, step=0.1)))
    model.add(LSTM(
        units=hp.Int('units_2', min_value=30, max_value=100, step=20)
    ))
    model.add(Dropout(rate=hp.Float('dropout_2', min_value=0.1, max_value=0.5, step=0.1)))
    model.add(Dense(1))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])),
        loss='mean_squared_error'
    )
    return model

if os.path.exists(BEST_LSTM_MODEL_PATH):
    print("Loading best saved LSTM model...")
    lstm_model = load_model(BEST_LSTM_MODEL_PATH, custom_objects={'mse': 'mean_squared_error'})
else:
    print("Tuning LSTM hyperparameters to find the best model...")
    tuner = kt.RandomSearch(
        build_model,
        objective='val_loss',
        max_trials=10,
        executions_per_trial=1,
        directory='keras_tuner_dir',
        project_name='demand_forecasting',
        overwrite=True
    )
    early_stop = EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)
    tuner.search(X_train, y_train, epochs=100, validation_data=(X_test, y_test), callbacks=[early_stop])
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    print(f"\nBest HPs: units_1={best_hps.get('units_1')}, dropout_1={best_hps.get('dropout_1')}, units_2={best_hps.get('units_2')}, dropout_2={best_hps.get('dropout_2')}, lr={best_hps.get('learning_rate')}")
    lstm_model = tuner.get_best_models(num_models=1)[0]
    lstm_model.save(BEST_LSTM_MODEL_PATH)
    print("\nBest LSTM model saved.")

lstm_predictions_scaled = lstm_model.predict(X_test)
dummy_array = np.zeros((len(lstm_predictions_scaled), len(features)))
dummy_array[:, target_col_idx] = lstm_predictions_scaled.flatten()
lstm_predictions = scaler.inverse_transform(dummy_array)[:, target_col_idx]

# --- 4. Model Evaluation & Comparison ---
print("\n--- 4. Model Evaluation & Comparison ---")
actuals = test_df['Order_Demand'].values[LOOK_BACK:]
arima_preds = arima_predictions.values[LOOK_BACK:]
lstm_preds = lstm_predictions

arima_rmse = np.sqrt(mean_squared_error(actuals, arima_preds))
arima_mape = mean_absolute_percentage_error(actuals, arima_preds)
lstm_rmse = np.sqrt(mean_squared_error(actuals, lstm_preds))
lstm_mape = mean_absolute_percentage_error(actuals, lstm_preds)

print("\nModel Performance on Test Set:")
print(f"ARIMA - RMSE: {arima_rmse:.2f}, MAPE: {arima_mape:.2f}%")
print(f"LSTM  - RMSE: {lstm_rmse:.2f}, MAPE: {lstm_mape:.2f}%")

# --- 5. Inventory Optimization ---
print("\n--- 5. Inventory Optimization ---")
if lstm_rmse < arima_rmse:
    print("✅ AI (LSTM) model performed better. Using its forecast.")
    best_forecast = lstm_preds
    forecast_error_std = np.std(actuals - lstm_preds)
else:
    print("Classic (ARIMA) model performed better. Using its forecast.")
    best_forecast = arima_preds
    forecast_error_std = np.std(actuals - arima_preds)

holding_cost_per_unit_per_year = 2.50
order_cost_per_order = 50.00
lead_time_in_months = 1
service_level_z_score = 1.65

forecasted_monthly_demand = best_forecast.mean()
annual_demand = forecasted_monthly_demand * 12
eoq = np.sqrt((2 * annual_demand * order_cost_per_order) / holding_cost_per_unit_per_year) if holding_cost_per_unit_per_year > 0 else 0
safety_stock = service_level_z_score * forecast_error_std * np.sqrt(lead_time_in_months)
reorder_point = (forecasted_monthly_demand * lead_time_in_months) + safety_stock

print("\nInventory Parameters (based on the best model):")
print(f"Forecasted Average Monthly Demand: {forecasted_monthly_demand:.0f} units")
print(f"Optimal Order Quantity (EOQ): {eoq:.0f} units per order")
print(f"Safety Stock: {safety_stock:.0f} units")
print(f"Reorder Point (ROP): {reorder_point:.0f} units")

# --- 6. Visualization ---
print("\n--- 6. Debugging Plot Data ---")
plot_index = test_df.index[LOOK_BACK:]
print(f"Plotting {len(plot_index)} data points.")
print(f"Actuals shape: {actuals.shape}")
print(f"ARIMA preds shape: {arima_preds.shape}")
print(f"LSTM preds shape: {lstm_preds.shape}")

plt.figure(figsize=(16, 8))
plt.title(f'Demand Forecast vs. Actuals for Product {top_product}')
plt.plot(plot_index, actuals, label='Actual Demand', color='blue', marker='o')
plt.plot(plot_index, arima_preds, label=f'ARIMA Forecast (MAPE: {arima_mape:.2f}%)', color='orange', linestyle='--')
plt.plot(plot_index, lstm_preds, label=f'AI/LSTM Forecast (MAPE: {lstm_mape:.2f}%)', color='green', linestyle='--')
plt.xlabel('Date')
plt.ylabel('Total Demand')
plt.legend()
plt.grid(True)
plt.show()