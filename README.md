# AI-Driven-Demand-Forecasting-and-Inventory-Optimization
A project comparing ARIMA and LSTM models for demand forecasting and inventory optimization using real-world data.

AI-Driven Demand Forecasting and Inventory Optimization
This project provides a comprehensive analysis comparing a classical statistical model (ARIMA) with an advanced AI model (LSTM) for demand forecasting. The study demonstrates how superior forecast accuracy from a well-configured AI model can be translated into actionable inventory optimization parameters (EOQ and ROP).

The full analysis, methodology, and findings are detailed in the ai_driven_demand_forcasting.docx report.

Project Highlights
Two-Phase Experiment: A direct comparison between a simple univariate ARIMA and LSTM, followed by an advanced multivariate LSTM to showcase the power of feature engineering.

AI Model Optimization: Utilizes Keras Tuner to automatically find the optimal hyperparameters for the LSTM architecture, ensuring peak performance.

Practical Application: Integrates the best forecast into a standard inventory management model to calculate key operational metrics.

Model Persistence: Saves trained models to disk to avoid time-consuming retraining on subsequent runs.

Key Results
The study concluded that while ARIMA provided a strong baseline, the advanced, multivariate LSTM model was significantly more accurate.

Final Model Comparison:

ARIMA: RMSE of 2,858,881.81, MAPE of 1035.72%

Advanced LSTM: RMSE of 2,649,523.91, MAPE of 877.76%

This represents a 7.3% improvement in RMSE and a 15.2% improvement in MAPE by using the optimized AI model.

How to Run This Project
1. Prerequisites
Python 3.9+

A virtual environment is recommended.

2. Clone the Repository
git clone https://github.com/[AaronTM44]/AI-Driven-Demand-Forecasting-and-Inventory-Optimization.git
cd AI-Driven-Demand-Forecasting-and-Inventory-Optimization

3. Install Dependencies
Install the required Python libraries using the requirements.txt file.

pip install -r requirements.txt



4. Download the Dataset
Download the "Forecasts for Product Demand" dataset from Kaggle and place the Historical Product Demand.csv file in the root of the project directory.

5. Run the Script
Execute the main Python script from your terminal.

python demand_forecasting.py
