# 🤖 AI-Driven Demand Forecasting and Inventory Optimization

This project compares a classical statistical model (**ARIMA**) with an advanced AI model (**LSTM**) for demand forecasting. The study shows how superior forecast accuracy from a well-configured AI model can lead to better inventory optimization parameters like Economic Order Quantity (**EOQ**) and Reorder Point (**ROP**).



---

## Project Highlights

* **Two-Phase Experiment**: A direct comparison between a simple univariate **ARIMA** and **LSTM**, followed by an advanced multivariate **LSTM** to showcase the power of feature engineering.
* **AI Model Optimization**: Utilizes **Keras Tuner** to automatically find the optimal hyperparameters for the **LSTM** architecture, ensuring peak performance.
* **Practical Application**: Integrates the best forecast into a standard inventory management model to calculate key operational metrics.
* **Model Persistence**: Saves trained models to disk to avoid time-consuming retraining on subsequent runs.

---

## 📊 Key Results

The study concluded that while **ARIMA** provided a strong baseline, the advanced, multivariate **LSTM** model was significantly more accurate.

### Final Model Comparison

| Model           | RMSE           | MAPE      |
| :-------------- | :------------- | :-------- |
| **ARIMA** | 2,858,881.81   | 1035.72%  |
| **Advanced LSTM** | 2,649,523.91   | 877.76%   |

This represents a **7.3% improvement** in Root Mean Squared Error (**RMSE**) and a **15.2% improvement** in Mean Absolute Percentage Error (**MAPE**) by using the optimized AI model.

---

## 🚀 How to Run This Project

### 1. Prerequisites
* Python 3.9+
* A virtual environment is recommended.
* 
---

## 2. Clone the Repository
```bash
git clone [https://github.com/](https://github.com/)[AaronTM44]/AI-Driven-Demand-Forecasting-and-Inventory-Optimization.git
cd AI-Driven-Demand-Forecasting-and-Inventory-Optimization

---

## 3. Install Dependencies
Install the required Python libraries using the `requirements.txt` file.
```bash
pip install -r requirements.txt

---

## 4. Download the Dataset
Download the "Forecasts for Product Demand" dataset from Kaggle and place the Historical Product Demand.csv file in the root of the project directory.

---

## 5.Run the Script
Execute the main Python script from your terminal.
```bash
python demand_forecasting.py
