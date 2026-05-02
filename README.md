# 📈 Algorithmic Trading Backtester (XGBoost)

## 📌 Overview
An end-to-end algorithmic trading pipeline that uses Machine Learning (XGBoost) to predict the daily price direction of the S&P 500 (SPY). The project features automated data fetching, feature engineering, and a custom backtesting engine to simulate trading performance against a Buy & Hold benchmark.

## 🛠️ Tech Stack
*   **Language:** Python
*   **Data Acquisition:** `yfinance`
*   **Data Manipulation:** `pandas`, `numpy`
*   **Machine Learning:** `xgboost`, `scikit-learn`
*   **Visualization:** `matplotlib`

## 🏗️ System Architecture
1.  **Data Pipeline:** Automatically downloads 10 years of historical daily data, handling missing values and cleaning the dataset.
2.  **Feature Engineering:** Translates raw price data into mathematical context for the AI. Features tested include:
    *   Simple Moving Averages (10-day & 50-day)
    *   Relative Strength Index (RSI - 14-day)
    *   MACD & Bollinger Band Width
3.  **Prediction Engine:** Uses an `XGBClassifier` to predict binary market direction. *Strict chronological splitting (80/20) was enforced to prevent Look-Ahead Bias.*
4.  **Backtest Simulation:** A custom Python loop that simulates a $10,000 starting portfolio, calculating compound returns and generating equity curves.

## 📊 Key Findings & Quant Insights
During the backtesting phase on unseen future data, the model achieved a **direction accuracy of >57%**. 

**Strategy & Feature Analysis:**
*   **The Danger of Shorting a Bull Market:** Initially, the model was programmed to short the market when predicting a downturn. Backtesting revealed massive beta-drag; shorting a strong bull market destroyed Alpha. Moving to a "Long/Cash" strategy vastly improved risk-adjusted returns.
*   **Simplicity vs. Complexity:** Hypothesis: Adding complex momentum/volatility indicators (MACD, Bollinger Bands) would improve returns. Result: The complex model actually underperformed ($12,119) compared to the simpler baseline model using only RSI and MAs ($12,615). Conclusion: Over-engineering features can introduce noise, causing the model to miss standard trend days. 

## 🚀 How to Run Locally
1. Clone the repository: `git clone https://github.com/yourusername/ml-trading-backtester.git`
2. Create a virtual environment: `python -m venv venv`
3. Activate the environment and install dependencies: `pip install -r requirements.txt`
4. Run the pipeline in order:
   ```bash
   python src/1_data_pipeline.py
   python src/2_feature_engineering.py
   python src/3_model_training.py
   python src/4_backtester.py