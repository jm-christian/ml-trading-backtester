import pandas as pd
import numpy as np

def calculate_rsi(data, window=14):
    """Calculates the Relative Strength Index (RSI)"""
    delta = data.diff()
    # Separate the gains and losses
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    
    # Calculate the RSI formula
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def engineer_features(file_path):
    print(f"Loading raw data from {file_path}...")
    df = pd.read_csv(file_path, index_col=0, parse_dates=True)
    print("Calculating advanced technical indicators...")
    
    # 1. Existing Features
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['RSI_14'] = calculate_rsi(df['Close'])
    df['Daily_Return'] = df['Close'].pct_change()

    # 2. NEW: MACD (Momentum)
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2

    # 3. NEW: Bollinger Band Width (Volatility)
    sma_20 = df['Close'].rolling(window=20).mean()
    std_20 = df['Close'].rolling(window=20).std()
    bb_upper = sma_20 + (std_20 * 2)
    bb_lower = sma_20 - (std_20 * 2)
    df['BB_Width'] = (bb_upper - bb_lower) / sma_20

    # 4. The Target
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)

    df = df.dropna()
    return df

if __name__ == "__main__":
    # Run the function on our downloaded data
    engineered_df = engineer_features("spy_10yr_data.csv")
    
    print("\nFeature Engineering Complete! Here is a sneak peek at what the AI will see:")
    # Print just a few columns to verify it worked
    print(engineered_df[['Close', 'SMA_10', 'RSI_14', 'Target']].tail())
    
    # Save this processed, perfectly clean data for the AI to train on
    engineered_df.to_csv("spy_engineered_data.csv")
    print("\nSaved ready-to-train data to spy_engineered_data.csv")