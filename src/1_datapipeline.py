import yfinance as yf
import pandas as pd

def download_market_data(ticker_symbol, years=10):
    print(f"Fetching {years} years of data for {ticker_symbol}...")
    
    # yfinance makes this incredibly easy. We just ask for the history.
    ticker = yf.Ticker(ticker_symbol)
    df = ticker.history(period=f"{years}y")
    
    # We only care about the actual prices and volume, so we drop extra columns
    # like 'Dividends' and 'Stock Splits' to keep our data clean for the AI.
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    
    # Remove any rows with missing data
    df = df.dropna()
    
    print("Data successfully downloaded!\n")
    return df

if __name__ == "__main__":
    # We will use SPY (the S&P 500 ETF) as our test subject
    target_ticker = "SPY"
    
    historical_data = download_market_data(target_ticker)
    
    # Print the oldest 5 days and the newest 5 days to verify
    print("Oldest Data:")
    print(historical_data.head())
    print("\nNewest Data:")
    print(historical_data.tail())
    
    # Save it to a CSV file so we don't have to re-download it every time we run our AI
    historical_data.to_csv("spy_10yr_data.csv")
    print("\nSaved raw data to spy_10yr_data.csv")