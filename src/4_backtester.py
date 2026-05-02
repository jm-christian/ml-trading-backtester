import pandas as pd
import matplotlib.pyplot as plt

def run_backtest(file_path):
    print(f"Loading predictions from {file_path}...")
    df = pd.read_csv(file_path, index_col=0, parse_dates=True)

    print("Simulating trades with $10,000 initial capital...")
    
    initial_capital = 10000

    # 1. Calculate "Buy and Hold" (The Benchmark)
    # What if we just bought SPY on day 1 and never touched it?
    # .cumprod() calculates compound growth over time.
    df['Buy_Hold_Value'] = initial_capital * (1 + df['Daily_Return']).cumprod()

    # 2. Calculate the Final AI Strategy (Long / Cash)
    # We remove the '-1' shorting logic. 
    # If the AI predicts 1, we earn the return. If it predicts 0, we earn 0% (Cash).
    df['AI_Return'] = df['Daily_Return'] * df['AI_Prediction'].shift(1)
    
    df['AI_Return'] = df['AI_Return'].fillna(0)
    df['AI_Strategy_Value'] = initial_capital * (1 + df['AI_Return']).cumprod()

    # 3. Print the Final Scoreboard
    bh_final = df['Buy_Hold_Value'].iloc[-1]
    ai_final = df['AI_Strategy_Value'].iloc[-1]

    print("\n--- 📈 FINAL BACKTEST RESULTS ---")
    print(f"Starting Capital:        ${initial_capital:,.2f}")
    print(f"Buy & Hold Final Value:  ${bh_final:,.2f}")
    print(f"AI Strategy Final Value: ${ai_final:,.2f}")
    
    # 4. Draw the Equity Curve
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['Buy_Hold_Value'], label='Buy & Hold (SPY)', color='gray', alpha=0.7)
    plt.plot(df.index, df['AI_Strategy_Value'], label='AI Trading Strategy', color='green', linewidth=2)
    
    plt.title('AI Trading Bot vs. Buy & Hold (Unseen Future Data)')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value ($)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save the chart as an image and pop it up on the screen
    plt.savefig("equity_curve.png")
    print("\nChart saved locally as equity_curve.png")
    plt.show()

if __name__ == "__main__":
    run_backtest("spy_predictions.csv")