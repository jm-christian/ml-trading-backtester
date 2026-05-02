import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score

def train_model(file_path):
    print(f"Loading engineered data from {file_path}...")
    df = pd.read_csv(file_path, index_col=0, parse_dates=True)

    # Define our Features (X) and our Target (y)
    # Define our UPGRADED Features (X) and our Target (y)
    features = ['Close', 'SMA_10', 'SMA_50', 'RSI_14', 'Daily_Return', 'MACD', 'BB_Width']
    X = df[features]
    y = df['Target']

    # CRITICAL: Chronological Split (No Look-Ahead Bias)
    # We train on the past (first 80%) and test on the future (last 20%).
    split_index = int(len(df) * 0.8)
    
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    print(f"Training on {len(X_train)} days, testing on {len(X_test)} days...")

    # Initialize the XGBoost model
    # We use a shallow tree (max_depth=3) to prevent the AI from over-memorizing the noise.
    model = xgb.XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)

    # Train the brain!
    model.fit(X_train, y_train)
    print("Model training complete!\n")

    # Make predictions on the unseen future data
    predictions = model.predict(X_test)
    
    # Check our baseline accuracy
    accuracy = accuracy_score(y_test, predictions)
    print(f"AI Accuracy on Unseen Data: {accuracy * 100:.2f}%")
    
    # Save the predictions alongside the actual prices for Phase 4 (Backtesting)
    test_results = df.iloc[split_index:].copy()
    test_results['AI_Prediction'] = predictions
    
    return test_results

if __name__ == "__main__":
    results_df = train_model("spy_engineered_data.csv")
    
    # Save to a new CSV for the final backtesting phase
    results_df.to_csv("spy_predictions.csv")
    print("\nSaved predictions to spy_predictions.csv")