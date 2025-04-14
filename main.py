import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from datetime import datetime, timedelta

sns.set(style="darkgrid")

# Deep Learning Model for Predicting Market Movement
class MarketPredictor(nn.Module):
    def __init__(self, input_dim):
        super(MarketPredictor, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.model(x)

class AutoTrader:
    def __init__(self, tickers, start_date, end_date):
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.data = self.fetch_data()
        self.model = None

    def fetch_data(self):
        raw = yf.download(self.tickers, start=self.start_date, end=self.end_date)['Adj Close']
        returns = np.log(raw / raw.shift(1)).dropna()
        return returns

    def engineer_features(self):
        df = self.data.copy()
        df['Volatility'] = df.rolling(window=5).std()
        df['Momentum'] = df - df.shift(5)
        df['Target'] = np.where(df.shift(-1).mean(axis=1) > df.mean(axis=1), 1, 0)
        df = df.dropna()
        return df

    def train_model(self):
        df = self.engineer_features()
        X = df.drop('Target', axis=1).values
        y = df['Target'].values

        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.LongTensor(y_train)
        X_test_tensor = torch.FloatTensor(X_test)
        y_test_tensor = torch.LongTensor(y_test)

        self.model = MarketPredictor(X_train.shape[1])
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

        for epoch in range(100):
            self.model.train()
            outputs = self.model(X_train_tensor)
            loss = criterion(outputs, y_train_tensor)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

        self.evaluate(X_test_tensor, y_test_tensor)

    def evaluate(self, X_test, y_test):
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_test)
            _, predicted = torch.max(outputs, 1)
        print("\nEvaluation Results:")
        print(confusion_matrix(y_test, predicted))
        print(classification_report(y_test, predicted))

    def make_decision(self):
        latest_data = self.engineer_features().iloc[-1:].drop('Target', axis=1)
        latest_data_scaled = StandardScaler().fit_transform(latest_data)
        latest_tensor = torch.FloatTensor(latest_data_scaled)
        self.model.eval()
        with torch.no_grad():
            prediction = self.model(latest_tensor)
        signal = torch.argmax(prediction).item()
        action = "BUY" if signal == 1 else "SELL"
        print(f"\nDecision based on AI model: {action}")

    def plot_data(self):
        self.data.plot(figsize=(12, 6))
        plt.title("Historical Log Returns")
        plt.ylabel("Log Return")
        plt.xlabel("Date")
        plt.show()

if __name__ == "__main__":
    tickers = ['AAPL', 'MSFT', 'GOOGL']
    start_date = '2015-01-01'
    end_date = datetime.today().strftime('%Y-%m-%d')

    bot = AutoTrader(tickers, start_date, end_date)
    print("\nTraining AI model to predict market direction...")
    bot.train_model()

    print("\nAI-Powered decision based on recent data:")
    bot.make_decision()

    print("\nPlotting market data...")
    bot.plot_data()
