
# 🧠 AutoTrader-AI: Autonomous Trading Agent with Deep Learning

AutoTrader-AI is a cutting-edge Python project that fuses finance, artificial intelligence, and deep learning to create an autonomous stock market prediction and decision-making system.

Built with `PyTorch`, `Yahoo Finance`, and real historical stock data, this AI-powered agent:
- Predicts market direction using a custom neural network
- Makes intelligent buy/sell decisions based on engineered indicators
- Visualizes performance and trends
- Is fully extensible for real-time deployment or dashboard visualization

---

## 🚀 Features

- 🔎 **Market Data Ingestion** — Fetches live historical stock prices using `yfinance`
- 🧮 **Feature Engineering** — Volatility, momentum, and log returns
- 🧠 **Neural Network Prediction** — Built in PyTorch for classifying price movement
- 📊 **Model Evaluation** — Confusion matrix, precision, recall, F1
- 📈 **Decision Engine** — Makes intelligent BUY/SELL decisions
- 🎨 **Visualization** — Plots historical returns and model output

---

## 🧑‍💻 Technologies Used

- Python 3.8+
- PyTorch
- Pandas / NumPy
- yFinance
- Matplotlib / Seaborn
- Scikit-learn

---

## 📂 Project Structure
```
autotrader-ai/ 
├── autotrader.py # Main script: modeling, training, evaluation, decision 
├── requirements.txt # All dependencies 
├── README.md # Project description 
└── models/ # (Optional) Saved model state_dict
```

---

## 🧪 Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/brettadams0/autotrader-ai.git
cd autotrader-ai
```
2. Install dependencies
```bash
pip install -r requirements.txt
```
3. Run the AI Trader

```bash
python autotrader.py
```

### 📜 License
MIT License. Use responsibly. This is for educational purposes only — not financial advice.
