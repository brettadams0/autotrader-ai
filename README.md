# autotrader-ai

A small PyTorch classifier that tries to predict whether tomorrow's average return across a basket
of tickers will beat today's, trained on log returns pulled from Yahoo Finance.

This is a learning exercise in wiring up the whole pipeline — fetch, feature-engineer, train,
evaluate, predict — not a trading system. The "Honest evaluation" section below explains why its
reported accuracy is not what it looks like.

## Running it

```sh
pip install -r requirements.txt
python main.py
```

Out of the box it pulls `AAPL`, `MSFT` and `GOOGL` from 2015-01-01 to today, trains for 100 epochs,
prints a confusion matrix and classification report, prints a BUY/SELL call for the most recent bar,
and opens a matplotlib window of the historical log returns. Edit the `tickers` and `start_date`
values at the bottom of `main.py` to change that.

Needs network access for the `yfinance` download.

## How it works

| Stage | |
|---|---|
| `fetch_data` | Downloads adjusted closes and converts them to daily log returns |
| `engineer_features` | Per ticker: the return, a 5-day rolling standard deviation, and 5-day momentum |
| `train_model` | Standard-scales, splits 80/20 without shuffling, trains a 128→64→2 MLP with dropout, Adam, cross-entropy |
| `evaluate` | Confusion matrix and per-class precision/recall/F1 on the held-out tail |
| `make_decision` | Applies the *training* scaler to the latest row and reports BUY or SELL |

The train/test split is deliberately `shuffle=False` — shuffling time series would let the model
train on data from after the test period.

## Honest evaluation

The accuracy this prints is inflated by how the label is constructed, and it is worth understanding
before reading anything into it.

`Target` is `1` when the mean return at `t+1` exceeds the mean return at `t`, while the features at
row `t` include the returns at `t`. Daily returns are close to zero-mean, so an unusually large
up-day at `t` makes "`t+1` is lower than `t`" nearly certain — regardless of anything about the
market. The model can score well by learning that arithmetic relationship.

Running this pipeline against a synthetic random walk — data with no predictable structure at all,
by construction — still yields about 71% accuracy. That is the size of the artifact. A genuine
evaluation would need a label defined purely on future data (for example, the sign of the return at
`t+1`), and a baseline comparison against always predicting the majority class.

## Scope

Educational. It has no backtest, no transaction costs, no position sizing, and no risk management,
and it does not save or reload a trained model. Not financial advice.

## License

MIT — see [LICENSE](LICENSE).
