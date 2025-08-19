# Prediction and Auto-Trading Script

```
 ░▒▓██████▓▒░░▒▓█▓▒░░▒▓█▓▒░▒▓████████▓▒░▒▓██████▓▒░▒▓████████▓▒░▒▓███████▓▒░ ░▒▓██████▓▒░░▒▓███████▓▒░░▒▓████████▓▒░ 
░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░  ░▒▓█▓▒░  ░▒▓█▓▒░░▒▓█▓▒░ ░▒▓█▓▒░   ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░        
░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░  ░▒▓█▓▒░  ░▒▓█▓▒░░▒▓█▓▒░ ░▒▓█▓▒░   ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░        
░▒▓████████▓▒░▒▓█▓▒░░▒▓█▓▒░  ░▒▓█▓▒░  ░▒▓█▓▒░░▒▓█▓▒░ ░▒▓█▓▒░   ░▒▓███████▓▒░░▒▓████████▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓██████▓▒░   
░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░  ░▒▓█▓▒░  ░▒▓█▓▒░░▒▓█▓▒░ ░▒▓█▓▒░   ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░        
░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░  ░▒▓█▓▒░  ░▒▓█▓▒░░▒▓█▓▒░ ░▒▓█▓▒░   ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░        
░▒▓█▓▒░░▒▓█▓▒░░▒▓██████▓▒░   ░▒▓█▓▒░   ░▒▓██████▓▒░  ░▒▓█▓▒░   ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓███████▓▒░░▒▓████████▓▒░ 

by GeorgeZip

Donate Me: https://zrzutka.pl/jbg3fz
```

## Overview

This Python script is designed for predicting Bitcoin (BTC) price movements using the XGBoost machine learning model and automating trades on MetaTrader5 (MT5) based on real-time data from Binance. It fetches historical and real-time data via WebSocket, applies technical indicators, optimizes the model periodically, and executes buy/sell orders with configurable stop-loss (SL) and take-profit (TP) methods.

The script supports asynchronous processing, countdown timers for operations, and graceful shutdown with `Ctrl + Esc`. It is optimized for various timeframes and includes fixes for common errors.

## Features

- **Asynchronous Real-Time Data Processing**: Uses `websockets` for live Binance kline data.
- **Countdown Timers**: Text-based countdowns for candle closures and other operations in "MM:SS" format.
- **Dynamic Bollinger Bands Parameters**: Adjusted based on timeframe for more frequent predictions.
- **Flexible History Fetching**: Configurable with `--history-days` (default: 10 days).
- **UTF-8 Encoding Support**: Handles special characters (e.g., Polish diacritics).
- **Predictions After Candle Close**: Minimal debug messages; one per unclosed candle.
- **Dynamic Data Limits**: Based on timeframe for data fetching and minimum rows.
- **Default Settings Option**: Use `--default-all` for quick start (`timeframe='1h'`, `sl_method='atr'`).
- **Duplicate Removal**: Eliminates duplicate timestamps in historical and WebSocket data.
- **Bug Fixes**: Addresses `RuntimeError`, `NameError`, `SyntaxError`, `OSError`, and more.
- **Graceful Shutdown**: `Ctrl + Esc` closes WebSocket, MT5, tasks, and saves logs; `Ctrl + C` for immediate exit.
- **Model Re-Optimization**: Every 24 hours for short timeframes (`1m` to `30m`), every 3 days for longer ones.
- **Optimizations**: Reduced console redundancies, improved loops, and code structure.
- **Trading Strategy**: Uses RSI, MACD, Bollinger Bands, EMA, ATR, OBV, Momentum, and Volume Ratio indicators.
- **SL/TP Methods**: Supports `atr`, `percent`, and `support_resistance`.
- **Confidence Threshold**: Skips trades below a configurable threshold (default: 0.75).
- **Position Management**: Limits max positions (default: 10), validates lot sizes, logs trades.

## Requirements

- Python 3.12 or compatible.
- Libraries (install via `pip`):
  - `pandas`
  - `pandas_ta`
  - `numpy`
  - `ccxt`
  - `xgboost`
  - `sklearn` (for preprocessing, model selection, metrics)
  - `optuna`
  - `MetaTrader5`
  - `colorama`
  - `websockets`
  - `pytz`
  - `keyboard`
  - `msvcrt` (Windows-specific for input)

No additional package installations are needed beyond these, as the script uses built-in or listed imports.

## Installation

1. Clone or download the script: `BTC_pred_autotrade.py`.
2. Install dependencies:
   ```
   pip install pandas pandas_ta numpy ccxt xgboost scikit-learn optuna MetaTrader5 colorama websockets pytz keyboard
   ```
3. Set up MetaTrader5 account details in the script ( `account`, `password`, `server` ).
4. Ensure MetaTrader5 is installed and the symbol (e.g., BTCUSD) is available.

## Usage

Run the script with Python:

```
python BTC_pred_autotrade.py [--default-all] [--history-days DAYS]
```

- `--default-all`: Use default settings (timeframe: 1h, SL/TP: atr).
- `--history-days`: Number of days of historical data to fetch (default: 10).

### Interactive Prompts

If not using `--default-all`, the script will prompt for:
- Timeframe (e.g., 1h, supported: 1m to 1d).
- SL/TP method (1: atr, 2: percent, 3: support_resistance).
- Symbol (default: BTCUSD).
- Lot size (default: 0.01).
- Max positions (default: 10).
- Confidence threshold (default: 0.75).

### Running

- The script tests MT5 connection.
- Fetches data, optimizes the model (20 trials with Optuna).
- Starts WebSocket for real-time updates.
- Executes trading strategy: Predicts direction, checks conditions (volatility, BB width, confidence), sends orders.
- Logs trades to `trade_log.txt` and WebSocket data to `websocket.log`.

Press `Ctrl + Esc` to gracefully shut down (closes connections, saves logs). `Ctrl + C` for immediate exit (may show errors).

## Configuration

- **Symbol**: `SYMBOL = "BTCUSD"` (configurable).
- **Lot Size**: `LOT_SIZE = 0.01` (validated against MT5 symbol specs).
- **Max Positions**: `MAX_POSITIONS = 10`.
- **Confidence Threshold**: `CONFIDENCE_THRESHOLD = 0.75`.
- **Time Zones**: Uses UTC and CEST (Poland).
- **MT5 Account**: Update `account`, `password`, `server`.
- **Binance**: Uses CCXT for data fetching (no API key needed for public data).

## Trading Logic

1. Fetch historical data from Binance.
2. Apply indicators (RSI, MACD, BB, EMA, ATR, etc.).
3. Prepare data for XGBoost (lagged features, scaling).
4. Optimize and train model.
5. Monitor real-time candles via WebSocket.
6. After each closed candle:
   - Update data.
   - Predict direction (1: up, 0: down).
   - Check filters (volatility, BB width, confidence, EMA diff).
   - Generate BUY/SELL/HOLD signal.
   - Send order to MT5 if valid.
   - Manage positions, log closures (TP/SL).
7. Re-optimize model periodically.

## Limitations

- No internet access for additional packages.
- Relies on public Binance data; no trading on Binance.
- MT5 must be configured correctly.
- Risk: Automated trading involves financial risk; use demo accounts.
- Windows-specific input handling with `msvcrt`.

## Donations

Support the developer: on zrzutka.pl [Donate Here](https://zrzutka.pl/jbg3fz) 
                       or suppi.pl [Donate Here](https://suppi.pl/georgezip)



## License

This project is open-source under the MIT License (assumed; add if needed).

## Author

GeorgeZip  
Current Date: July 11, 2025
