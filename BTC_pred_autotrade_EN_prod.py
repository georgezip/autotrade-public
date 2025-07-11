# coding: utf-8
# Script version with the following features and fixes:
# - Asynchronous real-time WebSocket data processing using the `websockets` library.
# - Text countdown to candle close in `candle_countdown` in the format "Remaining MM:SS", displayed in the same line (overwritten with `\r`) to minimize lines.
# - Text countdown in other operations (`countdown`) in the format "Remaining MM:SS", displayed in new lines.
# - Input parameters (timeframe, SL/TP) in the same line as the prompt, with text entered by the user.
# - Minimal number of empty lines in the console for readability.
# - Dynamic Bollinger Bands parameters (`length`, `std`, `width_threshold`) depending on the timeframe for more frequent predictions.
# - Flexible data history with the `--history-days` argument (default 10 days).
# - UTF-8 encoding fix for handling Polish diacritics.
# - Predictions after candle close with minimal debug messages (one message about unclosed candle per candle).
# - Dynamic data limit (`get_data_limit`) and minimum number of rows (`get_min_rows`) depending on the timeframe.
# - Support for `--default-all` for default settings (`timeframe='1h'`, `sl_method='atr`).
# - Removal of duplicate timestamps in historical and WebSocket data.
# - Bug fixes: `RuntimeError` (use of `websockets`, initialization of `trigger_prediction`), `NameError` (addition of `threading`), `SyntaxError` (UTF-8 declaration), `RuntimeError: This event loop is already running` (removal of nested `run_until_complete`), `OSError: [WinError 10038]` (use of `msvcrt` for asynchronous input in Windows), `SyntaxError: name 'model' is parameter and global` (use of local variables in `trading_strategy`).
# - Handling of `Ctrl+Esc` key for graceful program shutdown (closing WebSocket, MT5, asynchronous tasks, saving logs), working during input, countdown, model optimization, and trading strategy.
# - Program shutdown messages after pressing `Ctrl+Esc` displayed in red (`Fore.RED`), without unnecessary output (e.g., `0`).
# - Default `Ctrl+C` behavior (immediate program interruption, may generate console errors).
# - Fix for missing countdown after signal issuance in `trading_strategy`.
# - Model re-optimization: every 24 hours for short timeframes (`1m`, `3m`, `5m`, `15m`, `30m`), every 3 days for longer ones (`1h`, `2h`, `4h`, `6h`, `8h`, `12h`, `1d`).
# - Optimizations: Removed redundancies in console blocks, optimized loops, improved code structure for better readability and efficiency.
# - Fixes for short timeframe timeout issue:
#   - Increased `no_data` timeout to full timeframe duration.
#   - Added fallback to fetch historical data on WebSocket timeout.
#   - Relaxed timestamp filtering to include `>=` latest historical timestamp.
#   - Added detailed WebSocket message debugging.
#   - Synchronized candle countdown with WebSocket timestamps.
#   - Updated WebSocket URI to use default port.

import pandas as pd
import pandas_ta as ta
import numpy as np
import ccxt
from xgboost import XGBClassifier
from datetime import datetime, timedelta
import warnings
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score
import optuna
import time
import MetaTrader5 as mt5
from colorama import init, Fore, Style
import websockets
import json
import threading
import asyncio
import logging
import pytz
import sys
import argparse
import re
import keyboard
import msvcrt

# Configure logging
logging.basicConfig(filename='websocket.log', level=logging.INFO, format='%(asctime)s - %(message)s')

# Suppress warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Symbol and configuration
SYMBOL = "BTCUSD"
LOT_SIZE = 0.01  # Set to 0.01 for BTCUSD
MAX_POSITIONS = 10
CONFIDENCE_THRESHOLD = 0.75  # Prediction confidence threshold

# Time zone
UTC = pytz.UTC
CEST = pytz.timezone('Europe/Warsaw')  # CEST for Poland

# Account data (change to your own)
account = 12345678
password = "********"
server = "ICMarketsEU-Demo"

# Initialize Binance exchange
exchange = ccxt.binance({'rateLimit': 1200, 'enableRateLimit': True})

# Structure for storing WebSocket data
realtime_data = []
current_candle = None
candle_start_time = None
console_lock = threading.Lock()

# Global variables for asynchronous processing
model = None
scaler = None
sl_method = "atr"
is_running = True  # Flag to control program shutdown
last_optimization_time = None  # Tracking the time of the last optimization

# Logging the current system date for debugging
with console_lock:
    print(f"{Fore.CYAN}Current system date: {datetime.utcnow().astimezone(CEST)}{Style.RESET_ALL}")

def countdown(seconds):
    """Displays a countdown of seconds in the console in the format "Remaining MM:SS" in new lines."""
    with console_lock:
        print(f"{Fore.CYAN}Countdown to next operation: {seconds // 60:02d}:{seconds % 60:02d}{Style.RESET_ALL}")
    while seconds > 0 and is_running:
        with console_lock:
            print(f"{Fore.CYAN}Remaining {seconds // 60:02d}:{seconds % 60:02d}{Style.RESET_ALL}")
        time.sleep(1)
        seconds -= 1
    with console_lock:
        print(f"{Fore.CYAN}Countdown finished!{Style.RESET_ALL}")

def shutdown_mt5():
    """Closes the connection to MetaTrader5."""
    try:
        if mt5.initialize():
            mt5.shutdown()
            with console_lock:
                print(f"{Fore.GREEN}MT5 connection closed successfully.{Style.RESET_ALL}")
    except Exception as e:
        with console_lock:
            print(f"{Fore.RED}Error closing MT5: {e}{Style.RESET_ALL}")

def close_logs():
    """Closes logs and saves the program termination message."""
    timestamp = datetime.utcnow().astimezone(CEST)
    with open("trade_log.txt", "a") as f:
        f.write(f"Program terminated: {timestamp}\n")
        f.flush()  # Ensures write to file
    logging.info(f"Program terminated: {timestamp}")
    logging.shutdown()

async def cleanup(tasks):
    """Asynchronously closes the program, releasing resources."""
    global is_running
    is_running = False
    with console_lock:
        print(f"{Fore.RED}Closing program, please wait...{Style.RESET_ALL}")
    for task in tasks:
        task.cancel()
    try:
        await asyncio.gather(*tasks, return_exceptions=True)
    except asyncio.CancelledError:
        pass
    shutdown_mt5()
    close_logs()
    with console_lock:
        print(f"{Fore.RED}Program terminated correctly.{Style.RESET_ALL}")

def test_mt5_connection():
    with console_lock:
        print(f"{Fore.CYAN}Testing MT5 connection...{Style.RESET_ALL}")
    if not mt5.initialize():
        with console_lock:
            print(f"{Fore.RED}Error: MT5 initialization failed. Error code: {mt5.last_error()}{Style.RESET_ALL}")
        sys.exit(1)
    if not mt5.login(account, password=password, server=server):
        with console_lock:
            print(f"{Fore.RED}Error: MT5 login failed. Error code: {mt5.last_error()}{Style.RESET_ALL}")
        sys.exit(1)
    else:
        symbol_info = mt5.symbol_info(SYMBOL)
        if symbol_info is None:
            with console_lock:
                print(f"{Fore.RED}Error: Symbol {SYMBOL} unavailable on server {server}{Style.RESET_ALL}")
            sys.exit(1)
        with console_lock:
            print(f"{Fore.GREEN}MT5 login successful. Account: {account}, Symbol: {SYMBOL} available{Style.RESET_ALL}")
            print(f"{Fore.CYAN}Symbol specifications {SYMBOL}:")
            print(f"  Minimum lot: {symbol_info.volume_min}")
            print(f"  Maximum lot: {symbol_info.volume_max}")
            print(f"  Lot step: {symbol_info.volume_step}{Style.RESET_ALL}")

def validate_lot_size(symbol, lot_size):
    """Checks if lot_size is valid for the given symbol."""
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        with console_lock:
            print(f"{Fore.RED}Error: Unable to retrieve symbol information {symbol}{Style.RESET_ALL}")
        return None
    min_lot = symbol_info.volume_min
    max_lot = symbol_info.volume_max
    lot_step = symbol_info.volume_step
    if lot_size < min_lot or lot_size > max_lot:
        with console_lock:
            print(f"{Fore.RED}Error: Lot size {lot_size} out of range [{min_lot}, {max_lot}]{Style.RESET_ALL}")
        return None
    # Round lot_size to the nearest step
    adjusted_lot = round(round(lot_size / lot_step) * lot_step, 2)
    if adjusted_lot != lot_size:
        with console_lock:
            print(f"{Fore.YELLOW}Lot size {lot_size} adjusted to {adjusted_lot} (step: {lot_step}){Style.RESET_ALL}")
    return adjusted_lot

def normalize_timeframe(timeframe):
    """Normalizes timeframe format from Binance (e.g. '15m') to pandas (e.g. '15min')."""
    timeframe_map = {
        '1m': '1min', '3m': '3min', '5m': '5min', '15m': '15min', '30m': '30min',
        '1h': '1H', '2h': '2H', '4h': '4H', '6h': '6H', '8h': '8H', '12h': '12H', '1d': '1D'
    }
    return timeframe_map.get(timeframe, '1H')  # Default 1H

def get_bb_params(timeframe):
    """Returns Bollinger Bands parameters depending on the timeframe."""
    if timeframe in ['1m', '3m', '5m', '15m', '30m']:
        return {'length': 10, 'std': 1.5, 'width_threshold': 0.015}
    return {'length': 20, 'std': 2.0, 'width_threshold': 0.02}

async def async_input(prompt):
    """Asynchronously reads input from the console in Windows using msvcrt."""
    global is_running
    with console_lock:
        print(prompt, end='', flush=True)
    input_data = ""
    while is_running:
        if msvcrt.kbhit():
            char = msvcrt.getch().decode('utf-8', errors='ignore')
            if char == '\x1b':  # Esc
                pass
            if char == '\r':  # Enter
                break
            elif char == '\b':  # Backspace
                if input_data:
                    input_data = input_data[:-1]
                    with console_lock:
                        sys.stdout.write('\b \b')
                        sys.stdout.flush()
            else:
                input_data += char
                with console_lock:
                    sys.stdout.write(char)
                    sys.stdout.flush()
        await asyncio.sleep(0.1)
    with console_lock:
        print()  # New line after input completion
    return input_data

async def on_message(ws, timeframe, trigger_prediction):
    global realtime_data, is_running, current_candle
    pandas_timeframe = normalize_timeframe(timeframe)
    try:
        async for message in ws:
            if not is_running:
                await ws.close()
                break
            data = json.loads(message)
            if 'k' not in data:
                logging.info(f"WebSocket message without kline data: {data}")
                continue
            kline = data['k']
            timestamp = pd.to_datetime(kline['t'], unit='ms', utc=True).floor(pandas_timeframe)
            open_price = float(kline['o'])
            high_price = float(kline['h'])
            low_price = float(kline['l'])
            close_price = float(kline['c'])
            volume = float(kline['v'])
            kline_closed = kline['x']

            timestamp_cest = timestamp.astimezone(CEST)
            logging.info(f"WebSocket - Timestamp: {timestamp_cest}, Open: {open_price}, High: {high_price}, Low: {low_price}, Close: {close_price}, Volume: {volume}, Closed: {kline_closed}")

            if current_candle is None or timestamp != current_candle['timestamp']:
                if current_candle is not None:
                    # Add the previous current_candle as assumed closed (if closed message was missed)
                    realtime_data.append(current_candle)
                    prev_timestamp_cest = current_candle['timestamp'].astimezone(CEST)
                    with console_lock:
                        sys.stdout.write("\n")
                        print(f"{Fore.GREEN}Added assumed closed candle (missed close?): {prev_timestamp_cest}, Close: {current_candle['close']}{Style.RESET_ALL}")
                    trigger_prediction.set()
                # Start new current_candle
                current_candle = {
                    'timestamp': timestamp,
                    'open': open_price,
                    'high': high_price,
                    'low': low_price,
                    'close': close_price,
                    'volume': volume
                }
                with console_lock:
                    sys.stdout.write("\n")
                    print(f"{Fore.YELLOW}Started new candle (unclosed): {timestamp_cest}, Current Close: {close_price}{Style.RESET_ALL}")
            else:
                # Update current_candle with latest values
                current_candle['high'] = max(current_candle['high'], high_price)
                current_candle['low'] = min(current_candle['low'], low_price)
                current_candle['close'] = close_price
                current_candle['volume'] = volume

            if kline_closed:
                # Ensure final values are set
                current_candle['high'] = high_price
                current_candle['low'] = low_price
                current_candle['close'] = close_price
                current_candle['volume'] = volume
                realtime_data.append(current_candle)
                with console_lock:
                    sys.stdout.write("\n")
                    print(f"{Fore.GREEN}Added closed candle: {timestamp_cest}, Close: {close_price}{Style.RESET_ALL}")
                trigger_prediction.set()
                current_candle = None  # Reset for next candle
    except Exception as e:
        with console_lock:
            sys.stdout.write("\n")
            print(f"{Fore.RED}Error in on_message: {e}{Style.RESET_ALL}")

async def start_websocket(timeframe, trigger_prediction):
    global is_running
    # Updated WebSocket URI to use default port 443
    uri = f"wss://stream.binance.com:443/ws/{SYMBOL.lower().replace('usd','usdt')}@kline_{timeframe}"
    while is_running:
        try:
            async with websockets.connect(uri, ping_interval=20, ping_timeout=20) as ws:
                with console_lock:
                    sys.stdout.write("\n")
                    print(f"{Fore.GREEN}WebSocket connection open{Style.RESET_ALL}")
                await on_message(ws, timeframe, trigger_prediction)
        except Exception as e:
            if not is_running:
                break
            with console_lock:
                sys.stdout.write("\n")
                print(f"{Fore.RED}WebSocket error: {e}{Style.RESET_ALL}")
                print(f"{Fore.YELLOW}WebSocket connection closed, retrying in 5 seconds...{Style.RESET_ALL}")
            await asyncio.sleep(5)

def timeframe_to_seconds(timeframe):
    """Converts timeframe (e.g. '5m', '1h') to seconds."""
    valid_timeframes = ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d']
    if timeframe not in valid_timeframes:
        with console_lock:
            print(f"{Fore.RED}Invalid timeframe: {timeframe}, using default 1h{Style.RESET_ALL}")
        timeframe = '1h'
    match = re.match(r'^(\d+)([mhd])$', timeframe)
    if not match:
        return 3600
    value, unit = int(match.group(1)), match.group(2)
    multipliers = {'m': 60, 'h': 3600, 'd': 86400}
    return value * multipliers[unit]

def get_min_rows(timeframe):
    tf_seconds = timeframe_to_seconds(timeframe)
    scale_factor = 60 / (tf_seconds / 60)
    return max(30, int(200 * scale_factor))

def get_data_limit(timeframe):
    tf_seconds = timeframe_to_seconds(timeframe)
    scale_factor = 60 / (tf_seconds / 60)
    data_limit = max(1000, int(3000 * scale_factor * 2))
    with console_lock:
        print(f"{Fore.CYAN}Calculated data limit for {timeframe}: {data_limit} candles{Style.RESET_ALL}")
    return data_limit

def get_countdown_time(condition, timeframe):
    tf_seconds = timeframe_to_seconds(timeframe)
    # Increased no_data timeout to full timeframe duration
    base_times = {
        'no_data': tf_seconds,  # Changed from tf_seconds / 2
        'low_volatility': 60,
        'narrow_bb': 60,
        'low_confidence': 60,
        'hold_signal': 60,
        'order_success': 120,
        'order_failed': 60,
        'position_error': 15
    }
    base_time = base_times.get(condition, 60)
    scale_factor = tf_seconds / 3600
    return max(10, int(base_time * scale_factor))

async def candle_countdown(timeframe, trigger_prediction):
    global is_running, realtime_data
    tf_seconds = timeframe_to_seconds(timeframe)
    with console_lock:
        print(f"{Fore.CYAN}Countdown to candle close: {tf_seconds // 60:02d}:{tf_seconds % 60:02d}{Style.RESET_ALL}\033[K")
    seconds_remaining = tf_seconds
    while seconds_remaining > 0 and is_running and not trigger_prediction.is_set():
        # Synchronize countdown with WebSocket data
        if realtime_data:
            latest_candle = pd.DataFrame(realtime_data).set_index('timestamp').index.max()
            if latest_candle:
                next_candle_time = latest_candle + pd.Timedelta(seconds=tf_seconds)
                seconds_remaining = max(0, int((next_candle_time - pd.Timestamp.utcnow()).total_seconds()))
                with console_lock:
                    sys.stdout.write(f"\r{Fore.CYAN}Remaining {seconds_remaining // 60:02d}:{seconds_remaining % 60:02d}{Style.RESET_ALL}\033[K")
                    sys.stdout.flush()
        else:
            with console_lock:
                sys.stdout.write(f"\r{Fore.CYAN}Remaining {seconds_remaining // 60:02d}:{seconds_remaining % 60:02d}{Style.RESET_ALL}\033[K")
                sys.stdout.flush()
        await asyncio.sleep(1)
        seconds_remaining -= 1
    if not is_running:
        with console_lock:
            print(f"\r{Fore.RED}Interrupted candle close countdown (Ctrl + Esc).{' ' * 20}{Style.RESET_ALL}")
        raise SystemExit("Interrupted candle close countdown (Ctrl + Esc).")
    if trigger_prediction.is_set():
        with console_lock:
            print(f"\r{Fore.CYAN}Countdown finished, candle closed.{' ' * 20}{Style.RESET_ALL}")
    else:
        with console_lock:
            print(f"\r{Fore.RED}Timeout, no new data, waiting...{' ' * 20}{Style.RESET_ALL}")

def fetch_data(ccxt_symbol, timeframe, since=None, history_days=10):
    limit = get_data_limit(timeframe)
    with console_lock:
        print(f"{Fore.CYAN}Fetching data for {ccxt_symbol} in interval {timeframe} with limit {limit} and history {history_days} days...{Style.RESET_ALL}")
    
    ohlcv = []
    if since is None:
        since = int((datetime.utcnow() - timedelta(days=history_days)).timestamp() * 1000)
    with console_lock:
        since_cest = pd.to_datetime(since, unit='ms', utc=True).astimezone(CEST)
        print(f"{Fore.CYAN}Fetching data from: {since_cest}{Style.RESET_ALL}")

    try:
        current_since = since
        min_rows = get_min_rows(timeframe)
        while len(ohlcv) < limit and is_running:
            data = exchange.fetch_ohlcv(ccxt_symbol, timeframe, since=current_since, limit=min(limit - len(ohlcv), 1000))
            if not data:
                break
            ohlcv.extend(data)
            current_since = data[-1][0] + 1
            with console_lock:
                print(f"{Fore.CYAN}Fetched {len(data)} candles, total: {len(ohlcv)}/{limit}{Style.RESET_ALL}")
            if len(data) < 1000:
                break
        if len(ohlcv) < min_rows:
            with console_lock:
                print(f"{Fore.YELLOW}Too little data: {len(ohlcv)} candles, required minimum {min_rows}, extending period...{Style.RESET_ALL}")
            since = int((datetime.utcnow() - timedelta(days=history_days + 10)).timestamp() * 1000)
            current_since = since
            while len(ohlcv) < limit and is_running:
                data = exchange.fetch_ohlcv(ccxt_symbol, timeframe, since=current_since, limit=min(limit - len(ohlcv), 1000))
                if not data:
                    break
                ohlcv.extend(data)
                current_since = data[-1][0] + 1
                with console_lock:
                    print(f"{Fore.CYAN}Additionally fetched {len(data)} candles, total: {len(ohlcv)}/{limit}{Style.RESET_ALL}")
                if len(data) < 1000:
                    break
    except Exception as e:
        with console_lock:
            print(f"{Fore.RED}Data fetching error: {e}{Style.RESET_ALL}")

    pandas_timeframe = normalize_timeframe(timeframe)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True).dt.floor(pandas_timeframe)
    df = df.set_index('timestamp')
    df = df[~df.index.duplicated(keep='last')]

    with console_lock:
        print(f"{Fore.CYAN}Number of candles from API: {len(df)}{Style.RESET_ALL}")

    if realtime_data:
        with console_lock:
            print(f"{Fore.CYAN}Number of candles in realtime_data before joining: {len(realtime_data)}{Style.RESET_ALL}")
        df_realtime = pd.DataFrame(realtime_data).set_index('timestamp')
        df_realtime = df_realtime[~df_realtime.index.duplicated(keep='last')]
        latest_fetched_time = df.index.max() if not df.empty else pd.Timestamp.min.tz_localize('UTC')
        df_realtime = df_realtime[df_realtime.index > latest_fetched_time]
        df = pd.concat([df, df_realtime]).sort_index()
        df = df[~df.index.duplicated(keep='last')]
        with console_lock:
            print(f"{Fore.GREEN}Joined {len(df_realtime)} candles from WebSocket{Style.RESET_ALL}")

    if not df.empty:
        last_candle = df.iloc[-1]
        cest_time = last_candle.name.astimezone(CEST)
        with console_lock:
            print(f"{Fore.GREEN}Last candle: {cest_time}, Close: {last_candle['close']}{Style.RESET_ALL}")
            print(f"{Fore.GREEN}Number of rows before indicators: {len(df)}{Style.RESET_ALL}")
            first_date = df.index[0].astimezone(CEST)
            last_date = df.index[-1].astimezone(CEST)
            print(f"{Fore.GREEN}First date: {first_date}{Style.RESET_ALL}")
            print(f"{Fore.GREEN}Last date: {last_date}{Style.RESET_ALL}")
    
    with console_lock:
        print(f"{Fore.GREEN}Fetching completed: {len(df)} rows.{Style.RESET_ALL}")
    return df

def apply_indicators(df, timeframe):
    with console_lock:
        print(f"{Fore.CYAN}Adding indicators...{Style.RESET_ALL}")
    df = df.copy()
    df['RSI'] = ta.rsi(df['close'], length=3)
    macd = ta.macd(df['close'], fast=3, slow=6, signal=2)
    df[['MACD', 'MACD_signal']] = macd[['MACD_3_6_2', 'MACDs_3_6_2']]
    bb_params = get_bb_params(timeframe)
    bb = ta.bbands(df['close'], length=bb_params['length'], std=bb_params['std'])
    df[['BB_lower', 'BB_middle', 'BB_upper']] = bb[[f'BBL_{bb_params["length"]}_{bb_params["std"]}', f'BBM_{bb_params["length"]}_{bb_params["std"]}', f'BBU_{bb_params["length"]}_{bb_params["std"]}']]
    df['EMA_5'] = ta.ema(df['close'], length=5)
    df['EMA_10'] = ta.ema(df['close'], length=10)
    df['ATR'] = ta.atr(df['high'], df['low'], df['close'], length=3)
    df['OBV'] = ta.obv(df['close'], df['volume'])
    df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / df['BB_middle']
    df['Momentum'] = df['close'].diff(4)
    df['Volume_Ratio'] = df['volume'] / df['volume'].rolling(5).mean()
    df.dropna(inplace=True)
    with console_lock:
        print(f"{Fore.GREEN}Indicators added, number of rows after dropna: {len(df)}{Style.RESET_ALL}")
    return df

def prepare_data_for_model(df, horizon=1, timeframe='1h'):
    with console_lock:
        print(f"{Fore.CYAN}Preparing data for the model...{Style.RESET_ALL}")
    min_rows = get_min_rows(timeframe)
    if len(df) < min_rows:
        with console_lock:
            print(f"{Fore.RED}Too little data: {len(df)} rows, required minimum {min_rows}{Style.RESET_ALL}")
        return None, None, None
    feature_columns = [col for col in df.columns if col not in ['open', 'high', 'low', 'close', 'volume']]
    X = df[feature_columns].copy()
    for col in feature_columns:
        X[f'{col}_lag1'] = X[col].shift(1)
        X[f'{col}_lag2'] = X[col].shift(2)
    X.dropna(inplace=True)
    if X.empty:
        with console_lock:
            print(f"{Fore.RED}Empty DataFrame X after lags{Style.RESET_ALL}")
        return None, None, None
    y = np.where(df['close'].shift(-horizon) > df['close'], 1, 0)[:-horizon]
    X = X.iloc[:-horizon]
    min_length = min(len(X), len(y))
    if min_length == 0:
        with console_lock:
            print(f"{Fore.RED}No data after processing: {min_length} samples{Style.RESET_ALL}")
        return None, None, None
    X = X.iloc[:min_length]
    y = y[:min_length]
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    with console_lock:
        print(f"{Fore.GREEN}Data prepared: {X_scaled.shape[0]} samples, {X_scaled.shape[1]} features.{Style.RESET_ALL}")
    return X_scaled, y, scaler

def optimize_model(trial, X, y):
    global is_running
    if not is_running:
        raise SystemExit("Interrupted model optimization (Ctrl + Esc).")
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 200, 500),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.001, 0.1),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'subsample': trial.suggest_uniform('subsample', 0.8, 1.0),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.8, 1.0),
        'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-5, 1),
        'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-5, 1),
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 0.5, 2.0),
        'random_state': 42
    }
    model_opt = XGBClassifier(**params)
    tscv = TimeSeriesSplit(n_splits=5)
    accuracy_scores = []
    for train_idx, val_idx in tscv.split(X):
        if not is_running:
            raise SystemExit("Interrupted model optimization (Ctrl + Esc).")
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        model_opt.fit(X_train, y_train)
        y_pred = model_opt.predict(X_val)
        accuracy_scores.append(accuracy_score(y_val, y_pred))
    return np.mean(accuracy_scores)

async def get_sl_method(use_default=False):
    if use_default:
        return "atr"
    while True:
        with console_lock:
            print(f"{Fore.CYAN}Select SL/TP method (press Enter for default - atr):{Style.RESET_ALL}")
            print(f"{Fore.WHITE}1. atr (ATR-based, multiplier=4.0, TP=4xATR){Style.RESET_ALL}")
            print(f"{Fore.WHITE}2. percent (Percentage, max loss=0.5%, TP=1.5%){Style.RESET_ALL}")
            print(f"{Fore.WHITE}3. support_resistance (Support/Resistance, lookback=3){Style.RESET_ALL}")
        choice = await async_input(f"{Fore.CYAN}Enter method number (1-3 or Enter for 1): {Style.RESET_ALL}")
        if not is_running:
            raise SystemExit("Interrupted SL/TP method selection (Ctrl + Esc).")
        if choice == "" or choice == "1":
            return "atr"
        if choice == "2":
            return "percent"
        if choice == "3":
            return "support_resistance"
        with console_lock:
            print(f"{Fore.RED}Invalid choice. Try again.{Style.RESET_ALL}")

async def get_timeframe(use_default=False):
    valid_timeframes = ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d']
    if use_default:
        return "1h"
    while True:
        choice = await async_input(f"{Fore.CYAN}Enter timeframe (press Enter for default - 1h): {Style.RESET_ALL}")
        if not is_running:
            raise SystemExit("Interrupted timeframe selection (Ctrl + Esc).")
        if not choice:
            return "1h"
        if choice in valid_timeframes:
            return choice
        with console_lock:
            print(f"{Fore.RED}Invalid timeframe: {choice}. Supported: {', '.join(valid_timeframes)}{Style.RESET_ALL}")

async def get_symbol(use_default=False):
    if use_default:
        return "BTCUSD"
    while True:
        choice = await async_input(f"{Fore.CYAN}Enter symbol (press Enter for default - BTCUSD): {Style.RESET_ALL}")
        if not is_running:
            raise SystemExit("Interrupted symbol selection (Ctrl + Esc).")
        if not choice:
            return "BTCUSD"
        return choice.upper()

async def get_lot_size(use_default=False):
    if use_default:
        return 0.01
    while True:
        choice = await async_input(f"{Fore.CYAN}Enter lot size (press Enter for default - 0.01): {Style.RESET_ALL}")
        if not is_running:
            raise SystemExit("Interrupted lot size selection (Ctrl + Esc).")
        if not choice:
            return 0.01
        try:
            lot = float(choice)
            if lot > 0:
                return lot
            else:
                with console_lock:
                    print(f"{Fore.RED}Lot size must be greater than 0.{Style.RESET_ALL}")
        except ValueError:
            with console_lock:
                print(f"{Fore.RED}Invalid lot size. Must be a number.{Style.RESET_ALL}")

async def get_max_positions(use_default=False):
    if use_default:
        return 10
    while True:
        choice = await async_input(f"{Fore.CYAN}Enter max positions (press Enter for default - 10): {Style.RESET_ALL}")
        if not is_running:
            raise SystemExit("Interrupted max positions selection (Ctrl + Esc).")
        if not choice:
            return 10
        try:
            max_pos = int(choice)
            if max_pos > 0:
                return max_pos
            else:
                with console_lock:
                    print(f"{Fore.RED}Max positions must be greater than 0.{Style.RESET_ALL}")
        except ValueError:
            with console_lock:
                print(f"{Fore.RED}Invalid max positions. Must be an integer.{Style.RESET_ALL}")

async def get_confidence_threshold(use_default=False):
    if use_default:
        return 0.75
    while True:
        choice = await async_input(f"{Fore.CYAN}Enter prediction confidence threshold (press Enter for default - 0.75): {Style.RESET_ALL}")
        if not is_running:
            raise SystemExit("Interrupted confidence threshold selection (Ctrl + Esc).")
        if not choice:
            return 0.75
        try:
            thresh = float(choice)
            if 0 <= thresh <= 1:
                return thresh
            else:
                with console_lock:
                    print(f"{Fore.RED}Threshold must be between 0 and 1.{Style.RESET_ALL}")
        except ValueError:
            with console_lock:
                print(f"{Fore.RED}Invalid threshold. Must be a number.{Style.RESET_ALL}")

async def get_optuna_trials(use_default=False):
    if use_default:
        return 20
    while True:
        choice = await async_input(f"{Fore.CYAN}Enter number of Optuna trials (press Enter for default - 20): {Style.RESET_ALL}")
        if not is_running:
            raise SystemExit("Interrupted Optuna trials selection (Ctrl + Esc).")
        if not choice:
            return 20
        try:
            trials = int(choice)
            if trials > 0:
                return trials
            else:
                with console_lock:
                    print(f"{Fore.RED}Number of trials must be greater than 0.{Style.RESET_ALL}")
        except ValueError:
            with console_lock:
                print(f"{Fore.RED}Invalid number. Must be an integer.{Style.RESET_ALL}")

def calculate_sl_tp(signal, price, atr, df, sl_method):
    if sl_method == "atr":
        distance = atr * 4
        sl = price - distance if signal == "BUY" else price + distance
        tp = price + distance if signal == "BUY" else price - distance
    elif sl_method == "percent":
        sl_distance = price * 0.005
        tp_distance = price * 0.015
        sl = price - sl_distance if signal == "BUY" else price + sl_distance
        tp = price + tp_distance if signal == "BUY" else price - tp_distance
    elif sl_method == "support_resistance":
        lookback = 3
        if signal == "BUY":
            sl = min(df['low'].iloc[-lookback:].min(), price - atr * 4)
            tp = price + 4 * atr
        else:
            sl = max(df['high'].iloc[-lookback:].max(), price + atr * 4)
            tp = price - 4 * atr
    else:
        raise ValueError("Unknown SL/TP method")
    with console_lock:
        print(f"{Fore.YELLOW}SL: {sl:.2f}, TP: {tp:.2f}, SL Distance: {abs(price - sl):.2f}{Style.RESET_ALL}")
    return sl, tp

def send_order(signal, price, atr, df, sl_method):
    if not mt5.initialize() or not mt5.login(account, password=password, server=server):
        with console_lock:
            print(f"{Fore.RED}MT5 initialization/login error. Error code: {mt5.last_error()}{Style.RESET_ALL}")
        return None

    adjusted_lot = validate_lot_size(SYMBOL, LOT_SIZE)
    if adjusted_lot is None:
        return None

    positions = mt5.positions_get(symbol=SYMBOL)
    if positions is None or len(positions) >= MAX_POSITIONS:
        with console_lock:
            if positions is None:
                print(f"{Fore.RED}Error fetching positions. Error code: {mt5.last_error()}{Style.RESET_ALL}")
            else:
                print(f"{Fore.RED}Limit of {MAX_POSITIONS} orders reached.{Style.RESET_ALL}")
        return None

    sl, tp = calculate_sl_tp(signal, price, atr, df, sl_method)
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": SYMBOL,
        "volume": adjusted_lot,
        "type": mt5.ORDER_TYPE_BUY if signal == "BUY" else mt5.ORDER_TYPE_SELL,
        "sl": sl,
        "tp": tp,
        "comment": f"XGBoost {signal}",
        "magic": 123456,
        "type_filling": mt5.ORDER_FILLING_IOC
    }

    with console_lock:
        print(f"{Fore.YELLOW}Order request: {request}{Style.RESET_ALL}")
    result = mt5.order_send(request)
    if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
        with console_lock:
            if result is None:
                print(f"{Fore.RED}Error sending {signal} order: mt5.order_send returned None. Error code: {mt5.last_error()}{Style.RESET_ALL}")
            else:
                print(f"{Fore.RED}Error sending {signal} order: {result.comment}, Code: {result.retcode}{Style.RESET_ALL}")
        return None
    with console_lock:
        print(f"{Fore.GREEN}{signal} order: Success (Ticket: {result.order}){Style.RESET_ALL}")
    return result.order

async def trading_strategy(local_model, local_scaler, timeframe, trigger_prediction, history_days, optuna_trials, horizon=1, sl_method="atr"):
    global is_running, model, scaler, last_optimization_time, realtime_data
    ccxt_symbol = SYMBOL.replace("USD", "/USDT")
    df_history = fetch_data(ccxt_symbol, timeframe, history_days=history_days)
    if df_history.empty:
        with console_lock:
            print(f"{Fore.RED}No historical data in trading_strategy, terminating.{Style.RESET_ALL}")
        return

    current_time = datetime.utcnow().astimezone(UTC)
    last_candle_time = df_history.index.max().astimezone(UTC)
    if (current_time - last_candle_time).days > 30:
        with console_lock:
            print(f"{Fore.RED}Historical data outdated (last candle: {last_candle_time.astimezone(CEST)}), re-fetching...{Style.RESET_ALL}")
        df_history = fetch_data(ccxt_symbol, timeframe, since=int((current_time - timedelta(days=history_days)).timestamp() * 1000), history_days=history_days)

    short_timeframes = ['1m', '3m', '5m', '15m', '30m']
    optimization_interval = timedelta(hours=24) if timeframe in short_timeframes else timedelta(days=3)

    last_timestamp = None
    last_price = None
    open_positions = {}

    while is_running:
        current_time = datetime.utcnow().astimezone(UTC)
        if last_optimization_time is None or (current_time - last_optimization_time) >= optimization_interval:
            with console_lock:
                print(f"{Fore.CYAN}Performing model re-optimization...{Style.RESET_ALL}")
            df_history = fetch_data(ccxt_symbol, timeframe, history_days=history_days)
            if df_history.empty:
                with console_lock:
                    print(f"{Fore.RED}No data for re-optimization, skipping...{Style.RESET_ALL}")
                continue
            df_history = apply_indicators(df_history, timeframe)
            if df_history.empty:
                with console_lock:
                    print(f"{Fore.RED}No data after indicators for re-optimization, skipping...{Style.RESET_ALL}")
                continue
            X, y, scaler_new = prepare_data_for_model(df_history, horizon, timeframe)
            if X is None:
                with console_lock:
                    print(f"{Fore.RED}Failed to prepare data for re-optimization, skipping...{Style.RESET_ALL}")
                continue
            study = optuna.create_study(direction="maximize")
            trial_count = 0
            def update_progress(study, trial):
                nonlocal trial_count
                trial_count += 1
                with console_lock:
                    print(f"{Fore.CYAN}Trial {trial_count}/{optuna_trials}{Style.RESET_ALL}")
            try:
                study.optimize(lambda trial: optimize_model(trial, X, y), n_trials=optuna_trials, callbacks=[update_progress])
            except SystemExit as e:
                with console_lock:
                    print(f"{Fore.RED}{e}{Style.RESET_ALL}")
                await cleanup([])
                return
            best_params = study.best_params
            with console_lock:
                print(f"{Fore.GREEN}Best parameters after re-optimization: {best_params}{Style.RESET_ALL}")
            local_model = XGBClassifier(**best_params, objective='binary:logistic', random_state=42)
            local_model.fit(X, y)
            local_scaler = scaler_new
            model = local_model
            scaler = local_scaler
            last_optimization_time = current_time
            with console_lock:
                print(f"{Fore.CYAN}Re-optimization completed, model update finished.{Style.RESET_ALL}")

        try:
            await candle_countdown(timeframe, trigger_prediction)
        except SystemExit as e:
            with console_lock:
                print(f"{Fore.RED}{e}{Style.RESET_ALL}")
            await cleanup([])
            return

        try:
            await asyncio.wait_for(trigger_prediction.wait(), timeout=get_countdown_time('no_data', timeframe))
            trigger_prediction.clear()
        except asyncio.TimeoutError:
            with console_lock:
                print(f"{Fore.RED}Timeout, no new data, fetching historical data...{Style.RESET_ALL}")
            # Added fallback to fetch historical data
            df_new = fetch_data(ccxt_symbol, timeframe, history_days=1)
            if df_new.empty:
                with console_lock:
                    print(f"{Fore.RED}No new historical data, waiting...{Style.RESET_ALL}")
                await asyncio.sleep(10)
                continue
            df_new = df_new[df_new.index >= df_history.index.max()]
            if not df_new.empty:
                with console_lock:
                    print(f"{Fore.GREEN}Fetched {len(df_new)} new candles from historical data{Style.RESET_ALL}")
                realtime_data = df_new.reset_index().to_dict('records')
            else:
                with console_lock:
                    print(f"{Fore.YELLOW}No new candles after filtering, waiting...{Style.RESET_ALL}")
                await asyncio.sleep(10)
                continue
        except asyncio.CancelledError:
            with console_lock:
                print(f"{Fore.RED}Interrupted trading strategy (Ctrl + Esc).{Style.RESET_ALL}")
            await cleanup([])
            return
        except SystemExit as e:
            with console_lock:
                print(f"{Fore.RED}{e}{Style.RESET_ALL}")
            await cleanup([])
            return

        if not realtime_data:
            with console_lock:
                print(f"{Fore.RED}No data from WebSocket, fetching historical data...{Style.RESET_ALL}")
            df_new = fetch_data(ccxt_symbol, timeframe, history_days=history_days)
        else:
            df_new = pd.DataFrame(realtime_data).set_index('timestamp')
            df_new = df_new[~df_new.index.duplicated(keep='last')]

        if df_new.empty:
            with console_lock:
                print(f"{Fore.RED}No new data from WebSocket, waiting...{Style.RESET_ALL}")
            continue

        if not df_history.empty:
            # Relaxed timestamp filtering to >=
            df_new = df_new[df_new.index >= df_history.index.max()]
            with console_lock:
                print(f"{Fore.CYAN}Filtering df_new: {len(df_new)} new rows after filter{Style.RESET_ALL}")

        df_history = pd.concat([df_history, df_new]).sort_index()
        df_history = df_history[~df_history.index.duplicated(keep='last')]
        last_timestamp = df_history.index[-1]
        df_history = df_history.tail(get_data_limit(timeframe))
        realtime_data = []  # Clear after adding to history

        df = apply_indicators(df_history, timeframe)
        if df.empty:
            with console_lock:
                print(f"{Fore.RED}No data after indicators, waiting...{Style.RESET_ALL}")
            continue

        X_scaled, y, scaler_new = prepare_data_for_model(df, horizon, timeframe)
        if X_scaled is None:
            with console_lock:
                print(f"{Fore.RED}Failed to prepare data for model, waiting...{Style.RESET_ALL}")
            continue

        last_row = X_scaled[-1].reshape(1, -1)
        predicted_direction = local_model.predict(last_row)[0]
        confidence = local_model.predict_proba(last_row)[0][predicted_direction]
        current_price = df['close'].iloc[-1]
        atr = df['ATR'].iloc[-1]

        atr_mean = df['ATR'].rolling(3).mean().iloc[-1]
        if atr < atr_mean * 0.75:
            with console_lock:
                print(f"{Fore.YELLOW}Low volatility (ATR: {atr:.2f}), skipping trading.{Style.RESET_ALL}")
            await asyncio.sleep(get_countdown_time('low_volatility', timeframe))
            continue

        bb_width = df['BB_width'].iloc[-1]
        bb_params = get_bb_params(timeframe)
        if bb_width < bb_params['width_threshold']:
            with console_lock:
                print(f"{Fore.YELLOW}Narrow Bollinger Bands ({bb_width:.4f}, threshold: {bb_params['width_threshold']}), skipping trading.{Style.RESET_ALL}")
            await asyncio.sleep(get_countdown_time('narrow_bb', timeframe))
            continue

        if confidence < CONFIDENCE_THRESHOLD:
            with console_lock:
                print(f"{Fore.YELLOW}Low prediction confidence ({confidence:.2f}), skipping trading.{Style.RESET_ALL}")
            await asyncio.sleep(get_countdown_time('low_confidence', timeframe))
            continue

        ema_diff = df['EMA_5'].iloc[-1] - df['EMA_10'].iloc[-1]
        if predicted_direction == 1 and ema_diff > atr:
            signal = "BUY"
            color = Fore.GREEN
        elif predicted_direction == 0 and ema_diff < -atr:
            signal = "SELL"
            color = Fore.RED
        else:
            signal = "HOLD"
            color = Fore.YELLOW
            cest_time = last_timestamp.astimezone(CEST)
            with console_lock:
                print(f"{color}Price: {current_price}, Direction: {predicted_direction}, Confidence: {confidence:.2f}, Signal: {signal}, Timestamp: {cest_time}{Style.RESET_ALL}")
            await asyncio.sleep(get_countdown_time('hold_signal', timeframe))
            continue

        cest_time = last_timestamp.astimezone(CEST)
        with console_lock:
            print(f"{color}Price: {current_price}, Direction: {predicted_direction}, Confidence: {confidence:.2f}, Signal: {signal}, Timestamp: {cest_time}{Style.RESET_ALL}")
        ticket = send_order(signal, current_price, atr, df, sl_method)

        if ticket:
            open_positions[ticket] = {
                'signal': signal,
                'open_price': current_price,
                'sl': calculate_sl_tp(signal, current_price, atr, df, sl_method)[0],
                'tp': calculate_sl_tp(signal, current_price, atr, df, sl_method)[1]
            }
            with console_lock:
                print(f"{Fore.GREEN}Order placed, waiting...{Style.RESET_ALL}")
            await asyncio.sleep(get_countdown_time('order_success', timeframe))
        else:
            with console_lock:
                print(f"{Fore.YELLOW}Order failed, waiting...{Style.RESET_ALL}")
            await asyncio.sleep(get_countdown_time('order_failed', timeframe))

        positions = mt5.positions_get(symbol=SYMBOL)
        if positions is None:
            with console_lock:
                print(f"{Fore.RED}Error: Failed to fetch positions. Error code: {mt5.last_error()}{Style.RESET_ALL}")
            await asyncio.sleep(get_countdown_time('position_error', timeframe))
            continue
        closed_tickets = []
        for ticket, pos in list(open_positions.items()):
            if not any(p.ticket == ticket for p in positions):
                history = mt5.history_deals_get(ticket=ticket)
                if history:
                    close_price = history[-1].price
                    profit = history[-1].profit
                    reason = "TP" if abs(close_price - pos['tp']) < abs(close_price - pos['sl']) else "SL"
                    cest_time = last_timestamp.astimezone(CEST)
                    with open("trade_log.txt", "a") as f:
                        f.write(f"Closed: {cest_time}, Ticket: {ticket}, {pos['signal']}, Open: {pos['open_price']}, Close: {close_price}, Profit: {profit:.2f}, Reason: {reason}\n")
                        f.flush()
                    with console_lock:
                        print(f"{Fore.GREEN}Position closed: Ticket: {ticket}, {pos['signal']}, Open: {pos['open_price']}, Close: {close_price}, Profit: {profit:.2f}, Reason: {reason}{Style.RESET_ALL}")
                closed_tickets.append(ticket)
        for ticket in closed_tickets:
            del open_positions[ticket]

        if last_price is not None:
            actual_change = (current_price - last_price) / last_price * 100
            actual_direction = 1 if current_price > last_price else 0
            cest_time = last_timestamp.astimezone(CEST)
            with open("trade_log.txt", "a") as f:
                f.write(f"{cest_time}, {current_price}, Pred: {predicted_direction}, Confidence: {confidence:.2f}, {signal}, Actual: {last_price} -> {current_price} ({actual_change:.2f}%, Dir: {actual_direction})\n")
                f.flush()
            with console_lock:
                print(f"{Fore.GREEN}Saved prediction: {cest_time}, Price: {current_price}, Pred: {predicted_direction}, Confidence: {confidence:.2f}, Signal: {signal}, Actual: {last_price} -> {current_price} ({actual_change:.2f}%, Dir: {actual_direction}){Style.RESET_ALL}")
        last_price = current_price

async def main():
    parser = argparse.ArgumentParser(description="BTC Prediction and Auto-Trading Script")
    parser.add_argument('--default-all', action='store_true', help="Run with all default settings (timeframe: 1h, SL/TP method: atr)")
    parser.add_argument('--history-days', type=int, default=10, help="Number of days of historical data to fetch (default: 10)")
    parser.add_argument('--timeframe', type=str, help="Timeframe (e.g., 1m, 15m, 1h)")
    args = parser.parse_args()

    test_mt5_connection()

    with console_lock:
        print(f"{Fore.YELLOW}To exit the program, press Ctrl + Esc.{Style.RESET_ALL}")

    try:
        global last_optimization_time
        timeframe = args.timeframe if args.timeframe else await get_timeframe(use_default=args.default_all)
        with console_lock:
            print(f"{Fore.GREEN}Selected timeframe: {timeframe}{Style.RESET_ALL}")

        global sl_method
        sl_method = await get_sl_method(use_default=args.default_all)
        with console_lock:
            print(f"{Fore.GREEN}Selected SL/TP method: {sl_method}{Style.RESET_ALL}")

        global SYMBOL
        SYMBOL = await get_symbol(use_default=args.default_all)
        with console_lock:
            print(f"{Fore.GREEN}Selected symbol: {SYMBOL}{Style.RESET_ALL}")

        global LOT_SIZE
        LOT_SIZE = await get_lot_size(use_default=args.default_all)
        with console_lock:
            print(f"{Fore.GREEN}Selected lot size: {LOT_SIZE}{Style.RESET_ALL}")

        global MAX_POSITIONS
        MAX_POSITIONS = await get_max_positions(use_default=args.default_all)
        with console_lock:
            print(f"{Fore.GREEN}Selected max positions: {MAX_POSITIONS}{Style.RESET_ALL}")

        global CONFIDENCE_THRESHOLD
        CONFIDENCE_THRESHOLD = await get_confidence_threshold(use_default=args.default_all)
        with console_lock:
            print(f"{Fore.GREEN}Selected confidence threshold: {CONFIDENCE_THRESHOLD}{Style.RESET_ALL}")

        optuna_trials = await get_optuna_trials(use_default=args.default_all)
        with console_lock:
            print(f"{Fore.GREEN}Selected Optuna trials: {optuna_trials}{Style.RESET_ALL}")

        ccxt_symbol = SYMBOL.replace("USD", "/USDT")
        df = fetch_data(ccxt_symbol, timeframe, history_days=args.history_days)
        df = apply_indicators(df, timeframe)
        if df.empty:
            with console_lock:
                print(f"{Fore.RED}No data after indicators in main, terminating.{Style.RESET_ALL}")
            return

        global model, scaler
        X, y, scaler = prepare_data_for_model(df, timeframe=timeframe)
        if X is None:
            with console_lock:
                print(f"{Fore.RED}Failed to prepare data in main, terminating.{Style.RESET_ALL}")
            return

        with console_lock:
            print(f"{Fore.CYAN}Optimizing model with {optuna_trials} trials...{Style.RESET_ALL}")
        study = optuna.create_study(direction="maximize")
        trial_count = 0
        def update_progress(study, trial):
            nonlocal trial_count
            trial_count += 1
            with console_lock:
                print(f"{Fore.CYAN}Trial {trial_count}/{optuna_trials}{Style.RESET_ALL}")

        try:
            study.optimize(lambda trial: optimize_model(trial, X, y), n_trials=optuna_trials, callbacks=[update_progress])
        except SystemExit as e:
            with console_lock:
                print(f"{Fore.RED}{e}{Style.RESET_ALL}")
            await cleanup([])
            return

        best_params = study.best_params
        with console_lock:
            print(f"{Fore.GREEN}Best parameters: {best_params}{Style.RESET_ALL}")

        model = XGBClassifier(**best_params, objective='binary:logistic', random_state=42)
        model.fit(X, y)
        last_optimization_time = datetime.utcnow().astimezone(UTC)

        with console_lock:
            print(f"{Fore.CYAN}Launching trading strategy...{Style.RESET_ALL}")

        trigger_prediction = asyncio.Event()
        tasks = [
            asyncio.create_task(start_websocket(timeframe, trigger_prediction)),
            asyncio.create_task(trading_strategy(model, scaler, timeframe, trigger_prediction, history_days=args.history_days, optuna_trials=optuna_trials, sl_method=sl_method))
        ]

        def check_escape():
            global is_running
            keyboard.wait('ctrl+esc')
            with console_lock:
                print(f"{Fore.RED}Ctrl + Esc key pressed, closing program.{Style.RESET_ALL}")
            is_running = False

        threading.Thread(target=check_escape, daemon=True).start()

        try:
            await asyncio.gather(*tasks)
        except SystemExit as e:
            with console_lock:
                print(f"{Fore.RED}{e}{Style.RESET_ALL}")
            await cleanup(tasks)
    except SystemExit as e:
        with console_lock:
            print(f"{Fore.RED}{e}{Style.RESET_ALL}")
        await cleanup([])

if __name__ == "__main__":
    init()
    print(r"""
 ░▒▓██████▓▒░░▒▓█▓▒░░▒▓█▓▒░▒▓████████▓▒░▒▓██████▓▒░▒▓████████▓▒░▒▓███████▓▒░ ░▒▓██████▓▒░░▒▓███████▓▒░░▒▓████████▓▒░ 
░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░  ░▒▓█▓▒░  ░▒▓█▓▒░░▒▓█▓▒░ ░▒▓█▓▒░   ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░        
░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░  ░▒▓█▓▒░  ░▒▓█▓▒░░▒▓█▓▒░ ░▒▓█▓▒░   ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░        
░▒▓████████▓▒░▒▓█▓▒░░▒▓█▓▒░  ░▒▓█▓▒░  ░▒▓█▓▒░░▒▓█▓▒░ ░▒▓█▓▒░   ░▒▓███████▓▒░░▒▓████████▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓██████▓▒░   
░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░  ░▒▓█▓▒░  ░▒▓█▓▒░░▒▓█▓▒░ ░▒▓█▓▒░   ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░        
░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░  ░▒▓█▓▒░  ░▒▓█▓▒░░▒▓█▓▒░ ░▒▓█▓▒░   ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░        
░▒▓█▓▒░░▒▓█▓▒░░▒▓██████▓▒░   ░▒▓█▓▒░   ░▒▓██████▓▒░  ░▒▓█▓▒░   ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓███████▓▒░░▒▓████████▓▒░ 

by GeorgeZip
(formerly known as: AI Signal Trend)
Donate Me: https://zrzutka.pl/jbg3fz
    """)
    asyncio.run(main())
