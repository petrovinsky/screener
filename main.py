import pandas as pd
import numpy as np
from polygon import RESTClient
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
import re
import time
from collections import deque

# Load variables from .env file
load_dotenv()

pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_columns', None)

# ──────────────────────────────────────────────
# Rate Limiter (enforces ~5 requests per 60 seconds)
# ──────────────────────────────────────────────
class RateLimiter:
    def __init__(self, calls=5, period=60):
        self.calls = calls
        self.period = period
        self.timestamps = deque(maxlen=calls)

    def acquire(self):
        now = time.time()
        while len(self.timestamps) == self.calls:
            oldest = self.timestamps[0]
            if now - oldest < self.period:
                sleep_time = self.period - (now - oldest) + 0.2  # small buffer
                print(f"  Rate limit: sleeping {sleep_time:.1f}s to stay under {self.calls}/min...")
                time.sleep(sleep_time)
                now = time.time()
            self.timestamps.popleft()
        self.timestamps.append(now)

# ──────────────────────────────────────────────
# Fetch data with rate-limit retry
# ──────────────────────────────────────────────
def fetch_stock_data(ticker, api_key, years=2):
    client = RESTClient(api_key)
    end_date = datetime.today().strftime('%Y-%m-%d')
    start_date = (datetime.today() - timedelta(days=365 * years + 30)).strftime('%Y-%m-%d')

    for attempt in range(2):  # try twice max
        try:
            aggs = client.get_aggs(
                ticker,
                1,
                'day',
                start_date,
                end_date,
                adjusted=True
            )
            if not aggs:
                print(f"No data returned for {ticker}")
                return pd.DataFrame()

            data = [{
                'date': pd.to_datetime(a.timestamp, unit='ms'),
                'open': a.open,
                'high': a.high,
                'low': a.low,
                'close': a.close,
                'volume': a.volume
            } for a in aggs]

            df = pd.DataFrame(data).set_index('date').sort_index()
            return df

        except Exception as e:
            err_str = str(e).lower()
            if "429" in err_str or "too many" in err_str or "rate limit" in err_str:
                if attempt == 0:
                    print(f"  Rate limit (429) hit for {ticker} — sleeping 65s and retrying...")
                    time.sleep(65)
                    continue
                else:
                    print(f"  Retry failed for {ticker} after rate limit: {e}")
                    return pd.DataFrame()
            else:
                print(f"Error fetching {ticker}: {e}")
                return pd.DataFrame()

    return pd.DataFrame()

# ──────────────────────────────────────────────
# Indicator functions (unchanged)
# ──────────────────────────────────────────────

def calculate_sma(series, window):
    return series.rolling(window).mean()

def calculate_ema(series, window):
    return series.ewm(span=window, adjust=False).mean()

def calculate_rsi(series, window=14):
    delta = series.diff(1)
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ema_up = up.ewm(com=window - 1, adjust=False).mean()
    ema_down = down.ewm(com=window - 1, adjust=False).mean()
    rs = ema_up / ema_down
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(series, fast=12, slow=26, signal=9):
    fast_ema = calculate_ema(series, fast)
    slow_ema = calculate_ema(series, slow)
    macd_line = fast_ema - slow_ema
    macd_signal = calculate_ema(macd_line, signal)
    macd_hist = macd_line - macd_signal
    return macd_line, macd_signal, macd_hist

def calculate_bollinger_bands(series, window=20, dev=2):
    rolling_mean = series.rolling(window).mean()
    rolling_std = series.rolling(window).std()
    upper = rolling_mean + (rolling_std * dev)
    lower = rolling_mean - (rolling_std * dev)
    return upper, rolling_mean, lower

# ──────────────────────────────────────────────
# Bull Score
# ──────────────────────────────────────────────
def get_bull_score(df):
    if len(df) < 50:
        return 0
    score = 0
    close = df['close'].iloc[-1]

    ema50 = calculate_ema(df['close'], 50).iloc[-1]
    sma200 = calculate_sma(df['close'], 200).iloc[-1] if len(df) >= 200 else np.nan

    if close > ema50:
        score += 20
    if not np.isnan(sma200) and close > sma200:
        score += 25

    rsi_val = calculate_rsi(df['close']).iloc[-1]
    if rsi_val < 35:
        score += 25
    elif rsi_val > 70:
        score -= 20

    macd_line, macd_sig, _ = calculate_macd(df['close'])
    if macd_line.iloc[-1] > macd_sig.iloc[-1]:
        score += 15

    _, _, bb_lower = calculate_bollinger_bands(df['close'])
    if close <= bb_lower.iloc[-1] * 1.015:
        score += 20

    return min(max(score, 0), 100)

# ──────────────────────────────────────────────
# Current signals summary
# ──────────────────────────────────────────────
def get_current_signals(df):
    close = df['close'].iloc[-1]
    rsi = calculate_rsi(df['close']).iloc[-1]
    macd_line, macd_signal, _ = calculate_macd(df['close'])
    macd_line = macd_line.iloc[-1]
    macd_signal = macd_signal.iloc[-1]
    _, _, bb_lower = calculate_bollinger_bands(df['close'])
    bb_lower = bb_lower.iloc[-1]

    if close > bb_lower and df['close'].iloc[-2] <= bb_lower:
        signal = "BB Lower Bounce"
    elif rsi < 30:
        signal = "Oversold (RSI)"
    elif macd_line > macd_signal:
        signal = "MACD Bullish"
    else:
        signal = "Neutral"

    return {
        'Close': round(close, 2),
        'RSI': round(rsi, 2),
        'MACD Status': 'Bullish' if macd_line > macd_signal else 'Bearish/Neutral',
        'BB Position': 'Near Lower' if close <= bb_lower * 1.02 else 'Mid/Upper',
        'Bull Score': get_bull_score(df),
        'Signal': signal
    }

# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
api_key = os.getenv("POLYGON_API_KEY")
if not api_key:
    print("Error: POLYGON_API_KEY not found in .env file")
    exit(1)

input_str = input("Enter tickers (comma/space separated) or filename: ").strip()

if os.path.isfile(input_str):
    try:
        with open(input_str, 'r') as f:
            tickers = [line.strip().upper() for line in f if line.strip()]
        print(f"Loaded {len(tickers)} tickers from file.")
    except Exception as e:
        print(f"File error: {e}")
        tickers = []
else:
    tickers = [t.strip().upper() for t in re.split(r'[, ]', input_str) if t.strip()]
    print(f"Processing {len(tickers)} tickers.")

if not tickers:
    print("No tickers to process.")
    exit(0)

rate_limiter = RateLimiter(calls=5, period=60)

results = []
for i, ticker in enumerate(tickers, 1):
    print(f"[{i}/{len(tickers)}] Processing {ticker}...")
    rate_limiter.acquire()

    df = fetch_stock_data(ticker, api_key)
    if df.empty:
        continue

    sig = get_current_signals(df)
    sig['Ticker'] = ticker
    results.append(sig)

if not results:
    print("No valid data retrieved.")
else:
    df_results = pd.DataFrame(results)
    cols = ['Ticker', 'Close', 'RSI', 'MACD Status', 'BB Position', 'Bull Score', 'Signal']
    df_results = df_results[cols]

    # Sort: highest Bull Score → then lowest RSI (most oversold)
    df_results = df_results.sort_values(by=['Bull Score', 'RSI'], ascending=[False, True])

    print("\n" + "="*90)
    print("Screening Results (sorted by Bull Score desc, then RSI asc):")
    print(df_results.to_string(index=False))
    print("="*90 + "\n")

    today = datetime.today().strftime('%Y-%m-%d')
    filename = f"screen_{today}.csv"
    df_results.to_csv(filename, index=False)
    print(f"Results saved to: {filename}")

print("Done.")