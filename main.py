import pandas as pd
import numpy as np
from polygon import RESTClient
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
import re
import time
from collections import deque

import pandas_ta_classic as ta  # <-- the key import

# Load .env and pandas display options
load_dotenv()
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_columns', None)

# ──────────────────────────────────────────────
# Rate Limiter (5 calls / 60s – free Polygon tier safe)
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
                sleep_time = self.period - (now - oldest) + 0.2
                print(f"  Rate limit: sleeping {sleep_time:.1f}s ...")
                time.sleep(sleep_time)
                now = time.time()
            self.timestamps.popleft()
        self.timestamps.append(now)

# ──────────────────────────────────────────────
# Fetch daily OHLCV from Polygon (with 429 retry)
# ──────────────────────────────────────────────
def fetch_stock_data(ticker, api_key, years=2):
    client = RESTClient(api_key)
    end_date = datetime.today().strftime('%Y-%m-%d')
    start_date = (datetime.today() - timedelta(days=365 * years + 30)).strftime('%Y-%m-%d')

    for attempt in range(2):
        try:
            aggs = client.get_aggs(
                ticker,
                1, 'day',
                start_date, end_date,
                adjusted=True
            )
            if not aggs:
                print(f"No data for {ticker}")
                return pd.DataFrame()

            df = pd.DataFrame([{
                'date': pd.to_datetime(a.timestamp, unit='ms'),
                'open': a.open,
                'high': a.high,
                'low': a.low,
                'close': a.close,
                'volume': a.volume
            } for a in aggs]).set_index('date').sort_index()

            return df

        except Exception as e:
            err = str(e).lower()
            if "429" in err or "too many" in err or "rate limit" in err:
                if attempt == 0:
                    print(f"  429 Rate limit for {ticker} — sleeping 65s + retry...")
                    time.sleep(65)
                    continue
                else:
                    print(f"  Retry failed for {ticker}: {e}")
                    return pd.DataFrame()
            else:
                print(f"Fetch error {ticker}: {e}")
                return pd.DataFrame()

    return pd.DataFrame()

# ──────────────────────────────────────────────
# Bull Score using pandas_ta_classic indicators
# ──────────────────────────────────────────────
def get_bull_score(df):
    if len(df) < 50:
        return 0

    score = 0
    close = df['close'].iloc[-1]

    # Trend filters
    ema50 = df.ta.ema(length=50).iloc[-1]
    sma200 = df.ta.sma(length=200).iloc[-1] if len(df) >= 200 else np.nan

    if close > ema50:
        score += 20
    if not np.isnan(sma200) and close > sma200:
        score += 25

    # Momentum / Oversold
    rsi_val = df.ta.rsi(length=14).iloc[-1]
    if rsi_val < 35:
        score += 25
    elif rsi_val > 70:
        score -= 20

    # MACD
    macd = df.ta.macd()
    if macd['MACD_12_26_9'].iloc[-1] > macd['MACDs_12_26_9'].iloc[-1]:
        score += 15

    # Bollinger Bands position
    bb = df.ta.bbands(length=20, std=2)
    bb_lower = bb['BBL_20_2.0'].iloc[-1]
    if close <= bb_lower * 1.015:
        score += 20

    return min(max(score, 0), 100)

# ──────────────────────────────────────────────
# Current signals summary (using pandas_ta_classic)
# ──────────────────────────────────────────────
def get_current_signals(df):
    if len(df) < 30:
        return {'Error': 'Insufficient data (<30 bars)'}

    close = df['close'].iloc[-1]
    prev_close = df['close'].iloc[-2]

    rsi = df.ta.rsi(length=14).iloc[-1]
    macd = df.ta.macd()
    macd_line = macd['MACD_12_26_9'].iloc[-1]
    macd_signal = macd['MACDs_12_26_9'].iloc[-1]
    macd_hist = macd['MACDh_12_26_9'].iloc[-1]

    bb = df.ta.bbands(length=20, std=2)
    bb_lower = bb['BBL_20_2.0'].iloc[-1]

    signals = []
    if close > bb_lower >= prev_close:
        signals.append("BB Lower Bounce")
    if rsi < 30:
        signals.append("RSI Oversold")
    if macd_line > macd_signal and macd['MACD_12_26_9'].iloc[-2] <= macd['MACDs_12_26_9'].iloc[-2]:
        signals.append("MACD Bull Cross")
    elif macd_line > macd_signal:
        signals.append("MACD Bullish")

    signal_str = " + ".join(signals) if signals else "Neutral"

    return {
        'Close': round(close, 2),
        'RSI': round(rsi, 2),
        'MACD Status': 'Bullish' if macd_line > macd_signal else 'Bearish/Neutral',
        'MACD Hist': round(macd_hist, 2),
        'BB Position': 'Near Lower' if close <= bb_lower * 1.02 else 'Mid/Upper',
        'Bull Score': get_bull_score(df),
        'Signal': signal_str
    }
# ──────────────────────────────────────────────
# Main screening logic
# ──────────────────────────────────────────────
api_key = os.getenv("POLYGON_API_KEY")
if not api_key:
    print("Error: POLYGON_API_KEY missing in .env")
    exit(1)

input_str = input("Enter tickers (comma/space sep) or filename: ").strip()

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
    print("No tickers provided.")
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

    # Highest Bull Score first, then most oversold (lowest RSI)
    df_results = df_results.sort_values(by=['Bull Score', 'RSI'], ascending=[False, True])

    print("\n" + "="*90)
    print("Screening Results (Bull Score desc → RSI asc):")
    print(df_results.to_string(index=False))
    print("="*90 + "\n")

    today = datetime.today().strftime('%Y-%m-%d %H:%M:%S')
    os.makedirs("./CSV", exist_ok=True)
    filename = f"./CSV/screened picks {today}.csv"
    df_results.to_csv(filename, index=False)
    print(f"Saved to: {filename}")

print("Done.")