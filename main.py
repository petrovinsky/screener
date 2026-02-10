import pandas as pd
import numpy as np
from polygon import RESTClient
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
import re
import time
from collections import deque
import pandas_ta_classic as ta
import requests

# ──────────────────────────────────────────────
# ANSI color codes
# ──────────────────────────────────────────────
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
RESET = "\033[0m"


def colorize_signal(signal_str):
    text = signal_str.lower()
    if any(word in text for word in ["bullish", "bounce", "recovery"]):
        return f"{GREEN}{signal_str}{RESET}"
    elif "neutral" in text or signal_str == "Neutral":
        return f"{YELLOW}{signal_str}{RESET}"
    else:
        return f"{YELLOW}{signal_str}{RESET}"


def colorize_yes_no_net(value):
    val = str(value).lower()
    if val.startswith("yes"):
        return f"{GREEN}{value}{RESET}"
    elif val.startswith("no "):
        return f"{RED}{value}{RESET}"
    else:
        return f"{YELLOW}{value}{RESET}"


# Load .env and pandas display options
load_dotenv()
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_columns', None)

# Finnhub API key
FINNHUB_KEY = "d6503fhr01qqbln5cbngd6503fhr01qqbln5cbo0"


# ──────────────────────────────────────────────
# Rate Limiter (Polygon)
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
                print(f" Rate limit: sleeping {sleep_time:.1f}s ...")
                time.sleep(sleep_time)
                now = time.time()
            self.timestamps.popleft()
        self.timestamps.append(now)


# ──────────────────────────────────────────────
# Polygon OHLCV
# ──────────────────────────────────────────────
def fetch_stock_data(ticker, api_key, years=2):
    client = RESTClient(api_key)
    end_date = datetime.today().strftime('%Y-%m-%d')
    start_date = (datetime.today() - timedelta(days=365 * years + 30)).strftime('%Y-%m-%d')

    for attempt in range(2):
        try:
            aggs = client.get_aggs(ticker, 1, 'day', start_date, end_date, adjusted=True)
            if not aggs:
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
                    time.sleep(65)
                    continue
                else:
                    return pd.DataFrame()
            else:
                return pd.DataFrame()
    return pd.DataFrame()


# ──────────────────────────────────────────────
# Finnhub: Upcoming earnings in next 30 days
# ──────────────────────────────────────────────
def fetch_earnings_date(ticker):
    try:
        from_date = datetime.today().strftime('%Y-%m-%d')
        to_date = (datetime.today() + timedelta(days=30)).strftime('%Y-%m-%d')
        url = f"https://finnhub.io/api/v1/calendar/earnings?from={from_date}&to={to_date}&token={FINNHUB_KEY}"
        resp = requests.get(url, timeout=10)
        if resp.status_code != 200:
            print(f"Earnings HTTP {resp.status_code} for {ticker}")
            return None

        data = resp.json()
        earnings = data.get('earningsCalendar', [])
        if not earnings:
            return None

        # Filter for our ticker and find the next date
        ticker_events = [e for e in earnings if e.get('symbol', '').upper() == ticker.upper()]
        if not ticker_events:
            return None

        # Sort by date and take the soonest future one
        today = datetime.today().date()
        future = [e for e in ticker_events if datetime.strptime(e['date'], '%Y-%m-%d').date() > today]
        if not future:
            return None

        next_event = min(future, key=lambda x: x['date'])
        next_date = datetime.strptime(next_event['date'], '%Y-%m-%d')
        days = (next_date - datetime.today()).days

        # DEBBUG
        print(f"{ticker} Finnhub earnings response: {data}")

        if 0 < days <= 30:
            return days
        return None
    except Exception as e:
        print(f"Finnhub earnings error {ticker}: {str(e)[:80]}...")
        return None


# ──────────────────────────────────────────────
# Finnhub: Analyst recommendation trends – net bullish
# ──────────────────────────────────────────────
def fetch_analyst_changes(ticker):
    try:
        url = f"https://finnhub.io/api/v1/stock/recommendation?symbol={ticker}&token={FINNHUB_KEY}"
        resp = requests.get(url, timeout=10)
        if resp.status_code != 200:
            return f"HTTP {resp.status_code}"

        data = resp.json()
        if not data or not isinstance(data, list):
            return "No data"

        strong_buy = buy = hold = sell = strong_sell = 0
        for period in data:
            strong_buy += period.get('strongBuy', 0)
            buy += period.get('buy', 0)
            hold += period.get('hold', 0)
            sell += period.get('sell', 0)
            strong_sell += period.get('strongSell', 0)

        bullish = strong_buy + buy
        bearish_hold = strong_sell + sell + hold
        net = bullish - bearish_hold

        if net > 0:
            return f"Yes (+{net})"
        else:
            return f"No ({net})"
    except Exception as e:
        print(f"Finnhub analyst error {ticker}: {str(e)[:80]}...")
        return "Error"


# ──────────────────────────────────────────────
# Bull Score (unchanged)
# ──────────────────────────────────────────────
def get_bull_score(df):
    if len(df) < 50:
        return 0
    score = 0
    close = df['close'].iloc[-1]
    ema50 = df.ta.ema(length=50).iloc[-1]
    if close > ema50:
        score += 20
    if len(df) >= 200:
        sma200 = df.ta.sma(length=200).iloc[-1]
        if close > sma200:
            score += 18
    rsi_current = df.ta.rsi(length=14).iloc[-1]
    if rsi_current < 38:
        score += 22
    elif rsi_current > 70:
        score -= 20
    if len(df) >= 25:
        rsi_last_25 = df.ta.rsi(length=14).tail(25)
        min_rsi = rsi_last_25.min()
        curr_rsi = rsi_last_25.iloc[-1]
        if min_rsi <= 32 and curr_rsi > 38 and (curr_rsi - min_rsi) >= 11:
            score += 12
    macd = df.ta.macd()
    if macd['MACD_12_26_9'].iloc[-1] > macd['MACDs_12_26_9'].iloc[-1]:
        score += 15
    bb = df.ta.bbands(length=20, std=2)
    bb_lower = bb['BBL_20_2.0'].iloc[-1]
    if close <= bb_lower * 1.01:
        score += 18
    return min(max(score, 0), 100)


# ──────────────────────────────────────────────
# Signals summary
# ──────────────────────────────────────────────
def get_current_signals(df, ticker):
    if len(df) < 30:
        return {'Error': 'Insufficient data (<30 bars)'}

    close = df['close'].iloc[-1]
    rsi_current = df.ta.rsi(length=14).iloc[-1]
    macd = df.ta.macd()
    macd_line = macd['MACD_12_26_9'].iloc[-1]
    macd_signal = macd['MACDs_12_26_9'].iloc[-1]
    bb = df.ta.bbands(length=20, std=2)
    bb_lower = bb['BBL_20_2.0'].iloc[-1]

    rsi_recovery_str = "No"
    if len(df) >= 25:
        rsi_series = df.ta.rsi(length=14).tail(25)
        min_rsi_in_period = rsi_series.min()
        current_rsi = rsi_series.iloc[-1]
        if min_rsi_in_period <= 32 and current_rsi > 38 and current_rsi - min_rsi_in_period >= 11:
            rsi_recovery_str = "Yes"
        rsi_recovery_str += f" - RSI {round(current_rsi, 1)}"

    signals = []
    if close > bb_lower and df['close'].iloc[-2] <= bb_lower:
        signals.append("BB Lower Bounce")
    if rsi_current < 38:
        signals.append("RSI Weak")
    if macd_line > macd_signal:
        signals.append("MACD Bullish")
    if "Yes" in rsi_recovery_str:
        signals.append("RSI Recovery")
    signal_str = " + ".join(signals) if signals else "Neutral"

    days_to_earnings = fetch_earnings_date(ticker)
    earnings_soon = f"Yes (in {days_to_earnings} days)" if days_to_earnings else "No"

    analyst_upg = fetch_analyst_changes(ticker)

    return {
        'Close': round(close, 2),
        'RSI': round(rsi_current, 1),
        'RSI Recovery': rsi_recovery_str,
        'MACD Status': 'Bullish' if macd_line > macd_signal else 'Bearish/Neutral',
        'BB Position': 'Near Lower' if close <= bb_lower * 1.02 else 'Mid/Upper',
        'Earnings Soon': earnings_soon,
        'Analyst Upgrades': analyst_upg,
        'Bull Score': get_bull_score(df),
        'Signal': signal_str
    }


# ──────────────────────────────────────────────
# Main
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
    sig = get_current_signals(df, ticker)
    sig['Ticker'] = ticker
    results.append(sig)
    time.sleep(1.2)

if not results:
    print("No valid data retrieved.")
else:
    df_results = pd.DataFrame(results)
    cols = [
        'Ticker', 'Close', 'RSI', 'RSI Recovery', 'MACD Status',
        'BB Position', 'Earnings Soon', 'Analyst Upgrades',
        'Bull Score', 'Signal'
    ]
    df_results = df_results[cols]

    df_results = df_results.sort_values(by=['Bull Score', 'RSI'], ascending=[False, True])

    print("\n" + "=" * 170)
    print("Screening Results (Bull Score desc → RSI asc):")
    print("-" * 170)

    w = {
        'Ticker': 10,
        'Close': 10,
        'RSI': 8,
        'RSI Recovery': 25,
        'MACD Status': 20,
        'BB Position': 20,
        'Earnings Soon': 26,
        'Analyst Upgrades': 28,
        'Bull Score': 12,
        'Signal': 60
    }

    header = (
        f"{'Ticker':<{w['Ticker']}}"
        f"{'Close':<{w['Close']}}"
        f"{'RSI':<{w['RSI']}}"
        f"{'RSI Recovery':<{w['RSI Recovery']}}"
        f"{'MACD Status':<{w['MACD Status']}}"
        f"{'BB Position':<{w['BB Position']}}"
        f"{'Earnings Soon':<{w['Earnings Soon']}}"
        f"{'Analyst Upgrades':<{w['Analyst Upgrades']}}"
        f"{'Bull Score':<{w['Bull Score']}}"
        f"{'Signal':<{w['Signal']}}"
    )
    print(header)
    print("-" * len(header))

    for _, row in df_results.iterrows():
        earn_col = colorize_yes_no_net(row['Earnings Soon'])
        analyst_col = colorize_yes_no_net(row['Analyst Upgrades'])
        signal_col = colorize_signal(row['Signal'])

        line = (
            f"{row['Ticker']:<{w['Ticker']}}"
            f"{row['Close']:<{w['Close']}}"
            f"{row['RSI']:<{w['RSI']}}"
            f"{row['RSI Recovery']:<{w['RSI Recovery']}}"
            f"{row['MACD Status']:<{w['MACD Status']}}"
            f"{row['BB Position']:<{w['BB Position']}}"
            f"{earn_col:<{w['Earnings Soon']}}"
            f"{analyst_col:<{w['Analyst Upgrades']}}"
            f"{row['Bull Score']:<{w['Bull Score']}}"
            f"{signal_col:<{w['Signal']}}"
        )
        print(line)

    print("=" * len(header) + "\n")

    today = datetime.today().strftime('%Y-%m-%d %H:%M:%S')
    os.makedirs("./CSV", exist_ok=True)
    filename = f"./CSV/screened_picks_{today.replace(':', '-')}.csv"
    df_results.to_csv(filename, index=False)
    print(f"Saved to: {filename}")
    print("Done.")