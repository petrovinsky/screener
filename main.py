import pandas as pd
import numpy as np
from polygon import RESTClient
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
import re

# Load variables from your .env file
load_dotenv()   # looks for .env file in current directory

# Set display options for panda so it shows the entire contents of cells and table
pd.set_option('display.width', None)          # None = detect terminal width
pd.set_option('display.max_colwidth', None)   # show full cell contents
pd.set_option('display.max_columns', None)    # show all columns

def fetch_stock_data(ticker, api_key, years=2):
    client = RESTClient(api_key)
    end_date = datetime.today().strftime('%Y-%m-%d')
    start_date = (datetime.today() - timedelta(days=365 * years)).strftime('%Y-%m-%d')
    aggs = client.get_aggs(ticker, 1, 'day', start_date, end_date)
    data = [{'date': pd.to_datetime(a.timestamp, unit='ms'), 'open': a.open, 'high': a.high,
             'low': a.low, 'close': a.close, 'volume': a.volume} for a in aggs]
    df = pd.DataFrame(data).set_index('date').sort_index()
    return df


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


def calculate_ichimoku(df, tenkan_period=9, kijun_period=26, senkou_period=52):
    high = df['high']
    low = df['low']
    close = df['close']
    # Tenkan-sen
    tenkan_high = high.rolling(tenkan_period).max()
    tenkan_low = low.rolling(tenkan_period).min()
    tenkan = (tenkan_high + tenkan_low) / 2
    # Kijun-sen
    kijun_high = high.rolling(kijun_period).max()
    kijun_low = low.rolling(kijun_period).min()
    kijun = (kijun_high + kijun_low) / 2
    # Senkou Span A
    senkou_a = ((tenkan + kijun) / 2).shift(kijun_period)
    # Senkou Span B
    senkou_high = high.rolling(senkou_period).max()
    senkou_low = low.rolling(senkou_period).min()
    senkou_b = ((senkou_high + senkou_low) / 2).shift(kijun_period)
    # Chikou Span
    chikou = close.shift(-kijun_period)
    return tenkan, kijun, senkou_a, senkou_b, chikou


def calculate_adx(df, window=14):
    high = df['high']
    low = df['low']
    close = df['close']
    tr = pd.concat([high - low, abs(high - close.shift()), abs(low - close.shift())], axis=1).max(axis=1)
    dm_pos = high - high.shift()
    dm_neg = low.shift() - low
    dm_pos = dm_pos.where((dm_pos > dm_neg) & (dm_pos > 0), 0)
    dm_neg = dm_neg.where((dm_neg > dm_pos) & (dm_neg > 0), 0)
    tr_ema = tr.ewm(span=window, adjust=False).mean()
    dm_pos_ema = dm_pos.ewm(span=window, adjust=False).mean()
    dm_neg_ema = dm_neg.ewm(span=window, adjust=False).mean()
    dx = 100 * abs(dm_pos_ema - dm_neg_ema) / (dm_pos_ema + dm_neg_ema)
    adx = dx.ewm(span=window, adjust=False).mean()
    return adx


def calculate_uo(df, n1=7, n2=14, n3=28, ws=4, wm=2, wl=1):
    high = df['high']
    low = df['low']
    close = df['close']
    bp = close - low.rolling(n1).min().shift(1)
    tr = pd.concat([high - low, abs(high - close.shift()), abs(low - close.shift())], axis=1).max(axis=1).rolling(
        n1).sum().shift(1)
    avg7 = bp.rolling(n1).sum() / tr.rolling(n1).sum()
    bp = close - low.rolling(n2).min().shift(1)
    tr = pd.concat([high - low, abs(high - close.shift()), abs(low - close.shift())], axis=1).max(axis=1).rolling(
        n2).sum().shift(1)
    avg14 = bp.rolling(n2).sum() / tr.rolling(n2).sum()
    bp = close - low.rolling(n3).min().shift(1)
    tr = pd.concat([high - low, abs(high - close.shift()), abs(low - close.shift())], axis=1).max(axis=1).rolling(
        n3).sum().shift(1)
    avg28 = bp.rolling(n3).sum() / tr.rolling(n3).sum()
    uo = 100 * (ws * avg7 + wm * avg14 + wl * avg28) / (ws + wm + wl)
    return uo


def calculate_mfi(df, window=14):
    high = df['high']
    low = df['low']
    close = df['close']
    volume = df['volume']
    typical_price = (high + low + close) / 3
    mf = typical_price * volume
    mf_pos = mf.where(typical_price > typical_price.shift(1), 0).rolling(window).sum()
    mf_neg = mf.where(typical_price < typical_price.shift(1), 0).rolling(window).sum()
    mfi = 100 - (100 / (1 + mf_pos / mf_neg))
    return mfi


# Combination signal functions
def bb_l_adx_l_rsi_l_signal(df):
    bb_upper, bb_mid, bb_lower = calculate_bollinger_bands(df['close'])
    adx = calculate_adx(df)
    rsi = calculate_rsi(df['close'])
    # Buy signal: CROSS above lower BB + AUX confirmations
    buy = (df['close'] > bb_lower) & (df['close'].shift(1) <= bb_lower.shift(1)) & (adx > 25) & (rsi < 30)
    # Sell: CROSS below lower or above upper
    sell = ((df['close'] < bb_lower) & (df['close'].shift(1) >= bb_lower.shift(1))) | (
                (df['close'] > bb_upper) & (df['close'].shift(1) <= bb_upper.shift(1)))
    return buy, sell


def bb_l_bb_l_rsi_l_signal(df):
    bb_upper, bb_mid, bb_lower = calculate_bollinger_bands(df['close'])
    rsi = calculate_rsi(df['close'])
    # Buy: CROSS above lower + AUX below lower + RSI <30
    buy = (df['close'] > bb_lower) & (df['close'].shift(1) <= bb_lower.shift(1)) & (df['close'] < bb_lower) & (rsi < 30)
    # Sell: Standard BB CROSS
    sell = ((df['close'] < bb_lower) & (df['close'].shift(1) >= bb_lower.shift(1))) | (
                (df['close'] > bb_upper) & (df['close'].shift(1) <= bb_upper.shift(1)))
    return buy, sell


def bb_l_uo_l_mfi_l_signal(df):
    bb_upper, bb_mid, bb_lower = calculate_bollinger_bands(df['close'])
    uo = calculate_uo(df)
    mfi = calculate_mfi(df)
    # Buy: CROSS above lower + UO <30 + MFI <20
    buy = (df['close'] > bb_lower) & (df['close'].shift(1) <= bb_lower.shift(1)) & (uo < 30) & (mfi < 20)
    # Sell: Standard BB CROSS
    sell = ((df['close'] < bb_lower) & (df['close'].shift(1) >= bb_lower.shift(1))) | (
                (df['close'] > bb_upper) & (df['close'].shift(1) <= bb_upper.shift(1)))
    return buy, sell


def get_current_signals(df):
    close = df['close'].iloc[-1]
    # Compute base indicators
    ema50 = calculate_ema(df['close'], 50).iloc[-1]
    sma50 = calculate_sma(df['close'], 50).iloc[-1]
    sma200 = calculate_sma(df['close'], 200).iloc[-1] if len(df) >= 200 else np.nan
    rsi = calculate_rsi(df['close']).iloc[-1]
    macd_line, macd_signal, _ = calculate_macd(df['close'])
    macd_line, macd_signal = macd_line.iloc[-1], macd_signal.iloc[-1]
    bb_upper, bb_mid, bb_lower = calculate_bollinger_bands(df['close'])
    bb_upper, bb_mid, bb_lower = bb_upper.iloc[-1], bb_mid.iloc[-1], bb_lower.iloc[-1]
    tenkan, kijun, senkou_a, senkou_b, chikou = calculate_ichimoku(df)
    tenkan, kijun, senkou_a, senkou_b, chikou = tenkan.iloc[-1], kijun.iloc[-1], senkou_a.iloc[-1], senkou_b.iloc[-1], \
    chikou.iloc[-1]

    # Compute combo signals (only buy for current, as bullish focus)
    buy_adx_rsi, _ = bb_l_adx_l_rsi_l_signal(df)
    current_adx_rsi = 'Bullish Entry' if buy_adx_rsi.iloc[-1] else 'Neutral/No Signal'
    buy_bb_rsi, _ = bb_l_bb_l_rsi_l_signal(df)
    current_bb_rsi = 'Bullish Entry' if buy_bb_rsi.iloc[-1] else 'Neutral/No Signal'
    buy_uo_mfi, _ = bb_l_uo_l_mfi_l_signal(df)
    current_uo_mfi = 'Bullish Entry' if buy_uo_mfi.iloc[-1] else 'Neutral/No Signal'

    # Golden Cross and Death Cross detection (on last bar)
    sma50_series = calculate_sma(df['close'], 50)
    sma200_series = calculate_sma(df['close'], 200)
    golden_cross = (sma50_series.iloc[-1] > sma200_series.iloc[-1]) and (
                sma50_series.iloc[-2] <= sma200_series.iloc[-2]) if len(df) >= 200 else False
    death_cross = (sma50_series.iloc[-1] < sma200_series.iloc[-1]) and (
                sma50_series.iloc[-2] >= sma200_series.iloc[-2]) if len(df) >= 200 else False
    golden_status = 'Recent Golden Cross' if golden_cross else 'No Recent Cross'
    death_status = 'Recent Death Cross' if death_cross else 'No Recent Cross'

    # 20/100 Day Cross detection (on last bar)
    sma20_series = calculate_sma(df['close'], 20)
    sma100_series = calculate_sma(df['close'], 100)
    short_golden_cross = (sma20_series.iloc[-1] > sma100_series.iloc[-1]) and (
                sma20_series.iloc[-2] <= sma100_series.iloc[-2]) if len(df) >= 100 else False
    short_death_cross = (sma20_series.iloc[-1] < sma100_series.iloc[-1]) and (
                sma20_series.iloc[-2] >= sma100_series.iloc[-1]) if len(df) >= 100 else False
    short_golden_status = 'Recent Short Golden Cross' if short_golden_cross else 'No Recent Cross'
    short_death_status = 'Recent Short Death Cross' if short_death_cross else 'No Recent Cross'
    sma100 = sma100_series.iloc[-1] if len(df) >= 100 else np.nan
    sma20 = sma20_series.iloc[-1] if len(df) >= 20 else np.nan

    signals = {
        'Indicator': ['Ichimoku Cloud', 'EMA(50)', 'SMA(50)', 'RSI(14)', 'MACD', 'Bollinger Bands',
                      'BB_L_ADX_L_RSI_L', 'BB_L_BB_L_RSI_L', 'BB_L_UO_L_MFI_L', 'Golden Cross', 'Death Cross',
                      'Short Golden Cross (20/100)', 'Short Death Cross (20/100)'],
        'Current Value': [
            f'Price: {close:.2f}, Tenkan: {tenkan:.2f}, Kijun: {kijun:.2f}, Senkou A: {senkou_a:.2f}, Senkou B: {senkou_b:.2f}',
            f'{ema50:.2f}', f'{sma50:.2f}', f'{rsi:.2f}', f'MACD: {macd_line:.2f}, Signal: {macd_signal:.2f}',
            f'Upper: {bb_upper:.2f}, Mid: {bb_mid:.2f}, Lower: {bb_lower:.2f}',
            '-', '-', '-', f'SMA(200): {sma200:.2f}', f'SMA(200): {sma200:.2f}',
            f'SMA(100): {sma100:.2f}', f'SMA(100): {sma100:.2f}'],
        'Bullish Signal': ['Price > Cloud' if close > max(senkou_a, senkou_b) else 'Neutral/Bearish',
                           'Price > EMA(50)' if close > ema50 else 'Neutral/Bearish',
                           'Price > SMA(50)' if close > sma50 else 'Neutral/Bearish',
                           'RSI > 50' if rsi > 50 else 'Neutral/Bearish',
                           'MACD > Signal' if macd_line > macd_signal else 'Neutral/Bearish',
                           'Price near Upper' if close > bb_mid else 'Neutral/Bearish',
                           current_adx_rsi, current_bb_rsi, current_uo_mfi, golden_status, death_status,
                           short_golden_status, short_death_status]
    }
    return pd.DataFrame(signals)


def analyze_ticker(ticker, api_key):
    df = fetch_stock_data(ticker, api_key)
    print(f"Current Signals for {ticker}:")
    print(get_current_signals(df))


# Usage: Replace with your Polygon API key
# Create .env file in current directory with this line: POLYGON_API_KEY=<my_api_key>
api_key = os.getenv("POLYGON_API_KEY")
input_str = input("Enter tickers (comma or space separated) or filename: ").strip()

if os.path.isfile(input_str):
    try:
        with open(input_str, 'r') as f:
            tickers = [line.strip().upper() for line in f if line.strip()]
        print(f"Loaded {len(tickers)} tickers from file '{input_str}'.")
    except Exception as e:
        print(f"Error reading file '{input_str}': {e}")
        tickers = []
else:
    # Replace commas with spaces and split, filtering empty
    tickers = [t.strip().upper() for t in re.split(r'[, ]', input_str) if t.strip()]
    print(f"Processing {len(tickers)} tickers from input.")

for ticker in tickers:
    analyze_ticker(ticker, api_key)