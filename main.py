import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import re
import time
import os
import warnings
import numpy as np
import sys

# Suppress common warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Try to import pandas_ta – we'll catch issues later
try:
    import pandas_ta as ta
except ImportError as e:
    print(f"pandas_ta import failed: {e}")
    ta = None

from rich.console import Console
from rich.table import Table
from rich import box

console = Console()

GREEN = "#00ff00"
YELLOW = "#ffff00"
RED = "#ff5555"


def color_text(text, color):
    return f"[{color}]{text}[/{color}]"


# ──────────────────────────────────────────────
# Data Fetch
# ──────────────────────────────────────────────
def fetch_stock_data(ticker, period="2y", interval="1d"):
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
        if df.empty:
            return pd.DataFrame()
        df.index.name = 'date'
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
        df.columns = ['open', 'high', 'low', 'close', 'volume']
        return df
    except Exception as e:
        console.print(f"[red]yfinance error for {ticker}: {str(e)[:80]}[/red]")
        return pd.DataFrame()


# ──────────────────────────────────────────────
# Earnings Date (improved handling for 2026 yfinance)
# ──────────────────────────────────────────────
def get_next_earnings_days(ticker):
    try:
        stock = yf.Ticker(ticker)

        # Preferred: earnings_dates – strip tz to avoid comparison errors
        dates = stock.earnings_dates
        if dates is not None and not dates.empty:
            # Remove timezone from index (safest fix for dtype mismatch)
            dates.index = dates.index.tz_localize(None)
            now = pd.Timestamp.now().normalize()  # date-only comparison
            upcoming = dates[dates.index > now]
            if not upcoming.empty:
                next_date = upcoming.index.min()
                days = (next_date - pd.Timestamp.now()).days
                console.print(f"[dim]{ticker} earnings_dates found, next in {days}d[/dim]")
                if 0 < days <= 30:
                    return days

        # Fallback: calendar
        cal = stock.calendar
        if cal is None:
            return None

        ed = cal.get('Earnings Date')
        if ed is None:
            return None

        # Normalize various formats
        if isinstance(ed, list) and ed:
            ed = ed[0]
        elif isinstance(ed, dict):
            ed = ed.get('raw') or ed.get('fmt') or ed.get('Earnings Date')

        if isinstance(ed, str):
            # Parse ISO-like strings, ignore tz
            try:
                ed = pd.to_datetime(ed, utc=True).tz_localize(None)
            except:
                return None
        elif isinstance(ed, pd.Timestamp):
            ed = ed.tz_localize(None) if ed.tz else ed

        if not isinstance(ed, pd.Timestamp):
            return None

        now = pd.Timestamp.now().normalize()
        days = (ed - now).days
        if 0 < days <= 30:
            console.print(f"[dim]{ticker} calendar found, next in {days}d[/dim]")
            return days

        return None

    except Exception as e:
        console.print(f"[yellow]Earnings fetch failed for {ticker}: {str(e)[:120]}[/yellow]")
        return None
# ──────────────────────────────────────────────
# Analyst Rating (more robust fallback)
# ──────────────────────────────────────────────
def get_analyst_rating(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        # Try recommendation summary first
        rec = stock.recommendations_summary
        if rec is not None and not rec.empty:
            latest = rec.iloc[0]
            score = latest.get('score') or latest.get('RatingScore')
            if score is not None:
                score = float(score)
                if score >= 2.5:
                    return f"Yes (strong) [+{round(score - 2, 1)}]"
                elif score > 2.0:
                    return f"Yes (mod) [+{round(score - 2, 1)}]"
                else:
                    return f"No ({round(score - 2, 1)})"

        # Fallback to info keys
        key = info.get('recommendationKey', 'none').lower()
        score = info.get('recommendationMean', None)  # 1=strong buy, 5=strong sell

        if score is not None:
            score = float(score)
            if score <= 2.0:
                return "Yes (strong)"
            elif score <= 2.5:
                return "Yes (mod)"
            elif score <= 3.5:
                return "Neutral"
            else:
                return "No"

        if 'strong buy' in key or 'buy' in key:
            return "Yes (strong)"
        elif 'hold' in key:
            return "Neutral"
        elif 'sell' in key:
            return "No"
        return "N/A"
    except:
        return "Error"


# ──────────────────────────────────────────────
# Candlestick patterns
# ──────────────────────────────────────────────
def has_bullish_candle(df):
    if len(df) < 2:
        return False
    prev = df.iloc[-2]
    curr = df.iloc[-1]
    body = abs(curr['close'] - curr['open'])
    lower_wick = min(curr['open'], curr['close']) - curr['low']

    # Hammer
    if lower_wick > 2 * body and curr['close'] > curr['open']:
        return True

    # Bullish engulfing
    if (prev['close'] < prev['open'] and
            curr['close'] > curr['open'] and
            curr['open'] <= prev['close'] and
            curr['close'] > prev['open']):
        return True
    return False


# ──────────────────────────────────────────────
# Bull Score with NaN safety
# ──────────────────────────────────────────────
def get_bull_score(df):
    if len(df) < 60 or ta is None:
        return 0

    try:
        score = 0
        close = df['close'].iloc[-1]

        ema20 = ta.ema(df['close'], length=20)
        ema50 = ta.ema(df['close'], length=50)
        if pd.notna(ema20.iloc[-1]) and close > ema20.iloc[-1]:
            score += 15
        if pd.notna(ema50.iloc[-1]) and close > ema50.iloc[-1]:
            score += 20

        if len(df) >= 200:
            sma200 = ta.sma(df['close'], length=200)
            if pd.notna(sma200.iloc[-1]) and close > sma200.iloc[-1]:
                score += 18

        rsi = ta.rsi(df['close'], length=14)
        if rsi is None or pd.isna(rsi.iloc[-1]):
            rsi_now = 50.0
        else:
            rsi_now = rsi.iloc[-1]

        if rsi_now < 35:
            score += 25
        elif rsi_now > 70:
            score -= 18

        # RSI recovery
        if len(rsi) >= 30:
            recent_rsi = rsi.tail(30)
            min_rsi = recent_rsi.min()
            if pd.notna(min_rsi) and min_rsi <= 35 and rsi_now >= min_rsi + 9 and rsi_now > 36:
                score += 15

        # RSI divergence (fixed lookback)
        if len(df) >= 20:
            recent_price = df['close'].tail(20)
            recent_rsi = rsi.tail(20) if rsi is not None else pd.Series()
            if not recent_rsi.empty and pd.notna(recent_rsi).any():
                price_low_idx = recent_price.idxmin()
                rsi_low_idx = recent_rsi.idxmin()
                if price_low_idx != rsi_low_idx:
                    prev_price_low = recent_price.iloc[:-5].min()  # look earlier in window
                    prev_rsi_low = recent_rsi.iloc[:-5].min()
                    if (close < prev_price_low and
                            rsi_now > prev_rsi_low + 3):
                        score += 18

        macd = ta.macd(df['close'])
        if macd is not None and 'MACD_12_26_9' in macd and 'MACDs_12_26_9' in macd:
            if macd['MACD_12_26_9'].iloc[-1] > macd['MACDs_12_26_9'].iloc[-1]:
                score += 15

        bb = ta.bbands(df['close'], length=20, std=2)
        if bb is not None and 'BBL_20_2.0' in bb:
            if close <= bb['BBL_20_2.0'].iloc[-1] * 1.015:
                score += 20

        # Volume confirmation
        if len(df) >= 5:
            recent = df.tail(5)
            up_days = recent[recent['close'] > recent['open']]
            if not up_days.empty and up_days['volume'].mean() > recent['volume'].mean() * 1.15:
                score += 12

        if has_bullish_candle(df):
            score += 14

        return min(max(int(score), 0), 100)
    except Exception as e:
        console.print(f"[yellow]Score calc issue for {df.index.name}: {str(e)[:60]}[/yellow]")
        return 0


# ──────────────────────────────────────────────
# Signals
# ──────────────────────────────────────────────
def get_signals(df, ticker):
    if len(df) < 30 or ta is None:
        return {'Error': 'Insufficient data or pandas_ta missing'}

    try:
        close = df['close'].iloc[-1]
        rsi_series = ta.rsi(df['close'], length=14)
        rsi = rsi_series.iloc[-1] if pd.notna(rsi_series.iloc[-1]) else 50.0

        macd = ta.macd(df['close'])
        macd_bull = (macd is not None and
                     'MACD_12_26_9' in macd and
                     macd['MACD_12_26_9'].iloc[-1] > macd['MACDs_12_26_9'].iloc[-1])

        bb = ta.bbands(df['close'], length=20, std=2)
        near_lower_bb = (bb is not None and
                         'BBL_20_2.0' in bb and
                         close <= bb['BBL_20_2.0'].iloc[-1] * 1.02)

        rsi_str = "No"
        if rsi_series is not None and len(rsi_series) >= 30:
            recent_rsi = rsi_series.tail(30)
            min_rsi = recent_rsi.min()
            if pd.notna(min_rsi) and min_rsi <= 35 and rsi >= min_rsi + 9 and rsi > 36:
                rsi_str = f"Yes (RSI {rsi:.1f} after {min_rsi:.1f})"

        signals = []
        if near_lower_bb and df['close'].iloc[-2] > bb['BBL_20_2.0'].iloc[-2]:
            signals.append("BB Lower Bounce")
        if rsi < 35:
            signals.append("RSI Oversold")
        if macd_bull:
            signals.append("MACD Bull Cross")
        if "Yes" in rsi_str:
            signals.append("RSI Recovery")
        if has_bullish_candle(df):
            signals.append("Bull Candle (Hammer/Engulf)")

        signal_str = " + ".join(signals) if signals else "Neutral"

        days_to_earn = get_next_earnings_days(ticker)
        earnings_str = f"Yes (in {days_to_earn}d)" if days_to_earn else "No"

        analyst = get_analyst_rating(ticker)

        bull_score = get_bull_score(df)

        return {
            'Ticker': ticker,
            'Close': round(close, 2),
            'RSI': round(rsi, 1) if pd.notna(rsi) else "N/A",
            'RSI Recovery': rsi_str,
            'MACD': 'Bullish' if macd_bull else 'Neutral/Bear',
            'BB Pos': 'Near Lower' if near_lower_bb else 'Mid/Upper',
            'Earnings': earnings_str,
            'Analyst': analyst,
            'Bull Score': bull_score,
            'Signal': signal_str
        }
    except Exception as e:
        console.print(f"[red]Signal error for {ticker}: {str(e)[:80]}[/red]")
        return {'Ticker': ticker, 'Error': str(e)[:80]}


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
if __name__ == "__main__":
    # Quick dependency check
    console.print(f"[dim]Python: {sys.version.split()[0]} | yfinance: {yf.__version__}[/dim]")
    if ta is not None:
        console.print(f"[dim]pandas_ta loaded[/dim]")
    else:
        console.print("[red]pandas_ta NOT loaded – technical indicators disabled[/red]")

    input_str = input("\nEnter tickers (comma/space sep) or filename: ").strip()

    if os.path.isfile(input_str):
        try:
            with open(input_str, 'r') as f:
                tickers = [line.strip().upper() for line in f if line.strip()]
            console.print(f"Loaded {len(tickers)} tickers from file.")
        except Exception as e:
            console.print(f"[red]File read error: {e}[/red]")
            tickers = []
    else:
        tickers = [t.strip().upper() for t in re.split(r'[, ]', input_str) if t.strip()]
        console.print(f"Processing {len(tickers)} tickers.")

    if not tickers:
        console.print("[red]No tickers provided.[/red]")
        exit(0)

    results = []
    for i, ticker in enumerate(tickers, 1):
        console.print(f"[cyan]{i}/{len(tickers)}[/cyan] Fetching {ticker}...")
        df = fetch_stock_data(ticker)
        if df.empty:
            continue
        sig = get_signals(df, ticker)
        results.append(sig)
        time.sleep(0.9)  # be gentle to Yahoo

    if not results:
        console.print("[red]No valid data retrieved.[/red]")
    else:
        df_res = pd.DataFrame(results)
        # Move error rows to bottom
        df_res = df_res.sort_values(by=['Bull Score', 'RSI'], ascending=[False, True], na_position='last')

        table = Table(title="Bullish Reversal Screener", box=box.ROUNDED,
                      show_header=True, header_style="bold magenta")

        table.add_column("Ticker", style="cyan", no_wrap=True)
        table.add_column("Close", justify="right")
        table.add_column("RSI", justify="right")
        table.add_column("RSI Recovery", style="green")
        table.add_column("MACD", justify="center")
        table.add_column("BB Pos", justify="center")
        table.add_column("Earnings", justify="center")
        table.add_column("Analyst", justify="center")
        table.add_column("Bull Score", justify="right", style="bold yellow")
        table.add_column("Signal", overflow="fold")

        for _, row in df_res.iterrows():
            if 'Error' in row:
                table.add_row(row['Ticker'], "-", "-", "-", "-", "-", "-", "-", "-",
                              f"[red]{row.get('Error', '')}[/red]")
                continue

            earn_color = "green" if "Yes" in str(row['Earnings']) else "white"
            analyst_color = "green" if any(s in str(row['Analyst']).lower() for s in ['yes', 'strong', 'buy']) else \
                "red" if "no" in str(row['Analyst']).lower() else "yellow"
            signal_color = "green" if any(
                x in str(row['Signal']).lower() for x in ["bounce", "recovery", "bull", "hammer"]) else "yellow"

            table.add_row(
                row['Ticker'],
                f"{row['Close']:.2f}" if pd.notna(row['Close']) else "-",
                f"{row['RSI']:.1f}" if pd.notna(row['RSI']) else "-",
                str(row['RSI Recovery']),
                str(row['MACD']),
                str(row['BB Pos']),
                color_text(str(row['Earnings']), earn_color),
                color_text(str(row['Analyst']), analyst_color),
                str(row['Bull Score']),
                color_text(str(row['Signal']), signal_color)
            )

        console.print(table)

        today = datetime.now().strftime('%Y-%m-%d %H-%M')
        os.makedirs("./CSV", exist_ok=True)
        filename = f"./CSV/bull_reversal_{today}.csv"
        df_res.to_csv(filename, index=False)
        console.print(f"\n[green]Saved to: {filename}[/green]")
        console.print("[bold]Done.[/bold]")