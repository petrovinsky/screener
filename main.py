import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import re
import time
import os
import warnings
import numpy as np
import sys

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

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
RED = "#ff6666"  # brighter red for better readability

def color_text(text, color):
    return f"[{color}]{text}[/{color}]"

# Approximate industry average trailing P/E (early 2026 estimates)
INDUSTRY_PE_AVG = {
    'Technology': 32, 'Consumer Cyclical': 24, 'Communication Services': 22,
    'Healthcare': 28, 'Financial Services': 16, 'Industrials': 22,
    'Consumer Defensive': 20, 'Energy': 14, 'Basic Materials': 18,
    'Real Estate': 30, 'Utilities': 19,
}

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
# Company info + valuation metrics (P/FCF, EV/EBITDA)
# ──────────────────────────────────────────────
def get_company_info(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        name = info.get('longName', info.get('shortName', 'N/A'))
        pe = info.get('trailingPE')
        if pe is not None:
            pe = round(float(pe), 1)

        # P/FCF
        p_fcf = None
        if info.get('freeCashflow') and info.get('marketCap'):
            fcf = info['freeCashflow']
            if fcf > 0:
                p_fcf = round(info['marketCap'] / fcf, 1)

        # EV/EBITDA
        ev_ebitda = info.get('enterpriseToEbitda')
        if ev_ebitda is not None:
            ev_ebitda = round(float(ev_ebitda), 1)

        sector = info.get('sector', 'N/A')
        return name, pe, p_fcf, ev_ebitda, sector
    except:
        return 'N/A', None, None, None, 'N/A'

# ──────────────────────────────────────────────
# Earnings Date
# ──────────────────────────────────────────────
def get_next_earnings_days(ticker):
    try:
        stock = yf.Ticker(ticker)

        # Preferred: earnings_dates
        dates = stock.earnings_dates
        if dates is not None and not dates.empty:
            # Ensure index is timezone-naive datetime
            if dates.index.tz is not None:
                dates.index = dates.index.tz_localize(None)
            now = pd.Timestamp.now().normalize()
            upcoming = dates[dates.index > now]
            if not upcoming.empty:
                next_date = upcoming.index.min()
                days = (next_date - now).days
                if 0 < days <= 30:
                    return days

        # Fallback: calendar
        cal = stock.calendar
        if cal is None or 'Earnings Date' not in cal:
            return None

        ed = cal['Earnings Date']

        # Handle various weird formats from yfinance
        if isinstance(ed, (list, tuple)) and ed:
            ed = ed[0]
        elif isinstance(ed, dict):
            ed = ed.get('raw') or ed.get('fmt') or ed.get('Earnings Date')

        # Try to convert to pd.Timestamp safely
        if isinstance(ed, str):
            try:
                ed = pd.to_datetime(ed, utc=True, errors='coerce')
            except:
                console.print(f"[yellow]{ticker} earnings date parse failed (str): {ed}[/yellow]")
                return None
        elif hasattr(ed, 'year') and hasattr(ed, 'month') and hasattr(ed, 'day'):  # duck-type date/datetime
            try:
                ed = pd.Timestamp(ed)
            except:
                ed = None
        else:
            ed = None

        if ed is None or pd.isna(ed):
            return None

        # Normalize to date-only, timezone-free
        ed = pd.Timestamp(ed).normalize().tz_localize(None)
        now = pd.Timestamp.now().normalize().tz_localize(None)

        days = (ed - now).days
        if 0 < days <= 30:
            return days

        return None

    except Exception as e:
        console.print(f"[yellow]Earnings fetch failed for {ticker}: {type(e).__name__}: {str(e)[:120]}[/yellow]")
        return None

# ──────────────────────────────────────────────
# Analyst Rating
# ──────────────────────────────────────────────
def get_analyst_rating(ticker):
    try:
        info = yf.Ticker(ticker).info
        score = info.get('recommendationMean')
        if score is not None:
            score = float(score)
            if score <= 2.0: return "Yes (strong)"
            if score <= 2.5: return "Yes (mod)"
            if score <= 3.5: return "Neutral"
            return "No"
        key = info.get('recommendationKey', '').lower()
        if 'buy' in key: return "Yes (strong)"
        if 'hold' in key: return "Neutral"
        if 'sell' in key: return "No"
        return "N/A"
    except:
        return "Error"

# ──────────────────────────────────────────────
# Bullish Candle
# ──────────────────────────────────────────────
def has_bullish_candle(df):
    if len(df) < 2: return False
    prev, curr = df.iloc[-2], df.iloc[-1]
    body = abs(curr['close'] - curr['open'])
    lower_wick = min(curr['open'], curr['close']) - curr['low']
    return (lower_wick > 2 * body and curr['close'] > curr['open']) or \
           (prev['close'] < prev['open'] and curr['close'] > curr['open'] and
            curr['open'] <= prev['close'] and curr['close'] > prev['open'])

# ──────────────────────────────────────────────
# Bullish RSI Divergence
# ──────────────────────────────────────────────
def has_bullish_rsi_divergence(df, rsi_series, lookback=60):
    if rsi_series is None or len(df) < 20 or len(rsi_series) < lookback:
        return False
    prices = df['close'].tail(lookback)
    rsi_vals = rsi_series.tail(lookback)
    is_low = (prices.shift(1) > prices) & (prices.shift(-1) > prices)
    low_prices = prices[is_low]
    low_rsi = rsi_vals[is_low]
    if len(low_prices) < 2:
        return False
    last_low_price = low_prices.iloc[-1]
    last_low_rsi   = low_rsi.iloc[-1]
    prev_low_price = low_prices.iloc[-2]
    prev_low_rsi   = low_rsi.iloc[-2]
    return last_low_price < prev_low_price and last_low_rsi > prev_low_rsi + 2

# ──────────────────────────────────────────────
# Bull Score – rebalanced weights (refactored)
# ──────────────────────────────────────────────
def get_bull_score(df):
    if len(df) < 60 or ta is None:
        return 0
    try:
        score = 0
        close = df['close'].iloc[-1]

        # ─── Trend alignment – core (highest weight) ────────────────────────
        if len(df) >= 200:
            sma200 = ta.sma(df['close'], length=200).iloc[-1]
            if pd.notna(sma200):
                score += 22 if close > sma200 else -15  # stronger trend filter + penalty

        ema50 = ta.ema(df['close'], length=50).iloc[-1]
        if pd.notna(ema50) and close > ema50:
            score += 24  # highest positive weight – intermediate trend

        ema20 = ta.ema(df['close'], length=20).iloc[-1]
        if pd.notna(ema20) and close > ema20:
            score += 16

        # ─── Momentum / Oversold (tiered, reduced dominance) ────────────────
        rsi = ta.rsi(df['close'], length=14)
        rsi_now = rsi.iloc[-1] if rsi is not None and pd.notna(rsi.iloc[-1]) else 50.0

        if rsi_now < 30:
            score += 16
        elif rsi_now < 35:
            score += 9

        # RSI recovery
        if rsi is not None and len(rsi) >= 30:
            min_rsi = rsi.tail(30).min()
            if pd.notna(min_rsi) and min_rsi <= 34 and rsi_now >= min_rsi + 8 and rsi_now > 36:
                score += 13

        # ─── Reversal / Confirmation signals (higher relative weight) ───────
        if has_bullish_rsi_divergence(df, rsi):
            score += 18

        macd = ta.macd(df['close'])
        if macd is not None and 'MACD_12_26_9' in macd and 'MACDs_12_26_9' in macd:
            if macd['MACD_12_26_9'].iloc[-1] > macd['MACDs_12_26_9'].iloc[-1]:
                score += 13
            # recent cross bonus
            if len(macd) >= 10 and any(
                macd['MACD_12_26_9'].iloc[-i] > macd['MACDs_12_26_9'].iloc[-i] and
                macd['MACD_12_26_9'].iloc[-i-1] <= macd['MACDs_12_26_9'].iloc[-i-1]
                for i in range(1, 11)
            ):
                score += 15

        bb = ta.bbands(df['close'], length=20, std=2)
        if bb is not None and 'BBL_20_2.0' in bb:
            if close <= bb['BBL_20_2.0'].iloc[-1] * 1.015:
                score += 17
                if len(df) >= 2 and df['close'].iloc[-2] > bb['BBL_20_2.0'].iloc[-2]:
                    score += 10

        if len(df) >= 5:
            recent = df.tail(5)
            up_vol_mean = recent[recent['close'] > recent['open']]['volume'].mean()
            if pd.notna(up_vol_mean) and up_vol_mean > recent['volume'].mean() * 1.20:
                score += 12

        if has_bullish_candle(df):
            score += 14

        return min(max(int(score), 0), 100)
    except Exception as e:
        console.print(f"[yellow]Score calc issue: {str(e)[:60]}[/yellow]")
        return 0

# ──────────────────────────────────────────────
# Signals
# ──────────────────────────────────────────────
def get_signals(df, ticker):
    if len(df) < 30 or ta is None:
        return {'Ticker': ticker, 'Error': 'Insufficient data or pandas_ta missing'}
    try:
        close = df['close'].iloc[-1]
        rsi_series = ta.rsi(df['close'], length=14)
        rsi = rsi_series.iloc[-1] if pd.notna(rsi_series.iloc[-1]) else 50.0

        macd = ta.macd(df['close'])
        macd_sigs = []
        if macd is not None and 'MACD_12_26_9' in macd and 'MACDs_12_26_9' in macd:
            macd_line = macd['MACD_12_26_9']
            signal_line = macd['MACDs_12_26_9']
            if len(macd_line) < 15:
                if pd.notna(macd_line.iloc[-1]) and pd.notna(signal_line.iloc[-1]):
                    if macd_line.iloc[-1] > signal_line.iloc[-1]:
                        macd_sigs.append("MACD Bullish")
            else:
                current_macd = macd_line.iloc[-1]
                current_signal = signal_line.iloc[-1]
                if pd.notna(current_macd) and pd.notna(current_signal):
                    if current_macd > current_signal:
                        macd_sigs.append("MACD Bullish")
                    lookback = min(14, len(macd_line) - 1)
                    recent_macd = macd_line.iloc[-lookback-1:-1]
                    recent_signal = signal_line.iloc[-lookback-1:-1]
                    cross_found = False
                    days_since_cross = 0
                    for i in range(len(recent_macd)):
                        if (pd.notna(recent_macd.iloc[i]) and pd.notna(recent_signal.iloc[i]) and
                            recent_macd.iloc[i] <= recent_signal.iloc[i] and
                            macd_line.iloc[-(lookback - i)] > signal_line.iloc[-(lookback - i)]):
                            cross_found = True
                            days_since_cross = lookback - i + 1
                            break
                    if cross_found and current_macd > current_signal:
                        if days_since_cross == 1:
                            macd_sigs.append("MACD Cross Up (today)")
                        else:
                            macd_sigs.append(f"MACD Cross Up ({days_since_cross}d ago)")

        # ADX trend strength rising + +DI > -DI
        adx = ta.adx(df['high'], df['low'], df['close'], length=14)
        if adx is not None and len(adx) >= 3:
            if (adx['ADX_14'].iloc[-1] > adx['ADX_14'].iloc[-2] > 20 and
                adx['DMP_14'].iloc[-1] > adx['DMN_14'].iloc[-1]):
                macd_sigs.append("ADX Uptrend")

        bb = ta.bbands(df['close'], length=20, std=2)
        near_lower_bb = bb is not None and 'BBL_20_2.0' in bb and close <= bb['BBL_20_2.0'].iloc[-1] * 1.02
        bb_bounce = near_lower_bb and df['close'].iloc[-2] > bb['BBL_20_2.0'].iloc[-2]

        rsi_str = "No"
        if rsi_series is not None and len(rsi_series) >= 30:
            recent_rsi = rsi_series.tail(30)
            min_rsi = recent_rsi.min()
            if pd.notna(min_rsi) and min_rsi <= 35 and rsi >= min_rsi + 9 and rsi > 36:
                rsi_str = f"Yes (RSI {rsi:.1f} after {min_rsi:.1f})"

        signals = macd_sigs[:]
        if bb_bounce:
            signals.append("BB Lower Bounce")
        if rsi < 35:
            signals.append("RSI Oversold")
        if "Yes" in rsi_str:
            signals.append("RSI Recovery")
        if has_bullish_rsi_divergence(df, rsi_series):
            signals.append("Bullish RSI Div")
        if has_bullish_candle(df):
            signals.append("Bull Candle (Hammer/Engulf)")

        signal_str = " + ".join(signals) if signals else "Neutral"

        days_to_earn = get_next_earnings_days(ticker)
        earnings_str = f"Yes (in {days_to_earn}d)" if days_to_earn else "No"
        analyst = get_analyst_rating(ticker)
        bull_score_val = get_bull_score(df)

        name, pe, p_fcf, ev_ebitda, sector = get_company_info(ticker)

        return {
            'Ticker': ticker,
            'Name': name,
            'Close': round(close, 2),
            'P/E': pe,
            'P/FCF': p_fcf,
            'EV/EBITDA': ev_ebitda,
            'RSI': round(rsi, 1) if pd.notna(rsi) else "N/A",
            'RSI Recovery': rsi_str,
            'Earnings': earnings_str,
            'Analyst': analyst,
            'Bull Score': bull_score_val,
            'Signal': signal_str
        }
    except Exception as e:
        console.print(f"[red]Signal error for {ticker}: {str(e)[:80]}[/red]")
        return {'Ticker': ticker, 'Error': str(e)[:80]}

# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
if __name__ == "__main__":
    console.print(f"[dim]Python: {sys.version.split()[0]} | yfinance: {yf.__version__}[/dim]")
    if ta is not None:
        console.print("[dim]pandas_ta loaded[/dim]")
    else:
        console.print("[red]pandas_ta NOT loaded – indicators disabled[/red]")

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
        sys.exit(0)

    results = []
    for i, ticker in enumerate(tickers, 1):
        console.print(f"[cyan]{i}/{len(tickers)}[/cyan] Fetching {ticker}...")
        df = fetch_stock_data(ticker)
        if df.empty:
            continue
        sig = get_signals(df, ticker)
        results.append(sig)
        time.sleep(0.9)

    if not results:
        console.print("[red]No valid data retrieved.[/red]")
    else:
        df_res = pd.DataFrame(results)
        df_res = df_res.sort_values(by=['Bull Score', 'RSI'], ascending=[False, True], na_position='last')

        table = Table(title="Bullish Reversal Screener", box=box.ROUNDED,
                      show_header=True, header_style="bold magenta")
        table.add_column("Ticker", style="cyan", no_wrap=True)
        table.add_column("Name", style="white", overflow="fold")
        table.add_column("Close", justify="right")
        table.add_column("P/E", justify="right")
        table.add_column("P/FCF", justify="right")
        table.add_column("EV/EBITDA", justify="right")
        table.add_column("RSI", justify="right")
        table.add_column("RSI Recovery", style="green")
        table.add_column("Earnings", justify="center")
        table.add_column("Analyst", justify="center")
        table.add_column("Bull Score", justify="right", style="bold")
        table.add_column("Signal", overflow="fold")

        for _, row in df_res.iterrows():
            if 'Error' in row:
                table.add_row(row['Ticker'], "-", "-", "-", "-", "-", "-", "-", "-",
                              f"[red]{row.get('Error', '')}[/red]", "-")
                continue

            earn_color = "green" if "Yes" in str(row['Earnings']) else "white"
            analyst_color = "green" if any(s in str(row['Analyst']).lower() for s in ['yes', 'strong', 'buy']) else \
                            "red" if "no" in str(row['Analyst']).lower() else "yellow"
            signal_color = "green" if any(
                x in str(row['Signal']).lower() for x in ["bounce", "cross", "recovery", "bull", "hammer", "div", "adx"]) else "yellow"

            # Valuation colors
            def val_color(v, low, high):
                if pd.isna(v): return "white", "-"
                v = float(v)
                c = "green" if v < low else "red" if v > high else "yellow"
                return c, f"{v:.1f}"

            pe_val = row.get('P/E')
            pe_c, pe_s = val_color(pe_val, 15, 35)

            p_fcf_val = row.get('P/FCF')
            p_fcf_c, p_fcf_s = val_color(p_fcf_val, 15, 30)

            ev_ebitda_val = row.get('EV/EBITDA')
            ev_c, ev_s = val_color(ev_ebitda_val, 10, 18)

            score = row['Bull Score']
            score_str = str(int(score)) if pd.notna(score) else "-"
            score_color = "green" if score >= 76 else "yellow" if score >= 51 else "red" if pd.notna(score) else "white"

            table.add_row(
                row['Ticker'],
                str(row.get('Name', 'N/A')),
                f"{row['Close']:.2f}" if pd.notna(row['Close']) else "-",
                color_text(pe_s, pe_c),
                color_text(p_fcf_s, p_fcf_c),
                color_text(ev_s, ev_c),
                f"{row['RSI']:.1f}" if pd.notna(row['RSI']) else "-",
                str(row['RSI Recovery']),
                color_text(str(row['Earnings']), earn_color),
                color_text(str(row['Analyst']), analyst_color),
                color_text(score_str, score_color),
                color_text(str(row['Signal']), signal_color)
            )

        console.print(table)

        today = datetime.now().strftime('%Y-%m-%d %H-%M')
        os.makedirs("./CSV", exist_ok=True)
        filename = f"./CSV/bull_reversal_{today}.csv"
        df_res.to_csv(filename, index=False)
        console.print(f"\n[green]Saved to: {filename}[/green]")
        console.print("[bold]Done.[/bold]")