import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.utils
import json


def get_historical_data(symbol, period="2y"):
    """
    Fetch 2 years of historical data.
    
    WHY 2 years?
    → More data = more trades = more reliable accuracy %
    → 6 months gives maybe 5 trades, 2 years gives 20+ trades
    """
    if "." not in symbol:
        symbol = symbol + ".NS"

    stock = yf.Ticker(symbol)
    df    = stock.history(period=period)

    if df.empty:
        raise ValueError(f"No data found for {symbol}")

    return df, symbol


def calculate_signals(df):
    """
    Calculate BUY/SELL signals for every single day in history.
    
    HOW IT WORKS:
    - MA7  = average price of last 7 days  (fast moving)
    - MA20 = average price of last 20 days (slow moving)
    - MA50 = average price of last 50 days (very slow)
    
    SIGNAL LOGIC:
    - When fast (MA7) crosses ABOVE slow (MA20) → momentum going UP → BUY
    - When fast (MA7) crosses BELOW slow (MA20) → momentum going DOWN → SELL
    
    This is called "Moving Average Crossover Strategy"
    It's one of the most used strategies by real traders.
    """
    df = df.copy()

    # calculate moving averages
    df["MA7"]  = df["Close"].rolling(7).mean()
    df["MA20"] = df["Close"].rolling(20).mean()
    df["MA50"] = df["Close"].rolling(50).mean()

    # RSI — tells if stock is overbought or oversold
    delta = df["Close"].diff()
    gain  = delta.where(delta > 0, 0).rolling(14).mean()
    loss  = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs    = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))

    """
    SIGNAL COLUMN:
    - 1  = BUY signal on this day
    - -1 = SELL signal on this day
    - 0  = no signal, hold whatever you have
    
    HOW TO READ THE CROSSOVER:
    - today MA7 > MA20  AND  yesterday MA7 < MA20 → crossover just happened → BUY
    - today MA7 < MA20  AND  yesterday MA7 > MA20 → crossover just happened → SELL
    """
    df["Signal"] = 0

    # BUY condition — MA7 crosses above MA20
    df.loc[
        (df["MA7"] > df["MA20"]) &
        (df["MA7"].shift(1) <= df["MA20"].shift(1)),
        "Signal"
    ] = 1

    # SELL condition — MA7 crosses below MA20
    df.loc[
        (df["MA7"] < df["MA20"]) &
        (df["MA7"].shift(1) >= df["MA20"].shift(1)),
        "Signal"
    ] = -1

    return df


def simulate_trades(df, initial_capital=100000):
    """
    Simulate actual trades based on signals.
    
    WHAT WE'RE DOING:
    - Start with ₹1,00,000 (one lakh)
    - Every BUY signal  → spend all money to buy shares
    - Every SELL signal → sell all shares, get money back
    - Track profit/loss on each trade
    
    WHY initial_capital=100000?
    → Standard assumption for backtesting
    → Makes it easy to calculate % returns
    """

    trades        = []       # list of all completed trades
    capital       = initial_capital
    shares_held   = 0        # how many shares we currently own
    buy_price     = 0        # at what price we bought
    buy_date      = None
    position_open = False    # are we currently in a trade?

    for date, row in df.iterrows():

        # skip rows where MA not yet calculated (first 20 days)
        if pd.isna(row["MA20"]):
            continue

        # BUY SIGNAL — only buy if we don't already own shares
        if row["Signal"] == 1 and not position_open:
            shares_held   = capital / row["Close"]  # buy as many shares as we can
            buy_price     = row["Close"]
            buy_date      = date
            position_open = True

        # SELL SIGNAL — only sell if we own shares
        elif row["Signal"] == -1 and position_open:
            sell_price  = row["Close"]
            sell_value  = shares_held * sell_price
            profit_loss = sell_value - capital           # how much we made/lost
            profit_pct  = ((sell_price - buy_price) / buy_price) * 100

            trades.append({
                "buy_date":   str(buy_date.date()),
                "sell_date":  str(date.date()),
                "buy_price":  round(buy_price, 2),
                "sell_price": round(sell_price, 2),
                "profit_loss": round(profit_loss, 2),
                "profit_pct":  round(profit_pct, 2),
                "result":      "WIN" if profit_loss > 0 else "LOSS"
            })

            capital       = sell_value   # update capital
            shares_held   = 0
            position_open = False

    return trades, capital


def calculate_metrics(trades, initial_capital, final_capital):
    """
    Calculate overall performance metrics.
    
    WHAT EACH METRIC MEANS:
    
    total_return % → if you started with ₹1L and now have ₹1.18L → return = 18%
    
    win_rate % → out of 10 trades, 7 were profit → win rate = 70%
                 Good strategies have win rate > 55%
    
    max_drawdown → biggest loss you faced at any point
                   e.g. capital went from ₹1L to ₹85K → drawdown = -15%
                   Lower is better. Professional funds aim for < 20%
    
    avg_profit → average profit on winning trades
    avg_loss   → average loss on losing trades
    
    profit_factor → total profit / total loss
                    > 1 means you made more than you lost → good
                    > 2 means you made 2x what you lost  → great
    """

    if not trades:
        return {
            "total_trades":  0,
            "win_rate":      0,
            "total_return":  0,
            "profit_factor": 0,
            "max_drawdown":  0,
            "avg_profit":    0,
            "avg_loss":      0,
        }

    total_trades = len(trades)
    wins         = [t for t in trades if t["result"] == "WIN"]
    losses       = [t for t in trades if t["result"] == "LOSS"]

    win_rate     = round((len(wins) / total_trades) * 100, 2)
    total_return = round(((final_capital - initial_capital) / initial_capital) * 100, 2)

    total_profit = sum(t["profit_loss"] for t in wins)   if wins   else 0
    total_loss   = abs(sum(t["profit_loss"] for t in losses)) if losses else 1
    profit_factor = round(total_profit / total_loss, 2)

    avg_profit = round(total_profit / len(wins),   2) if wins   else 0
    avg_loss   = round(total_loss   / len(losses), 2) if losses else 0

    # max drawdown calculation
    # → track capital after each trade and find biggest drop
    capital_curve = [initial_capital]
    running       = initial_capital
    for t in trades:
        running += t["profit_loss"]
        capital_curve.append(running)

    peak        = initial_capital
    max_drawdown = 0
    for c in capital_curve:
        if c > peak:
            peak = c
        drawdown = ((peak - c) / peak) * 100
        if drawdown > max_drawdown:
            max_drawdown = drawdown

    return {
        "total_trades":  total_trades,
        "winning_trades": len(wins),
        "losing_trades":  len(losses),
        "win_rate":       win_rate,
        "total_return":   total_return,
        "profit_factor":  profit_factor,
        "max_drawdown":   round(max_drawdown, 2),
        "avg_profit":     avg_profit,
        "avg_loss":       avg_loss,
        "final_capital":  round(final_capital, 2),
    }


def generate_backtest_chart(df, trades, symbol):
    """
    Generate chart showing:
    1. Stock price over 2 years
    2. Where BUY signals happened (green triangles)
    3. Where SELL signals happened (red triangles)
    
    This lets you VISUALLY see if the strategy makes sense.
    """
    fig = go.Figure()

    # price line
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df["Close"],
        name="Price",
        line=dict(color="#3b82f6", width=1.5),
        opacity=0.8
    ))

    # BUY markers — green triangles pointing up
    buy_signals = df[df["Signal"] == 1]
    fig.add_trace(go.Scatter(
        x=buy_signals.index,
        y=buy_signals["Close"],
        mode="markers",
        name="BUY Signal",
        marker=dict(symbol="triangle-up", size=12, color="#22c55e")
    ))

    # SELL markers — red triangles pointing down
    sell_signals = df[df["Signal"] == -1]
    fig.add_trace(go.Scatter(
        x=sell_signals.index,
        y=sell_signals["Close"],
        mode="markers",
        name="SELL Signal",
        marker=dict(symbol="triangle-down", size=12, color="#ef4444")
    ))

    fig.update_layout(
        title=f"{symbol} — Backtest Signals (2 Years)",
        xaxis_title="Date",
        yaxis_title="Price (₹)",
        template="plotly_dark",
        height=450,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#94a3b8"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
    )

    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)


def run_backtest(symbol, initial_capital=100000):
    """
    MAIN FUNCTION — runs the full backtest pipeline.
    
    Pipeline:
    1. Fetch 2 years data
    2. Calculate signals on every day
    3. Simulate trades
    4. Calculate metrics
    5. Generate chart
    6. Return everything
    """
    symbol = symbol.upper().strip()

    # Step 1 — get data
    df, symbol = get_historical_data(symbol)

    # Step 2 — calculate signals
    df = calculate_signals(df)

    # Step 3 — simulate trades
    trades, final_capital = simulate_trades(df, initial_capital)

    # Step 4 — calculate metrics
    metrics = calculate_metrics(trades, initial_capital, final_capital)

    # Step 5 — generate chart
    chart = generate_backtest_chart(df, trades, symbol)

    return {
        "symbol":          symbol.replace(".NS", ""),
        "trades":          trades,
        "metrics":         metrics,
        "chart":           chart,
        "initial_capital": initial_capital,
    }