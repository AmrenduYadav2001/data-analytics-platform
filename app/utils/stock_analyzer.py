import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go
import plotly.utils
import json


def get_stock_data(symbol, period="6mo"):
    # Only add .NS if it's not a US stock and has no exchange suffix
    if not symbol.endswith(".NS") and not symbol.endswith(".BO") and "." not in symbol:
        # Try NSE first
        nse_symbol = symbol + ".NS"
        stock = yf.Ticker(nse_symbol)
        df = stock.history(period=period)
        
        if df.empty:
            # fallback to US stock
            stock = yf.Ticker(symbol)
            df = stock.history(period=period)
        else:
            symbol = nse_symbol
    else:
        stock = yf.Ticker(symbol)
        df = stock.history(period=period)

    if df.empty:
        raise ValueError(f"No data found for {symbol}. Try: RELIANCE, TCS, INFY for Indian stocks or AAPL, TSLA for US stocks.")

    return df, stock.info


def calculate_indicators(df):
    df["MA7"]  = df["Close"].rolling(7).mean()
    df["MA20"] = df["Close"].rolling(20).mean()
    df["MA50"] = df["Close"].rolling(50).mean()
    delta = df["Close"].diff()
    gain  = delta.where(delta > 0, 0).rolling(14).mean()
    loss  = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs    = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))
    df["Volatility"] = df["Close"].pct_change().rolling(20).std() * 100
    return df


def generate_signal(df):
    latest = df.iloc[-1]
    prev   = df.iloc[-2]
    score  = 0
    if latest["MA7"] > latest["MA20"]:
        score += 1
    if latest["MA20"] > latest["MA50"]:
        score += 1
    rsi = latest["RSI"]
    if rsi < 35:
        score += 2
    elif rsi > 70:
        score -= 2
    if latest["Close"] > prev["Close"]:
        score += 1
    if score >= 3:
        return "BUY", "green"
    elif score <= 0:
        return "SELL", "red"
    else:
        return "HOLD", "orange"


def risk_management(current_price, signal):
    if signal == "BUY":
        stop_loss = round(current_price * 0.95, 2)
        target_1  = round(current_price * 1.10, 2)
        target_2  = round(current_price * 1.20, 2)
    else:
        stop_loss = round(current_price * 1.05, 2)
        target_1  = round(current_price * 0.90, 2)
        target_2  = round(current_price * 0.80, 2)
    max_loss   = round(abs(current_price - stop_loss), 2)
    max_profit = round(abs(current_price - target_1), 2)
    rr_ratio   = round(max_profit / max_loss, 2) if max_loss > 0 else 0
    return {
        "stop_loss":  stop_loss,
        "target_1":   target_1,
        "target_2":   target_2,
        "max_loss":   max_loss,
        "max_profit": max_profit,
        "rr_ratio":   rr_ratio
    }


def predict_next_7_days(df):
    df   = df.copy().dropna()
    data = df["Close"].tail(60).values
    X    = np.arange(len(data)).reshape(-1, 1)
    y    = data
    model = LinearRegression()
    model.fit(X, y)
    future_X    = np.arange(len(data), len(data) + 7).reshape(-1, 1)
    predictions = model.predict(future_X)
    direction   = "UP 📈" if predictions[-1] > predictions[0] else "DOWN 📉"
    change      = round(((predictions[-1] - data[-1]) / data[-1]) * 100, 2)
    return {
        "predictions": [round(p, 2) for p in predictions],
        "direction":   direction,
        "change_pct":  change
    }


def generate_chart(df, symbol):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["Open"],
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        name=symbol
    ))
    fig.add_trace(go.Scatter(x=df.index, y=df["MA20"], name="MA20", line=dict(color="blue", width=1)))
    fig.add_trace(go.Scatter(x=df.index, y=df["MA50"], name="MA50", line=dict(color="orange", width=1)))
    fig.update_layout(
        title=f"{symbol} Price Chart",
        xaxis_title="Date",
        yaxis_title="Price (₹)",
        template="plotly_white",
        height=400,
        xaxis_rangeslider_visible=False
    )
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)


def analyze_stock(symbol):
    df, info      = get_stock_data(symbol)
    df            = calculate_indicators(df)
    signal, color = generate_signal(df)
    current_price = round(df["Close"].iloc[-1], 2)
    risk          = risk_management(current_price, signal)
    prediction    = predict_next_7_days(df)
    chart         = generate_chart(df, symbol)
    return {
        "symbol":        symbol.replace(".NS", ""),
        "current_price": current_price,
        "signal":        signal,
        "signal_color":  color,
        "rsi":           round(df["RSI"].iloc[-1], 2),
        "volatility":    round(df["Volatility"].iloc[-1], 2),
        "risk":          risk,
        "prediction":    prediction,
        "chart":         chart,
        "company_name":  info.get("longName", symbol),
        "sector":        info.get("sector", "N/A"),
        "market_cap":    info.get("marketCap", "N/A"),
    }