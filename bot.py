import os
import asyncio
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes

import oandapyV20
import oandapyV20.endpoints.instruments as instruments

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
OANDA_TOKEN = os.getenv("OANDA_API_KEY")
CHAT_ID = os.getenv("CHAT_ID")

PAIR = "EUR_USD"
client = oandapyV20.API(access_token=OANDA_TOKEN)

# ===== DATA =====
def get_candles(tf, count=200):
    try:
        r = instruments.InstrumentsCandles(
            instrument=PAIR,
            params={"granularity": tf, "count": count, "price": "M"}
        )
        client.request(r)

        data = []
        for c in r.response["candles"]:
            if c["complete"]:
                data.append({
                    "open": float(c["mid"]["o"]),
                    "high": float(c["mid"]["h"]),
                    "low": float(c["mid"]["l"]),
                    "close": float(c["mid"]["c"]),
                })
        return pd.DataFrame(data)
    except:
        return pd.DataFrame()

# ===== INDICATORS =====
def add_indicators(df):
    df["ema20"] = df["close"].ewm(span=20).mean()
    df["ema50"] = df["close"].ewm(span=50).mean()

    delta = df["close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss
    df["rsi"] = 100 - (100 / (1 + rs))

    df["atr"] = (df["high"] - df["low"]).rolling(14).mean()

    return df

# ===== SUPPORT / RESISTANCE =====
def levels(df):
    recent = df.tail(20)
    support = recent["low"].min()
    resistance = recent["high"].max()
    return support, resistance

# ===== PRICE ACTION =====
def candle_dir(df):
    c = df.iloc[-3:]
    bull = sum(c["close"] > c["open"])
    bear = sum(c["close"] < c["open"])
    if bull >= 2: return 1
    if bear >= 2: return -1
    return 0

def engulfing(df):
    c1, c2 = df.iloc[-1], df.iloc[-2]
    if c1["close"] > c1["open"] and c2["close"] < c2["open"]:
        if c1["close"] > c2["open"]: return 1
    if c1["close"] < c1["open"] and c2["close"] > c2["open"]:
        if c1["close"] < c2["open"]: return -1
    return 0

def pin_bar(df):
    c = df.iloc[-1]
    body = abs(c["close"] - c["open"])
    wick = c["high"] - c["low"]
    if body < wick * 0.3:
        return 1 if c["close"] > c["open"] else -1
    return 0

# ===== FEATURE ENGINEERING =====
def build_features(df1, df5, df15):
    return np.array([
        df1["close"].iloc[-1] - df1["close"].iloc[-3],
        df5["close"].iloc[-1] - df5["close"].iloc[-3],
        df15["close"].iloc[-1] - df15["close"].iloc[-3],
        df5["ema20"].iloc[-1] - df5["ema50"].iloc[-1],
        df5["rsi"].iloc[-1],
        df5["atr"].iloc[-1]
    ])

# ===== MODEL =====
def train():
    df = add_indicators(get_candles("M5", 600))
    X, y = [], []

    for i in range(50, len(df)-3):
        sub = df.iloc[:i]
        X.append([
            sub["close"].iloc[-1] - sub["close"].iloc[-3],
            sub["ema20"].iloc[-1] - sub["ema50"].iloc[-1],
            sub["rsi"].iloc[-1],
            sub["atr"].iloc[-1]
        ])
        y.append(1 if df["close"].iloc[i+2] > df["close"].iloc[i] else 0)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    model = RandomForestClassifier(n_estimators=200)
    model.fit(X, y)

    joblib.dump((model, scaler), "model.pkl")

def load_model():
    if not os.path.exists("model.pkl"):
        train()
    return joblib.load("model.pkl")

# ===== SIGNAL =====
def signal():
    df1 = add_indicators(get_candles("M1"))
    df5 = add_indicators(get_candles("M5"))
    df15 = add_indicators(get_candles("M15"))

    if df5.empty:
        return None, "NO DATA"

    model, scaler = load_model()

    feat = build_features(df1, df5, df15).reshape(1, -1)
    feat = scaler.transform(feat)

    prob = model.predict_proba(feat)[0][1]
    conf = int(prob * 100)

    score = 0

    # ML
    score += (prob - 0.5) * 6

    # Trend
    score += 2 if df5["ema20"].iloc[-1] > df5["ema50"].iloc[-1] else -2

    # PA
    score += candle_dir(df1) + candle_dir(df5) + candle_dir(df15)

    # Patterns
    score += engulfing(df5) * 2 + pin_bar(df5)

    # Levels
    sup, res = levels(df5)
    price = df5["close"].iloc[-1]
    if price < sup * 1.002:
        score += 1
    if price > res * 0.998:
        score -= 1

    # Kill filter
    if abs(score) < 3:
        return None, "NO EDGE"

    direction = "BUY" if score > 0 else "SELL"

    atr = df5["atr"].iloc[-1]
    sl = price - atr if direction == "BUY" else price + atr
    tp = price + atr * 1.7 if direction == "BUY" else price - atr * 1.7

    return direction, conf, round(tp,5), round(sl,5)

# ===== JOURNAL =====
def log_trade(res):
    with open("trades.log", "a") as f:
        f.write(f"{datetime.now()} | {res}\n")

# ===== TELEGRAM =====
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🔥 V50 PRO MAX")

async def sig(update: Update, context: ContextTypes.DEFAULT_TYPE):
    res = signal()

    if res[0] is None:
        await update.message.reply_text(f"❌ {res[1]}")
        return

    log_trade(res)

    d, c, tp, sl = res

    await update.message.reply_text(
        f"{d}\nCONF: {c}%\nTP: {tp}\nSL: {sl}"
    )

# ===== AUTO =====
async def auto(app):
    while True:
        try:
            res = signal()
            if res[0]:
                log_trade(res)
                await app.bot.send_message(
                    chat_id=CHAT_ID,
                    text=f"{res[0]} | {res[1]}%"
                )
        except:
            pass

        await asyncio.sleep(300)

# ===== MAIN =====
async def main():
    app = ApplicationBuilder().token(TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("signal", sig))

    await app.initialize()
    await app.start()

    asyncio.create_task(auto(app))

    await app.updater.start_polling()
    await asyncio.Event().wait()

if __name__ == "__main__":
    asyncio.run(main())
