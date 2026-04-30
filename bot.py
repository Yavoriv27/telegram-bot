# ===== DEBUG START =====
print("🔥 FILE STARTED")

import os
import asyncio
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from collections import deque

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ApplicationBuilder, CommandHandler, CallbackQueryHandler, ContextTypes

import oandapyV20
import oandapyV20.endpoints.instruments as instruments

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# ===== CONFIG =====
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
OANDA_TOKEN = os.getenv("OANDA_API_KEY")
CHAT_ID = os.getenv("CHAT_ID")

print("TOKEN:", "OK" if TOKEN else "NONE")
print("OANDA:", "OK" if OANDA_TOKEN else "NONE")
print("CHAT_ID:", CHAT_ID)

PAIR = "EUR_USD"
LOG_FILE = "trades.csv"
MODEL_FILE = "model.pkl"

client = oandapyV20.API(access_token=OANDA_TOKEN)

# ===== STATE =====
results = deque(maxlen=300)
balance = 1000.0
loss_streak = 0
last_signal = None

weights = {
    "ml": 1.0,
    "pa": 1.0,
    "trend": 1.0
}

# ===== SESSION FILTER =====
def session_filter():
    hour = datetime.utcnow().hour
    if 6 <= hour <= 12:
        return "LONDON"
    elif 13 <= hour <= 18:
        return "NEWYORK"
    return "DEAD"

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
    except Exception as e:
        print("DATA ERROR:", e)
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

# ===== LOGIC (без змін) =====
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

# ===== MODEL (без змін) =====
def train():
    df1 = add_indicators(get_candles("M1", 800))
    df5 = add_indicators(get_candles("M5", 800))
    df15 = add_indicators(get_candles("M15", 800))

    X, y = [], []

    for i in range(50, len(df5)-3):
        X.append([
            df1["close"].iloc[i] - df1["close"].iloc[i-3],
            df5["close"].iloc[i] - df5["close"].iloc[i-3],
            df15["close"].iloc[i] - df15["close"].iloc[i-3],
            df5["ema20"].iloc[i] - df5["ema50"].iloc[i],
            df5["rsi"].iloc[i],
            df5["atr"].iloc[i]
        ])

        y.append(1 if df5["close"].iloc[i+2] > df5["close"].iloc[i] else 0)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    model = RandomForestClassifier(n_estimators=250)
    model.fit(X, y)

    joblib.dump((model, scaler), MODEL_FILE)

def load_model():
    if os.path.exists(MODEL_FILE):
        print("DELETE OLD MODEL")
        os.remove(MODEL_FILE)

    print("TRAIN NEW MODEL")
    train()

    return joblib.load(MODEL_FILE)

# ===== SIGNAL (без змін логіки) =====
def signal():
    global loss_streak

    if loss_streak >= 3:
        return None, "PAUSE AFTER LOSSES"

    if session_filter() == "DEAD":
        return None, "BAD SESSION"

    df1 = add_indicators(get_candles("M1"))
    df5 = add_indicators(get_candles("M5"))
    df15 = add_indicators(get_candles("M15"))

    if df5.empty:
        return None, "NO DATA"

    model, scaler = load_model()

    feat = np.array([
        df1["close"].iloc[-1] - df1["close"].iloc[-3],
        df5["close"].iloc[-1] - df5["close"].iloc[-3],
        df15["close"].iloc[-1] - df15["close"].iloc[-3],
        df5["ema20"].iloc[-1] - df5["ema50"].iloc[-1],
        df5["rsi"].iloc[-1],
        df5["atr"].iloc[-1]
    ]).reshape(1, -1)

    feat = scaler.transform(feat)
    prob = model.predict_proba(feat)[0][1]
    conf = int(prob * 100)

    score = (prob - 0.5) * 6 * weights["ml"]
    score += (2 if df5["ema20"].iloc[-1] > df5["ema50"].iloc[-1] else -2) * weights["trend"]
    score += (candle_dir(df1) + candle_dir(df5) + candle_dir(df15)) * weights["pa"]
    score += engulfing(df5)*2 + pin_bar(df5)

    if df5["atr"].iloc[-1] < df5["atr"].mean()*0.7:
        return None, "LOW VOL"

    if abs(score) < 3:
        return None, "NO EDGE"

    direction = "BUY" if score > 0 else "SELL"

    price = df5["close"].iloc[-1]
    atr = df5["atr"].iloc[-1]

    risk = 0.05 if loss_streak > 0 else 0.1

    tp = price + atr*1.8 if direction == "BUY" else price - atr*1.8
    sl = price - atr if direction == "BUY" else price + atr

    return direction, conf, round(tp,5), round(sl,5), risk

# ===== TELEGRAM =====
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🚀 BOT WORKING")

async def signal_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        res = signal()

        if res[0] is None:
            await update.message.reply_text(f"❌ {res[1]}")
            return

        d, c, tp, sl, r = res

        await update.message.reply_text(
            f"{d}\nCONF: {c}%\nTP: {tp}\nSL: {sl}\nRISK: {int(r*100)}%"
        )
    except Exception as e:
        print("SIGNAL ERROR:", e)

# ===== MAIN =====
def main():
    try:
        app = ApplicationBuilder().token(TOKEN).build()

        app.add_handler(CommandHandler("start", start))
        app.add_handler(CommandHandler("signal", signal_cmd))

        print("🚀 BOT STARTED")

        app.run_polling()

    except Exception as e:
        print("MAIN ERROR:", e)


if __name__ == "__main__":
    main()
