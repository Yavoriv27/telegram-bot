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

# ===== LOGIC =====
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

# ===== MODEL =====
def train():
    df = add_indicators(get_candles("M5", 800))
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

    model = RandomForestClassifier(n_estimators=250)
    model.fit(X, y)

    joblib.dump((model, scaler), MODEL_FILE)

def load_model():
    if not os.path.exists(MODEL_FILE):
        train()
    return joblib.load(MODEL_FILE)

# ===== ONLINE LEARNING =====
def retrain_if_needed():
    if not os.path.exists(LOG_FILE):
        return

    df = pd.read_csv(LOG_FILE)

    if len(df) < 50:
        return

    if len(df) % 20 == 0:
        train()

# ===== ADAPTIVE =====
def adaptive_threshold():
    if len(results) < 30:
        return 3

    wr = sum(results)/len(results)

    if wr > 0.65:
        return 2.5
    elif wr < 0.4:
        return 4
    return 3

def adjust_weights(win):
    if win:
        weights["pa"] += 0.05
    else:
        weights["ml"] += 0.05

# ===== SIGNAL =====
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

    if abs(score) < adaptive_threshold():
        return None, "NO EDGE"

    direction = "BUY" if score > 0 else "SELL"

    price = df5["close"].iloc[-1]
    atr = df5["atr"].iloc[-1]

    # dynamic risk
    risk = 0.05 if loss_streak > 0 else 0.1

    tp = price + atr*1.8 if direction == "BUY" else price - atr*1.8
    sl = price - atr if direction == "BUY" else price + atr

    return direction, conf, round(tp,5), round(sl,5), risk

# ===== JOURNAL =====
def log_trade(direction, conf, result):
    df = pd.DataFrame([{
        "time": datetime.now(),
        "direction": direction,
        "confidence": conf,
        "result": result,
        "balance": balance
    }])
    if not os.path.exists(LOG_FILE):
        df.to_csv(LOG_FILE, index=False)
    else:
        df.to_csv(LOG_FILE, mode="a", header=False, index=False)

# ===== TELEGRAM =====
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🚀 V90 FINAL READY")

async def signal_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global last_signal

    retrain_if_needed()

    res = signal()

    if res[0] is None:
        await update.message.reply_text(f"❌ {res[1]}")
        return

    last_signal = res

    keyboard = [[
        InlineKeyboardButton("✅ Плюс", callback_data="win"),
        InlineKeyboardButton("❌ Мінус", callback_data="loss")
    ]]

    d, c, tp, sl, r = res

    await update.message.reply_text(
        f"{d}\nCONF: {c}%\nTP: {tp}\nSL: {sl}\nRISK: {int(r*100)}%",
        reply_markup=InlineKeyboardMarkup(keyboard)
    )

async def result_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global balance, loss_streak

    query = update.callback_query
    await query.answer()

    d, c, tp, sl, r = last_signal

    if query.data == "win":
        results.append(1)
        balance += balance*r
        loss_streak = 0
        adjust_weights(True)
        log_trade(d, c, "win")
    else:
        results.append(0)
        balance -= balance*r
        loss_streak += 1
        adjust_weights(False)
        log_trade(d, c, "loss")

    total = len(results)
    wins = sum(results)
    wr = int((wins/total)*100) if total else 0

    await query.edit_message_text(
        f"💰 Balance: {round(balance,2)}\nTrades: {total}\nWinrate: {wr}%\nLoss streak: {loss_streak}"
    )

# ===== AUTO =====
async def auto(app):
    while True:
        try:
            res = signal()
            if res[0]:
                await app.bot.send_message(chat_id=CHAT_ID, text=f"{res[0]} | {res[1]}%")
        except:
            pass

        await asyncio.sleep(300)

# ===== MAIN =====
async def main():
    app = ApplicationBuilder().token(TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("signal", signal_cmd))
    app.add_handler(CallbackQueryHandler(result_handler))

    print("🚀 BOT STARTED")

    await app.run_polling()
