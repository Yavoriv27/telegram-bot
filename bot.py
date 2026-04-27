import os
import asyncio
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

import oandapyV20
import oandapyV20.endpoints.instruments as instruments

from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# ===== ENV =====
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
OANDA_TOKEN = os.getenv("OANDA_API_KEY")
ENV = os.getenv("OANDA_ENV", "practice")

chat_ids_env = os.getenv("CHAT_IDS", "")
CHAT_IDS = [int(x.strip()) for x in chat_ids_env.split(",") if x.strip().isdigit()]

PAIR = "EUR_USD"

MODEL_FILE = "model_v29.pkl"
CALIB_FILE = "calib_v29.pkl"

client = oandapyV20.API(access_token=OANDA_TOKEN, environment=ENV)

# ===== STATE =====
balance = 3000
peak_balance = 3000
daily_loss = 0
loss_streak = 0
stats = {"win": 0, "loss": 0}

# ===== DATA =====
def get_candles(tf="M5", count=300):
    params = {"granularity": tf, "count": count, "price": "M"}
    r = instruments.InstrumentsCandles(instrument=PAIR, params=params)
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

# ===== INDICATORS =====
def add_indicators(df):
    df["ema20"] = df["close"].ewm(span=20).mean()
    df["ema50"] = df["close"].ewm(span=50).mean()

    delta = df["close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df["rsi"] = 100 - (100 / (1 + rs))

    return df

# ===== TIME =====
def best_time():
    h = datetime.utcnow().hour
    return (8 <= h <= 11) or (13 <= h <= 16)

# ===== VOL =====
def volatility_filter(df):
    atr = (df["high"] - df["low"]).rolling(14).mean().iloc[-1]
    return 0.0003 < atr < 0.003

# ===== TREND =====
def trend(df):
    if df["ema20"].iloc[-1] > df["ema50"].iloc[-1]:
        return "UP"
    elif df["ema20"].iloc[-1] < df["ema50"].iloc[-1]:
        return "DOWN"
    return "RANGE"

# ===== PRICE ACTION =====
def engulfing(df):
    c1 = df.iloc[-2]
    c2 = df.iloc[-1]

    if c2["close"] > c2["open"] and c1["close"] < c1["open"]:
        if c2["close"] > c1["open"]:
            return "BUY"

    if c2["close"] < c2["open"] and c1["close"] > c1["open"]:
        if c2["close"] < c1["open"]:
            return "SELL"

    return None

# ===== FEATURES =====
def features(df):
    t = 1 if trend(df) == "UP" else -1

    momentum = df["close"].iloc[-1] - df["close"].iloc[-5]
    vol = (df["high"] - df["low"]).rolling(14).mean().iloc[-1]
    ema_dist = df["close"].iloc[-1] - df["ema20"].iloc[-1]
    rsi = df["rsi"].iloc[-1]

    return np.array([t, momentum, vol, ema_dist, rsi])

# ===== TRAIN =====
def train_model():
    df = get_candles("M5", 1500)
    df = add_indicators(df)

    X, y = [], []

    for i in range(50, len(df)-5):
        sub = df.iloc[:i]
        X.append(features(sub))

        future = df["close"].iloc[i+4]
        current = df["close"].iloc[i]

        y.append(1 if future > current else 0)

    model = RandomForestClassifier(n_estimators=300)
    model.fit(X, y)

    probs = model.predict_proba(X)[:,1]
    calib = LogisticRegression()
    calib.fit(probs.reshape(-1,1), y)

    joblib.dump(model, MODEL_FILE)
    joblib.dump(calib, CALIB_FILE)

# ===== LOAD =====
def load_model():
    if not os.path.exists(MODEL_FILE):
        train_model()
    return joblib.load(MODEL_FILE), joblib.load(CALIB_FILE)

# ===== RR =====
def rr(entry, tp, sl):
    risk = abs(entry - sl)
    reward = abs(tp - entry)
    return 0 if risk == 0 else round(reward / risk, 2)

# ===== SIGNAL =====
def generate_signal():
    if not best_time():
        return None, "WAIT TIME", None, None, None, None, None, None

    df = get_candles("M5", 100)
    df = add_indicators(df)

    if not volatility_filter(df):
        return None, "BAD VOL", None, None, None, None, None, None

    pa = engulfing(df)
    if not pa:
        return None, "NO PA", None, None, None, None, None, None

    # RSI FILTER
    if pa == "BUY" and df["rsi"].iloc[-1] < 50:
        return None, "RSI WEAK", None, None, None, None, None, None

    if pa == "SELL" and df["rsi"].iloc[-1] > 50:
        return None, "RSI WEAK", None, None, None, None, None, None

    model, calib = load_model()
    f = features(df).reshape(1,-1)

    raw = model.predict_proba(f)[0][1]
    prob = calib.predict_proba([[raw]])[0][1]

    conf = int(prob * 100)

    price = df.iloc[-1]["close"]

    sl = df["low"].rolling(20).min().iloc[-1] if pa == "BUY" else df["high"].rolling(20).max().iloc[-1]

    if pa == "BUY":
        tp = price + (price - sl) * 2
    else:
        tp = price - (sl - price) * 2

    r = rr(price, tp, sl)

    if conf >= 80 and r >= 2:
        decision = "ENTER"
        strength = "🔥 STRONG"
    elif conf >= 70 and r >= 1.5:
        decision = "WAIT"
        strength = "⚖️ NORMAL"
    else:
        decision = "SKIP"
        strength = "⚠️ WEAK"

    return pa, conf, tp, sl, r, strength, decision, price

# ===== TELEGRAM =====
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🚀 V29 PRO ML")

async def signal(update: Update, context: ContextTypes.DEFAULT_TYPE):
    d, c, tp, sl, r, s, dec, p = generate_signal()

    if d is None:
        await update.message.reply_text(f"❌ {c}")
        return

    await update.message.reply_text(
        f"{'🔼 BUY' if d=='BUY' else '🔻 SELL'}\n"
        f"📊 {c}%\n"
        f"📍 {round(p,5)}\n"
        f"🎯 {round(tp,5)}\n"
        f"🛑 {round(sl,5)}\n"
        f"RR 1:{r}\n"
        f"{s}\n"
        f"🧠 {dec}"
    )

# ===== AUTO =====
async def auto(app):
    while True:
        try:
            d, c, tp, sl, r, s, dec, p = generate_signal()

            if d and dec == "ENTER" and CHAT_IDS:
                for chat_id in CHAT_IDS:
                    await app.bot.send_message(
                        chat_id=chat_id,
                        text=f"{d} | {c}% | RR 1:{r} | {s}"
                    )
        except Exception as e:
            print("ERR:", e)

        await asyncio.sleep(240)

# ===== MAIN =====
async def post_init(app):
    asyncio.create_task(auto(app))

def main():
    app = ApplicationBuilder().token(TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("signal", signal))

    app.post_init = post_init
    app.run_polling()

if __name__ == "__main__":
    main()
