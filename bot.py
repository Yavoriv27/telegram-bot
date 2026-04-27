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
CHAT_ID = os.getenv("CHAT_ID")
OANDA_TOKEN = os.getenv("OANDA_API_KEY")
ENV = os.getenv("OANDA_ENV", "practice")

if not TOKEN:
    raise ValueError("❌ TELEGRAM TOKEN не знайдено")
if not OANDA_TOKEN:
    raise ValueError("❌ OANDA TOKEN не знайдено")

PAIR = "EUR_USD"

MODEL_FILE = "model_v2.pkl"
CALIB_FILE = "calib_v2.pkl"

client = oandapyV20.API(access_token=OANDA_TOKEN, environment=ENV)

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
                "volume": c["volume"]
            })

    return pd.DataFrame(data)

# ===== FEATURES =====
def compute_features(df):
    last = df.iloc[-1]

    high = df["high"].rolling(20).max().iloc[-1]
    low = df["low"].rolling(20).min().iloc[-1]

    liquidity_dist = min(abs(last["close"] - high), abs(last["close"] - low))
    trend = np.sign(df["close"].iloc[-1] - df["close"].iloc[-10])
    vol = (df["high"] - df["low"]).rolling(14).mean().iloc[-1]
    momentum = last["close"] - df["close"].iloc[-5]

    return np.array([liquidity_dist, trend, vol, momentum])

# ===== MTF =====
def mtf_features():
    df1 = get_candles("M1")
    df5 = get_candles("M5")
    df15 = get_candles("M15")

    f1 = compute_features(df1)
    f5 = compute_features(df5)
    f15 = compute_features(df15)

    return np.concatenate([f1, f5, f15]), df5

# ===== TRAIN =====
def train_model():
    print("⚠️ Навчання моделі...")

    df = get_candles("M5", 1500)

    X, y = [], []

    for i in range(50, len(df)-5):
        sub = df.iloc[:i]

        # 🔥 MTF як у сигналу
        f1 = compute_features(sub)
        f5 = compute_features(sub)
        f15 = compute_features(sub)

        feat = np.concatenate([f1, f5, f15])

        future = df["close"].iloc[i+3]
        current = df["close"].iloc[i]

        label = 1 if future > current else 0

        X.append(feat)
        y.append(label)

    model = RandomForestClassifier(n_estimators=200)
    model.fit(X, y)

    probs = model.predict_proba(X)[:,1]
    calib = LogisticRegression()
    calib.fit(probs.reshape(-1,1), y)

    joblib.dump(model, MODEL_FILE)
    joblib.dump(calib, CALIB_FILE)

    print("✅ Модель готова")

# ===== LOAD =====
def load_model():
    if not os.path.exists(MODEL_FILE):
        train_model()

    return joblib.load(MODEL_FILE), joblib.load(CALIB_FILE)

# ===== SIGNAL =====
def generate_signal():
    feat, df = mtf_features()

    model, calib = load_model()

    raw = model.predict_proba(feat.reshape(1,-1))[0][1]
    prob = calib.predict_proba([[raw]])[0][1]

    conf = int(prob * 100)

    if conf > 65:
        direction = "BUY"
    elif conf < 35:
        direction = "SELL"
        conf = 100 - conf
    else:
        return None, "Немає edge", None, None

    last = df.iloc[-1]

    if direction == "BUY":
        sl = last["low"]
        tp = last["close"] + (last["close"] - sl) * 2
    else:
        sl = last["high"]
        tp = last["close"] - (sl - last["close"]) * 2

    return direction, conf, tp, sl

# ===== TELEGRAM =====
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🚀 BOT READY")

async def signal(update: Update, context: ContextTypes.DEFAULT_TYPE):
    direction, conf, tp, sl = generate_signal()

    if direction is None:
        await update.message.reply_text(f"❌ {conf}")
        return

    await update.message.reply_text(
        f"{'🔼 BUY' if direction=='BUY' else '🔻 SELL'}\n"
        f"📊 {conf}%\n"
        f"🎯 TP {round(tp,5)}\n"
        f"🛑 SL {round(sl,5)}"
    )

# ===== AUTO =====
async def auto(app):
    while True:
        direction, conf, tp, sl = generate_signal()

        if direction and conf > 70 and CHAT_ID:
            await app.bot.send_message(
                chat_id=CHAT_ID,
                text=f"{direction} | {conf}%"
            )

        await asyncio.sleep(300)

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
