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
ACCOUNT_ID = os.getenv("OANDA_ACCOUNT_ID")
ENV = os.getenv("OANDA_ENV", "practice")

# 🔒 перевірка
if not TOKEN:
    raise ValueError("❌ TELEGRAM TOKEN не знайдено")

if not OANDA_TOKEN:
    raise ValueError("❌ OANDA TOKEN не знайдено")

PAIR = "EUR_USD"

MODEL_FILE = "model_v25.pkl"
CALIB_FILE = "calib_v25.pkl"
BUFFER_FILE = "buffer_v25.pkl"

client = oandapyV20.API(
    access_token=OANDA_TOKEN,
    environment=ENV
)

# ===== STATE =====
balance = 3000
max_dd = 0.2
daily_loss = 0
loss_streak = 0

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

# ===== SESSION FILTER =====
def session_filter():
    h = datetime.utcnow().hour
    return 7 <= h <= 17

# ===== VOLATILITY FILTER =====
def volatility_filter(df):
    atr = (df["high"] - df["low"]).rolling(14).mean().iloc[-1]
    avg = (df["high"] - df["low"]).mean()
    return atr < avg * 2

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

# ===== MODEL =====
def load_model():
    return joblib.load(MODEL_FILE), joblib.load(CALIB_FILE)

# ===== BUFFER =====
def load_buffer():
    try:
        return joblib.load(BUFFER_FILE)
    except:
        return {"X": [], "y": []}

def save_buffer(buf):
    joblib.dump(buf, BUFFER_FILE)

# ===== ONLINE LEARNING =====
def update_model(features, result):
    buf = load_buffer()

    buf["X"].append(features)
    buf["y"].append(result)

    if len(buf["y"]) >= 20:
        model = RandomForestClassifier(n_estimators=200)
        model.fit(buf["X"], buf["y"])

        probs = model.predict_proba(buf["X"])[:,1]
        calib = LogisticRegression()
        calib.fit(probs.reshape(-1,1), buf["y"])

        joblib.dump(model, MODEL_FILE)
        joblib.dump(calib, CALIB_FILE)

        buf = {"X": [], "y": []}

    save_buffer(buf)

# ===== RISK =====
def get_bet():
    global balance, loss_streak

    bet = balance * 0.1

    if loss_streak >= 2:
        bet *= 0.5

    if daily_loss > balance * max_dd:
        return 0

    return round(bet, 2)

# ===== TP/SL =====
def calculate_tp_sl(df, direction):
    last = df.iloc[-1]

    if direction == "BUY":
        sl = last["low"]
        tp = last["close"] + (last["close"] - sl) * 2
    else:
        sl = last["high"]
        tp = last["close"] - (sl - last["close"]) * 2

    return tp, sl

# ===== SIGNAL =====
def generate_signal():
    if not session_filter():
        return None, "Поза сесією", None, None, None

    feat, df = mtf_features()

    if not volatility_filter(df):
        return None, "Spike / новини", None, None, None

    model, calib = load_model()

    raw = model.predict_proba(feat.reshape(1,-1))[0][1]
    prob = calib.predict_proba([[raw]])[0][1]

    confidence = int(prob * 100)

    if confidence > 65:
        direction = "BUY"
    elif confidence < 35:
        direction = "SELL"
        confidence = 100 - confidence
    else:
        return None, "Немає edge", None, None, None

    tp, sl = calculate_tp_sl(df, direction)
    bet = get_bet()

    return direction, confidence, tp, sl, bet

# ===== TELEGRAM =====
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🚀 V25 FINAL BOSS")

async def signal(update: Update, context: ContextTypes.DEFAULT_TYPE):
    direction, conf, tp, sl, bet = generate_signal()

    if direction is None:
        await update.message.reply_text(f"❌ {conf}")
        return

    context.user_data["last_feat"] = mtf_features()[0]

    await update.message.reply_text(
        f"{'🔼 BUY' if direction=='BUY' else '🔻 SELL'}\n"
        f"💵 {bet}\n"
        f"📊 {conf}%\n"
        f"🎯 TP: {round(tp,5)}\n"
        f"🛑 SL: {round(sl,5)}"
    )

async def win(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global balance, loss_streak

    feat = context.user_data.get("last_feat")

    if feat is not None:
        update_model(feat, 1)

    balance *= 1.08
    loss_streak = 0

    await update.message.reply_text("✅ Win")

async def loss(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global balance, loss_streak, daily_loss

    feat = context.user_data.get("last_feat")

    if feat is not None:
        update_model(feat, 0)

    balance *= 0.9
    loss_streak += 1
    daily_loss += 1

    await update.message.reply_text("❌ Loss")

# ===== AUTO =====
async def auto(app):
    while True:
        direction, conf, tp, sl, bet = generate_signal()

        if direction and conf > 70:
            await app.bot.send_message(
                chat_id=CHAT_ID,
                text=f"{direction} | {conf}% | TP {round(tp,5)} | SL {round(sl,5)}"
            )

        await asyncio.sleep(300)

# ===== MAIN =====
def main():
    app = ApplicationBuilder().token(TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("signal", signal))
    app.add_handler(CommandHandler("win", win))
    app.add_handler(CommandHandler("loss", loss))

    app.create_task(auto(app))

    app.run_polling()

if __name__ == "__main__":
    main()
