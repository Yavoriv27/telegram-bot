print("🔥 FILE STARTED")

import os
import asyncio
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from collections import deque

from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes

import oandapyV20
import oandapyV20.endpoints.instruments as instruments

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# ===== CONFIG =====
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
OANDA_TOKEN = os.getenv("OANDA_API_KEY")

PAIR = "EUR_USD"
MODEL_FILE = "model.pkl"

client = oandapyV20.API(access_token=OANDA_TOKEN)

# ===== USERS =====
users = set()

# ===== STATE =====
last_signal_sent = None
last_signal_time = None

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

# ===== MODEL =====
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

    model = RandomForestClassifier(n_estimators=200)
    model.fit(X, y)

    joblib.dump((model, scaler), MODEL_FILE)

def load_model():
    if not os.path.exists(MODEL_FILE):
        print("TRAIN MODEL...")
        train()
    return joblib.load(MODEL_FILE)

# ===== SIGNAL (твоя логіка збережена) =====
def signal():
    df1 = add_indicators(get_candles("M1"))
    df5 = add_indicators(get_candles("M5"))
    df15 = add_indicators(get_candles("M15"))

    if df5.empty:
        return None

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

    score = (prob - 0.5) * 6

    if abs(score) < 3:
        return None

    direction = "BUY" if score > 0 else "SELL"

    price = df5["close"].iloc[-1]
    atr = df5["atr"].iloc[-1]

    tp = price + atr*1.8 if direction == "BUY" else price - atr*1.8
    sl = price - atr if direction == "BUY" else price + atr

    return direction, conf, round(tp,5), round(sl,5)

# ===== STRONG FILTER =====
def is_strong_signal(res):
    if not res:
        return False

    direction, conf, tp, sl = res

    # 🔥 тільки сильні сигнали
    return conf >= 65

# ===== AUTO SIGNALS =====
async def auto_signals(app):
    global last_signal_sent, last_signal_time

    while True:
        try:
            res = signal()

            if is_strong_signal(res):
                now = datetime.utcnow()

                # ❗ анти-спам (1 сигнал раз в 5 хв)
                if last_signal_time and (now - last_signal_time).seconds < 300:
                    await asyncio.sleep(30)
                    continue

                # ❗ не повторювати той самий сигнал
                if res == last_signal_sent:
                    await asyncio.sleep(30)
                    continue

                last_signal_sent = res
                last_signal_time = now

                d, c, tp, sl = res

                msg = f"🔥 СИЛЬНИЙ СИГНАЛ\n\n{d}\nCONF: {c}%\nTP: {tp}\nSL: {sl}"

                for user in users:
                    try:
                        await app.bot.send_message(chat_id=user, text=msg)
                    except Exception as e:
                        print("SEND ERROR:", e)

        except Exception as e:
            print("AUTO ERROR:", e)

        await asyncio.sleep(60)

# ===== TELEGRAM =====
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_chat.id
    users.add(user_id)

    await update.message.reply_text("✅ Ти підключений. Чекай сильні сигнали 🔥")

async def signal_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    res = signal()

    if not res:
        await update.message.reply_text("❌ Немає сигналу")
        return

    d, c, tp, sl = res

    await update.message.reply_text(
        f"{d}\nCONF: {c}%\nTP: {tp}\nSL: {sl}"
    )

# ===== MAIN =====
async def post_init(app):
    asyncio.create_task(auto_signals(app))

def main():
    app = ApplicationBuilder().token(TOKEN).post_init(post_init).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("signal", signal_cmd))

    print("🚀 BOT STARTED")

    app.run_polling()

if __name__ == "__main__":
    main()
