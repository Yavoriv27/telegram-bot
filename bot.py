print("🔥 FILE STARTED")

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

# ===== CONFIG =====
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
OANDA_TOKEN = os.getenv("OANDA_API_KEY")

PAIR = "EUR_USD"
MODEL_FILE = "model.pkl"

client = oandapyV20.API(
    access_token=OANDA_TOKEN,
    environment="practice"
)

# ===== USERS =====
users = set()

# ===== STATE =====
last_signal_sent = None
last_signal_time = None

# ===== DATA =====
def get_candles(tf, count=200):
    for attempt in range(3):
        try:
            r = instruments.InstrumentsCandles(
                instrument=PAIR,
                params={"granularity": tf, "count": count, "price": "M"}
            )
            client.request(r)

            if not isinstance(r.response, dict):
                continue

            if "candles" not in r.response:
                continue

            data = []
            for c in r.response["candles"]:
                if c["complete"]:
                    data.append({
                        "open": float(c["mid"]["o"]),
                        "high": float(c["mid"]["h"]),
                        "low": float(c["mid"]["l"]),
                        "close": float(c["mid"]["c"]),
                    })

            if len(data) == 0:
                continue

            return pd.DataFrame(data)

        except Exception as e:
            print("DATA ERROR:", e)

        import time
        time.sleep(2)

    return pd.DataFrame()

# ===== INDICATORS =====
def add_indicators(df):
    if df is None or df.empty or "close" not in df:
        return pd.DataFrame()

    df["ema20"] = df["close"].ewm(span=20).mean()
    df["ema50"] = df["close"].ewm(span=50).mean()

    delta = df["close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss
    df["rsi"] = 100 - (100 / (1 + rs))

    df["atr"] = (df["high"] - df["low"]).rolling(14).mean()
    return df

# ===== АНТИ-ІМПУЛЬС =====
def strong_impulse_filter(df):
    try:
        last = df.iloc[-1]
        body = abs(last["close"] - last["open"])
        atr = df["atr"].iloc[-1]

        if body > atr * 2:
            return True
        return False
    except:
        return False

# ===== ТРЕНД =====
def get_trend(df):
    return "UP" if df["ema20"].iloc[-1] > df["ema50"].iloc[-1] else "DOWN"

# ===== РІВНІ =====
def get_levels(df):
    try:
        recent = df.tail(30)
        resistance = recent["high"].max()
        support = recent["low"].min()
        return support, resistance
    except:
        return None, None

def level_filter(direction, price, support, resistance, atr):
    if support is None or resistance is None:
        return True

    buffer = atr * 0.3

    if direction == "BUY":
        if price > support + buffer:
            return False

    if direction == "SELL":
        if price < resistance - buffer:
            return False

    return True

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
        train()
    return joblib.load(MODEL_FILE)

# ===== SIGNAL =====
def signal():
    df1 = add_indicators(get_candles("M1"))
    df5 = add_indicators(get_candles("M5"))
    df15 = add_indicators(get_candles("M15"))

    if df1.empty or df5.empty or df15.empty:
        return None

    if strong_impulse_filter(df5):
        return None

    trend = get_trend(df15)

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

    if abs(score) < 1.5:
        return None

    direction = "BUY" if score > 0 else "SELL"

    if direction == "BUY" and trend == "DOWN" and conf < 80:
        return None
    if direction == "SELL" and trend == "UP" and conf < 80:
        return None

    price = df5["close"].iloc[-1]
    atr = df5["atr"].iloc[-1]

    # 🔥 РІВНІ
    support, resistance = get_levels(df5)

    if not level_filter(direction, price, support, resistance, atr):
        return None

    tp = price + atr*1.8 if direction == "BUY" else price - atr*1.8
    sl = price - atr if direction == "BUY" else price + atr

    return direction, conf, round(tp,5), round(sl,5)

# ===== FILTER =====
def is_strong_signal(res):
    return res and res[1] >= 55

# ===== AUTO =====
async def auto_signals(app):
    global last_signal_sent, last_signal_time

    while True:
        try:
            res = signal()

            if is_strong_signal(res):
                now = datetime.utcnow()

                if last_signal_time and (now - last_signal_time).seconds < 180:
                    await asyncio.sleep(20)
                    continue

                if res == last_signal_sent:
                    await asyncio.sleep(20)
                    continue

                last_signal_sent = res
                last_signal_time = now

                d, c, tp, sl = res

                msg = f"🔥 СИГНАЛ\n\n{d}\nCONF: {c}%\nTP: {tp}\nSL: {sl}"

                for user in users:
                    await app.bot.send_message(chat_id=user, text=msg)

        except Exception as e:
            print("AUTO ERROR:", e)

        await asyncio.sleep(90)

# ===== TELEGRAM =====
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    users.add(update.effective_chat.id)
    await update.message.reply_text("✅ Бот активний")

async def signal_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    df5 = get_candles("M5")

    if df5 is None or df5.empty:
        await update.message.reply_text("❌ Немає даних")
        return

    price = df5["close"].iloc[-1]
    res = signal()

    if not res:
        await update.message.reply_text(f"📊 {round(price,5)}\n❌ Немає сигналу")
        return

    d, c, tp, sl = res

    await update.message.reply_text(
        f"📊 {round(price,5)}\n\n{d}\nCONF: {c}%\nTP: {tp}\nSL: {sl}"
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
