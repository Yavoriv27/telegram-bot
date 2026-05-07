print("🔥 FILE STARTED")

import os
import asyncio
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import time

from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes

import oandapyV20
import oandapyV20.endpoints.instruments as instruments

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# ===== CONFIG =====
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
OANDA_TOKEN = os.getenv("OANDA_API_KEY")

PAIRS = [
    "EUR_USD",
    "GBP_USD",
    "USD_JPY"
]
MODEL_FILE = "model.pkl"

client = oandapyV20.API(access_token=OANDA_TOKEN, environment="practice")

users = set()
last_signal_sent = None
last_signal_time = None

# ===== DATA =====
def get_candles(pair, tf, count=200):
    for _ in range(2):
        try:
            r = instruments.InstrumentsCandles(
                instrument=pair,
                params={"granularity": tf, "count": count, "price": "M"}
            )
            client.request(r)
            
            if not isinstance(r.response, dict):
                return pd.DataFrame()

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

            if data:
                return pd.DataFrame(data)

        except Exception as e:
            pass

        time.sleep(2)

    return pd.DataFrame()

# ===== INDICATORS =====
def add_indicators(df):
    if df is None or df.empty:
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

# ===== SESSION =====
def session_filter():
    h = datetime.utcnow().hour
    return 6 <= h <= 18

# ===== IMPULSE =====
def strong_impulse_filter(df):
    last = df.iloc[-1]
    body = abs(last["close"] - last["open"])
    atr = df["atr"].iloc[-1]
    return body > atr * 2

# ===== TREND =====
def get_trend(df):
    return "UP" if df["ema20"].iloc[-1] > df["ema50"].iloc[-1] else "DOWN"

# ===== ZONES =====
def get_zones(df):
    recent = df.tail(40)
    high = recent["high"].max()
    low = recent["low"].min()
    zone = (high - low) * 0.4
    return (low, low + zone), (high - zone, high)

def zone_filter(direction, price, sz, rz):
    s_low, s_high = sz
    r_low, r_high = rz

    buffer = (r_high - s_low) * 0.15

    if direction == "BUY":
        return price <= s_high + buffer

    if direction == "SELL":
        return price >= r_low - buffer

    return True

# ===== FAKE BREAKOUT =====
def fake_breakout(df):
    last = df.iloc[-1]
    prev = df.iloc[-2]

    if prev["low"] < df["low"].tail(20).min() and last["close"] > prev["low"]:
        return "BUY"

    if prev["high"] > df["high"].tail(20).max() and last["close"] < prev["high"]:
        return "SELL"

    return None

# ===== MODEL =====
def train():
    df = add_indicators(get_candles(PAIRS[0], "M5", 800))

    X, y = [], []

    for i in range(50, len(df)-3):
        X.append([
            df["close"].iloc[i] - df["close"].iloc[i-3],
            df["ema20"].iloc[i] - df["ema50"].iloc[i],
            df["rsi"].iloc[i],
            df["atr"].iloc[i]
        ])
        y.append(1 if df["close"].iloc[i+2] > df["close"].iloc[i] else 0)

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
def signal(pair):
    # if not session_filter():
#     return None

    df = add_indicators(get_candles(pair,"M5"))
    
    if df.empty or len(df) < 60:
        return None

    if strong_impulse_filter(df):
        return None

    trend = get_trend(df)
    model, scaler = load_model()

    feat = np.array([
        df["close"].iloc[-1] - df["close"].iloc[-3],
        df["ema20"].iloc[-1] - df["ema50"].iloc[-1],
        df["rsi"].iloc[-1],
        df["atr"].iloc[-1]
    ]).reshape(1, -1)

    feat = scaler.transform(feat)
    prob = model.predict_proba(feat)[0][1]
    conf = int(prob * 100)

    score = (prob - 0.5) * 6
    if abs(score) < 1.6:
        return None

    direction = "BUY" if score > 0 else "SELL"

        price = df["close"].iloc[-1]
    atr = df["atr"].iloc[-1]

    # 🔥 НЕ ВХОДИТИ В КІНЦІ ІМПУЛЬСУ
    last = df.iloc[-1]

    body = abs(last["close"] - last["open"])

    if body > atr * 0.7:
        return None

    # TREND FILTER
    if direction == "BUY" and trend == "DOWN" and conf < 70:
        return None
    if direction == "SELL" and trend == "UP" and conf < 70:
        return None

    # ZONES
    sz, rz = get_zones(df)
    if not zone_filter(direction, price, sz, rz):
        return None

    # FAKE BREAKOUT
    fb = fake_breakout(df)
    if fb and fb == direction:
        conf += 5

    tp = price + atr*1.8 if direction == "BUY" else price - atr*1.8
    sl = price - atr if direction == "BUY" else price + atr

    return direction, conf, round(tp,5), round(sl,5)

# ===== FILTER =====
def is_strong(res):
    return res and res[1] >= 60

# ===== AUTO =====
async def auto(app):
    global last_signal_sent, last_signal_time

    while True:
        try:
            best = None

            for pair in PAIRS:
                res = signal(pair)

                if res:
                    if not best or res[1] > best[1]:
                        best = (*res, pair)

            res = best

            if is_strong(res):
                now = datetime.utcnow()

                if last_signal_time and (now - last_signal_time).seconds < 180:
                    await asyncio.sleep(20)
                    continue

                if res == last_signal_sent:
                    await asyncio.sleep(20)
                    continue

                last_signal_sent = res
                last_signal_time = now

                d, c, tp, sl, pair = res
                msg = f"🔥 SIGNAL {pair}\n\n{d}\nCONF: {c}%\nTP: {tp}\nSL: {sl}"

                for u in users:
                    await app.bot.send_message(chat_id=u, text=msg)

        except Exception as e:
            print("AUTO ERROR:", e)

        await asyncio.sleep(130)

# ===== TELEGRAM =====
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    users.add(update.effective_chat.id)
    await update.message.reply_text("✅ BOT WORKING")

async def signal_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    df = get_candles(PAIRS[0], "M5")
    if df.empty:
        await update.message.reply_text("❌ No data")
        return

    price = df["close"].iloc[-1]
    res = signal(PAIRS[0])

    if not res:
        await update.message.reply_text(f"📊 {round(price,5)}\n❌ No signal")
        return

    d, c, tp, sl = res

    await update.message.reply_text(
        f"📊 {round(price,5)}\n\n{d}\nCONF: {c}%\nTP: {tp}\nSL: {sl}"
    )

# ===== MAIN =====
async def post_init(app):
    asyncio.create_task(auto(app))

def main():
    app = ApplicationBuilder().token(TOKEN).post_init(post_init).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("signal", signal_cmd))

    print("🚀 BOT STARTED")
    app.run_polling()

if __name__ == "__main__":
    main()
