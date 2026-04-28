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

CHAT_IDS = [int(x) for x in os.getenv("CHAT_IDS", "").split(",") if x.strip().isdigit()]

PAIR = "EUR_USD"
MODEL_FILE = "model_v31.pkl"
CALIB_FILE = "calib_v31.pkl"
BUFFER_FILE = "buffer.pkl"

client = oandapyV20.API(access_token=OANDA_TOKEN, environment=ENV)

# ===== DATA =====
def get_candles(tf="M5", count=200):
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

# ===== INDICATORS =====
def add_indicators(df):
    df["ema20"] = df["close"].ewm(span=20).mean()
    df["ema50"] = df["close"].ewm(span=50).mean()

    delta = df["close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss
    df["rsi"] = 100 - (100 / (1 + rs))

    # ADX
    df["tr"] = df["high"] - df["low"]
    df["dm_plus"] = np.maximum(df["high"].diff(), 0)
    df["dm_minus"] = np.maximum(df["low"].diff(), 0)

    tr14 = df["tr"].rolling(14).mean()
    dm_plus14 = df["dm_plus"].rolling(14).mean()
    dm_minus14 = df["dm_minus"].rolling(14).mean()

    di_plus = 100 * (dm_plus14 / tr14)
    di_minus = 100 * (dm_minus14 / tr14)

    df["adx"] = (abs(di_plus - di_minus) / (di_plus + di_minus)) * 100

    return df

# ===== PATTERNS =====
def engulfing(df):
    c1, c2 = df.iloc[-2], df.iloc[-1]
    if c2["close"] > c2["open"] and c1["close"] < c1["open"] and c2["close"] > c1["open"]:
        return "BUY"
    if c2["close"] < c2["open"] and c1["close"] > c1["open"] and c2["close"] < c1["open"]:
        return "SELL"
    return None

def pin_bar(df):
    c = df.iloc[-1]
    body = abs(c["close"] - c["open"])
    wick_up = c["high"] - max(c["close"], c["open"])
    wick_down = min(c["close"], c["open"]) - c["low"]

    if wick_down > body * 2:
        return "BUY"
    if wick_up > body * 2:
        return "SELL"
    return None

def breakout(df):
    high = df["high"].rolling(20).max().iloc[-2]
    low = df["low"].rolling(20).min().iloc[-2]
    last = df.iloc[-1]

    if last["close"] > high:
        return "BUY"
    if last["close"] < low:
        return "SELL"
    return None

def fake_breakout(df):
    high = df["high"].rolling(20).max().iloc[-2]
    low = df["low"].rolling(20).min().iloc[-2]
    last = df.iloc[-1]

    if last["high"] > high and last["close"] < high:
        return "SELL"
    if last["low"] < low and last["close"] > low:
        return "BUY"
    return None

# ===== MTF =====
def mtf_data():
    df1 = add_indicators(get_candles("M1", 120))
    df5 = add_indicators(get_candles("M5", 120))
    df15 = add_indicators(get_candles("M15", 120))
    return df1, df5, df15

# ===== FEATURES =====
def features(df1, df5, df15):
    def f(df):
        return [
            df["close"].iloc[-1] - df["close"].iloc[-5],
            df["ema20"].iloc[-1] - df["ema50"].iloc[-1],
            df["rsi"].iloc[-1],
            df["adx"].iloc[-1]
        ]
    return np.array(f(df1) + f(df5) + f(df15))

# ===== SCORE =====
def indicator_score(df):
    score = 0

    score += 3 if df["ema20"].iloc[-1] > df["ema50"].iloc[-1] else -3

    if df["rsi"].iloc[-1] > 60:
        score += 2
    elif df["rsi"].iloc[-1] < 40:
        score -= 2

    if df["adx"].iloc[-1] > 25:
        score += 2

    if df["close"].iloc[-1] > df["close"].iloc[-3]:
        score += 1
    else:
        score -= 1

    return score

# ===== TRAIN =====
def train_model():
    df = add_indicators(get_candles("M5", 1200))

    X, y = [], []
    for i in range(50, len(df)-5):
        sub = df.iloc[:i]
        X.append(features(sub, sub, sub))

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

def load_model():
    if not os.path.exists(MODEL_FILE):
        train_model()
    return joblib.load(MODEL_FILE), joblib.load(CALIB_FILE)

# ===== RR =====
def rr(entry, tp, sl):
    return round(abs(tp-entry)/abs(entry-sl),2) if entry!=sl else 0

# ===== SIGNAL =====
def generate_signal():
    df1, df5, df15 = mtf_data()

    # --- PATTERN ---
    pa = engulfing(df5) or pin_bar(df5) or breakout(df5) or fake_breakout(df5)

    if not pa:
        return None, "NO SETUP", None, None, None, None, None, None

    # --- TREND FILTER ---
    trend_up = df5["ema20"].iloc[-1] > df5["ema50"].iloc[-1]

    if pa == "BUY" and not trend_up:
        return None, "AGAINST TREND", None, None, None, None, None, None

    if pa == "SELL" and trend_up:
        return None, "AGAINST TREND", None, None, None, None, None, None

    # --- SCORE ---
    if indicator_score(df5) < 0:
        return None, "WEAK SCORE", None, None, None, None, None, None

    # --- ML ---
    model, calib = load_model()
    f = features(df1, df5, df15).reshape(1, -1)

    raw = model.predict_proba(f)[0][1]
    prob = calib.predict_proba([[raw]])[0][1]
    conf = int(prob * 100)

    # --- PRICE ---
    price = df5.iloc[-1]["close"]

    sl = df5["low"].rolling(20).min().iloc[-1] if pa == "BUY" else df5["high"].rolling(20).max().iloc[-1]

    if pa == "BUY":
        tp = price + (price - sl) * 2
    else:
        tp = price - (sl - price) * 2

    r = rr(price, tp, sl)

    # --- DECISION ---
    if conf >= 70 and r >= 1.5:
        return pa, conf, tp, sl, r, "🔥 STRONG", "ENTER", price

    elif conf >= 60 and r >= 1.3:
        return pa, conf, tp, sl, r, "⚖️ NORMAL", "WAIT", price

    else:
        return pa, conf, tp, sl, r, "⚠️ WEAK", "SKIP", price
# ===== TELEGRAM =====
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🚀 V31 READY")

async def signal(update: Update, context: ContextTypes.DEFAULT_TYPE):
    res = generate_signal()
    if res[0] is None:
        await update.message.reply_text(f"❌ {res[1]}")
        return

    d,c,tp,sl,r,s,dec,p = res
    await update.message.reply_text(
        f"{d}\n{c}%\nTP {round(tp,5)}\nSL {round(sl,5)}\nRR 1:{r}\n{s}\n{dec}"
    )

# ===== AUTO =====
async def auto(app):
    while True:
        try:
            res = generate_signal()
            if res[0] and res[6]=="ENTER":
                for chat_id in CHAT_IDS:
                    await app.bot.send_message(chat_id=chat_id, text=f"{res[0]} | {res[1]}% | RR {res[4]}")
        except Exception as e:
            print("ERR:", e)

        await asyncio.sleep(240)

# ===== MAIN =====
async def run_bot():
    while True:
        try:
            app = ApplicationBuilder().token(TOKEN).build()

            app.add_handler(CommandHandler("start", start))
            app.add_handler(CommandHandler("signal", signal))

            await app.initialize()
            await app.start()

            asyncio.create_task(auto(app))

            print("✅ BOT STARTED")

            await app.updater.start_polling()
            await asyncio.Event().wait()

        except Exception as e:
            print("❌ BOT CRASH:", e)
            await asyncio.sleep(5)  # пауза і рестарт


if __name__ == "__main__":
    asyncio.run(run_bot())
