import os
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime

from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes

import oandapyV20
import oandapyV20.endpoints.instruments as instruments

# ===== CONFIG =====
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
OANDA_TOKEN = os.getenv("OANDA_API_KEY")

PAIR = "EUR_USD"

# 👉 ДОДАЙ СЮДИ CHAT ID
USERS = [
    123456789,  # ти
    987654321   # брат
]

client = oandapyV20.API(access_token=OANDA_TOKEN, environment="practice")

last_signal = None
last_time = None

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

    df["adx"] = abs(df["ema20"] - df["ema50"]) / df["atr"]

    return df

# ===== PRICE ACTION =====
def price_action(df):
    last3 = df.tail(3)

    bull = all(last3["close"] > last3["open"])
    bear = all(last3["close"] < last3["open"])

    body = abs(df["close"].iloc[-1] - df["open"].iloc[-1])
    atr = df["atr"].iloc[-1]

    strong = body > atr * 0.8

    if bull and strong:
        return "BUY"
    if bear and strong:
        return "SELL"

    return None

# ===== LEVELS =====
def get_levels(df):
    recent = df.tail(30)
    return recent["low"].min(), recent["high"].max()

# ===== TREND =====
def get_trend(df):
    return "UP" if df["ema20"].iloc[-1] > df["ema50"].iloc[-1] else "DOWN"

# ===== FILTER =====
def market_filter(df):
    return df["adx"].iloc[-1] > 0.3

# ===== AI FILTER (простий) =====
def ai_filter(score, rsi, trend):
    # імітація "розуму"
    if abs(score) < 4:
        return False

    if trend == "UP" and rsi > 75:
        return False

    if trend == "DOWN" and rsi < 25:
        return False

    return True

# ===== SIGNAL =====
def generate_signal():
    df1 = add_indicators(get_candles("M1"))
    df5 = add_indicators(get_candles("M5"))
    df15 = add_indicators(get_candles("M15"))

    if df1.empty or df5.empty or df15.empty:
        return None

    if not market_filter(df5):
        return None

    trend = get_trend(df15)
    pa = price_action(df5)

    support, resistance = get_levels(df5)

    price = df5["close"].iloc[-1]
    atr = df5["atr"].iloc[-1]

    score = 0

    # EMA
    score += 2 if df5["ema20"].iloc[-1] > df5["ema50"].iloc[-1] else -2

    # RSI
    if df5["rsi"].iloc[-1] < 30:
        score += 1
    if df5["rsi"].iloc[-1] > 70:
        score -= 1

    # TREND
    score += 2 if trend == "UP" else -2

    # PA
    if pa == "BUY":
        score += 2
    if pa == "SELL":
        score -= 2

    # AI FILTER
    if not ai_filter(score, df5["rsi"].iloc[-1], trend):
        return None

    if score >= 4:
        direction = "BUY"
    elif score <= -4:
        direction = "SELL"
    else:
        return None

    # LEVEL CHECK
    if direction == "BUY" and price > support + atr:
        return None
    if direction == "SELL" and price < resistance - atr:
        return None

    confidence = min(90, abs(score) * 10)

    tp = price + atr * 1.5 if direction == "BUY" else price - atr * 1.5
    sl = price - atr if direction == "BUY" else price + atr

    return direction, confidence, round(price,5), round(tp,5), round(sl,5)

# ===== AUTO =====
async def auto(app):
    global last_signal, last_time

    while True:
        try:
            res = generate_signal()

            if res:
                now = datetime.utcnow()

                if last_time and (now - last_time).seconds < 120:
                    await asyncio.sleep(20)
                    continue

                if res == last_signal:
                    await asyncio.sleep(20)
                    continue

                last_signal = res
                last_time = now

                d, c, price, tp, sl = res

                msg = f"""🔥 СИГНАЛ

{d}
📊 {price}
📈 CONF: {c}%
🎯 TP: {tp}
🛑 SL: {sl}
⏱ 2 хв"""

                for u in USERS:
                    await app.bot.send_message(chat_id=u, text=msg)

        except Exception as e:
            print("ERROR:", e)

        await asyncio.sleep(60)

# ===== TELEGRAM =====
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("✅ Бот працює")

async def signal_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    res = generate_signal()

    if not res:
        await update.message.reply_text("❌ Немає сигналу")
        return

    d, c, price, tp, sl = res

    await update.message.reply_text(
        f"{d}\n📊 {price}\nCONF: {c}%\nTP: {tp}\nSL: {sl}"
    )

# ===== MAIN =====
async def post_init(app):
    asyncio.create_task(auto(app))

def main():
    app = ApplicationBuilder().token(TOKEN).post_init(post_init).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("signal", signal_cmd))

    print("🚀 V14 BOT STARTED")

    app.run_polling()

if __name__ == "__main__":
    main()
