import os
import asyncio
import schedule
import time
from datetime import datetime
from dotenv import load_dotenv

import pandas as pd
from ta.trend import EMAIndicator, MACD, ADXIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange

import oandapyV20
from oandapyV20.endpoints import instruments

from telegram import Bot

load_dotenv()

# ================= CONFIG =================
PAIRS = ["EUR_USD", "GBP_USD", "USD_JPY"]

MIN_M1_CONFLUENCE = 65
MIN_M5_CONFIRM = 55

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

OANDA_API_KEY = os.getenv("OANDA_API_KEY")
ACCOUNT_ID = os.getenv("OANDA_ACCOUNT_ID")

bot = Bot(token=TOKEN)
client = oandapyV20.API(access_token=OANDA_API_KEY)


# ================= DATA =================
def get_candles(pair, granularity, count=150):
    params = {"granularity": granularity, "count": count, "price": "M"}
    r = instruments.InstrumentsCandles(instrument=pair, params=params)
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


# ================= ANALYSIS =================
def analyze(df):
    close = df["close"]
    high = df["high"]
    low = df["low"]

    df["ema8"] = EMAIndicator(close, 8).ema_indicator()
    df["ema21"] = EMAIndicator(close, 21).ema_indicator()
    df["ema50"] = EMAIndicator(close, 50).ema_indicator()

    macd = MACD(close)
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()

    df["rsi"] = RSIIndicator(close).rsi()

    stoch = StochasticOscillator(high, low, close)
    df["stoch_k"] = stoch.stoch()
    df["stoch_d"] = stoch.stoch_signal()

    bb = BollingerBands(close)
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_lower"] = bb.bollinger_lband()

    adx = ADXIndicator(high, low, close)
    df["adx"] = adx.adx()

    df["atr"] = AverageTrueRange(high, low, close).average_true_range()

    last = df.iloc[-1]

    buy = 0
    sell = 0

    # EMA
    if last["ema8"] > last["ema21"] > last["ema50"]:
        buy += 1
    elif last["ema8"] < last["ema21"] < last["ema50"]:
        sell += 1

    # MACD
    if last["macd"] > last["macd_signal"]:
        buy += 1
    else:
        sell += 1

    # RSI
    if last["rsi"] < 30:
        buy += 1
    elif last["rsi"] > 70:
        sell += 1

    # Stochastic
    if last["stoch_k"] < 20:
        buy += 1
    elif last["stoch_k"] > 80:
        sell += 1

    # Bollinger
    if last["close"] <= last["bb_lower"]:
        buy += 1
    elif last["close"] >= last["bb_upper"]:
        sell += 1

    # ADX тренд
    if last["adx"] > 25:
        if buy > sell:
            buy += 1
        else:
            sell += 1

    total = buy + sell
    if total == 0:
        return None

    buy_score = (buy / total) * 100
    sell_score = (sell / total) * 100

    if buy_score > sell_score:
        return "BUY", buy_score
    elif sell_score > buy_score:
        return "SELL", sell_score

    return None


# ================= SIGNAL =================
def generate_signal(pair):
    m1 = get_candles(pair, "M1")
    m5 = get_candles(pair, "M5")

    if m1.empty or m5.empty:
        return None

    m1_signal = analyze(m1)
    m5_signal = analyze(m5)

    if not m1_signal:
        return None

    direction, score = m1_signal

    if score < MIN_M1_CONFLUENCE:
        return None

    # фільтр M5
    if m5_signal:
        m5_dir, m5_score = m5_signal
        if m5_dir != direction and m5_score > MIN_M5_CONFIRM:
            return None

    return {
        "pair": pair,
        "direction": direction,
        "score": score
    }


# ================= SEND =================
async def send_signal(signal):
    text = f"""
🚀 SIGNAL

📊 {signal['pair'].replace('_','/')}
{'🟢 BUY' if signal['direction']=='BUY' else '🔴 SELL'}

⏱ Експірація: 2 хв
📈 Ймовірність: {signal['score']:.0f}%
🕒 {datetime.utcnow().strftime('%H:%M:%S')}
"""
    await bot.send_message(chat_id=CHAT_ID, text=text)


# ================= MAIN =================
async def run():
    print("Checking market...")

    best = None

    for pair in PAIRS:
        signal = generate_signal(pair)
        if signal:
            if not best or signal["score"] > best["score"]:
                best = signal

    if best:
        print("SIGNAL:", best)
        await send_signal(best)
    else:
        print("No signal")


def job():
    asyncio.run(run())


# ================= START =================
print("🚀 BOT STARTED (2-min strategy)")

schedule.every(2).minutes.do(job)

while True:
    schedule.run_pending()
    time.sleep(1)
