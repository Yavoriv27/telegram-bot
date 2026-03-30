import os
import asyncio
from datetime import datetime
from dotenv import load_dotenv

import pandas as pd
import yfinance as yf

from ta.trend import EMAIndicator, MACD, ADXIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands

from telegram import Update, InlineKeyboardMarkup, InlineKeyboardButton
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes

load_dotenv()

PAIRS = ["EURUSD=X", "GBPUSD=X", "USDJPY=X"]

MIN_M1_CONFLUENCE = 65
MIN_M5_CONFIRM = 55

BOT_TOKEN = os.getenv("BOT_TOKEN")

# ================= DATA =================
def get_data(pair, interval):
    try:
        df = yf.download(pair, period="1d", interval=interval, progress=False)
        df.dropna(inplace=True)
        return df
    except:
        return pd.DataFrame()

# ================= ANALYSIS =================
def analyze(df):
    if len(df) < 50:
        return None

    close = df["Close"]
    high = df["High"]
    low = df["Low"]

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

    last = df.iloc[-1]

    buy = 0
    sell = 0

    if last["ema8"] > last["ema21"] > last["ema50"]:
        buy += 1
    elif last["ema8"] < last["ema21"] < last["ema50"]:
        sell += 1

    if last["macd"] > last["macd_signal"]:
        buy += 1
    else:
        sell += 1

    if last["rsi"] < 30:
        buy += 1
    elif last["rsi"] > 70:
        sell += 1

    if last["stoch_k"] < 20:
        buy += 1
    elif last["stoch_k"] > 80:
        sell += 1

    if last["Close"] <= last["bb_lower"]:
        buy += 1
    elif last["Close"] >= last["bb_upper"]:
        sell += 1

    if last["adx"] > 20:
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
def generate_signal():
    best = None

    for pair in PAIRS:
        m1 = get_data(pair, "1m")
        m5 = get_data(pair, "5m")

        if m1.empty or m5.empty:
            continue

        s1 = analyze(m1)
        s5 = analyze(m5)

        if not s1:
            continue

        direction, score = s1

        if score < MIN_M1_CONFLUENCE:
            continue

        if s5:
            d5, s5_score = s5
            if d5 != direction and s5_score > MIN_M5_CONFIRM:
                continue

        if not best or score > best["score"]:
            best = {
                "pair": pair,
                "direction": direction,
                "score": score
            }

    return best

# ================= HANDLERS =================
def main_keyboard():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("📈 Прогноз", callback_data="signal")]
    ])

def result_keyboard():
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("✅ WIN", callback_data="win"),
            InlineKeyboardButton("❌ LOSS", callback_data="loss")
        ]
    ])

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🤖 Бот запущений", reply_markup=main_keyboard())

async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    if query.data == "signal":
        signal = generate_signal()

        if not signal:
            await query.message.reply_text("❌ Немає сигналу", reply_markup=main_keyboard())
            return

        text = f"""
🚀 SIGNAL

📊 {signal['pair'].replace('=X','')}
{'🟢 BUY' if signal['direction']=='BUY' else '🔴 SELL'}

⏱ 2 хв
📊 {signal['score']:.0f}%
🕒 {datetime.utcnow().strftime('%H:%M:%S')}
"""
        await query.message.reply_text(text, reply_markup=result_keyboard())

    elif query.data in ["win", "loss"]:
        await query.message.reply_text("Результат збережено", reply_markup=main_keyboard())

# ================= MAIN =================
def main():
    app = Application.builder().token(BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CallbackQueryHandler(button_handler))

    print("🚀 BOT STARTED")
    app.run_polling()

if __name__ == "__main__":
    main()
