import os
import asyncio
import time
from datetime import datetime
from collections import deque
from dotenv import load_dotenv

import yfinance as yf

from telegram import Update, InlineKeyboardMarkup, InlineKeyboardButton
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes

load_dotenv()

TOKEN = os.getenv("BOT_TOKEN")

PAIRS = ["EURUSD=X", "GBPUSD=X", "USDJPY=X"]

BALANCE = 1000
WIN = 0
LOSS = 0

memory = {
    "BUY": 1.0,
    "SELL": 1.0
}

# ================= DATA =================
def get_price(pair):
    df = yf.download(pair, period="1d", interval="1m", progress=False)
    return df["Close"].iloc[-1]

# ================= CANDLES =================
class CandleBuilder:
    def __init__(self):
        self.m1 = deque(maxlen=100)
        self.m5 = deque(maxlen=100)
        self.last_min = None
        self.current = None

    def update(self, price):
        now = datetime.utcnow()

        minute = now.minute

        if self.last_min != minute:
            if self.current:
                self.m1.append(self.current)

            self.current = {
                "open": price,
                "high": price,
                "low": price,
                "close": price
            }
            self.last_min = minute
        else:
            self.current["close"] = price
            self.current["high"] = max(self.current["high"], price)
            self.current["low"] = min(self.current["low"], price)

        if len(self.m1) >= 5:
            last5 = list(self.m1)[-5:]
            m5 = {
                "open": last5[0]["open"],
                "high": max(x["high"] for x in last5),
                "low": min(x["low"] for x in last5),
                "close": last5[-1]["close"]
            }
            self.m5.append(m5)

builder = {p: CandleBuilder() for p in PAIRS}

# ================= ANALYSIS =================
def analyze(candles):
    if len(candles) < 10:
        return None

    closes = [c["close"] for c in candles]

    buy = 0
    sell = 0

    # simple momentum
    if closes[-1] > closes[-2]:
        buy += 1
    else:
        sell += 1

    # trend
    if closes[-1] > sum(closes[-5:]) / 5:
        buy += 1
    else:
        sell += 1

    # price action (3 candles)
    last3 = candles[-3:]
    if all(c["close"] > c["open"] for c in last3):
        buy += 2
    elif all(c["close"] < c["open"] for c in last3):
        sell += 2

    total = buy + sell
    if total == 0:
        return None

    buy_score = buy / total * 100
    sell_score = sell / total * 100

    if buy_score > sell_score:
        return "BUY", buy_score * memory["BUY"]
    else:
        return "SELL", sell_score * memory["SELL"]

# ================= SIGNAL =================
def generate_signal():
    best = None

    for pair in PAIRS:
        b = builder[pair]

        if len(b.m1) < 10 or len(b.m5) < 5:
            continue

        m1 = analyze(list(b.m1))
        m5 = analyze(list(b.m5))

        if not m1:
            continue

        direction, score = m1

        if m5:
            d5, s5 = m5
            if d5 != direction:
                continue

        if not best or score > best["score"]:
            best = {
                "pair": pair,
                "direction": direction,
                "score": score
            }

    return best

# ================= TELEGRAM =================
def main_kb():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("📈 Прогноз", callback_data="signal")],
        [InlineKeyboardButton("🤖 Авто", callback_data="auto")]
    ])

def result_kb():
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("✅", callback_data="win"),
            InlineKeyboardButton("❌", callback_data="loss")
        ]
    ])

AUTO = False

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🚀 PRO BOT", reply_markup=main_kb())

async def button(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global AUTO, BALANCE, WIN, LOSS

    q = update.callback_query
    await q.answer()

    if q.data == "signal":
        s = generate_signal()

        if not s:
            await q.message.reply_text("❌ Немає сигналу", reply_markup=main_kb())
            return

        msg = f"""
{ s['pair'] }
{ s['direction'] }

📊 {s['score']:.0f}%
💰 {BALANCE}
"""
        await q.message.reply_text(msg, reply_markup=result_kb())

    elif q.data == "win":
        WIN += 1
        BALANCE += BALANCE * 0.1
        memory["BUY"] += 0.05
        memory["SELL"] += 0.05
        await q.message.reply_text(f"✅ WIN | {BALANCE}", reply_markup=main_kb())

    elif q.data == "loss":
        LOSS += 1
        BALANCE -= BALANCE * 0.1
        memory["BUY"] *= 0.95
        memory["SELL"] *= 0.95
        await q.message.reply_text(f"❌ LOSS | {BALANCE}", reply_markup=main_kb())

    elif q.data == "auto":
        AUTO = not AUTO
        await q.message.reply_text(f"AUTO: {AUTO}", reply_markup=main_kb())

# ================= LOOP =================
async def market_loop(app):
    while True:
        for p in PAIRS:
            try:
                price = get_price(p)
                builder[p].update(price)
            except:
                pass

        if AUTO:
            s = generate_signal()
            if s:
                await app.bot.send_message(
                    chat_id=os.getenv("CHAT_ID"),
                    text=f"{s['pair']} {s['direction']} {s['score']:.0f}%"
                )

        await asyncio.sleep(10)

# ================= MAIN =================
def main():
    app = Application.builder().token(TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CallbackQueryHandler(button))

    print("🚀 PRO BOT STARTED")

    app.create_task(market_loop(app))
    app.run_polling()

if __name__ == "__main__":
    main()
