import os
import asyncio
from datetime import datetime
from dotenv import load_dotenv
import requests

from telegram import Update, InlineKeyboardMarkup, InlineKeyboardButton
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes

import oandapyV20
from oandapyV20.endpoints import instruments, orders

load_dotenv()

# ===== CONFIG =====
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
OANDA_KEY = os.getenv("OANDA_API_KEY")
ACCOUNT_ID = os.getenv("OANDA_ACCOUNT_ID")
NEWS_KEY = os.getenv("NEWS_API_KEY")

client = oandapyV20.API(access_token=OANDA_KEY)

PAIRS = ["EUR_USD", "GBP_USD", "USD_JPY"]

BALANCE = 1000
BET_PERCENT = 0.1

AUTO = False
REAL_TRADING = False
LAST_SIGNAL = None

ML_MEMORY = {"BUY": 1.0, "SELL": 1.0}

# ===== LOG =====
def log(text):
    with open("trades.log", "a") as f:
        f.write(f"{datetime.utcnow()} | {text}\n")

# ===== NEWS API =====
def check_news():
    try:
        url = f"https://newsapi.org/v2/everything?q=forex OR usd OR eur&apiKey={NEWS_KEY}"
        r = requests.get(url, timeout=5).json()

        for article in r.get("articles", [])[:5]:
            t = article["title"].lower()
            if any(x in t for x in ["fed", "inflation", "cpi", "rate", "nfp", "ecb"]):
                return True
    except:
        pass

    return False

# ===== DATA =====
def get_prices(pair, tf="M1", count=50):
    params = {"granularity": tf, "count": count, "price": "M"}
    r = instruments.InstrumentsCandles(instrument=pair, params=params)
    client.request(r)
    return [float(c["mid"]["c"]) for c in r.response["candles"] if c["complete"]]

# ===== STRATEGY =====
def analyze(prices):
    if len(prices) < 30:
        return None

    buy = 0
    sell = 0

    # TREND
    ema_fast = sum(prices[-5:]) / 5
    ema_slow = sum(prices[-25:]) / 25

    if abs(ema_fast - ema_slow) < 0.00005:
        return None

    trend = "BUY" if ema_fast > ema_slow else "SELL"

    if trend == "BUY":
        buy += 3
    else:
        sell += 3

    # MOMENTUM
    if prices[-1] > prices[-2] > prices[-3]:
        buy += 2
    elif prices[-1] < prices[-2] < prices[-3]:
        sell += 2

    # SUPPORT / RESISTANCE
    support = min(prices[-25:])
    resistance = max(prices[-25:])
    current = prices[-1]

    zone = (resistance - support) * 0.15

    if abs(current - support) < zone:
        buy += 3
    elif abs(current - resistance) < zone:
        sell += 3
    else:
        return None

    # RSI
    gains = [prices[i] - prices[i-1] for i in range(1, len(prices)) if prices[i] > prices[i-1]]
    losses = [prices[i-1] - prices[i] for i in range(1, len(prices)) if prices[i] < prices[i-1]]

    avg_gain = sum(gains[-14:]) / 14 if len(gains) >= 14 else 0
    avg_loss = sum(losses[-14:]) / 14 if len(losses) >= 14 else 0

    rsi = 50 if avg_loss == 0 else 100 - (100 / (1 + (avg_gain / avg_loss)))

    if rsi < 40:
        buy += 1
    elif rsi > 60:
        sell += 1

    total = buy + sell
    if total < 7:
        return None

    direction = "BUY" if buy > sell else "SELL"
    score = (max(buy, sell) / total) * 100

    return direction, score * ML_MEMORY.get(direction, 1.0)

# ===== SIGNAL =====
def generate_signal():
    global LAST_SIGNAL

    if check_news():
        return None

    best = None

    for pair in PAIRS:
        m1 = get_prices(pair, "M1")
        m5 = get_prices(pair, "M5")

        s1 = analyze(m1)
        s5 = analyze(m5)

        if not s1:
            continue

        direction, score = s1

        if s5:
            d5, score5 = s5
            if d5 != direction or score5 < 60:
                continue

        if not best or score > best["score"]:
            best = {"pair": pair, "direction": direction, "score": score}

    if best == LAST_SIGNAL:
        return None

    LAST_SIGNAL = best
    return best

# ===== TRADE =====
def trade(pair, direction):
    if not REAL_TRADING:
        return "SIM"

    units = 1000 if direction == "BUY" else -1000

    data = {
        "order": {
            "units": str(units),
            "instrument": pair,
            "type": "MARKET",
            "timeInForce": "FOK",
            "positionFill": "DEFAULT"
        }
    }

    r = orders.OrderCreate(ACCOUNT_ID, data=data)
    client.request(r)
    return "LIVE"

# ===== UI =====
def main_kb():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("📈 Прогноз", callback_data="signal")],
        [InlineKeyboardButton("🤖 Авто", callback_data="auto")],
        [InlineKeyboardButton("💰 Trade", callback_data="trade")]
    ])

def result_kb():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("✅", callback_data="win"),
         InlineKeyboardButton("❌", callback_data="loss")]
    ])

# ===== HANDLERS =====
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🚀 MAX BOT", reply_markup=main_kb())

async def btn(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global AUTO, BALANCE, REAL_TRADING

    q = update.callback_query
    await q.answer()

    if q.data == "signal":
        s = generate_signal()

        if not s:
            await q.message.reply_text("❌ Немає сигналу", reply_markup=main_kb())
            return

        bet = BALANCE * BET_PERCENT
        res = trade(s["pair"], s["direction"])

        log(f"{s} | {res}")

        await q.message.reply_text(
            f"{s['pair']} {s['direction']} {s['score']:.0f}%\n💵 {bet:.2f}\n{res}",
            reply_markup=result_kb()
        )

    elif q.data == "win":
        BALANCE += BALANCE * BET_PERCENT * 0.8
        ML_MEMORY["BUY"] += 0.05
        ML_MEMORY["SELL"] += 0.05
        log("WIN")

        await q.message.reply_text(f"✅ {BALANCE:.2f}", reply_markup=main_kb())

    elif q.data == "loss":
        BALANCE -= BALANCE * BET_PERCENT
        ML_MEMORY["BUY"] *= 0.95
        ML_MEMORY["SELL"] *= 0.95
        log("LOSS")

        await q.message.reply_text(f"❌ {BALANCE:.2f}", reply_markup=main_kb())

    elif q.data == "auto":
        AUTO = not AUTO
        await q.message.reply_text(f"AUTO: {AUTO}", reply_markup=main_kb())

    elif q.data == "trade":
        REAL_TRADING = not REAL_TRADING
        await q.message.reply_text(f"REAL: {REAL_TRADING}", reply_markup=main_kb())

# ===== AUTO =====
async def auto_loop(app):
    while True:
        if AUTO:
            s = generate_signal()
            if s:
                await app.bot.send_message(
                    chat_id=os.getenv("CHAT_ID"),
                    text=f"{s['pair']} {s['direction']} {s['score']:.0f}%"
                )
        await asyncio.sleep(120)

# ===== MAIN =====
def main():
    app = Application.builder().token(TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CallbackQueryHandler(btn))

    app.create_task(auto_loop(app))

    print("🚀 MAX BOT STARTED")
    app.run_polling()

if __name__ == "__main__":
    main()
