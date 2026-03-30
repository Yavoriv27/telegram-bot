import os
import asyncio
from datetime import datetime
from dotenv import load_dotenv

from telegram import Update, InlineKeyboardMarkup, InlineKeyboardButton
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes

import oandapyV20
from oandapyV20.endpoints import instruments, orders

load_dotenv()

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
OANDA_KEY = os.getenv("OANDA_API_KEY")
ACCOUNT_ID = os.getenv("OANDA_ACCOUNT_ID")

client = oandapyV20.API(access_token=OANDA_KEY)

PAIRS = ["EUR_USD", "GBP_USD", "USD_JPY"]

BALANCE = 1000
BET_PERCENT = 0.1

AUTO = False
REAL_TRADING = False  # ⚠️ ВКЛ тільки коли готовий

LAST_SIGNAL = None

NEWS_TIMES = ["15:30", "16:00", "17:00"]

# ================= LOG =================
def log_trade(text):
    with open("trades.log", "a") as f:
        f.write(f"{datetime.utcnow()} | {text}\n")

# ================= NEWS FILTER =================
def is_news_time():
    now = datetime.utcnow().strftime("%H:%M")
    return now in NEWS_TIMES

# ================= DATA =================
def get_prices(pair, tf="M1", count=50):
    params = {"granularity": tf, "count": count, "price": "M"}
    r = instruments.InstrumentsCandles(instrument=pair, params=params)
    client.request(r)

    return [float(c["mid"]["c"]) for c in r.response["candles"] if c["complete"]]

# ================= STRATEGY =================
def analyze(prices):
    if len(prices) < 10:
        return None

    buy = 0
    sell = 0

    if prices[-1] > prices[-2]:
        buy += 1
    else:
        sell += 1

    if prices[-1] > sum(prices[-5:]) / 5:
        buy += 1
    else:
        sell += 1

    if prices[-1] > prices[-2] > prices[-3]:
        buy += 2
    elif prices[-1] < prices[-2] < prices[-3]:
        sell += 2

    total = buy + sell
    if total == 0:
        return None

    if buy > sell:
        return "BUY", (buy / total) * 100
    else:
        return "SELL", (sell / total) * 100

# ================= SIGNAL =================
def generate_signal():
    global LAST_SIGNAL

    best = None

    for pair in PAIRS:
        m1 = get_prices(pair, "M1")
        m5 = get_prices(pair, "M5")

        s1 = analyze(m1)
        s5 = analyze(m5)

        if not s1:
            continue

        direction, score = s1

        if s5 and s5[0] != direction:
            continue

        if not best or score > best["score"]:
            best = {"pair": pair, "direction": direction, "score": score}

    # анти-флуд
    if best and LAST_SIGNAL == best:
        return None

    LAST_SIGNAL = best
    return best

# ================= OANDA TRADE =================
def place_trade(pair, direction):
    if not REAL_TRADING:
        return "SIMULATION"

    units = 1000 if direction == "BUY" else -1000

    data = {
        "order": {
            "units": str(units),
            "instrument": pair,
            "timeInForce": "FOK",
            "type": "MARKET",
            "positionFill": "DEFAULT"
        }
    }

    r = orders.OrderCreate(ACCOUNT_ID, data=data)
    client.request(r)

    return "ORDER PLACED"

# ================= UI =================
def main_kb():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("📈 Прогноз", callback_data="signal")],
        [InlineKeyboardButton("🤖 Авто", callback_data="auto")],
        [InlineKeyboardButton("💰 Trade ON/OFF", callback_data="trade")]
    ])

def result_kb():
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("✅", callback_data="win"),
            InlineKeyboardButton("❌", callback_data="loss")
        ]
    ])

# ================= HANDLERS =================
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🚀 PRO BOT+", reply_markup=main_kb())

async def buttons(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global AUTO, BALANCE, REAL_TRADING

    q = update.callback_query
    await q.answer()

    if q.data == "signal":

        if is_news_time():
            await q.message.reply_text("📰 Новини — пропускаємо")
            return

        s = generate_signal()

        if not s:
            await q.message.reply_text("❌ Немає сигналу", reply_markup=main_kb())
            return

        bet = BALANCE * BET_PERCENT

        result = place_trade(s["pair"], s["direction"])

        log_trade(f"{s} | {result}")

        await q.message.reply_text(
            f"{s['pair']} {s['direction']} {s['score']:.0f}%\n💵 {bet:.2f}\n{result}",
            reply_markup=result_kb()
        )

    elif q.data == "win":
        profit = BALANCE * BET_PERCENT * 0.8
        BALANCE += profit
        log_trade("WIN")

        await q.message.reply_text(f"✅ {BALANCE:.2f}", reply_markup=main_kb())

    elif q.data == "loss":
        loss = BALANCE * BET_PERCENT
        BALANCE -= loss
        log_trade("LOSS")

        await q.message.reply_text(f"❌ {BALANCE:.2f}", reply_markup=main_kb())

    elif q.data == "auto":
        AUTO = not AUTO
        await q.message.reply_text(f"AUTO: {AUTO}", reply_markup=main_kb())

    elif q.data == "trade":
        REAL_TRADING = not REAL_TRADING
        await q.message.reply_text(f"REAL TRADING: {REAL_TRADING}", reply_markup=main_kb())

# ================= AUTO =================
async def auto_loop(app):
    while True:
        if AUTO and not is_news_time():
            s = generate_signal()
            if s:
                await app.bot.send_message(
                    chat_id=os.getenv("CHAT_ID"),
                    text=f"{s['pair']} {s['direction']} {s['score']:.0f}%"
                )
        await asyncio.sleep(120)

# ================= MAIN =================
def main():
    app = Application.builder().token(TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CallbackQueryHandler(buttons))

    app.create_task(auto_loop(app))

    print("🚀 BOT PRO+ STARTED")
    app.run_polling()

if __name__ == "__main__":
    main()
