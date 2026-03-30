import os
import asyncio
from datetime import datetime
from dotenv import load_dotenv

from telegram import Update, InlineKeyboardMarkup, InlineKeyboardButton
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes

import oandapyV20
from oandapyV20.endpoints import instruments

load_dotenv()

# ================= CONFIG =================
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
OANDA_KEY = os.getenv("OANDA_API_KEY")

PAIRS = ["EUR_USD", "GBP_USD", "USD_JPY"]

BET_PERCENT = 0.1

# ================= STATE =================
BALANCE = 1000
WIN = 0
LOSS = 0
STREAK = 0
AUTO = False

client = oandapyV20.API(access_token=OANDA_KEY)

# ================= DATA =================
def get_prices(pair, tf="M1", count=50):
    params = {"granularity": tf, "count": count, "price": "M"}
    r = instruments.InstrumentsCandles(instrument=pair, params=params)
    client.request(r)

    data = []
    for c in r.response["candles"]:
        if c["complete"]:
            data.append(float(c["mid"]["c"]))
    return data

# ================= STRATEGY =================
def analyze(prices):
    if len(prices) < 10:
        return None

    buy = 0
    sell = 0

    # momentum
    if prices[-1] > prices[-2]:
        buy += 1
    else:
        sell += 1

    # trend
    if prices[-1] > sum(prices[-5:]) / 5:
        buy += 1
    else:
        sell += 1

    # price action (3 candles)
    if prices[-1] > prices[-2] > prices[-3]:
        buy += 2
    elif prices[-1] < prices[-2] < prices[-3]:
        sell += 2

    total = buy + sell
    if total == 0:
        return None

    buy_score = buy / total * 100
    sell_score = sell / total * 100

    if buy_score > sell_score:
        return "BUY", buy_score
    else:
        return "SELL", sell_score

# ================= SIGNAL =================
def generate_signal():
    best = None

    for pair in PAIRS:
        m1 = get_prices(pair, "M1")
        m5 = get_prices(pair, "M5")

        s1 = analyze(m1)
        s5 = analyze(m5)

        if not s1:
            continue

        direction, score = s1

        # M5 filter
        if s5:
            d5, _ = s5
            if d5 != direction:
                continue

        if not best or score > best["score"]:
            best = {
                "pair": pair,
                "direction": direction,
                "score": score
            }

    return best

# ================= UI =================
def main_kb():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("📈 Прогноз", callback_data="signal")],
        [InlineKeyboardButton("🤖 Авто ON/OFF", callback_data="auto")]
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
    await update.message.reply_text("🚀 BOT READY", reply_markup=main_kb())

async def buttons(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global AUTO, BALANCE, WIN, LOSS, STREAK

    q = update.callback_query
    await q.answer()

    # ===== SIGNAL =====
    if q.data == "signal":
        s = generate_signal()

        if not s:
            await q.message.reply_text("❌ Немає сигналу", reply_markup=main_kb())
            return

        bet = BALANCE * BET_PERCENT

        msg = f"""
📊 {s['pair']}
{'🟢 BUY' if s['direction']=='BUY' else '🔴 SELL'}

📈 {s['score']:.0f}%
💵 Ставка: {bet:.2f}
💰 Баланс: {BALANCE:.2f}
⏱ 2 хв
🕒 {datetime.utcnow().strftime('%H:%M:%S')}
"""
        await q.message.reply_text(msg, reply_markup=result_kb())

    # ===== WIN =====
    elif q.data == "win":
        bet = BALANCE * BET_PERCENT
        profit = bet * 0.8

        BALANCE += profit
        WIN += 1
        STREAK = max(1, STREAK + 1)

        await q.message.reply_text(f"✅ WIN\n💰 {BALANCE:.2f}", reply_markup=main_kb())

    # ===== LOSS =====
    elif q.data == "loss":
        bet = BALANCE * BET_PERCENT

        BALANCE -= bet
        LOSS += 1
        STREAK = min(-1, STREAK - 1)

        await q.message.reply_text(f"❌ LOSS\n💰 {BALANCE:.2f}", reply_markup=main_kb())

    # ===== AUTO =====
    elif q.data == "auto":
        AUTO = not AUTO
        await q.message.reply_text(f"🤖 AUTO: {AUTO}", reply_markup=main_kb())

# ================= AUTO LOOP =================
async def auto_loop(app):
    global AUTO

    while True:
        if AUTO:
            s = generate_signal()
            if s:
                await app.bot.send_message(
                    chat_id=os.getenv("CHAT_ID"),
                    text=f"{s['pair']} {'BUY' if s['direction']=='BUY' else 'SELL'} {s['score']:.0f}%"
                )
        await asyncio.sleep(120)

# ================= MAIN =================
def main():
    app = Application.builder().token(TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CallbackQueryHandler(buttons))

    app.create_task(auto_loop(app))

    print("🚀 BOT STARTED")
    app.run_polling()

if __name__ == "__main__":
    main()
