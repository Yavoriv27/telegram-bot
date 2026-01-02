import os
import json
import time
import queue
import math
import threading
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional, List, Dict, Any

import requests
import pytz
from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import Application, CommandHandler, ContextTypes

# ---------------- CONFIG ----------------

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("telegram_oanda_bot")

KYIV_TZ = pytz.timezone("Europe/Kyiv")

# ---------------- UTILS ----------------

def now_utc() -> datetime:
    return datetime.now(timezone.utc)

def fmt_kyiv(dt: datetime) -> str:
    return dt.astimezone(KYIV_TZ).strftime("%Y-%m-%d %H:%M:%S")

def mean(xs): return sum(xs) / max(1, len(xs))

def stdev(xs):
    if len(xs) < 2: return 0.0
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))

# ---------------- CANDLES ----------------

@dataclass
class Candle:
    tf_sec: int
    start_ts: float
    open: float
    high: float
    low: float
    close: float

    @property
    def dir(self):
        if self.close > self.open: return "UP"
        if self.close < self.open: return "DOWN"
        return "FLAT"

class InternalCandleBuilder:
    def __init__(self, tf):
        self.tf = tf
        self.current = None
        self.last_closed = None

    def _bucket(self, ts): return ts - (ts % self.tf)

    def on_tick(self, ts, price):
        b = self._bucket(ts)
        if not self.current:
            self.current = Candle(self.tf, b, price, price, price, price)
            return
        if b == self.current.start_ts:
            self.current.close = price
            self.current.high = max(self.current.high, price)
            self.current.low = min(self.current.low, price)
        else:
            self.last_closed = self.current
            self.current = Candle(self.tf, b, price, price, price, price)

class CandleHistory:
    def __init__(self, maxlen=400):
        self.items = []

    def add(self, c):
        self.items.append(c)
        self.items = self.items[-400:]

# ---------------- INDICATORS ----------------

def ema(vals, p):
    if len(vals) < p: return None
    k = 2 / (p + 1)
    e = mean(vals[:p])
    for v in vals[p:]: e = v * k + e * (1 - k)
    return e

def rsi(vals, p=14):
    if len(vals) < p + 1: return None
    gains, losses = [], []
    for i in range(-p, 0):
        d = vals[i] - vals[i-1]
        gains.append(max(0, d))
        losses.append(max(0, -d))
    if sum(losses) == 0: return 100
    rs = (sum(gains)/p) / (sum(losses)/p)
    return 100 - (100 / (1 + rs))

def macd(vals):
    if len(vals) < 35: return None
    fast = ema(vals, 12)
    slow = ema(vals, 26)
    if fast is None or slow is None: return None
    hist = fast - slow
    return hist

def adx(highs, lows, closes, p=14):
    if len(closes) < p + 1: return None
    trs = []
    for i in range(1, len(closes)):
        trs.append(max(
            highs[i]-lows[i],
            abs(highs[i]-closes[i-1]),
            abs(lows[i]-closes[i-1])
        ))
    return mean(trs[-p:])

# ---------------- SUBSCRIBERS ----------------

class Subscribers:
    def __init__(self, path):
        self.path = path
        self.ids = []
        if os.path.exists(path):
            self.ids = json.load(open(path)).get("ids", [])

    def save(self):
        json.dump({"ids": self.ids}, open(self.path, "w"))

    def add(self, cid):
        if cid not in self.ids:
            self.ids.append(cid)
            self.save()
            return True
        return False

    def list(self): return self.ids

# ---------------- SIGNAL ENGINE ----------------

class SignalEngine:
    def __init__(self):
        self.q = queue.Queue()
        self.b1 = InternalCandleBuilder(60)
        self.b5 = InternalCandleBuilder(300)
        self.h1 = CandleHistory()
        self.h5 = CandleHistory()

    def start_stream(self):
        threading.Thread(target=self.fake_stream, daemon=True).start()

    def fake_stream(self):
        # ЗАМІНА OANDA ДЛЯ СТАРТУ (щоб бот працював)
        price = 1.1
        while True:
            price += (math.sin(time.time()) * 0.00001)
            ts = time.time()
            self.b1.on_tick(ts, price)
            self.b5.on_tick(ts, price)
            if self.b1.last_closed: self.h1.add(self.b1.last_closed)
            if self.b5.last_closed: self.h5.add(self.b5.last_closed)
            time.sleep(1)

    def compute(self):
        if len(self.h5.items) < 50 or len(self.h1.items) < 50:
            return None
        c1 = self.h1.items
        c5 = self.h5.items
        closes5 = [c.close for c in c5]
        highs5 = [c.high for c in c5]
        lows5 = [c.low for c in c5]

        buy = sell = 0
        if ema(closes5,20) > ema(closes5,50): buy += 1
        if rsi(closes5) > 55: buy += 1
        if macd(closes5) > 0: buy += 1
        if adx(highs5,lows5,closes5) > 0: buy += 1
        if c1[-1].dir == "UP": buy += 1

        if buy >= 4:
            return "BUY"
        return None

# ---------------- GLOBALS ----------------

ENGINE = SignalEngine()
SUBS = Subscribers("/app/subscribers.json")

# ---------------- TELEGRAM ----------------

async def start(update: Update, ctx):
    await update.message.reply_text("✅ Бот запущено\n/subscribe /signal")

async def subscribe(update: Update, ctx):
    if SUBS.add(update.effective_chat.id):
        await update.message.reply_text("✅ Підписано")
    else:
        await update.message.reply_text("ℹ️ Вже підписаний")

async def signal(update: Update, ctx):
    s = ENGINE.compute()
    if not s:
        await update.message.reply_text("⚠️ Немає сигналу")
    else:
        await update.message.reply_text(f"📈 {s} EUR/USD", parse_mode=ParseMode.HTML)

# ---------------- MAIN ----------------

def main():
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN missing")

    ENGINE.start_stream()

    app = Application.builder().token(token).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("subscribe", subscribe))
    app.add_handler(CommandHandler("signal", signal))
    app.run_polling()

if __name__ == "__main__":
    main()
