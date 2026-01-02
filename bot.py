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

def fmt_kyiv(dt_utc: datetime) -> str:
    return dt_utc.astimezone(KYIV_TZ).strftime("%Y-%m-%d %H:%M:%S")

def mean(xs: List[float]) -> float:
    return sum(xs) / max(1, len(xs))

def stdev(xs: List[float]) -> float:
    if len(xs) < 2:
        return 0.0
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
    ticks: int = 0

    def update(self, price: float):
        self.close = price
        self.high = max(self.high, price)
        self.low = min(self.low, price)
        self.ticks += 1

    @property
    def dir(self) -> str:
        if self.close > self.open:
            return "UP"
        if self.close < self.open:
            return "DOWN"
        return "FLAT"

class InternalCandleBuilder:
    def __init__(self, tf_sec: int):
        self.tf_sec = tf_sec
        self.current: Optional[Candle] = None
        self.last_closed: Optional[Candle] = None

    def _bucket_start(self, ts: float) -> float:
        return ts - (ts % self.tf_sec)

    def on_tick(self, ts: float, price: float):
        b = self._bucket_start(ts)
        if self.current is None:
            self.current = Candle(self.tf_sec, b, price, price, price, price, ticks=1)
            return
        if b == self.current.start_ts:
            self.current.update(price)
            return
        self.last_closed = self.current
        self.current = Candle(self.tf_sec, b, price, price, price, price, ticks=1)

class CandleHistory:
    def __init__(self, maxlen: int = 400):
        self.maxlen = maxlen
        self._items: List[Candle] = []

    def append(self, c: Candle):
        self._items.append(c)
        if len(self._items) > self.maxlen:
            self._items = self._items[-self.maxlen:]

    def items(self) -> List[Candle]:
        return list(self._items)

# ---------------- OANDA STREAM ----------------

class OandaPriceStream(threading.Thread):
    def __init__(self, api_key, account_id, instrument, out_q, practice=True):
        super().__init__(daemon=True)
        self.api_key = api_key
        self.account_id = account_id
        self.instrument = instrument
        self.out_q = out_q
        self.practice = practice

    def run(self):
        base = "https://stream-fxpractice.oanda.com" if self.practice else "https://stream-fxtrade.oanda.com"
        url = f"{base}/v3/accounts/{self.account_id}/pricing/stream"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        params = {"instruments": self.instrument}

        while True:
            try:
                with requests.get(url, headers=headers, params=params, stream=True, timeout=30) as r:
                    r.raise_for_status()
                    for line in r.iter_lines():
                        if not line:
                            continue
                        msg = json.loads(line.decode())
                        if msg.get("type") == "PRICE":
                            bid = float(msg["bids"][0]["price"])
                            ask = float(msg["asks"][0]["price"])
                            mid = (bid + ask) / 2
                            self.out_q.put({
                                "ts": time.time(),
                                "bid": bid,
                                "ask": ask,
                                "mid": mid
                            })
            except Exception as e:
                log.warning("OANDA stream error: %s", e)
                time.sleep(5)

# ---------------- INDICATORS ----------------

def ema(values: List[float], period: int):
    if len(values) < period:
        return None
    k = 2 / (period + 1)
    e = mean(values[:period])
    for v in values[period:]:
        e = v * k + e * (1 - k)
    return e

def rsi(values: List[float], period: int = 14):
    if len(values) < period + 1:
        return None
    gains, losses = [], []
    for i in range(-period, 0):
        diff = values[i] - values[i - 1]
        gains.append(max(0, diff))
        losses.append(max(0, -diff))
    if sum(losses) == 0:
        return 100.0
    rs = (sum(gains) / period) / (sum(losses) / period)
    return 100 - (100 / (1 + rs))

def macd(values: List[float]):
    if len(values) < 26:
        return None
    return ema(values, 12) - ema(values, 26)

def adx(highs, lows, closes, period=14):
    if len(closes) < period + 1:
        return None
    trs = []
    for i in range(1, len(closes)):
        trs.append(max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1])
        ))
    return mean(trs[-period:])

# ---------------- SUBSCRIBERS ----------------

class Subscribers:
    def __init__(self, path):
        self.path = path
        self.ids = []
        if os.path.exists(path):
            self.ids = json.load(open(path)).get("chat_ids", [])

    def add(self, cid):
        if cid not in self.ids:
            self.ids.append(cid)
            json.dump({"chat_ids": self.ids}, open(self.path, "w"))
            return True
        return False

    def list(self):
        return self.ids

# ---------------- SIGNAL ENGINE ----------------

class SignalEngine:
    def __init__(self):
        self.q = queue.Queue()
        self.builder_1m = InternalCandleBuilder(60)
        self.builder_5m = InternalCandleBuilder(300)
        self.h1 = CandleHistory()
        self.h5 = CandleHistory()

    def start_stream(self):
        api_key = os.getenv("OANDA_API_KEY")
        account_id = os.getenv("OANDA_ACCOUNT_ID")
        env = os.getenv("OANDA_ENV", "practice")
        stream = OandaPriceStream(
            api_key,
            account_id,
            "EUR_USD",
            self.q,
            practice=(env == "practice")
        )
        stream.start()
        threading.Thread(target=self._pump, daemon=True).start()

    def _pump(self):
        while True:
            item = self.q.get()
            ts = item["ts"]
            price = item["mid"]
            self.builder_1m.on_tick(ts, price)
            self.builder_5m.on_tick(ts, price)
            if self.builder_1m.last_closed:
                self.h1.append(self.builder_1m.last_closed)
            if self.builder_5m.last_closed:
                self.h5.append(self.builder_5m.last_closed)

    def compute(self):
        if len(self.h5.items()) < 60:
            return None
        closes = [c.close for c in self.h5.items()]
        highs = [c.high for c in self.h5.items()]
        lows = [c.low for c in self.h5.items()]

        score = 0
        if ema(closes, 20) > ema(closes, 50): score += 1
        if rsi(closes) > 55: score += 1
        if macd(closes) > 0: score += 1
        if adx(highs, lows, closes) > 0: score += 1
        if self.h1.items()[-1].dir == "UP": score += 1

        if score >= 4:
            return "BUY"
        return None

# ---------------- GLOBALS ----------------

ENGINE = SignalEngine()
SUBS = Subscribers("/app/subscribers.json")

# ---------------- TELEGRAM COMMANDS ----------------

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "✅ Бот запущено.\n\n"
        "Команди:\n"
        "/status — стан ринку\n"
        "/signal — сигнал зараз\n"
        "/auto_on — авто ON\n"
        "/auto_off — авто OFF\n"
        "/subscribe — підписка\n"
        "/unsubscribe — відписка\n"
        "/subs — кількість підписників"
    )

async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    snap = ENGINE.snapshot()
    await update.message.reply_text(
        f"📊 Статус:\n"
        f"1M свічок: {len(snap['h1'])}\n"
        f"5M свічок: {len(snap['h5'])}\n"
        f"Авто: {'ON' if ENGINE.auto_enabled else 'OFF'}"
    )

async def cmd_signal(update: Update, context: ContextTypes.DEFAULT_TYPE):
    sig = ENGINE.compute_signal()
    await update.message.reply_text(fmt_signal(sig), parse_mode=ParseMode.HTML)

async def cmd_auto_on(update: Update, context: ContextTypes.DEFAULT_TYPE):
    ENGINE.auto_enabled = True
    await update.message.reply_text("✅ Автосигнали УВІМКНЕНО")

async def cmd_auto_off(update: Update, context: ContextTypes.DEFAULT_TYPE):
    ENGINE.auto_enabled = False
    await update.message.reply_text("⛔ Автосигнали ВИМКНЕНО")

async def cmd_subscribe(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if SUBS.add(update.effective_chat.id):
        await update.message.reply_text("✅ Підписано на автосигнали")
    else:
        await update.message.reply_text("ℹ️ Ви вже підписані")

async def cmd_unsubscribe(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if SUBS.remove(update.effective_chat.id):
        await update.message.reply_text("❌ Відписано")
    else:
        await update.message.reply_text("ℹ️ Ви не були підписані")

async def cmd_subs(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(f"👥 Підписників: {len(SUBS.list())}")


# ---------------- MAIN ----------------

def main():
    token = (os.getenv("TELEGRAM_BOT_TOKEN") or os.getenv("BOT_TOKEN") or "").strip()
    if not token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN / BOT_TOKEN missing")

    ENGINE.start_stream()

    app = Application.builder().token(token).build()

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(CommandHandler("signal", cmd_signal))

    app.add_handler(CommandHandler("auto_on", cmd_auto_on))
    app.add_handler(CommandHandler("auto_off", cmd_auto_off))

    app.add_handler(CommandHandler("subscribe", cmd_subscribe))
    app.add_handler(CommandHandler("unsubscribe", cmd_unsubscribe))
    app.add_handler(CommandHandler("subs", cmd_subs))

    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
