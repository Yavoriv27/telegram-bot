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
from dotenv import load_dotenv
import pytz

from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import Application, CommandHandler, ContextTypes

load_dotenv()

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("telegram_oanda_bot")

KYIV_TZ = pytz.timezone("Europe/Kyiv")


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
    v = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return math.sqrt(v)


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
        if price > self.high:
            self.high = price
        if price < self.low:
            self.low = price
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
            self._items = self._items[-self.maxlen :]

    def items(self) -> List[Candle]:
        return list(self._items)


# ---------------- OANDA STREAM ----------------

class OandaPriceStream(threading.Thread):
    def __init__(self, api_key: str, account_id: str, instrument: str, out_q: queue.Queue, practice: bool = True):
        super().__init__(daemon=True)
        self.api_key = api_key
        self.account_id = account_id
        self.instrument = instrument
        self.out_q = out_q
        self.practice = practice
        self._stop = threading.Event()

    def stop(self):
        self._stop.set()

    def run(self):
        base = "https://stream-fxpractice.oanda.com" if self.practice else "https://stream-fxtrade.oanda.com"
        url = f"{base}/v3/accounts/{self.account_id}/pricing/stream"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        params = {"instruments": self.instrument}

        backoff = 2
        while not self._stop.is_set():
            try:
                log.info("performing request %s", url)
                with requests.get(url, headers=headers, params=params, stream=True, timeout=30) as r:
                    r.raise_for_status()
                    backoff = 2
                    for line in r.iter_lines():
                        if self._stop.is_set():
                            break
                        if not line:
                            continue
                        msg = json.loads(line.decode("utf-8"))
                        if msg.get("type") == "PRICE":
                            bid = float(msg["bids"][0]["price"])
                            ask = float(msg["asks"][0]["price"])
                            mid = (bid + ask) / 2.0
                            self.out_q.put({
                                "ts": time.time(),
                                "bid": bid,
                                "ask": ask,
                                "mid": mid
                            })
            except Exception as e:
                log.warning("Stream error: %s (reconnect in %ss)", e, backoff)
                time.sleep(backoff)
                backoff = min(backoff * 2, 30)


# ---------------- INDICATORS ----------------

def ema(values: List[float], period: int) -> Optional[float]:
    if len(values) < period:
        return None
    k = 2 / (period + 1)
    e = mean(values[:period])
    for v in values[period:]:
        e = v * k + e * (1 - k)
    return e


def rsi(values: List[float], period: int = 14) -> Optional[float]:
    if len(values) < period + 1:
        return None
    gains, losses = [], []
    for i in range(-period, 0):
        diff = values[i] - values[i - 1]
        gains.append(max(diff, 0))
        losses.append(max(-diff, 0))
    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def adx(highs: List[float], lows: List[float], closes: List[float], period: int = 14) -> Optional[float]:
    if len(closes) < period + 1:
        return None

    trs, p_dm, m_dm = [], [], []
    for i in range(1, len(closes)):
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1])
        )
        trs.append(tr)
        up = highs[i] - highs[i - 1]
        down = lows[i - 1] - lows[i]
        p_dm.append(up if up > down and up > 0 else 0)
        m_dm.append(down if down > up and down > 0 else 0)

    if len(trs) < period:
        return None

    tr14 = sum(trs[:period])
    p14 = sum(p_dm[:period])
    m14 = sum(m_dm[:period])

    dxs = []
    for i in range(period, len(trs)):
        tr14 = tr14 - (tr14 / period) + trs[i]
        p14 = p14 - (p14 / period) + p_dm[i]
        m14 = m14 - (m14 / period) + m_dm[i]
        if tr14 == 0:
            continue
        di_p = 100 * (p14 / tr14)
        di_m = 100 * (m14 / tr14)
        if di_p + di_m == 0:
            continue
        dxs.append(100 * abs(di_p - di_m) / (di_p + di_m))

    if len(dxs) < period:
        return None

    adx_val = sum(dxs[:period]) / period
    for v in dxs[period:]:
        adx_val = (adx_val * (period - 1) + v) / period
    return adx_val
# ---------------- SIGNAL ENGINE ----------------

class SignalEngine:
    def __init__(self):
        self.symbol = os.getenv("SYMBOL", "EUR_USD")

        self.auto_enabled = os.getenv("AUTO_ENABLED", "true").lower() == "true"
        self.auto_every_sec = int(os.getenv("AUTO_EVERY_SEC", "300"))

        # таймфрейми
        self.tf_fast = 60     # 30 секунд
        self.tf_slow = 600    # 5 хвилин

        self._q = queue.Queue(maxsize=20000)
        self._lock = threading.Lock()

        self.builder_fast = InternalCandleBuilder(self.tf_fast)
        self.builder_slow = InternalCandleBuilder(self.tf_slow)

        self.hist_fast = CandleHistory(maxlen=400)
        self.hist_slow = CandleHistory(maxlen=400)

        self._last_fast_ts = None
        self._last_slow_ts = None

        self.last_tick = None
        self._stream = None

    # ---------- STREAM ----------
    def start_stream(self):
        api_key = (os.getenv("OANDA_API_KEY") or "").strip()
        account_id = (os.getenv("OANDA_ACCOUNT_ID") or "").strip()
        env = (os.getenv("OANDA_ENV") or "practice").lower()

        if not api_key or not account_id:
            raise RuntimeError("OANDA_API_KEY / OANDA_ACCOUNT_ID missing")

        practice = env == "practice"

        self._stream = OandaPriceStream(
            api_key=api_key,
            account_id=account_id,
            instrument=self.symbol,
            out_q=self._q,
            practice=practice,
        )
        self._stream.start()

        threading.Thread(target=self._pump_ticks, daemon=True).start()

    def _pump_ticks(self):
        while True:
            item = self._q.get()
            ts = float(item["ts"])
            mid = float(item["mid"])

            with self._lock:
                self.last_tick = item

                self.builder_fast.on_tick(ts, mid)
                self.builder_slow.on_tick(ts, mid)

                c_fast = self.builder_fast.last_closed
                if c_fast and c_fast.start_ts != self._last_fast_ts:
                    self._last_fast_ts = c_fast.start_ts
                    self.hist_fast.append(c_fast)

                c_slow = self.builder_slow.last_closed
                if c_slow and c_slow.start_ts != self._last_slow_ts:
                    self._last_slow_ts = c_slow.start_ts
                    self.hist_slow.append(c_slow)

    # ---------- SNAPSHOT ----------
    def snapshot(self):
        with self._lock:
            return {
                "last": self.last_tick,
                "fast": self.hist_fast.items(),
                "slow": self.hist_slow.items(),
            }

    # ---------- SIGNAL LOGIC ----------
    def compute_signal(self):
    snap = self.snapshot()
    last = snap["last"]
    fast = snap["fast"]
    slow = snap["slow"]

    if not last or len(fast) < 30 or len(slow) < 30:
        return {"ok": False, "reason": "NOT_ENOUGH_DATA"}

    closes = [c.close for c in slow]
    highs = [c.high for c in slow]
    lows = [c.low for c in slow]

    rsi_v = rsi(closes, 14)
    adx_v = adx(highs, lows, closes, 14)

    if rsi_v is None or adx_v is None:
        return {"ok": False, "reason": "NO_DATA"}

    # захист: мертвий ринок
    if adx_v < 18:
        return {"ok": False, "reason": "MARKET_FLAT"}

    # тренд є, але не перегрітий
    if adx_v > 35:
        return {"ok": False, "reason": "OVERHEATED"}

    # ---- BUY ----
    if 56 <= rsi_v <= 66:
        return {
            "ok": True,
            "direction": "BUY",
            "expiry_sec": 600,
            "rsi": round(rsi_v, 1),
            "adx": round(adx_v, 1)
        }

    # ---- SELL ----
    if 34 <= rsi_v <= 44:
        return {
            "ok": True,
            "direction": "SELL",
            "expiry_sec": 600,
            "rsi": round(rsi_v, 1),
            "adx": round(adx_v, 1)
        }

    return {"ok": False, "reason": "NO_SIGNAL"}


# ---------------- SUBSCRIBERS ----------------

class Subscribers:
    def __init__(self, path: str):
        self.path = path
        self._lock = threading.Lock()
        self._ids = []
        self._load()

    def _load(self):
        try:
            if os.path.exists(self.path):
                with open(self.path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self._ids = list(set(data.get("chat_ids", [])))
        except Exception:
            self._ids = []

    def _save(self):
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump({"chat_ids": self._ids}, f)

    def add(self, chat_id: int) -> bool:
        with self._lock:
            if chat_id in self._ids:
                return False
            self._ids.append(chat_id)
            self._save()
            return True

    def remove(self, chat_id: int) -> bool:
        with self._lock:
            if chat_id not in self._ids:
                return False
            self._ids.remove(chat_id)
            self._save()
            return True

    def list(self):
        return list(self._ids)

ENGINE = SignalEngine()
SUBS = Subscribers(os.getenv("SUBSCRIBERS_FILE", "/app/subscribers.json"))

# ---------------- TELEGRAM TEXT FORMAT ----------------

def fmt_manual_signal(sig: dict) -> str:
    t = fmt_kyiv(now_utc())

    if sig.get("ok") and sig.get("direction") in ("BUY", "SELL"):
        arrow = "🟢 BUY" if sig["direction"] == "BUY" else "🔴 SELL"
        return (
            f"{arrow}\n"
            f"⏱ <b>Експірація:</b> 10 хв\n"
            f"🕒 <b>Kyiv:</b> {t}\n"
            f"<b>RSI(14):</b> {sig['rsi']}\n"
            f"<b>ADX(14):</b> {sig['adx']}"
        )

    reasons = {
        "NOT_ENOUGH_DATA": "⏳ Недостатньо свічок (бот тільки запустився)",
        "ADX_FILTER": "⚠️ Немає нормального тренду (ADX)",
        "NO_SIGNAL": "😐 RSI у середині — немає переваги",
        "NO_DATA": "❌ Індикатори не порахувались"
    }

    return (
        "❌ <b>Сигналу немає</b>\n"
        f"🕒 <b>Kyiv:</b> {t}\n"
        f"{reasons.get(sig.get('reason'), sig.get('reason'))}"
    )



# ---------------- COMMANDS ----------------

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "✅ Бот запущено.\n\n"
        "Команди:\n"
        "/status — стан ринку/свічок\n"
        "/signal — сигнал зараз\n"
        "/auto_on — авто ON\n"
        "/auto_off — авто OFF\n"
        "/subscribe — отримувати автосигнали (для брата теж)\n"
        "/unsubscribe — відписатися\n"
        "/subs — список підписників (count)"
    )


async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    snap = ENGINE.snapshot()
    t = fmt_kyiv(now_utc())

    fast = snap["fast"]
    slow = snap["slow"]
    last = snap.get("last")

    msg = (
        "<b>СТАТУС БОТА</b>\n"
        f"🕒 <b>Kyiv:</b> {t}\n"
        f"⚙️ <b>Авто:</b> {'ON' if ENGINE.auto_enabled else 'OFF'}\n"
        f"⏱️ <b>Інтервал:</b> {ENGINE.auto_every_sec} сек\n"
        f"🕯️ <b>Свічки:</b> fast={len(fast)} | slow={len(slow)}"
    )

    if last:
        msg += f"\nTick: bid={last['bid']:.5f} ask={last['ask']:.5f}"

    await update.message.reply_text(msg, parse_mode="HTML")




async def cmd_signal(update: Update, context: ContextTypes.DEFAULT_TYPE):
    sig = ENGINE.compute_signal()
    text = fmt_manual_signal(sig)
    await update.message.reply_text(text, parse_mode="HTML")

async def cmd_auto_on(update: Update, context: ContextTypes.DEFAULT_TYPE):
    ENGINE.auto_enabled = True
    await update.message.reply_text("✅ Автосигнали: ON")


async def cmd_auto_off(update: Update, context: ContextTypes.DEFAULT_TYPE):
    ENGINE.auto_enabled = False
    await update.message.reply_text("⛔ Автосигнали: OFF")


async def cmd_subscribe(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    if SUBS.add(chat_id):
        await update.message.reply_text("✅ Підписано на автосигнали.")
    else:
        await update.message.reply_text("ℹ️ Уже підписаний.")


async def cmd_unsubscribe(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    if SUBS.remove(chat_id):
        await update.message.reply_text("✅ Відписано.")
    else:
        await update.message.reply_text("ℹ️ Не був підписаний.")


async def cmd_subs(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(f"👥 Підписників: {len(SUBS.list())}")


# ---------------- AUTO JOB ----------------

async def auto_job(context: ContextTypes.DEFAULT_TYPE):
    if not ENGINE.auto_enabled:
        return

    sig = ENGINE.compute_signal()
    if not sig.get("ok"):
        return

    msg = fmt_manual_signal(sig)
    for cid in SUBS.list():
        await context.bot.send_message(cid, msg, parse_mode=ParseMode.HTML)


# ---------------- MAIN ----------------

def main():
    import sys

    LOCK_FILE = "/tmp/bot.lock"

    if os.path.exists(LOCK_FILE):
        print("Bot already running, exit.")
        sys.exit(0)

    with open(LOCK_FILE, "w") as f:
        f.write(str(os.getpid()))

    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN missing")

    ENGINE.start_stream()

    app = Application.builder().token(token).build()

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(CommandHandler("signal", cmd_signal))
    app.add_handler(CommandHandler("auto_on", cmd_auto_on))
    app.add_handler(CommandHandler("auto_off", cmd_auto_off))
    app.add_handler(CommandHandler("subscribe", cmd_subscribe))
    app.add_handler(CommandHandler("unsubscribe", cmd_unsubscribe))

    app.job_queue.run_repeating(
        auto_job,
        interval=ENGINE.auto_every_sec,
        first=10
    )

    try:
        app.run_polling(drop_pending_updates=True)
    finally:
        if os.path.exists(LOCK_FILE):
            os.remove(LOCK_FILE)


if __name__ == "__main__":
    main()

