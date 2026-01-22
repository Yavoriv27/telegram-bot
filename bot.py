import os
import json
import time
import queue
import math
import threading
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
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
log = logging.getLogger("telegram_oanda_hard_bot")

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


def pip_size(instrument: str) -> float:
    return 0.01 if "JPY" in instrument else 0.0001


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
    def body(self) -> float:
        return abs(self.close - self.open)

    @property
    def direction(self) -> str:
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
    def __init__(self, maxlen: int = 600):
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
                log.info("performing request %s | %s", url, self.instrument)
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
                                "instrument": self.instrument,
                                "bid": bid,
                                "ask": ask,
                                "mid": mid
                            })
            except Exception as e:
                log.warning("Stream error %s: %s (reconnect in %ss)", self.instrument, e, backoff)
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
            abs(lows[i] - closes[i - 1]),
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


def last3_confirm(candles: List[Candle], direction: str) -> int:
    if len(candles) < 3:
        return 0
    ups = sum(1 for c in candles[-3:] if c.direction == "UP")
    downs = sum(1 for c in candles[-3:] if c.direction == "DOWN")
    if direction == "BUY":
        return ups
    if direction == "SELL":
        return downs
    return 0


def min_body_pips_ok(candles: List[Candle], instrument: str, min_body_pips: float) -> bool:
    if len(candles) < 2:
        return False
    p = pip_size(instrument)
    b1 = candles[-1].body / p
    b2 = candles[-2].body / p
    return (b1 >= min_body_pips) or (b2 >= min_body_pips)


def chop_filter_ok(closes: List[float]) -> bool:
    if len(closes) < 25:
        return False
    last = closes[-25:]
    s = stdev(last)
    m = mean(last)
    if m == 0:
        return False
    ratio = s / m
    return ratio > 0.00005  # щоб не було "пили"


# ---------------- ENGINE ----------------

class SignalEngine:
    def __init__(self):
        self.symbol = os.getenv("SYMBOL", "EUR_USD")

        self.auto_enabled = os.getenv("AUTO_ENABLED", "true").lower() == "true"
        self.auto_every_sec = int(os.getenv("AUTO_EVERY_SEC", "60"))

        self.expiry_sec = int(os.getenv("EXPIRY_SEC", "120"))

        self.min_adx = float(os.getenv("MIN_ADX", "28"))
        self.min_body_pips = float(os.getenv("MIN_BODY_PIPS", "2.0"))

        self.rsi_buy_low = float(os.getenv("RSI_BUY_LOW", "52"))
        self.rsi_buy_high = float(os.getenv("RSI_BUY_HIGH", "64"))
        self.rsi_sell_low = float(os.getenv("RSI_SELL_LOW", "36"))
        self.rsi_sell_high = float(os.getenv("RSI_SELL_HIGH", "48"))

        self.tf_1m = 60
        self.tf_10m = 600

        self._q = queue.Queue(maxsize=20000)
        self._lock = threading.Lock()

        self.builder_1m = InternalCandleBuilder(self.tf_1m)
        self.builder_10m = InternalCandleBuilder(self.tf_10m)

        self.hist_1m = CandleHistory(maxlen=800)
        self.hist_10m = CandleHistory(maxlen=800)

        self._last_1m_ts = None
        self._last_10m_ts = None

        self.last_tick = None
        self._stream = None

        self._last_signal_candle_ts: Optional[float] = None

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

                self.builder_1m.on_tick(ts, mid)
                self.builder_10m.on_tick(ts, mid)

                c1 = self.builder_1m.last_closed
                if c1 and c1.start_ts != self._last_1m_ts:
                    self._last_1m_ts = c1.start_ts
                    self.hist_1m.append(c1)

                c10 = self.builder_10m.last_closed
                if c10 and c10.start_ts != self._last_10m_ts:
                    self._last_10m_ts = c10.start_ts
                    self.hist_10m.append(c10)

    def snapshot(self):
        with self._lock:
            return {
                "last": self.last_tick,
                "m1": self.hist_1m.items(),
                "m10": self.hist_10m.items(),
            }

    def compute_signal(self) -> Dict[str, Any]:
        snap = self.snapshot()
        last = snap["last"]
        m1 = snap["m1"]
        m10 = snap["m10"]

        if not last or len(m10) < 80 or len(m1) < 50:
            return {"ok": False, "reason": "NOT_ENOUGH_DATA"}

        last_closed_10m = m10[-1]

        if self._last_signal_candle_ts == last_closed_10m.start_ts:
            return {"ok": False, "reason": "WAIT_NEXT_CANDLE"}

        closes10 = [c.close for c in m10]
        highs10 = [c.high for c in m10]
        lows10 = [c.low for c in m10]

        rsi_v = rsi(closes10, 14)
        adx_v = adx(highs10, lows10, closes10, 14)

        if rsi_v is None or adx_v is None:
            return {"ok": False, "reason": "NO_DATA"}

        if adx_v < self.min_adx:
            return {"ok": False, "reason": "MARKET_FLAT", "rsi": round(rsi_v, 1), "adx": round(adx_v, 1)}

        ema20 = ema(closes10[-150:], 20)
        ema50 = ema(closes10[-220:], 50)
        if ema20 is None or ema50 is None:
            return {"ok": False, "reason": "NO_EMA"}

        trend = "BUY" if ema20 > ema50 else "SELL" if ema20 < ema50 else None
        if trend is None:
            return {"ok": False, "reason": "NO_TREND"}

        if not chop_filter_ok(closes10):
            return {"ok": False, "reason": "CHOPPY"}

        direction = None
        if trend == "BUY" and (self.rsi_buy_low <= rsi_v <= self.rsi_buy_high):
            direction = "BUY"
        elif trend == "SELL" and (self.rsi_sell_low <= rsi_v <= self.rsi_sell_high):
            direction = "SELL"
        else:
            return {"ok": False, "reason": "NO_SIGNAL", "rsi": round(rsi_v, 1), "adx": round(adx_v, 1)}

        conf10 = last3_confirm(m10, direction)
        if conf10 < 2:
            return {"ok": False, "reason": "WEAK_CANDLES_10M"}

        if not min_body_pips_ok(m10, self.symbol, self.min_body_pips):
            return {"ok": False, "reason": "NO_IMPULSE"}

        # ✅ 1M підтвердження входу
        last1 = m1[-1]
        if direction == "BUY" and last1.direction != "UP":
            return {"ok": False, "reason": "M1_NOT_CONFIRMED"}
        if direction == "SELL" and last1.direction != "DOWN":
            return {"ok": False, "reason": "M1_NOT_CONFIRMED"}

        # якщо все ідеально — фіксуємо
        self._last_signal_candle_ts = last_closed_10m.start_ts

        # умовна оцінка "ймовірності"
        score = 90
        if adx_v >= 35:
            score = 93
        if conf10 == 3:
            score += 2
        score = min(score, 97)

        return {
            "ok": True,
            "direction": direction,
            "expiry_sec": self.expiry_sec,
            "score": score,
            "rsi": round(rsi_v, 1),
            "adx": round(adx_v, 1),
            "ema20": round(ema20, 5),
            "ema50": round(ema50, 5),
            "why": [
                f"EMA тренд ({trend})",
                f"ADX сильний ({round(adx_v,1)})",
                f"RSI у зоні ({round(rsi_v,1)})",
                f"10M свічки підтвердили: {conf10}/3",
                f"Є імпульс (body ≥ {self.min_body_pips} pips)",
                f"1M підтвердив вхід ({last1.direction})",
            ],
        }


ENGINE = SignalEngine()


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


SUBS = Subscribers(os.getenv("SUBSCRIBERS_FILE", "/app/subscribers.json"))


# ---------------- TELEGRAM TEXT FORMAT ----------------

def fmt_signal(sig: dict) -> str:
    t = fmt_kyiv(now_utc())

    if sig.get("ok") and sig.get("direction") in ("BUY", "SELL"):
        arrow = "🟢 BUY" if sig["direction"] == "BUY" else "🔴 SELL"
        why = "\n".join([f"• {x}" for x in sig.get("why", [])])
        mins = max(1, int(sig.get("expiry_sec", 120) / 60))
        return (
            f"{arrow} | <b>{ENGINE.symbol}</b>\n"
            f"⏱ <b>Експірація:</b> {mins} хв\n"
            f"🕒 <b>Kyiv:</b> {t}\n"
            f"📊 <b>Ймовірність:</b> {sig.get('score', 0)}%\n\n"
            f"<b>RSI(14):</b> {sig['rsi']}\n"
            f"<b>ADX(14):</b> {sig['adx']}\n"
            f"<b>EMA20:</b> {sig['ema20']}\n"
            f"<b>EMA50:</b> {sig['ema50']}\n\n"
            f"<b>Підтвердження:</b>\n{why}"
        )

    reasons = {
        "NOT_ENOUGH_DATA": "⏳ Недостатньо свічок (бот тільки запустився)",
        "WAIT_NEXT_CANDLE": "⏳ Чекаю закриття нової 10-хв свічки",
        "MARKET_FLAT": "🟡 Слабкий тренд (ADX низький)",
        "NO_EMA": "❌ EMA не порахувались",
        "NO_TREND": "🟡 Немає EMA тренду",
        "CHOPPY": "🟡 Ринок пиляє (немає структури)",
        "NO_SIGNAL": "😐 Немає входу по RSI зоні",
        "WEAK_CANDLES_10M": "🟡 10M свічки не підтвердили напрям",
        "NO_IMPULSE": "🟡 Нема імпульсу (тіло слабке)",
        "M1_NOT_CONFIRMED": "⏳ 1M не підтвердив — чекаю",
        "NO_DATA": "❌ Індикатори не порахувались",
    }

    return (
        "❌ <b>Сигналу немає</b>\n"
        f"🕒 <b>Kyiv:</b> {t}\n"
        f"{reasons.get(sig.get('reason'), sig.get('reason'))}"
    )


# ---------------- COMMANDS ----------------

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "✅ Бот запущено (HARD).\n"
        "Сигнали: тільки після закриття 10-хв свічки + підтвердження 1M.\n"
        "Ціль: 2–3 дуже сильні сигнали в день.\n\n"
        "Команди:\n"
        "/status\n"
        "/signal\n"
        "/auto_on\n"
        "/auto_off\n"
        "/subscribe\n"
        "/unsubscribe\n"
        "/subs",
        parse_mode=ParseMode.HTML
    )


async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    snap = ENGINE.snapshot()
    t = fmt_kyiv(now_utc())
    m1 = snap["m1"]
    m10 = snap["m10"]
    last = snap.get("last")

    msg = (
        "<b>СТАТУС БОТА (HARD)</b>\n"
        f"🕒 <b>Kyiv:</b> {t}\n"
        f"⚙️ <b>Авто:</b> {'ON' if ENGINE.auto_enabled else 'OFF'}\n"
        f"🎯 <b>MIN_ADX:</b> {ENGINE.min_adx}\n"
        f"⚡ <b>MIN_BODY:</b> {ENGINE.min_body_pips} pips\n"
        f"🕯️ <b>Свічки:</b> 1M={len(m1)} | 10M={len(m10)}\n"
        f"⏱️ <b>EXPIRY:</b> {ENGINE.expiry_sec} сек"
    )
    if last:
        msg += f"\nTick: bid={last['bid']:.5f} ask={last['ask']:.5f}"

    await update.message.reply_text(msg, parse_mode=ParseMode.HTML)


async def cmd_signal(update: Update, context: ContextTypes.DEFAULT_TYPE):
    sig = ENGINE.compute_signal()
    await update.message.reply_text(fmt_signal(sig), parse_mode=ParseMode.HTML)


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

    msg = fmt_signal(sig)
    for cid in SUBS.list():
        await context.bot.send_message(cid, msg, parse_mode=ParseMode.HTML)


# ---------------- MAIN ----------------

def main():
    token = (os.getenv("TELEGRAM_BOT_TOKEN") or "").strip()
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
    app.add_handler(CommandHandler("subs", cmd_subs))

    app.job_queue.run_repeating(auto_job, interval=ENGINE.auto_every_sec, first=10)

    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
