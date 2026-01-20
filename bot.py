import os
import json
import time
import queue
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
log = logging.getLogger("telegram_oanda_multi_bot")

KYIV_TZ = pytz.timezone("Europe/Kyiv")


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def fmt_kyiv(dt_utc: datetime) -> str:
    return dt_utc.astimezone(KYIV_TZ).strftime("%Y-%m-%d %H:%M:%S")


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


def ema(values: List[float], period: int) -> Optional[float]:
    if len(values) < period:
        return None
    k = 2 / (period + 1)
    e = values[0]
    for v in values[1:]:
        e = v * k + e * (1 - k)
    return e


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


def price_action_score(last3: List[Candle], direction: str) -> int:
    if len(last3) < 3:
        return 0

    ups = sum(1 for c in last3 if c.close > c.open)
    downs = sum(1 for c in last3 if c.close < c.open)

    if direction == "BUY":
        if ups == 3:
            return 25
        if ups == 2:
            return 15
        return 0

    if direction == "SELL":
        if downs == 3:
            return 25
        if downs == 2:
            return 15
        return 0

    return 0


def pip_size(instrument: str) -> float:
    return 0.01 if "JPY" in instrument else 0.0001


def candle_body_pips(c: Candle, instrument: str) -> float:
    return abs(c.close - c.open) / pip_size(instrument)


# ---------------- SINGLE PAIR ENGINE ----------------

class PairEngine:
    def __init__(self, instrument: str, tf_fast: int = 60, tf_slow: int = 600):
        self.instrument = instrument
        self.tf_fast = tf_fast
        self.tf_slow = tf_slow

        self.builder_fast = InternalCandleBuilder(self.tf_fast)
        self.builder_slow = InternalCandleBuilder(self.tf_slow)

        self.hist_fast = CandleHistory(maxlen=400)
        self.hist_slow = CandleHistory(maxlen=400)

        self._last_fast_ts = None
        self._last_slow_ts = None

        self.last_tick: Optional[dict] = None
        self._last_signal_candle_ts: Optional[float] = None

    def on_tick(self, item: dict):
        ts = float(item["ts"])
        mid = float(item["mid"])
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

    def compute_signal(self, min_score: int, min_adx: float, rsi_buy: float, rsi_sell: float, min_body_pips: float) -> Dict[str, Any]:
        slow = self.hist_slow.items()
        if not self.last_tick or len(slow) < 60:
            return {"ok": False, "reason": "NOT_ENOUGH_DATA", "instrument": self.instrument}

        last_closed_slow = slow[-1]
        if self._last_signal_candle_ts == last_closed_slow.start_ts:
            return {"ok": False, "reason": "WAIT_NEXT_CANDLE", "instrument": self.instrument}

        body = candle_body_pips(last_closed_slow, self.instrument)
        if body < min_body_pips:
            return {"ok": False, "reason": "LOW_IMPULSE", "instrument": self.instrument}

        closes = [c.close for c in slow]
        highs = [c.high for c in slow]
        lows = [c.low for c in slow]

        rsi_v = rsi(closes, 14)
        adx_v = adx(highs, lows, closes, 14)

        if rsi_v is None or adx_v is None:
            return {"ok": False, "reason": "NO_DATA", "instrument": self.instrument}

        if adx_v < min_adx:
            return {"ok": False, "reason": "MARKET_FLAT", "instrument": self.instrument}

        ema20 = ema(closes[-120:], 20)
        ema50 = ema(closes[-160:], 50)
        if ema20 is None or ema50 is None:
            return {"ok": False, "reason": "NO_EMA", "instrument": self.instrument}

        trend = "BUY" if ema20 > ema50 else "SELL" if ema20 < ema50 else None
        if trend is None:
            return {"ok": False, "reason": "NO_TREND", "instrument": self.instrument}

        direction = None
        if trend == "BUY" and rsi_v >= rsi_buy:
            direction = "BUY"
        elif trend == "SELL" and rsi_v <= rsi_sell:
            direction = "SELL"
        else:
            return {"ok": False, "reason": "NO_SIGNAL", "instrument": self.instrument}

        score = 0
        parts = []

        score += 25
        parts.append("ADX сильний")

        score += 25
        parts.append("EMA тренд підтвердив")

        score += 25
        parts.append("RSI екстремум по тренду")

        pa = price_action_score(slow[-3:], direction)
        score += pa
        if pa >= 25:
            parts.append("3/3 свічки за напрямком")
        elif pa >= 15:
            parts.append("2/3 свічки за напрямком")
        else:
            parts.append("Свічки слабко підтвердили")

        if score < min_score:
            return {
                "ok": False,
                "reason": "WEAK_SCORE",
                "instrument": self.instrument,
                "score": score,
                "rsi": round(rsi_v, 1),
                "adx": round(adx_v, 1),
            }

        self._last_signal_candle_ts = last_closed_slow.start_ts

        return {
            "ok": True,
            "instrument": self.instrument,
            "direction": direction,
            "expiry_sec": 600,
            "score": score,
            "why": parts,
            "rsi": round(rsi_v, 1),
            "adx": round(adx_v, 1),
            "ema20": round(ema20, 5),
            "ema50": round(ema50, 5),
        }


# ---------------- MULTI SIGNAL ENGINE ----------------

class MultiSignalEngine:
    def __init__(self):
        self.auto_enabled = os.getenv("AUTO_ENABLED", "true").lower() == "true"
        self.auto_every_sec = int(os.getenv("AUTO_EVERY_SEC", "15"))

        self.instruments = ["EUR_USD", "GBP_USD", "USD_JPY"]

        self.min_adx = float(os.getenv("MIN_ADX", "30"))
        self.rsi_buy = float(os.getenv("RSI_BUY", "68"))
        self.rsi_sell = float(os.getenv("RSI_SELL", "32"))
        self.min_score = int(os.getenv("MIN_SCORE", "88"))
        self.min_body_pips = float(os.getenv("MIN_BODY_PIPS", "2.0"))

        self.cooldown_after_losses = int(os.getenv("COOLDOWN_AFTER_LOSSES", "2"))
        self.cooldown_minutes = int(os.getenv("COOLDOWN_MINUTES", "30"))
        self.consecutive_losses = 0
        self.cooldown_until_ts: Optional[float] = None

        self._q = queue.Queue(maxsize=20000)
        self._lock = threading.Lock()

        self.pairs: Dict[str, PairEngine] = {ins: PairEngine(ins) for ins in self.instruments}
        self._streams: List[OandaPriceStream] = []

    def start_streams(self):
        api_key = (os.getenv("OANDA_API_KEY") or "").strip()
        account_id = (os.getenv("OANDA_ACCOUNT_ID") or "").strip()
        env = (os.getenv("OANDA_ENV") or "practice").lower()

        if not api_key or not account_id:
            raise RuntimeError("OANDA_API_KEY / OANDA_ACCOUNT_ID missing")

        practice = env == "practice"

        for ins in self.instruments:
            stream = OandaPriceStream(api_key, account_id, ins, self._q, practice=practice)
            stream.start()
            self._streams.append(stream)

        threading.Thread(target=self._pump_ticks, daemon=True).start()

    def _pump_ticks(self):
        while True:
            item = self._q.get()
            ins = item.get("instrument")
            if ins not in self.pairs:
                continue
            with self._lock:
                self.pairs[ins].on_tick(item)

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            data = {}
            for ins, pe in self.pairs.items():
                data[ins] = {
                    "last": pe.last_tick,
                    "slow": pe.hist_slow.items()
                }
            return data

    def compute_best_signal(self) -> Dict[str, Any]:
        now_ts = time.time()
        if self.cooldown_until_ts and now_ts < self.cooldown_until_ts:
            return {"ok": False, "reason": "COOLDOWN"}

        with self._lock:
            candidates = []
            for ins, pe in self.pairs.items():
                sig = pe.compute_signal(
                    self.min_score,
                    self.min_adx,
                    self.rsi_buy,
                    self.rsi_sell,
                    self.min_body_pips
                )
                if sig.get("ok"):
                    candidates.append(sig)

            if not candidates:
                return {"ok": False, "reason": "NO_BEST_SIGNAL"}

            candidates.sort(key=lambda x: x.get("score", 0), reverse=True)
            return candidates[0]

    def record_win(self):
        self.consecutive_losses = 0

    def record_loss(self):
        self.consecutive_losses += 1
        if self.consecutive_losses >= self.cooldown_after_losses:
            self.cooldown_until_ts = time.time() + self.cooldown_minutes * 60
            self.consecutive_losses = 0


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


ENGINE = MultiSignalEngine()
SUBS = Subscribers(os.getenv("SUBSCRIBERS_FILE", "/app/subscribers.json"))


# ---------------- TELEGRAM TEXT FORMAT ----------------

def fmt_signal(sig: dict) -> str:
    t = fmt_kyiv(now_utc())

    if sig.get("reason") == "COOLDOWN":
        return (
            "⏸ <b>Пауза</b>\n"
            f"🕒 <b>Kyiv:</b> {t}\n"
            "Пауза після серії мінусів (захист депозиту)."
        )

    if sig.get("ok") and sig.get("direction") in ("BUY", "SELL"):
        arrow = "🟢 BUY" if sig["direction"] == "BUY" else "🔴 SELL"
        why = "\n".join([f"• {x}" for x in sig.get("why", [])])
        ins = sig.get("instrument", "UNKNOWN")
        return (
            f"{arrow} | <b>{ins}</b>\n"
            f"⏱ <b>Експірація:</b> 10 хв\n"
            f"🕒 <b>Kyiv:</b> {t}\n"
            f"📊 <b>Ймовірність:</b> {sig.get('score', 0)}%\n\n"
            f"<b>RSI(14):</b> {sig['rsi']}\n"
            f"<b>ADX(14):</b> {sig['adx']}\n"
            f"<b>EMA20:</b> {sig['ema20']}\n"
            f"<b>EMA50:</b> {sig['ema50']}\n\n"
            f"<b>Підтвердження:</b>\n{why}"
        )

    reasons = {
        "NO_BEST_SIGNAL": "Сильного сетапу немає (Score < 88%)",
        "LOW_IMPULSE": "Слабкий імпульс (свічка дуже маленька) — пропускаю",
    }

    return (
        "❌ <b>Сигналу немає</b>\n"
        f"🕒 <b>Kyiv:</b> {t}\n"
        f"{reasons.get(sig.get('reason'), 'Немає сильного сигналу')}"
    )


# ---------------- COMMANDS ----------------

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "✅ Бот запущено (3 пари).\n"
        "Сигнали: тільки після закриття 10-хв свічки.\n"
        "Фільтр: Score ≥ 88%, ADX ≥ 30, імпульс-фільтр.\n"
        "Є захист депозиту: пауза після 2 мінусів.\n\n"
        "Пари: EUR_USD, GBP_USD, USD_JPY\n\n"
        "Команди:\n"
        "/status\n"
        "/signal\n"
        "/auto_on\n"
        "/auto_off\n"
        "/subscribe\n"
        "/unsubscribe\n"
        "/subs\n"
        "/win\n"
        "/loss"
    )


async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    snap = ENGINE.snapshot()
    t = fmt_kyiv(now_utc())

    lines = [
        "<b>СТАТУС БОТА</b>",
        f"🕒 <b>Kyiv:</b> {t}",
        f"⚙️ <b>Авто:</b> {'ON' if ENGINE.auto_enabled else 'OFF'}",
        f"🎯 <b>Поріг сигналу:</b> {ENGINE.min_score}%",
        f"📈 <b>MIN_ADX:</b> {ENGINE.min_adx}",
        f"⚡ <b>MIN_BODY:</b> {ENGINE.min_body_pips} pips",
        ""
    ]

    for ins, d in snap.items():
        n = len(d.get("slow", []))
        lines.append(f"🕯️ <b>{ins}:</b> 10-хв свічок = {n}")

    await update.message.reply_text("\n".join(lines), parse_mode=ParseMode.HTML)


async def cmd_signal(update: Update, context: ContextTypes.DEFAULT_TYPE):
    sig = ENGINE.compute_best_signal()
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


async def cmd_win(update: Update, context: ContextTypes.DEFAULT_TYPE):
    ENGINE.record_win()
    await update.message.reply_text("✅ Записано: ПЛЮС")


async def cmd_loss(update: Update, context: ContextTypes.DEFAULT_TYPE):
    ENGINE.record_loss()
    if ENGINE.cooldown_until_ts and time.time() < ENGINE.cooldown_until_ts:
        await update.message.reply_text(f"❌ Записано: МІНУС\n⏸ Пауза {ENGINE.cooldown_minutes} хв")
    else:
        await update.message.reply_text("❌ Записано: МІНУС")


# ---------------- AUTO JOB ----------------

async def auto_job(context: ContextTypes.DEFAULT_TYPE):
    if not ENGINE.auto_enabled:
        return

    sig = ENGINE.compute_best_signal()
    if not sig.get("ok"):
        return

    msg = fmt_signal(sig)
    for cid in SUBS.list():
        await context.bot.send_message(cid, msg, parse_mode=ParseMode.HTML)


# ---------------- MAIN ----------------

def acquire_lock(lock_path: str):
    if os.path.exists(lock_path):
        try:
            with open(lock_path, "r", encoding="utf-8") as f:
                old_pid = int((f.read() or "").strip())
            os.kill(old_pid, 0)
            log.error("Bot already running with PID=%s", old_pid)
            raise SystemExit(0)
        except ProcessLookupError:
            try:
                os.remove(lock_path)
            except Exception:
                pass
        except Exception:
            try:
                os.remove(lock_path)
            except Exception:
                pass

    with open(lock_path, "w", encoding="utf-8") as f:
        f.write(str(os.getpid()))


def main():
    token = (os.getenv("TELEGRAM_BOT_TOKEN") or "").strip()
    if not token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN missing")

    lock_path = "/tmp/telegram_bot.lock"
    acquire_lock(lock_path)

    try:
        ENGINE.start_streams()

        app = Application.builder().token(token).build()

        app.add_handler(CommandHandler("start", cmd_start))
        app.add_handler(CommandHandler("status", cmd_status))
        app.add_handler(CommandHandler("signal", cmd_signal))
        app.add_handler(CommandHandler("auto_on", cmd_auto_on))
        app.add_handler(CommandHandler("auto_off", cmd_auto_off))
        app.add_handler(CommandHandler("subscribe", cmd_subscribe))
        app.add_handler(CommandHandler("unsubscribe", cmd_unsubscribe))
        app.add_handler(CommandHandler("subs", cmd_subs))
        app.add_handler(CommandHandler("win", cmd_win))
        app.add_handler(CommandHandler("loss", cmd_loss))

        app.job_queue.run_repeating(auto_job, interval=ENGINE.auto_every_sec, first=10)

        app.run_polling(drop_pending_updates=True)
    finally:
        try:
            if os.path.exists(lock_path):
                os.remove(lock_path)
        except Exception:
            pass


if __name__ == "__main__":
    main()
