import os
import json
import time
import math
import queue
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, List, Dict
from collections import deque

import requests
from dotenv import load_dotenv
import pytz

from telegram import Update, InlineKeyboardMarkup, InlineKeyboardButton
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes

load_dotenv()

# ============== TIMEZONE ==============
KYIV = pytz.timezone("Europe/Kyiv")

def now():
    return datetime.now(timezone.utc).astimezone(KYIV)

def now_str():
    return now().strftime("%H:%M:%S")

# ============== TRADING SESSIONS ==============
def get_session():
    """Always active session"""
    return "ACTIVE"


def is_trading_time():
    """Trading always allowed"""
    return True


# High-impact news times (Kyiv time) - avoid 15 min before/after
NEWS_TIMES = ["15:30", "16:00", "17:00", "11:00"]

def is_news_time():
    """Avoid trading around news releases"""
    n = now()
    for t in NEWS_TIMES:
        hh, mm = map(int, t.split(":"))
        news = n.replace(hour=hh, minute=mm, second=0, microsecond=0)
        diff = abs((n - news).total_seconds()) / 60
        if diff <= 15:
            return True
    return False

# ============== UTILITY FUNCTIONS ==============
def mean(x):
    return sum(x) / len(x) if x else 0

def stdev(x):
    if not x or len(x) < 2:
        return 0
    m = mean(x)
    return math.sqrt(sum((i - m) ** 2 for i in x) / (len(x) - 1))

def pip_value(symbol):
    return 0.01 if "JPY" in symbol else 0.0001

# ============== STRATEGY PARAMETERS ==============
# RSI Settings
RSI_PERIOD = 14
RSI_OVERBOUGHT = 72
RSI_OVERSOLD = 32
RSI_STRONG_OB = 81
RSI_STRONG_OS = 22

# MACD Settings
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

# Bollinger Bands
BB_PERIOD = 20
BB_STD = 2.0

# ATR Filter
ATR_PERIOD = 14
ATR_MIN_MULTIPLIER = 0.5
ATR_MAX_MULTIPLIER = 2.5

# Confluence Requirements
MIN_CONFLUENCE_SCORE = 4  # Out of 6 indicators
MIN_PROBABILITY = 0.72

# ============== MONEY MANAGEMENT ==============
INITIAL_BALANCE = 3000
BALANCE = INITIAL_BALANCE
RISK_PERCENT = 0.04  # 5% per trade (conservative)
MAX_RISK_PERCENT = 0.9  # Max 10% on high-confidence trades
PAYOUT = 0.80

STATE = {
    "wins": 0,
    "losses": 0,
    "streak": 0,
    "daily_trades": 0,
    "daily_pnl": 0,
    "last_trade_day": None
}

def reset_daily_stats():
    """Reset daily statistics"""
    today = now().date()
    if STATE["last_trade_day"] != today:
        STATE["daily_trades"] = 0
        STATE["daily_pnl"] = 0
        STATE["last_trade_day"] = today

def get_bet(confidence: float):
    """Calculate bet size based on Kelly Criterion modified"""
    reset_daily_stats()
    
    # Base risk
    base_risk = RISK_PERCENT
    
    # Increase risk for high confidence
    if confidence >= 0.80:
        base_risk = MAX_RISK_PERCENT
    elif confidence >= 0.75:
        base_risk = 0.07
    
    # Reduce risk after losses
    if STATE["streak"] < 0:
        base_risk *= max(0.5, 1 + STATE["streak"] * 0.1)
    
    bet = round(BALANCE * base_risk, 2)
    return max(10, min(bet, BALANCE * 0.15))  # Min $10, Max 15% of balance

def risk_block():
    """Check if trading should be blocked"""
    reset_daily_stats()
    
    profit_pct = (BALANCE - INITIAL_BALANCE) / INITIAL_BALANCE
    
    # Daily loss limit: -10%
    if STATE["daily_pnl"] <= -INITIAL_BALANCE * 0.10:
        return "⛔ ДЕННИЙ ЛІМІТ ВТРАТ"
    
    # Daily profit target: +15%
    if STATE["daily_pnl"] >= INITIAL_BALANCE * 0.15:
        return "💰 ДЕННА ЦІЛЬ ДОСЯГНУТА"
    
    # Max 6 trades per day
    if STATE["daily_trades"] >= 6:
        return "⛔ ЛІМІТ УГОД НА ДЕНЬ"
    
    # Losing streak protection
    if STATE["streak"] <= -3:
        return "⛔ СТОП (серія втрат)"
    
    # Overall drawdown protection
    if profit_pct <= -0.25:
        return "⛔ КРИТИЧНИЙ DRAWDOWN"
    
    return None

# ============== CANDLE DATA ==============
@dataclass
class Candle:
    t: float
    o: float
    h: float
    l: float
    c: float
    v: float = 0  # tick volume
    
    def update(self, p):
        self.c = p
        self.h = max(self.h, p)
        self.l = min(self.l, p)
        self.v += 1
    
    @property
    def body(self):
        return abs(self.c - self.o)
    
    @property
    def upper_wick(self):
        return self.h - max(self.o, self.c)
    
    @property
    def lower_wick(self):
        return min(self.o, self.c) - self.l
    
    @property
    def range(self):
        return self.h - self.l
    
    @property
    def is_bullish(self):
        return self.c > self.o
    
    @property
    def is_bearish(self):
        return self.c < self.o
    
    @property
    def is_doji(self):
        return self.body < self.range * 0.1 if self.range > 0 else True

class CandleBuilder:
    def __init__(self, timeframe_seconds):
        self.tf = timeframe_seconds
        self.current: Optional[Candle] = None
        self.completed: Optional[Candle] = None
    
    def bucket(self, ts):
        return ts - ts % self.tf
    
    def tick(self, ts, price):
        b = self.bucket(ts)
        
        if not self.current:
            self.current = Candle(b, price, price, price, price)
            return False
        
        if self.current.t == b:
            self.current.update(price)
            return False
        else:
            self.completed = self.current
            self.current = Candle(b, price, price, price, price)
            return True  # New candle completed

# ============== TECHNICAL INDICATORS ==============
def ema(prices: List[float], period: int) -> Optional[float]:
    """Exponential Moving Average"""
    if len(prices) < period:
        return None
    k = 2 / (period + 1)
    ema_val = prices[0]
    for price in prices[1:]:
        ema_val = price * k + ema_val * (1 - k)
    return ema_val

def sma(prices: List[float], period: int) -> Optional[float]:
    """Simple Moving Average"""
    if len(prices) < period:
        return None
    return mean(prices[-period:])

def rsi(prices: List[float], period: int = RSI_PERIOD) -> Optional[float]:
    """Relative Strength Index"""
    if len(prices) < period + 1:
        return None
    
    gains = []
    losses = []
    
    for i in range(1, len(prices)):
        change = prices[i] - prices[i-1]
        if change > 0:
            gains.append(change)
            losses.append(0)
        else:
            gains.append(0)
            losses.append(abs(change))
    
    if len(gains) < period:
        return None
    
    avg_gain = mean(gains[-period:])
    avg_loss = mean(losses[-period:])
    
    if avg_loss == 0:
        return 100
    
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def macd(prices: List[float]) -> Optional[Dict]:
    """MACD with signal and histogram"""
    if len(prices) < MACD_SLOW + MACD_SIGNAL:
        return None
    
    ema_fast = ema(prices, MACD_FAST)
    ema_slow = ema(prices, MACD_SLOW)
    
    if ema_fast is None or ema_slow is None:
        return None
    
    macd_line = ema_fast - ema_slow
    
    # Calculate MACD history for signal line
    macd_history = []
    for i in range(MACD_SLOW, len(prices) + 1):
        ef = ema(prices[:i], MACD_FAST)
        es = ema(prices[:i], MACD_SLOW)
        if ef and es:
            macd_history.append(ef - es)
    
    if len(macd_history) < MACD_SIGNAL:
        return None
    
    signal_line = ema(macd_history, MACD_SIGNAL)
    
    if signal_line is None:
        return None
    
    histogram = macd_line - signal_line
    
    # Previous values for crossover detection
    prev_macd = macd_history[-2] if len(macd_history) >= 2 else macd_line
    prev_signal = ema(macd_history[:-1], MACD_SIGNAL) if len(macd_history) > MACD_SIGNAL else signal_line
    prev_histogram = prev_macd - prev_signal if prev_signal else 0
    
    return {
        "macd": macd_line,
        "signal": signal_line,
        "histogram": histogram,
        "prev_histogram": prev_histogram,
        "crossover_up": prev_macd < prev_signal and macd_line > signal_line,
        "crossover_down": prev_macd > prev_signal and macd_line < signal_line
    }

def bollinger_bands(prices: List[float], period: int = BB_PERIOD, std_mult: float = BB_STD) -> Optional[Dict]:
    """Bollinger Bands"""
    if len(prices) < period:
        return None
    
    middle = sma(prices, period)
    if middle is None:
        return None
    
    std = stdev(prices[-period:])
    upper = middle + std_mult * std
    lower = middle - std_mult * std
    
    current_price = prices[-1]
    
    # Bandwidth for squeeze detection
    bandwidth = (upper - lower) / middle if middle != 0 else 0
    
    # %B indicator
    percent_b = (current_price - lower) / (upper - lower) if upper != lower else 0.5
    
    return {
        "upper": upper,
        "middle": middle,
        "lower": lower,
        "bandwidth": bandwidth,
        "percent_b": percent_b,
        "squeeze": bandwidth < 0.02,  # Squeeze condition
        "at_upper": current_price >= upper,
        "at_lower": current_price <= lower
    }

def atr(highs: List[float], lows: List[float], closes: List[float], period: int = ATR_PERIOD) -> Optional[float]:
    """Average True Range"""
    if len(closes) < period + 1:
        return None
    
    true_ranges = []
    for i in range(1, len(closes)):
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i-1]),
            abs(lows[i] - closes[i-1])
        )
        true_ranges.append(tr)
    
    if len(true_ranges) < period:
        return None
    
    return mean(true_ranges[-period:])

def support_resistance(candles: List[Candle], lookback: int = 50) -> Dict:
    """Detect support and resistance levels"""
    if len(candles) < lookback:
        return {"support": [], "resistance": []}
    
    recent = candles[-lookback:]
    
    highs = [c.h for c in recent]
    lows = [c.l for c in recent]
    
    # Find swing highs and lows
    resistance_levels = []
    support_levels = []
    
    for i in range(2, len(recent) - 2):
        # Swing high
        if highs[i] > highs[i-1] and highs[i] > highs[i-2] and \
           highs[i] > highs[i+1] and highs[i] > highs[i+2]:
            resistance_levels.append(highs[i])
        
        # Swing low
        if lows[i] < lows[i-1] and lows[i] < lows[i-2] and \
           lows[i] < lows[i+1] and lows[i] < lows[i+2]:
            support_levels.append(lows[i])
    
    return {
        "support": sorted(support_levels)[-3:] if support_levels else [],
        "resistance": sorted(resistance_levels)[:3] if resistance_levels else []
    }

def detect_candlestick_pattern(candles: List[Candle]) -> Optional[Dict]:
    """Detect candlestick patterns"""
    if len(candles) < 3:
        return None
    
    c1 = candles[-3]  # Oldest
    c2 = candles[-2]  # Middle
    c3 = candles[-1]  # Current
    
    patterns = []
    
    # Bullish Engulfing
    if c2.is_bearish and c3.is_bullish:
        if c3.o < c2.c and c3.c > c2.o:
            patterns.append({"name": "BULLISH_ENGULFING", "direction": "BUY", "strength": 0.8})
    
    # Bearish Engulfing
    if c2.is_bullish and c3.is_bearish:
        if c3.o > c2.c and c3.c < c2.o:
            patterns.append({"name": "BEARISH_ENGULFING", "direction": "SELL", "strength": 0.8})
    
    # Bullish Pin Bar (Hammer)
    if c3.lower_wick > c3.body * 2 and c3.upper_wick < c3.body * 0.5:
        patterns.append({"name": "HAMMER", "direction": "BUY", "strength": 0.7})
    
    # Bearish Pin Bar (Shooting Star)
    if c3.upper_wick > c3.body * 2 and c3.lower_wick < c3.body * 0.5:
        patterns.append({"name": "SHOOTING_STAR", "direction": "SELL", "strength": 0.7})
    
    # Morning Star (Bullish reversal)
    if c1.is_bearish and c2.is_doji and c3.is_bullish:
        if c3.c > (c1.o + c1.c) / 2:
            patterns.append({"name": "MORNING_STAR", "direction": "BUY", "strength": 0.85})
    
    # Evening Star (Bearish reversal)
    if c1.is_bullish and c2.is_doji and c3.is_bearish:
        if c3.c < (c1.o + c1.c) / 2:
            patterns.append({"name": "EVENING_STAR", "direction": "SELL", "strength": 0.85})
    
    # Three White Soldiers
    if all(c.is_bullish for c in [c1, c2, c3]):
        if c2.c > c1.c and c3.c > c2.c:
            patterns.append({"name": "THREE_WHITE_SOLDIERS", "direction": "BUY", "strength": 0.75})
    
    # Three Black Crows
    if all(c.is_bearish for c in [c1, c2, c3]):
        if c2.c < c1.c and c3.c < c2.c:
            patterns.append({"name": "THREE_BLACK_CROWS", "direction": "SELL", "strength": 0.75})
    
    return patterns[0] if patterns else None

def detect_divergence(prices: List[float], rsi_values: List[float]) -> Optional[str]:
    """Detect RSI divergence"""
    if len(prices) < 10 or len(rsi_values) < 10:
        return None
    
    # Compare last 5 candles with previous 5
    recent_price_high = max(prices[-5:])
    prev_price_high = max(prices[-10:-5])
    recent_price_low = min(prices[-5:])
    prev_price_low = min(prices[-10:-5])
    
    recent_rsi_high = max(rsi_values[-5:])
    prev_rsi_high = max(rsi_values[-10:-5])
    recent_rsi_low = min(rsi_values[-5:])
    prev_rsi_low = min(rsi_values[-10:-5])
    
    # Bullish divergence: price lower low, RSI higher low
    if recent_price_low < prev_price_low and recent_rsi_low > prev_rsi_low:
        return "BULLISH_DIVERGENCE"
    
    # Bearish divergence: price higher high, RSI lower high
    if recent_price_high > prev_price_high and recent_rsi_high < prev_rsi_high:
        return "BEARISH_DIVERGENCE"
    
    return None

# ============== TRADING ENGINE ==============
class TradingEngine:
    def __init__(self, symbol):
        self.symbol = symbol
        self.queue = queue.Queue()

    def start(self):
        threading.Thread(target=self._stream, daemon=True).start()

    def _stream(self):
        url = f"https://stream-fxpractice.oanda.com/v3/accounts/{os.getenv('OANDA_ACCOUNT_ID')}/pricing/stream"
        headers = {"Authorization": f"Bearer {os.getenv('OANDA_API_KEY')}"}
        params = {"instruments": self.symbol}

        while True:
            try:
                r = requests.get(
                    url,
                    headers=headers,
                    params=params,
                    stream=True
                )

                if r.status_code != 200:
                    print(f"HTTP Error {r.status_code}")
                    time.sleep(5)
                    continue

                last_tick = time.time()

                for line in r.iter_lines():
                    if line:
                        last_tick = time.time()
                    else:
                        continue

                    try:
                        decoded = line.decode().strip()
                        if not decoded:
                            continue

                        data = json.loads(decoded)
                    except Exception:
                        continue

                    if data.get("type") == "PRICE":
                        bid = float(data["bids"][0]["price"])
                        ask = float(data["asks"][0]["price"])
                        mid = (bid + ask) / 2
                        self.queue.put((time.time(), mid))

                    if time.time() - last_tick > 60:
                        print(f"Stream stale → reconnect {self.symbol}")
                        break

            except Exception as e:
                print(f"ERROR {self.symbol}: {e}")
                print(f"Reconnect {self.symbol}...")
                time.sleep(5)

# ============== BOT SETUP ==============
SYMBOLS = ["EUR_USD", "GBP_USD", "USD_JPY"]
ENGINES = [TradingEngine(s) for s in SYMBOLS]

# Start all engines
for engine in ENGINES:
    engine.start()

LAST_SIGNAL = None

def get_best_signal() -> Optional[Dict]:
    """Get the best signal from all pairs"""
    global LAST_SIGNAL
    
    best = None
    for engine in ENGINES:
        signal = engine.generate_signal()
        if signal:
            if not best or signal["probability"] > best["probability"]:
                best = signal
    
    LAST_SIGNAL = best
    return best

def record_result(win: bool):
    """Record trade result"""
    global BALANCE
    
    bet = get_bet(LAST_SIGNAL["probability"] / 100 if LAST_SIGNAL else 0.75)
    
    if win:
        profit = bet * PAYOUT
        BALANCE += profit
        STATE["wins"] += 1
        STATE["streak"] = max(1, STATE["streak"] + 1)
        STATE["daily_pnl"] += profit
    else:
        BALANCE -= bet
        STATE["losses"] += 1
        STATE["streak"] = min(-1, STATE["streak"] - 1)
        STATE["daily_pnl"] -= bet
    
    STATE["daily_trades"] += 1

def get_stats() -> str:
    """Get trading statistics"""
    total = STATE["wins"] + STATE["losses"]
    win_rate = (STATE["wins"] / total * 100) if total > 0 else 0
    profit_pct = ((BALANCE - INITIAL_BALANCE) / INITIAL_BALANCE) * 100

    return f"""📊 СТАТИСТИКА

______________

💰 Баланс: ${BALANCE:.2f}
📈 P/L: {profit_pct:+.1f}%
✅ Виграші: {STATE['wins']}
❌ Програші: {STATE['losses']}
🎯 Win Rate: {win_rate:.1f}%
🔥 Серія: {STATE['streak']}
📊 Угод сьогодні: {STATE['daily_trades']}
"""


def format_signal(signal: Optional[Dict]) -> str:
    """Format signal for display"""
    block = risk_block()
    if block:
        return f"{block}\n\n{get_stats()}"
    
    if not signal:
        session = get_session()
        status = "🟢 АКТИВНА" if is_trading_time() else "🔴 НЕАКТИВНА"
        news = "⚠️ НОВИНИ" if is_news_time() else "✅ Чисто"
        
        return (
            f"❌ Немає сигналу\n"
            f"━━━━━━━━━━━━━━━\n"
            f"🕒 {now_str()}\n"
            f"📍 Сесія: {session} {status}\n"
            f"📰 Новини: {news}\n\n"
            f"Очікую якісний сетап..."
        )
    
    direction_emoji = "🟢" if signal["direction"] == "BUY" else "🔴"
    confidence = "🔥🔥🔥" if signal["probability"] >= 80 else "🔥🔥" if signal["probability"] >= 75 else "🔥"
    
    reasons_text = "\n".join(f"  • {r}" for r in signal["reasons"][:4])
    bet = get_bet(signal["probability"] / 100)
    
    return (
        f"{direction_emoji} {signal['direction']} {signal['symbol'].replace('_', '/')}\n"
        f"━━━━━━━━━━━━━━━\n"
        f"📊 Ймовірність: {signal['probability']}% {confidence}\n"
        f"🎯 Confluence: {signal['confluence_score']}/6\n"
        f"💵 Ставка: ${bet}\n"
        f"⏱ Експірація: 2 хв\n"
        f"💰 Баланс: ${BALANCE:.2f}\n"
        f"━━━━━━━━━━━━━━━\n"
        f"📋 Причини:\n{reasons_text}\n"
        f"━━━━━━━━━━━━━━━\n"
        f"RSI: {signal.get('rsi', 'N/A')} | ATR: {signal.get('atr_pips', 'N/A')} pips\n"
        f"🕒 {signal['time']}"
    )


def keyboard():
    """Create result keyboard"""
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("✅ WIN", callback_data="win"),
            InlineKeyboardButton("❌ LOSS", callback_data="loss")
        ]
    ])


# ============== TELEGRAM HANDLERS ==============

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🤖 FOREX TRADING BOT\n"
        "━━━━━━━━━━━━━━━\n"
        "Multi-Indicator Confluence Strategy\n"
        "Pairs: EUR/USD, GBP/USD, USD/JPY\n"
        "━━━━━━━━━━━━━━━\n"
        "Commands:\n"
        "/signal - Get current signal\n"
        "/stats - View statistics\n"
        "/auto - Start auto signals\n"
        "/stop - Stop auto signals"
    )


async def cmd_signal(update: Update, context: ContextTypes.DEFAULT_TYPE):
    signal = get_best_signal()
    await update.message.reply_text(
        format_signal(signal),
        reply_markup=keyboard() if signal else None
    )


async def cmd_stats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(get_stats())


async def callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    
    record_result(query.data == "win")
    result = "✅ WIN записано" if query.data == "win" else "❌ LOSS записано"
    
    await query.edit_message_text(f"{result}\n\n{get_stats()}")


async def auto_signal(context: ContextTypes.DEFAULT_TYPE):
    """Auto signal job"""
    if not is_trading_time() or is_news_time():
        return
    
    signal = get_best_signal()
    if signal:
        await context.bot.send_message(
            context.job.chat_id,
            format_signal(signal),
            reply_markup=keyboard()
        )


async def cmd_auto(update: Update, context: ContextTypes.DEFAULT_TYPE):
    current_jobs = context.job_queue.get_jobs_by_name(f"auto_{update.effective_chat.id}")
    
    for job in current_jobs:
        job.schedule_removal()
    
    context.job_queue.run_repeating(
        auto_signal,
        interval=120,
        first=10,
        chat_id=update.effective_chat.id,
        name=f"auto_{update.effective_chat.id}"
    )
    
    await update.message.reply_text("✅ Авто-сигнали увімкнено")


async def cmd_stop(update: Update, context: ContextTypes.DEFAULT_TYPE):
    current_jobs = context.job_queue.get_jobs_by_name(f"auto_{update.effective_chat.id}")
    
    for job in current_jobs:
        job.schedule_removal()
    
    await update.message.reply_text("🛑 Авто-сигнали вимкнено")

async def cmd_signal(update: Update, context: ContextTypes.DEFAULT_TYPE):
    signal = get_best_signal()
    await update.message.reply_text(format_signal(signal), reply_markup=keyboard() if signal else None)

async def cmd_stats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(get_stats())

async def callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    
    record_result(query.data == "win")
    result = "✅ WIN записано" if query.data == "win" else "❌ LOSS записано"
    await query.edit_message_text(f"{result}\n\n{get_stats()}")

async def auto_signal(context: ContextTypes.DEFAULT_TYPE):
    """Auto signal job"""
    if not is_trading_time() or is_news_time():
        return
    
    signal = get_best_signal()
    if signal:
        await context.bot.send_message(
            context.job.chat_id,
            format_signal(signal),
            reply_markup=keyboard()
        )

async def cmd_auto(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Remove existing jobs
    current_jobs = context.job_queue.get_jobs_by_name(f"auto_{update.effective_chat.id}")
    for job in current_jobs:
        job.schedule_removal()
    
    context.job_queue.run_repeating(
        auto_signal,
        interval=120,  # Check every 2 minutes
        first=10,
        chat_id=update.effective_chat.id,
        name=f"auto_{update.effective_chat.id}"
    )
    await update.message.reply_text("✅ Авто-сигнали увімкнено")

async def cmd_stop(update: Update, context: ContextTypes.DEFAULT_TYPE):
    current_jobs = context.job_queue.get_jobs_by_name(f"auto_{update.effective_chat.id}")
    for job in current_jobs:
        job.schedule_removal()
    await update.message.reply_text("🛑 Авто-сигнали вимкнено")

def main():
    """Main entry point"""
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        print("ERROR: TELEGRAM_BOT_TOKEN not set")
        return
    
    app = Application.builder().token(token).build()
    
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("signal", cmd_signal))
    app.add_handler(CommandHandler("stats", cmd_stats))
    app.add_handler(CommandHandler("auto", cmd_auto))
    app.add_handler(CommandHandler("stop", cmd_stop))
    app.add_handler(CallbackQueryHandler(callback_handler))
    
    print("🚀 Bot started...")
    app.run_polling(drop_pending_updates=True)

if __name__ == "__main__":
    main()
