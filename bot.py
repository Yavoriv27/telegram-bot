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
KYIV = pytz.timezone(\"Europe/Kyiv\")

def now():
    return datetime.now(timezone.utc).astimezone(KYIV)

def now_str():
    return now().strftime(\"%H:%M:%S\")

# ============== TRADING SESSIONS ==============
def get_session():
    \"\"\"Identify current trading session\"\"\"
    n = now()
    h = n.hour
    
    # London Session: 10:00-18:00 Kyiv (best for EUR pairs)
    if 10 <= h < 18:
        return \"LONDON\"
    # New York overlap: 15:30-18:00 Kyiv (highest volatility)
    if 15 <= h < 18:
        return \"NY_OVERLAP\"
    # Asian Session: 02:00-10:00 Kyiv (best for JPY)
    if 2 <= h < 10:
        return \"ASIAN\"
    return \"OFF\"

def is_trading_time():
    \"\"\"Only trade during optimal sessions\"\"\"
    n = now()
    h = n.hour
    m = n.minute
    weekday = n.weekday()
    
    # No trading on weekends
    if weekday >= 5:
        return False
    
    # London session: 10:00-12:30 and 14:00-17:30
    if 10 <= h < 12:
        return True
    if h == 12 and m < 30:
        return True
    if 14 <= h < 17:
        return True
    if h == 17 and m < 30:
        return True
    
    return False

# High-impact news times (Kyiv time) - avoid 15 min before/after
NEWS_TIMES = [\"15:30\", \"16:00\", \"17:00\", \"11:00\"]

def is_news_time():
    \"\"\"Avoid trading around news releases\"\"\"
    n = now()
    for t in NEWS_TIMES:
        hh, mm = map(int, t.split(\":\"))
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
    return 0.01 if \"JPY\" in symbol else 0.0001

# ============== STRATEGY PARAMETERS ==============
# RSI Settings
RSI_PERIOD = 14
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30
RSI_STRONG_OB = 80
RSI_STRONG_OS = 20

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
RISK_PERCENT = 0.05  # 5% per trade (conservative)
MAX_RISK_PERCENT = 0.10  # Max 10% on high-confidence trades
PAYOUT = 0.80

STATE = {
    \"wins\": 0,
    \"losses\": 0,
    \"streak\": 0,
    \"daily_trades\": 0,
    \"daily_pnl\": 0,
    \"last_trade_day\": None
}

def reset_daily_stats():
    \"\"\"Reset daily statistics\"\"\"
    today = now().date()
    if STATE[\"last_trade_day\"] != today:
        STATE[\"daily_trades\"] = 0
        STATE[\"daily_pnl\"] = 0
        STATE[\"last_trade_day\"] = today

def get_bet(confidence: float):
    \"\"\"Calculate bet size based on Kelly Criterion modified\"\"\"
    reset_daily_stats()
    
    # Base risk
    base_risk = RISK_PERCENT
    
    # Increase risk for high confidence
    if confidence >= 0.80:
        base_risk = MAX_RISK_PERCENT
    elif confidence >= 0.75:
        base_risk = 0.07
    
    # Reduce risk after losses
    if STATE[\"streak\"] < 0:
        base_risk *= max(0.5, 1 + STATE[\"streak\"] * 0.1)
    
    bet = round(BALANCE * base_risk, 2)
    return max(10, min(bet, BALANCE * 0.15))  # Min $10, Max 15% of balance

def risk_block():
    \"\"\"Check if trading should be blocked\"\"\"
    reset_daily_stats()
    
    profit_pct = (BALANCE - INITIAL_BALANCE) / INITIAL_BALANCE
    
    # Daily loss limit: -10%
    if STATE[\"daily_pnl\"] <= -INITIAL_BALANCE * 0.10:
        return \"⛔ ДЕННИЙ ЛІМІТ ВТРАТ\"
    
    # Daily profit target: +15%
    if STATE[\"daily_pnl\"] >= INITIAL_BALANCE * 0.15:
        return \"💰 ДЕННА ЦІЛЬ ДОСЯГНУТА\"
    
    # Max 6 trades per day
    if STATE[\"daily_trades\"] >= 6:
        return \"⛔ ЛІМІТ УГОД НА ДЕНЬ\"
    
    # Losing streak protection
    if STATE[\"streak\"] <= -3:
        return \"⛔ СТОП (серія втрат)\"
    
    # Overall drawdown protection
    if profit_pct <= -0.25:
        return \"⛔ КРИТИЧНИЙ DRAWDOWN\"
    
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
    \"\"\"Exponential Moving Average\"\"\"
    if len(prices) < period:
        return None
    k = 2 / (period + 1)
    ema_val = prices[0]
    for price in prices[1:]:
        ema_val = price * k + ema_val * (1 - k)
    return ema_val

def sma(prices: List[float], period: int) -> Optional[float]:
    \"\"\"Simple Moving Average\"\"\"
    if len(prices) < period:
        return None
    return mean(prices[-period:])

def rsi(prices: List[float], period: int = RSI_PERIOD) -> Optional[float]:
    \"\"\"Relative Strength Index\"\"\"
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
    \"\"\"MACD with signal and histogram\"\"\"
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
        \"macd\": macd_line,
        \"signal\": signal_line,
        \"histogram\": histogram,
        \"prev_histogram\": prev_histogram,
        \"crossover_up\": prev_macd < prev_signal and macd_line > signal_line,
        \"crossover_down\": prev_macd > prev_signal and macd_line < signal_line
    }

def bollinger_bands(prices: List[float], period: int = BB_PERIOD, std_mult: float = BB_STD) -> Optional[Dict]:
    \"\"\"Bollinger Bands\"\"\"
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
        \"upper\": upper,
        \"middle\": middle,
        \"lower\": lower,
        \"bandwidth\": bandwidth,
        \"percent_b\": percent_b,
        \"squeeze\": bandwidth < 0.02,  # Squeeze condition
        \"at_upper\": current_price >= upper,
        \"at_lower\": current_price <= lower
    }

def atr(highs: List[float], lows: List[float], closes: List[float], period: int = ATR_PERIOD) -> Optional[float]:
    \"\"\"Average True Range\"\"\"
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
    \"\"\"Detect support and resistance levels\"\"\"
    if len(candles) < lookback:
        return {\"support\": [], \"resistance\": []}
    
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
        \"support\": sorted(support_levels)[-3:] if support_levels else [],
        \"resistance\": sorted(resistance_levels)[:3] if resistance_levels else []
    }

def detect_candlestick_pattern(candles: List[Candle]) -> Optional[Dict]:
    \"\"\"Detect candlestick patterns\"\"\"
    if len(candles) < 3:
        return None
    
    c1 = candles[-3]  # Oldest
    c2 = candles[-2]  # Middle
    c3 = candles[-1]  # Current
    
    patterns = []
    
    # Bullish Engulfing
    if c2.is_bearish and c3.is_bullish:
        if c3.o < c2.c and c3.c > c2.o:
            patterns.append({\"name\": \"BULLISH_ENGULFING\", \"direction\": \"BUY\", \"strength\": 0.8})
    
    # Bearish Engulfing
    if c2.is_bullish and c3.is_bearish:
        if c3.o > c2.c and c3.c < c2.o:
            patterns.append({\"name\": \"BEARISH_ENGULFING\", \"direction\": \"SELL\", \"strength\": 0.8})
    
    # Bullish Pin Bar (Hammer)
    if c3.lower_wick > c3.body * 2 and c3.upper_wick < c3.body * 0.5:
        patterns.append({\"name\": \"HAMMER\", \"direction\": \"BUY\", \"strength\": 0.7})
    
    # Bearish Pin Bar (Shooting Star)
    if c3.upper_wick > c3.body * 2 and c3.lower_wick < c3.body * 0.5:
        patterns.append({\"name\": \"SHOOTING_STAR\", \"direction\": \"SELL\", \"strength\": 0.7})
    
    # Morning Star (Bullish reversal)
    if c1.is_bearish and c2.is_doji and c3.is_bullish:
        if c3.c > (c1.o + c1.c) / 2:
            patterns.append({\"name\": \"MORNING_STAR\", \"direction\": \"BUY\", \"strength\": 0.85})
    
    # Evening Star (Bearish reversal)
    if c1.is_bullish and c2.is_doji and c3.is_bearish:
        if c3.c < (c1.o + c1.c) / 2:
            patterns.append({\"name\": \"EVENING_STAR\", \"direction\": \"SELL\", \"strength\": 0.85})
    
    # Three White Soldiers
    if all(c.is_bullish for c in [c1, c2, c3]):
        if c2.c > c1.c and c3.c > c2.c:
            patterns.append({\"name\": \"THREE_WHITE_SOLDIERS\", \"direction\": \"BUY\", \"strength\": 0.75})
    
    # Three Black Crows
    if all(c.is_bearish for c in [c1, c2, c3]):
        if c2.c < c1.c and c3.c < c2.c:
            patterns.append({\"name\": \"THREE_BLACK_CROWS\", \"direction\": \"SELL\", \"strength\": 0.75})
    
    return patterns[0] if patterns else None

def detect_divergence(prices: List[float], rsi_values: List[float]) -> Optional[str]:
    \"\"\"Detect RSI divergence\"\"\"
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
        return \"BULLISH_DIVERGENCE\"
    
    # Bearish divergence: price higher high, RSI lower high
    if recent_price_high > prev_price_high and recent_rsi_high < prev_rsi_high:
        return \"BEARISH_DIVERGENCE\"
    
    return None

# ============== TRADING ENGINE ==============
class TradingEngine:
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.queue = queue.Queue()
        
        # Multi-timeframe candle builders
        self.m1_builder = CandleBuilder(60)      # 1 minute
        self.m5_builder = CandleBuilder(300)     # 5 minutes
        self.m15_builder = CandleBuilder(900)    # 15 minutes
        
        # Candle history
        self.m1_candles: List[Candle] = []
        self.m5_candles: List[Candle] = []
        self.m15_candles: List[Candle] = []
        
        # RSI history for divergence
        self.rsi_history: List[float] = []
        
        # Signal cooldown
        self.last_signal_time = 0
        self.signal_cooldown = 300  # 5 minutes between signals
        
        # Current price
        self.current_price = 0
        
    def start(self):
        threading.Thread(target=self._stream, daemon=True).start()
        threading.Thread(target=self._process, daemon=True).start()
    
    def _stream(self):
        \"\"\"Stream price data from OANDA\"\"\"
        url = f\"https://stream-fxpractice.oanda.com/v3/accounts/{os.getenv('OANDA_ACCOUNT_ID')}/pricing/stream\"
        headers = {\"Authorization\": f\"Bearer {os.getenv('OANDA_API_KEY')}\"}
        params = {\"instruments\": self.symbol}
        
        while True:
            try:
                r = requests.get(url, headers=headers, params=params, stream=True, timeout=30)
                for line in r.iter_lines():
                    if not line:
                        continue
                    data = json.loads(line.decode())
                    if data.get(\"type\") == \"PRICE\":
                        bid = float(data[\"bids\"][0][\"price\"])
                        ask = float(data[\"asks\"][0][\"price\"])
                        mid = (bid + ask) / 2
                        self.queue.put((time.time(), mid))
            except Exception as e:
                print(f\"Stream error {self.symbol}: {e}\")
                time.sleep(5)
    
    def _process(self):
        \"\"\"Process incoming price data\"\"\"
        while True:
            try:
                ts, price = self.queue.get(timeout=5)
                self.current_price = price
                
                # Update all timeframes
                if self.m1_builder.tick(ts, price):
                    if self.m1_builder.completed:
                        self.m1_candles.append(self.m1_builder.completed)
                        self.m1_candles = self.m1_candles[-200:]
                
                if self.m5_builder.tick(ts, price):
                    if self.m5_builder.completed:
                        self.m5_candles.append(self.m5_builder.completed)
                        self.m5_candles = self.m5_candles[-100:]
                
                if self.m15_builder.tick(ts, price):
                    if self.m15_builder.completed:
                        self.m15_candles.append(self.m15_builder.completed)
                        self.m15_candles = self.m15_candles[-50:]
                
                # Update RSI history
                if len(self.m1_candles) >= RSI_PERIOD + 1:
                    closes = [c.c for c in self.m1_candles]
                    r = rsi(closes)
                    if r:
                        self.rsi_history.append(r)
                        self.rsi_history = self.rsi_history[-50:]
                        
            except queue.Empty:
                continue
            except Exception as e:
                print(f\"Process error {self.symbol}: {e}\")
    
    def analyze(self) -> Optional[Dict]:
        \"\"\"Comprehensive market analysis\"\"\"
        if len(self.m1_candles) < 50 or len(self.m5_candles) < 20:
            return None
        
        # Get price data
        m1_closes = [c.c for c in self.m1_candles]
        m5_closes = [c.c for c in self.m5_candles]
        
        m1_highs = [c.h for c in self.m1_candles]
        m1_lows = [c.l for c in self.m1_candles]
        
        # Calculate indicators
        analysis = {
            \"symbol\": self.symbol,
            \"price\": self.current_price,
            \"timestamp\": now_str()
        }
        
        # 1. RSI Analysis
        rsi_val = rsi(m1_closes)
        if rsi_val:
            analysis[\"rsi\"] = {
                \"value\": round(rsi_val, 2),
                \"overbought\": rsi_val >= RSI_OVERBOUGHT,
                \"oversold\": rsi_val <= RSI_OVERSOLD,
                \"strong_ob\": rsi_val >= RSI_STRONG_OB,
                \"strong_os\": rsi_val <= RSI_STRONG_OS
            }
        
        # 2. MACD Analysis
        macd_data = macd(m1_closes)
        if macd_data:
            analysis[\"macd\"] = macd_data
        
        # 3. Bollinger Bands
        bb_data = bollinger_bands(m1_closes)
        if bb_data:
            analysis[\"bollinger\"] = bb_data
        
        # 4. ATR for volatility
        atr_val = atr(m1_highs, m1_lows, m1_closes)
        if atr_val:
            pip_size = pip_value(self.symbol)
            atr_pips = atr_val / pip_size
            analysis[\"atr\"] = {
                \"value\": atr_val,
                \"pips\": round(atr_pips, 1),
                \"normal\": ATR_MIN_MULTIPLIER < atr_pips < ATR_MAX_MULTIPLIER * 10
            }
        
        # 5. Support/Resistance
        sr_levels = support_resistance(self.m5_candles)
        analysis[\"sr_levels\"] = sr_levels
        
        # 6. Candlestick Pattern
        pattern = detect_candlestick_pattern(self.m1_candles)
        if pattern:
            analysis[\"pattern\"] = pattern
        
        # 7. Divergence
        if len(self.rsi_history) >= 10:
            divergence = detect_divergence(m1_closes[-20:], self.rsi_history[-20:])
            if divergence:
                analysis[\"divergence\"] = divergence
        
        # 8. Trend (M5 EMA)
        ema20 = ema(m5_closes, 20)
        ema50 = ema(m5_closes, 50)
        if ema20 and ema50:
            analysis[\"trend\"] = {
                \"direction\": \"UP\" if ema20 > ema50 else \"DOWN\",
                \"strength\": abs(ema20 - ema50) / self.current_price * 10000  # In pips
            }
        
        return analysis
    
    def generate_signal(self) -> Optional[Dict]:
        \"\"\"Generate trading signal based on confluence\"\"\"
        
        # Check trading conditions
        if not is_trading_time():
            return None
        if is_news_time():
            return None
        if risk_block():
            return None
        
        # Signal cooldown
        now_ts = time.time()
        if now_ts - self.last_signal_time < self.signal_cooldown:
            return None
        
        # Get analysis
        analysis = self.analyze()
        if not analysis:
            return None
        
        # Confluence scoring
        buy_score = 0
        sell_score = 0
        reasons_buy = []
        reasons_sell = []
        
        # 1. RSI (weight: 1)
        if \"rsi\" in analysis:
            rsi_data = analysis[\"rsi\"]
            if rsi_data[\"strong_os\"]:
                buy_score += 1.5
                reasons_buy.append(f\"RSI перепроданий ({rsi_data['value']})\")
            elif rsi_data[\"oversold\"]:
                buy_score += 1
                reasons_buy.append(f\"RSI низький ({rsi_data['value']})\")
            
            if rsi_data[\"strong_ob\"]:
                sell_score += 1.5
                reasons_sell.append(f\"RSI перекуплений ({rsi_data['value']})\")
            elif rsi_data[\"overbought\"]:
                sell_score += 1
                reasons_sell.append(f\"RSI високий ({rsi_data['value']})\")
        
        # 2. MACD (weight: 1)
        if \"macd\" in analysis:
            macd_data = analysis[\"macd\"]
            if macd_data[\"crossover_up\"]:
                buy_score += 1.5
                reasons_buy.append(\"MACD бичаче перетин\")
            elif macd_data[\"histogram\"] > 0 and macd_data[\"histogram\"] > macd_data[\"prev_histogram\"]:
                buy_score += 0.5
                reasons_buy.append(\"MACD зростає\")
            
            if macd_data[\"crossover_down\"]:
                sell_score += 1.5
                reasons_sell.append(\"MACD ведмеже перетин\")
            elif macd_data[\"histogram\"] < 0 and macd_data[\"histogram\"] < macd_data[\"prev_histogram\"]:
                sell_score += 0.5
                reasons_sell.append(\"MACD падає\")
        
        # 3. Bollinger Bands (weight: 1)
        if \"bollinger\" in analysis:
            bb_data = analysis[\"bollinger\"]
            if bb_data[\"at_lower\"] and not bb_data[\"squeeze\"]:
                buy_score += 1
                reasons_buy.append(\"Ціна на нижній BB\")
            if bb_data[\"at_upper\"] and not bb_data[\"squeeze\"]:
                sell_score += 1
                reasons_sell.append(\"Ціна на верхній BB\")
            if bb_data[\"squeeze\"]:
                # Squeeze - wait for breakout
                buy_score -= 0.5
                sell_score -= 0.5
        
        # 4. Candlestick Pattern (weight: 1)
        if \"pattern\" in analysis:
            pattern = analysis[\"pattern\"]
            if pattern[\"direction\"] == \"BUY\":
                buy_score += pattern[\"strength\"]
                reasons_buy.append(f\"Патерн: {pattern['name']}\")
            else:
                sell_score += pattern[\"strength\"]
                reasons_sell.append(f\"Патерн: {pattern['name']}\")
        
        # 5. Divergence (weight: 1.5)
        if \"divergence\" in analysis:
            if analysis[\"divergence\"] == \"BULLISH_DIVERGENCE\":
                buy_score += 1.5
                reasons_buy.append(\"Бича дивергенція\")
            else:
                sell_score += 1.5
                reasons_sell.append(\"Ведмежа дивергенція\")
        
        # 6. Trend Alignment (weight: 1)
        if \"trend\" in analysis:
            trend = analysis[\"trend\"]
            if trend[\"direction\"] == \"UP\" and trend[\"strength\"] > 2:
                buy_score += 1
                reasons_buy.append(\"Висхідний тренд\")
            elif trend[\"direction\"] == \"DOWN\" and trend[\"strength\"] > 2:
                sell_score += 1
                reasons_sell.append(\"Низхідний тренд\")
        
        # Check ATR for normal volatility
        if \"atr\" in analysis and not analysis[\"atr\"][\"normal\"]:
            return None  # Skip abnormal volatility
        
        # Determine direction and check confluence
        if buy_score >= MIN_CONFLUENCE_SCORE and buy_score > sell_score:
            direction = \"BUY\"
            score = buy_score
            reasons = reasons_buy
        elif sell_score >= MIN_CONFLUENCE_SCORE and sell_score > buy_score:
            direction = \"SELL\"
            score = sell_score
            reasons = reasons_sell
        else:
            return None  # Insufficient confluence
        
        # Calculate probability
        max_possible_score = 7.5  # Maximum possible confluence score
        probability = min(0.95, 0.6 + (score / max_possible_score) * 0.35)
        
        if probability < MIN_PROBABILITY:
            return None
        
        # Signal generated
        self.last_signal_time = now_ts
        
        return {
            \"symbol\": self.symbol,
            \"direction\": direction,
            \"probability\": round(probability * 100, 1),
            \"confluence_score\": round(score, 1),
            \"reasons\": reasons,
            \"price\": analysis[\"price\"],
            \"time\": analysis[\"timestamp\"],
            \"rsi\": analysis.get(\"rsi\", {}).get(\"value\"),
            \"atr_pips\": analysis.get(\"atr\", {}).get(\"pips\")
        }

# ============== BOT SETUP ==============
SYMBOLS = [\"EUR_USD\", \"GBP_USD\", \"USD_JPY\"]
ENGINES = [TradingEngine(s) for s in SYMBOLS]

# Start all engines
for engine in ENGINES:
    engine.start()

LAST_SIGNAL = None

def get_best_signal() -> Optional[Dict]:
    \"\"\"Get the best signal from all pairs\"\"\"
    global LAST_SIGNAL
    
    best = None
    for engine in ENGINES:
        signal = engine.generate_signal()
        if signal:
            if not best or signal[\"probability\"] > best[\"probability\"]:
                best = signal
    
    LAST_SIGNAL = best
    return best

def record_result(win: bool):
    \"\"\"Record trade result\"\"\"
    global BALANCE
    
    bet = get_bet(LAST_SIGNAL[\"probability\"] / 100 if LAST_SIGNAL else 0.75)
    
    if win:
        profit = bet * PAYOUT
        BALANCE += profit
        STATE[\"wins\"] += 1
        STATE[\"streak\"] = max(1, STATE[\"streak\"] + 1)
        STATE[\"daily_pnl\"] += profit
    else:
        BALANCE -= bet
        STATE[\"losses\"] += 1
        STATE[\"streak\"] = min(-1, STATE[\"streak\"] - 1)
        STATE[\"daily_pnl\"] -= bet
    
    STATE[\"daily_trades\"] += 1

def get_stats() -> str:
    \"\"\"Get trading statistics\"\"\"
    total = STATE[\"wins\"] + STATE[\"losses\"]
    win_rate = (STATE[\"wins\"] / total * 100) if total > 0 else 0
    profit_pct = ((BALANCE - INITIAL_BALANCE) / INITIAL_BALANCE) * 100
    
    return (
        f\"📊 СТАТИСТИКА
\"
        f\"━━━━━━━━━━━━━━━
\"
        f\"💰 Баланс: ${BALANCE:.2f}
\"
        f\"📈 P/L: {profit_pct:+.1f}%
\"
        f\"✅ Виграші: {STATE['wins']}
\"
        f\"❌ Програші: {STATE['losses']}
\"
        f\"🎯 Win Rate: {win_rate:.1f}%
\"
        f\"🔥 Серія: {STATE['streak']}
\"
        f\"📅 Угод сьогодні: {STATE['daily_trades']}\"
    )

def format_signal(signal: Optional[Dict]) -> str:
    \"\"\"Format signal for display\"\"\"
    block = risk_block()
    if block:
        return f\"{block}

{get_stats()}\"
    
    if not signal:
        session = get_session()
        status = \"🟢 АКТИВНА\" if is_trading_time() else \"🔴 НЕАКТИВНА\"
        news = \"⚠️ НОВИНИ\" if is_news_time() else \"✅ Чисто\"
        
        return (
            f\"❌ Немає сигналу
\"
            f\"━━━━━━━━━━━━━━━
\"
            f\"🕒 {now_str()}
\"
            f\"📍 Сесія: {session} {status}
\"
            f\"📰 Новини: {news}

\"
            f\"Очікую якісний сетап...\"
        )
    
    direction_emoji = \"🟢\" if signal[\"direction\"] == \"BUY\" else \"🔴\"
    confidence = \"🔥🔥🔥\" if signal[\"probability\"] >= 80 else \"🔥🔥\" if signal[\"probability\"] >= 75 else \"🔥\"
    
    reasons_text = \"
\".join(f\"  • {r}\" for r in signal[\"reasons\"][:4])
    
    bet = get_bet(signal[\"probability\"] / 100)
    
    return (
        f\"{direction_emoji} {signal['direction']} {signal['symbol'].replace('_', '/')}
\"
        f\"━━━━━━━━━━━━━━━
\"
        f\"📊 Ймовірність: {signal['probability']}% {confidence}
\"
        f\"🎯 Confluence: {signal['confluence_score']}/6
\"
        f\"💵 Ставка: ${bet}
\"
        f\"⏱ Експірація: 2 хв
\"
        f\"💰 Баланс: ${BALANCE:.2f}
\"
        f\"━━━━━━━━━━━━━━━
\"
        f\"📋 Причини:
{reasons_text}
\"
        f\"━━━━━━━━━━━━━━━
\"
        f\"RSI: {signal.get('rsi', 'N/A')} | ATR: {signal.get('atr_pips', 'N/A')} pips
\"
        f\"🕒 {signal['time']}\"
    )

def keyboard():
    \"\"\"Create result keyboard\"\"\"
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton(\"✅ WIN\", callback_data=\"win\"),
            InlineKeyboardButton(\"❌ LOSS\", callback_data=\"loss\")
        ]
    ])

# ============== TELEGRAM HANDLERS ==============
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        \"🤖 FOREX TRADING BOT
\"
        \"━━━━━━━━━━━━━━━
\"
        \"Multi-Indicator Confluence Strategy
\"
        \"Pairs: EUR/USD, GBP/USD, USD/JPY
\"
        \"━━━━━━━━━━━━━━━
\"
        \"Commands:
\"
        \"/signal - Get current signal
\"
        \"/stats - View statistics
\"
        \"/auto - Start auto signals
\"
        \"/stop - Stop auto signals\"
    )

async def cmd_signal(update: Update, context: ContextTypes.DEFAULT_TYPE):
    signal = get_best_signal()
    await update.message.reply_text(format_signal(signal), reply_markup=keyboard() if signal else None)

async def cmd_stats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(get_stats())

async def callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    
    record_result(query.data == \"win\")
    result = \"✅ WIN записано\" if query.data == \"win\" else \"❌ LOSS записано\"
    await query.edit_message_text(f\"{result}

{get_stats()}\")

async def auto_signal(context: ContextTypes.DEFAULT_TYPE):
    \"\"\"Auto signal job\"\"\"
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
    current_jobs = context.job_queue.get_jobs_by_name(f\"auto_{update.effective_chat.id}\")
    for job in current_jobs:
        job.schedule_removal()
    
    context.job_queue.run_repeating(
        auto_signal,
        interval=120,  # Check every 2 minutes
        first=10,
        chat_id=update.effective_chat.id,
        name=f\"auto_{update.effective_chat.id}\"
    )
    await update.message.reply_text(\"✅ Авто-сигнали увімкнено\")

async def cmd_stop(update: Update, context: ContextTypes.DEFAULT_TYPE):
    current_jobs = context.job_queue.get_jobs_by_name(f\"auto_{update.effective_chat.id}\")
    for job in current_jobs:
        job.schedule_removal()
    await update.message.reply_text(\"🛑 Авто-сигнали вимкнено\")

def main():
    \"\"\"Main entry point\"\"\"
    token = os.getenv(\"TELEGRAM_BOT_TOKEN\")
    if not token:
        print(\"ERROR: TELEGRAM_BOT_TOKEN not set\")
        return
    
    app = Application.builder().token(token).build()
    
    app.add_handler(CommandHandler(\"start\", cmd_start))
    app.add_handler(CommandHandler(\"signal\", cmd_signal))
    app.add_handler(CommandHandler(\"stats\", cmd_stats))
    app.add_handler(CommandHandler(\"auto\", cmd_auto))
    app.add_handler(CommandHandler(\"stop\", cmd_stop))
    app.add_handler(CallbackQueryHandler(callback_handler))
    
    print(\"🚀 Bot started...\")
    app.run_polling(drop_pending_updates=True)

if __name__ == \"__main__\":
    main()
