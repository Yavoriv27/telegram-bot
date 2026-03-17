import os, json, time, math, queue, threading
from dataclasses import dataclass
from datetime import datetime, timezone

import requests
from dotenv import load_dotenv
import pytz

from telegram import Update, InlineKeyboardMarkup, InlineKeyboardButton
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes

load_dotenv()
KYIV = pytz.timezone("Europe/Kyiv")

def now():
    return datetime.now(timezone.utc).astimezone(KYIV)

def now_str():
    return now().strftime("%H:%M:%S")

def is_trading_time():
    n=now()
    h=n.hour; m=n.minute
    if 9<=h<12: return True
    if (h==15 and m>=30) or (16<=h<18): return True
    return False

NEWS_TIMES=["15:30","17:00"]

def is_news_time():
    n=now()
    for t in NEWS_TIMES:
        hh,mm=map(int,t.split(":"))
        news=n.replace(hour=hh,minute=mm,second=0,microsecond=0)
        if abs((n-news).total_seconds())/60<=15:
            return True
    return False

def mean(x): return sum(x)/len(x) if x else 0
def stdev(x): return math.sqrt(mean([(i-mean(x))**2 for i in x])) if x else 0
def pip(s): return 0.01 if "JPY" in s else 0.0001
def sigmoid(x): return 1/(1+math.exp(-x))

RSI_BUY=55
RSI_SELL=45
ATR_FILTER=0.0005
MIN_PROB=0.78

BALANCE=3000
START_BALANCE=3000
RISK=0.1
PAYOUT=0.8

STATE={"wins":0,"losses":0,"streak":0,"bias":0}

def get_bet():
    return round(min(BALANCE*RISK, BALANCE*0.15),2)

def risk_block():
    profit=(BALANCE-START_BALANCE)/START_BALANCE
    if profit<=-0.2: return "⛔ STOP DAY"
    if profit>=0.3: return "💰 STOP DAY"
    if STATE["streak"]<=-3: return "⛔ STOP (серія)"
    return None

@dataclass
class Candle:
    t:float;o:float;h:float;l:float;c:float
    def update(self,p):
        self.c=p;self.h=max(self.h,p);self.l=min(self.l,p)
    @property
    def dir(self):
        return "UP" if self.c>self.o else "DOWN"

class Builder:
    def __init__(self,tf):
        self.tf=tf;self.cur=None;self.last=None
    def bucket(self,ts): return ts-ts%self.tf
    def tick(self,ts,p):
        b=self.bucket(ts)
        if not self.cur:
            self.cur=Candle(b,p,p,p,p);return
        if self.cur.t==b:self.cur.update(p)
        else:
            self.last=self.cur
            self.cur=Candle(b,p,p,p,p)

def ema(a,p):
    if len(a)<p:return None
    k=2/(p+1);e=a[0]
    for v in a[1:]:e=v*k+e*(1-k)
    return e

def rsi(a,p=14):
    if len(a)<p+1:return None
    g=l=0
    for i in range(-p,0):
        d=a[i]-a[i-1]
        if d>0:g+=d
        else:l-=d
    if l==0:return 100
    return 100-(100/(1+(g/l)))

def atr(h,l,c,p=14):
    trs=[]
    for i in range(1,len(c)):
        trs.append(max(h[i]-l[i],abs(h[i]-c[i-1]),abs(l[i]-c[i-1])))
    return mean(trs[-p:]) if len(trs)>=p else None

class Engine:
    def __init__(self,s):
        self.s=s
        self.q=queue.Queue()

        self.b1=Builder(60)
        self.b5=Builder(300)
        self.b15=Builder(900)
        self.b60=Builder(3600)

        self.m1=[];self.m5=[];self.m15=[];self.m60=[]

        self.last_signal_time=0

    def start(self):
        threading.Thread(target=self.stream,daemon=True).start()
        threading.Thread(target=self.loop,daemon=True).start()

    def stream(self):
        url=f"https://stream-fxpractice.oanda.com/v3/accounts/{os.getenv('OANDA_ACCOUNT_ID')}/pricing/stream"
        headers={"Authorization":f"Bearer {os.getenv('OANDA_API_KEY')}"}
        params={"instruments":self.s}

        while True:
            try:
                r=requests.get(url,headers=headers,params=params,stream=True)
                for l in r.iter_lines():
                    if not l:continue
                    d=json.loads(l.decode())
                    if d.get("type")=="PRICE":
                        p=float(d["bids"][0]["price"])
                        self.q.put((time.time(),p))
            except:
                time.sleep(3)

    def loop(self):
        while True:
            ts,p=self.q.get()

            for b in [self.b1,self.b5,self.b15,self.b60]:
                b.tick(ts,p)

            if self.b1.last:self.m1.append(self.b1.last)
            if self.b5.last:self.m5.append(self.b5.last)
            if self.b15.last:self.m15.append(self.b15.last)
            if self.b60.last:self.m60.append(self.b60.last)

            self.m1=self.m1[-200:]
            self.m5=self.m5[-200:]
            self.m15=self.m15[-200:]
            self.m60=self.m60[-200:]

    def signal(self):
        if len(self.m15)<60:return None
        if not is_trading_time() or is_news_time(): return None
        if risk_block(): return None

        now_ts=time.time()
        if now_ts-self.last_signal_time<180: return None

        closes=[c.c for c in self.m15]
        e20=ema(closes,20)
        e50=ema(closes,50)
        if not e20 or not e50:return None

        direction="BUY" if e20>e50 else "SELL"

        # H1
        closes60=[c.c for c in self.m60]
        e20h=ema(closes60,20)
        e50h=ema(closes60,50)
        if not e20h or not e50h:return None
        if ("BUY" if e20h>e50h else "SELL")!=direction:
            return None

        r=rsi(closes)
        if not r:return None

        last3=self.m5[-3:]
        if all(c.dir=="UP" for c in last3): return None
        if all(c.dir=="DOWN" for c in last3): return None

        score=1
        if direction=="BUY" and r>RSI_BUY: score+=1
        if direction=="SELL" and r<RSI_SELL: score+=1

        prob=sigmoid(score)

        if prob<MIN_PROB:return None

        self.last_signal_time=now_ts

        # 🔥 ПІДКАЗКА ВХОДУ
        entry = "🔔 Входити зараз" if prob>0.8 else "⏳ Дочекатись свічки"

        return {
            "dir":direction,
            "prob":round(prob*100,1),
            "symbol":self.s,
            "time":now_str(),
            "entry":entry
        }

SYMBOLS=["EUR_USD","GBP_USD","USD_JPY"]
ENGINES=[Engine(s) for s in SYMBOLS]
for e in ENGINES:e.start()

LAST=None

def best():
    global LAST
    best=None
    for e in ENGINES:
        s=e.signal()
        if not s:continue
        if not best or s["prob"]>best["prob"]:
            best=s
    LAST=best
    return best

def train(win):
    global BALANCE
    bet=get_bet()
    if win:
        BALANCE+=bet*PAYOUT
        STATE["wins"]+=1
        STATE["streak"]+=1
    else:
        BALANCE-=bet
        STATE["losses"]+=1
        STATE["streak"]-=1

def kb():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("✅",callback_data="win"),
         InlineKeyboardButton("❌",callback_data="loss")]
    ])

def fmt(s):
    block=risk_block()
    if block:return block
    if not s:return f"❌ Нема сигналу\n🕒 {now_str()}"

    return (
        f"{'🟢 BUY' if s['dir']=='BUY' else '🔴 SELL'} {s['symbol']}\n"
        f"📊 {s['prob']}%\n"
        f"{s['entry']}\n"
        f"💵 {get_bet()}\n"
        f"💰 {round(BALANCE,2)}\n"
        f"📉 {STATE['streak']}\n"
        f"⏱ 2 хв\n"
        f"🕒 {s['time']}"
    )

async def signal(update:Update,context:ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(fmt(best()),reply_markup=kb())

async def callback(update:Update,context:ContextTypes.DEFAULT_TYPE):
    q=update.callback_query
    await q.answer()
    train(q.data=="win")
    await q.edit_message_text("Результат збережено")

async def start(update:Update,context:ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🔥 BOT ULTIMATE READY")

async def auto(context):
    if not is_trading_time() or is_news_time(): return
    s=best()
    if s:
        await context.bot.send_message(context.job.chat_id,fmt(s),reply_markup=kb())

def main():
    app=Application.builder().token(os.getenv("TELEGRAM_BOT_TOKEN")).build()

    app.add_handler(CommandHandler("start",start))
    app.add_handler(CommandHandler("signal",signal))
    app.add_handler(CallbackQueryHandler(callback))

    async def start_auto(update:Update,context:ContextTypes.DEFAULT_TYPE):
        context.job_queue.run_repeating(auto,interval=120,first=10,chat_id=update.effective_chat.id)
        await update.message.reply_text("✅ Авто ON")

    app.add_handler(CommandHandler("auto",start_auto))

    app.run_polling(drop_pending_updates=True)

if __name__=="__main__":
    main()
