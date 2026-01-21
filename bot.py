import os
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List, Tuple

from dotenv import load_dotenv
import pytz

from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from PIL import Image, ImageOps, ImageFilter
import pytesseract

load_dotenv()

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("po_screenshot_bot")

KYIV_TZ = pytz.timezone("Europe/Kyiv")

EXPIRY_MINUTES = 10

MIN_CONFIDENCE = int(os.getenv("MIN_CONFIDENCE", "65"))
MAX_CONFIDENCE = int(os.getenv("MAX_CONFIDENCE", "75"))

ADX_MIN = float(os.getenv("ADX_MIN", "25"))
ADX_FLAT = float(os.getenv("ADX_FLAT", "22"))

RSI_SELL_BLOCK = float(os.getenv("RSI_SELL_BLOCK", "28"))
RSI_BUY_BLOCK = float(os.getenv("RSI_BUY_BLOCK", "72"))

STATE_FILE = os.getenv("PO_STATE_FILE", "/app/po_state.json")


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def fmt_kyiv(dt_utc: datetime) -> str:
    return dt_utc.astimezone(KYIV_TZ).strftime("%Y-%m-%d %H:%M:%S")


def safe_float(x: str) -> Optional[float]:
    try:
        return float(x.replace(",", ".").strip())
    except Exception:
        return None


def load_state() -> dict:
    try:
        if os.path.exists(STATE_FILE):
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return {"pending": {}}


def save_state(state: dict):
    try:
        os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
    except Exception:
        pass
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False)


STATE = load_state()


@dataclass
class POAnalysis:
    direction: str
    confidence: int
    risk: int
    why: List[str]
    rsi: Optional[float] = None
    adx: Optional[float] = None


def _ocr_extract_key_value(text: str, key: str) -> Optional[float]:
    i = text.find(key)
    if i == -1:
        return None
    chunk = text[i:i + 50]
    nums = []
    cur = ""
    for ch in chunk:
        if ch.isdigit() or ch in ".,": 
            cur += ch
        else:
            if cur:
                nums.append(cur)
                cur = ""
    if cur:
        nums.append(cur)
    for n in nums:
        v = safe_float(n)
        if v is not None and 0 <= v <= 100:
            return v
    return None


def extract_rsi_adx_from_image(path: str) -> Tuple[Optional[float], Optional[float]]:
    try:
        img = Image.open(path).convert("RGB")
        w, h = img.size

        roi = img.crop((0, int(h * 0.70), int(w * 0.70), int(h * 0.98)))

        gray = ImageOps.grayscale(roi)
        gray = gray.resize((int(gray.size[0] * 1.6), int(gray.size[1] * 1.6)))
        gray = gray.filter(ImageFilter.GaussianBlur(1))

        thr = gray.point(lambda p: 255 if p > 140 else 0)

        txt = pytesseract.image_to_string(thr, config=r"--oem 3 --psm 6")
        txt = txt.upper().replace(" ", "").replace("\n", "")

        rsi_v = _ocr_extract_key_value(txt, "RSI")
        adx_v = _ocr_extract_key_value(txt, "ADX")

        return rsi_v, adx_v
    except Exception as e:
        log.warning("OCR error: %s", e)
        return None, None


def clamp(x: int, a: int, b: int) -> int:
    return max(a, min(b, x))


def analyze_strong(tf15_path: str, tf1_path: str) -> POAnalysis:
    why: List[str] = []
    confidence = 50

    rsi15, adx15 = extract_rsi_adx_from_image(tf15_path)
    rsi1, adx1 = extract_rsi_adx_from_image(tf1_path)

    rsi_v = rsi1 if rsi1 is not None else rsi15
    adx_v = adx1 if adx1 is not None else adx15

    if adx_v is None:
        why.append("ADX не прочитався")
        confidence -= 6
    else:
        why.append(f"ADX: {adx_v:.1f}")
        if adx_v >= ADX_MIN:
            confidence += 10
            why.append("ADX сильний → рух є")
        elif adx_v < ADX_FLAT:
            confidence -= 12
            why.append("Флет → пропуск")
        else:
            confidence -= 4
            why.append("Рух слабкий")

    if rsi_v is None:
        why.append("RSI не прочитався")
        confidence -= 6
    else:
        why.append(f"RSI: {rsi_v:.1f}")

    direction = "NEUTRAL"

    if rsi_v is not None:
        if rsi_v >= 65:
            direction = "SELL"
            confidence += 6
            why.append("RSI високий → можливий відкат вниз")
        elif rsi_v <= 35:
            direction = "BUY"
            confidence += 6
            why.append("RSI низький → можливий відскок вгору")
        else:
            direction = "NEUTRAL"
            confidence -= 6
            why.append("RSI середній → немає перекосу")

    if direction == "SELL" and rsi_v is not None and rsi_v < RSI_SELL_BLOCK:
        direction = "NEUTRAL"
        confidence -= 14
        why.append(f"SELL заборонено: RSI<{RSI_SELL_BLOCK} (часто відскок)")

    if direction == "BUY" and rsi_v is not None and rsi_v > RSI_BUY_BLOCK:
        direction = "NEUTRAL"
        confidence -= 14
        why.append(f"BUY заборонено: RSI>{RSI_BUY_BLOCK} (часто відкат)")

    confidence = clamp(int(confidence), 40, MAX_CONFIDENCE)

    if direction != "NEUTRAL" and confidence < MIN_CONFIDENCE:
        direction = "NEUTRAL"
        confidence = min(confidence, 60)
        why.append("Сетап недостатньо сильний → NEUTRAL")

    risk = 100 - confidence
    return POAnalysis(direction=direction, confidence=confidence, risk=risk, why=why, rsi=rsi_v, adx=adx_v)


def fmt_signal(a: POAnalysis) -> str:
    t = fmt_kyiv(now_utc())

    if a.direction == "NEUTRAL":
        reasons = "\n".join([f"• {x}" for x in a.why])
        return (
            "⚪ <b>NEUTRAL</b>\n"
            f"⏱ <b>Експірація:</b> {EXPIRY_MINUTES} хв\n"
            f"🕒 <b>Kyiv:</b> {t}\n"
            "⚠️ <b>Сильного входу немає</b>\n\n"
            f"<b>Деталі:</b>\n{reasons}"
        )

    arrow = "🟢 BUY" if a.direction == "BUY" else "🔴 SELL"
    reasons = "\n".join([f"• {x}" for x in a.why])
    enter_hint = "🔔 Рекомендовано входити негайно" if a.confidence >= 70 else "⏳ Краще дочекатися закриття 1M свічки"

    return (
        f"{arrow}\n"
        f"⏱ <b>Експірація:</b> {EXPIRY_MINUTES} хв\n"
        f"🕒 <b>Kyiv:</b> {t}\n"
        f"📊 <b>Ймовірність:</b> {a.confidence}%\n"
        f"⚠️ <b>Ризик:</b> {a.risk}%\n\n"
        f"{enter_hint}\n\n"
        f"<b>Підтвердження:</b>\n{reasons}"
    )


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "✅ PocketOption бот запущено.\n\n"
        "Як користуватись:\n"
        "1) Напиши /signal\n"
        "2) Надішли 2 скріни з PocketOption: 15M → 1M\n"
        "3) Бот дасть BUY/SELL або NEUTRAL\n\n"
        "Команди:\n"
        "/signal\n"
        "/reset\n",
    )


async def cmd_reset(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = str(update.effective_chat.id)
    STATE["pending"].pop(chat_id, None)
    save_state(STATE)
    await update.message.reply_text("✅ Скинуто. Напиши /signal")


async def cmd_signal(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = str(update.effective_chat.id)
    STATE["pending"][chat_id] = {"step": 1, "tf15": None, "tf1": None}
    save_state(STATE)
    await update.message.reply_text(
        "📸 Надішли 2 скріни з PocketOption:\n"
        "1) 15M (тренд)\n"
        "2) 1M (вхід)\n\n"
        "Кидай по черзі 2 фото підряд."
    )


async def on_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = str(update.effective_chat.id)
    pending = STATE["pending"].get(chat_id)

    if not pending:
        await update.message.reply_text("Напиши /signal і потім надішли 2 скріни (15M і 1M).")
        return

    photos = update.message.photo
    if not photos:
        return

    file = await context.bot.get_file(photos[-1].file_id)
    os.makedirs("/tmp/po", exist_ok=True)

    if pending["step"] == 1:
        path = f"/tmp/po/{chat_id}_15m.jpg"
        await file.download_to_drive(path)
        pending["tf15"] = path
        pending["step"] = 2
        save_state(STATE)
        await update.message.reply_text("✅ Прийняв 15M. Тепер надішли 1M скрін.")
        return

    if pending["step"] == 2:
        path = f"/tmp/po/{chat_id}_1m.jpg"
        await file.download_to_drive(path)
        pending["tf1"] = path
        save_state(STATE)

        try:
            a = analyze_strong(pending["tf15"], pending["tf1"])
            await update.message.reply_text(fmt_signal(a), parse_mode=ParseMode.HTML)
        except Exception as e:
            log.exception("Analyze error: %s", e)
            await update.message.reply_text("❌ Помилка аналізу. Спробуй ще раз /signal")

        STATE["pending"].pop(chat_id, None)
        save_state(STATE)
        return


def acquire_lock(lock_path: str):
    if os.path.exists(lock_path):
        try:
            with open(lock_path, "r", encoding="utf-8") as f:
                old_pid = int((f.read() or "").strip())
            os.kill(old_pid, 0)
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
        app = Application.builder().token(token).build()

        app.add_handler(CommandHandler("start", cmd_start))
        app.add_handler(CommandHandler("signal", cmd_signal))
        app.add_handler(CommandHandler("reset", cmd_reset))
        app.add_handler(MessageHandler(filters.PHOTO, on_photo))

        app.run_polling(drop_pending_updates=True)
    finally:
        try:
            if os.path.exists(lock_path):
                os.remove(lock_path)
        except Exception:
            pass


if __name__ == "__main__":
    main()
