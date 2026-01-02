import os
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes

TOKEN = os.getenv("8490611446:AAFZwLYoVEJrsvtapZKY4NcX8XBtHIoC4Oo")

if not TOKEN:
    raise RuntimeError("BOT_TOKEN не заданий")

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Бот запущений і працює ✅")

app = ApplicationBuilder().token(TOKEN).build()
app.add_handler(CommandHandler("start", start))

print("Bot started")
app.run_polling()
