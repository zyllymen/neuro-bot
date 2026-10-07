import os
import httpx
import telebot

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

bot = telebot.TeleBot(TELEGRAM_TOKEN)

SYSTEM_PROMPT = "Ты — циничный, но очень умный и полезный ИИ-ассистент. Отвечай кратко, по делу, иногда можешь добавить сарказма, но всегда помогай пользователю."

def ask_groq(user_message: str) -> str:
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "llama3-8b-8192",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message}
        ],
        "temperature": 0.7
    }
    
    response = httpx.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers=headers,
        json=data,
        timeout=30.0
    )
    result = response.json()
    return result["choices"][0]["message"]["content"]

@bot.message_handler(commands=['start'])
def cmd_start(message):
    bot.reply_to(message, "Привет! Я крутая нейросеть на базе Llama 3. Задай мне любой вопрос!")

@bot.message_handler(func=lambda message: True)
def handle_message(message):
    bot.send_chat_action(message.chat.id, "typing")
    try:
        response = ask_groq(message.text)
        bot.reply_to(message, response)
    except Exception as e:
        bot.reply_to(message, f"Ошибка при обращении к мозгам: {e}")

print("Бот запущен!")
bot.infinity_polling()
