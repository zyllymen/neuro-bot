import os
import httpx
from flask import Flask, request, jsonify
import telebot

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

app = Flask(__name__)
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
    try:
        response = ask_groq(message.text)
        bot.reply_to(message, response)
    except Exception as e:
        bot.reply_to(message, f"Ошибка: {e}")

@app.route('/' + TELEGRAM_TOKEN, methods=['POST'])
def webhook():
    update = telebot.types.Update.de_json(request.stream.read().decode('utf-8'))
    bot.process_new_updates([update])
    return '', 200

@app.route('/')
def home():
    return 'Бот работает!'

if __name__ == '__main__':
    # Устанавливаем webhook при запуске
    webhook_url = f"https://neuro-bot-sdko.onrender.com/{TELEGRAM_TOKEN}"
    bot.remove_webhook()
    bot.set_webhook(url=webhook_url)
    print(f"Webhook установлен: {webhook_url}")
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 10000)))
