import os
import httpx
import telebot
from flask import Flask
import threading
import json

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

app = Flask(__name__)
bot = telebot.TeleBot(TELEGRAM_TOKEN, threaded=False)

SYSTEM_PROMPT = "Ты — циничный, но очень умный и полезный ИИ-ассистент. Отвечай кратко, по делу, иногда можешь добавить сарказма, но всегда помогай пользователю."

def ask_groq(user_message: str) -> str:
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "qwen/qwen3.8-27b",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message}
        ],
        "temperature": 0.7
    }
    
    try:
        response = httpx.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json=data,
            timeout=30.0
        )
        
        print(f"Groq status: {response.status_code}")
        
        if response.status_code != 200:
            return f"Ошибка Groq: {response.status_code} - {response.text[:200]}"
        
        result = response.json()
        
        if "choices" not in result or len(result["choices"]) == 0:
            return f"Пустой ответ от Groq: {json.dumps(result, ensure_ascii=False)[:200]}"
        
        return result["choices"][0]["message"]["content"]
        
    except httpx.TimeoutException:
        return "Превышено время ожидания"
    except Exception as e:
        return f"Ошибка: {str(e)}"

@bot.message_handler(commands=['start'])
def cmd_start(message):
    bot.reply_to(message, "Привет! Я нейросеть Qwen 3 на базе 27B параметров. Задай мне любой вопрос!")

@bot.message_handler(func=lambda message: True)
def handle_message(message):
    try:
        response = ask_groq(message.text)
        bot.reply_to(message, response)
    except Exception as e:
        bot.reply_to(message, f"Критическая ошибка: {str(e)}")

@app.route('/')
def home():
    return 'Бот работает!'

@app.route('/health')
def health():
    return 'OK'

def run_bot():
    print("Бот запущен!")
    bot.infinity_polling()

if __name__ == '__main__':
    bot_thread = threading.Thread(target=run_bot, daemon=True)
    bot_thread.start()
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 10000)))
