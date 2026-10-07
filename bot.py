import asyncio
import logging
import os
from aiogram import Bot, Dispatcher, types, F
from aiogram.filters import Command
from groq import Groq

# Берем токены из переменных окружения (так безопаснее)
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Инициализация
bot = Bot(token=TELEGRAM_TOKEN)
dp = Dispatcher()
client = Groq(api_key=GROQ_API_KEY)

# Системный промпт: задаем характер нейросети
SYSTEM_PROMPT = "Ты — циничный, но очень умный и полезный ИИ-ассистент. Отвечай кратко, по делу, иногда можешь добавить сарказма, но всегда помогай пользователю."

@dp.message(Command("start"))
async def cmd_start(message: types.Message):
    await message.answer("Привет! Я крутая нейросеть на базе Llama 3. Задай мне любой вопрос!")

@dp.message(F.text)
async def handle_message(message: types.Message):
    # Показываем, что бот "думает"
    await bot.send_chat_action(message.chat.id, "typing")
    
    try:
        # Запрос к нейросети
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": message.text}
            ],
            model="llama3-8b-8192", # Используем Llama 3
            temperature=0.7,
        )
        response = chat_completion.choices[0].message.content
        await message.answer(response)
    except Exception as e:
        await message.answer(f"Ошибка при обращении к мозгам: {e}")

async def main():
    logging.basicConfig(level=logging.INFO)
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
