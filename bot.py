import os
import asyncio
import logging
import json
from typing import List

import aiosqlite
import httpx
from duckduckgo_search import DDGS
from aiogram import Bot, Dispatcher, types, F
from aiogram.filters import Command
from aiogram.enums import ParseMode

# --- НАСТРОЙКИ ---
logging.basicConfig(level=logging.INFO)

# 1. БЕРЕМ КЛЮЧИ ИЗ ОКРУЖЕНИЯ (или ставим заглушки, чтобы код не падал, если ключа нет)
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
GROQ_API_KEY = os.getenv("LLM_API_KEY")  # Сюда вставим ключ от Groq
NVIDIA_NIM_API_KEY = os.getenv("NVIDIA_NIM_API_KEY", "") # Если нет - будет пустая строка
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "") # Если нет - будет пустая строка

# Проверка на наличие главного ключа
if not TELEGRAM_TOKEN or not GROQ_API_KEY:
    logging.error("❌ ОШИБКА: Не заданы TELEGRAM_TOKEN или LLM_API_KEY (Groq Key)!")
    logging.error("Задай их в Render или запусти локально с export ...")
    exit(1)

# Цепочка моделей. Бот пойдет по списку сверху вниз.
# Если у первого (NVIDIA) нет ключа или лимит кончился -> он возьмет Groq.
# Если и там лимит -> возьмет OpenRouter.
MODELS_CHAIN = [
    {"provider": "nvidia", "model": "nvidia/nemotron-3.5-lightning-30b-a3b", "key": NVIDIA_NIM_API_KEY},
    {"provider": "groq", "model": "qwen/qwen3.8-27b", "key": GROQ_API_KEY},
    {"provider": "openrouter", "model": "qwen/qwen3.8-27b:free", "key": OPENROUTER_API_KEY},
]

SYSTEM_PROMPT = "Ты полезный ассистент. Отвечай кратко, по делу, с кодом если просят. Без воды."
DB_PATH = "bot_memory.db"

# --- БАЗА ДАННЫХ ---
async def init_db():
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("PRAGMA journal_mode=WAL")
        await db.execute("""CREATE TABLE IF NOT EXISTS history (
            user_id INTEGER, role TEXT, content TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )""")
        # Оставляем только последние 20 сообщений на пользователя
        await db.execute("""DELETE FROM history WHERE id NOT IN (
            SELECT id FROM history ORDER BY created_at DESC LIMIT 20
        )""")
        await db.commit()

async def get_history(user_id: int):
    async with aiosqlite.connect(DB_PATH) as db:
        cursor = await db.execute("SELECT role, content FROM history WHERE user_id = ? ORDER BY created_at ASC", (user_id,))
        return [{"role": r, "content": c} for r, c in await cursor.fetchall()]

async def save_history(user_id: int, role: str, content: str):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("INSERT INTO history (user_id, role, content) VALUES (?, ?, ?)", (user_id, role, content))
        await db.commit()

# --- ПОИСК (БЕЗЛИМИТНЫЙ) ---
async def perform_search(query: str) -> str:
    try:
        with DDGS() as ddgs:
            results = ddgs.text(query, max_results=3)
            if results:
                text = "\n\n🔍 **Поиск:**\n"
                for r in results:
                    text += f"- {r['title']}\n  {r['href']}\n"
                return text
    except:
        return ""
    return ""

# --- ВЫЗОВ МОДЕЛИ (С АВТОМАТИЧЕСКИМ ПЕРЕКЛЮЧЕНИЕМ) ---
async def get_ai_response(messages: List[dict], search_context: str = "") -> str:
    for m in MODELS_CHAIN:
        if not m["key"]:
            continue # Пропускаем, если ключа нет
        
        url = ""
        headers = {"Authorization": f"Bearer {m['key']}", "Content-Type": "application/json"}
        payload = {
            "model": m["model"],
            "messages": messages,
            "stream": True,
            "max_tokens": 1000
        }

        if m["provider"] == "groq":
            url = "https://api.groq.com/openai/v1/chat/completions"
        elif m["provider"] == "nvidia":
            url = "https://api.nvidia.com/v1/chat/completions"
        elif m["provider"] == "openrouter":
            url = "https://openrouter.ai/v1/chat/completions"
            headers["HTTP-Referer"] = "https://localhost"

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                async with client.stream("POST", url, json=payload, headers=headers) as resp:
                    if resp.status_code != 200:
                        logging.warning(f"{m['provider']} вернул ошибку {resp.status_code}. Пробуем следующую модель.")
                        continue # Переходим к следующей модели в списке
                    
                    full_text = ""
                    async for line in resp.aiter_lines():
                        if line.startswith("data: "):
                            chunk_data = line[6:]
                            if chunk_data == "[DONE]": break
                            try:
                                chunk = json.loads(chunk_data)
                                content = chunk.get("choices", [{}]).get("delta", {}).get("content", "")
                                if content: full_text += content
                            except: pass
                    
                    if len(full_text) > 10:
                        return full_text
        except Exception as e:
            logging.error(f"Ошибка связи с {m['provider']}: {e}")
            continue
            
    return "❌ Не удалось получить ответ от AI. Проверьте лимиты."

# --- БОТ (AIORGRAM) ---
bot = Bot(token=TELEGRAM_TOKEN)
dp = Dispatcher()

@dp.message(Command("start"))
async def cmd_start(message: types.Message):
    await message.answer("Привет! Я готов кодить. Напиши что-нибудь.")

@dp.message(F.text)
async def handle_message(message: types.Message):
    user_id = message.from_user.id
    user_text = message.text
    
    # 1. Получаем историю
    history = await get_history(user_id)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history
    messages.append({"role": "user", "content": user_text})

    # 2. Если в тексте есть слова "найди", "гугли", "ошибка" -> делаем поиск
    search_res = ""
    if any(word in user_text.lower() for word in ["найди", "гугли", "ошибка", "error", "не работает"]):
        await message.answer("🔍 Ищу информацию в интернете...")
        search_res = await perform_search(user_text)

    # 3. Отправляем в AI
    await message.answer("🤖 Думаю... (использую Groq, если занят — переключусь на резерв)")
    response = await get_ai_response(messages, search_res)
    
    # 4. Отправляем ответ (разбиваем, если длинный)
    if len(response) > 4000:
        for i in range(0, len(response), 4000):
            await message.answer(response[i:i+4000])
    else:
        await message.answer(response)

    # 5. Сохраняем в историю
    await save_history(user_id, "user", user_text)
    await save_history(user_id, "assistant", response)

async def main():
    await init_db()
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
