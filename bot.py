import os
import asyncio
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime
from contextlib import asynccontextmanager

import aiosqlite
import httpx
from aiogram import Bot, Dispatcher, types, F
from aiogram.filters import Command
from aiogram.enums import ParseMode
from aiohttp import web

# --- Configuration & Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

REQUIRED_VARS = ["TELEGRAM_TOKEN", "OPENROUTER_API_KEY", "TAVILY_API_KEY"]
for var in REQUIRED_VARS:
    if not os.getenv(var):
        logger.error(f"❌ Missing required environment variable: {var}")
        raise SystemExit(1)

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
PORT = int(os.getenv("PORT", "8000"))

DB_PATH = "bot_db.sqlite"

# --- Database Setup ---
async def init_db():
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("PRAGMA journal_mode=WAL")
        await db.execute("PRAGMA synchronous=NORMAL")
        
        # Facts Table
        await db.execute("""
            CREATE TABLE IF NOT EXISTS facts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                fact_type TEXT NOT NULL,
                fact_content TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        await db.execute("CREATE INDEX IF NOT EXISTS idx_facts_user ON facts(user_id)")

        # History Table
        await db.execute("""
            CREATE TABLE IF NOT EXISTS history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        await db.execute("CREATE INDEX IF NOT EXISTS idx_history_user ON history(user_id)")

        # User Settings Table
        await db.execute("""
            CREATE TABLE IF NOT EXISTS user_settings (
                user_id INTEGER PRIMARY KEY,
                preferred_model TEXT DEFAULT 'qwen/qwen3-coder:free'
            )
        """)
        await db.commit()

async def get_user_model(user_id: int) -> str:
    async with aiosqlite.connect(DB_PATH) as db:
        cursor = await db.execute(
            "SELECT preferred_model FROM user_settings WHERE user_id = ?", (user_id,)
        )
        row = await cursor.fetchone()
        if row:
            return row
        await db.execute(
            "INSERT OR IGNORE INTO user_settings (user_id, preferred_model) VALUES (?, ?)",
            (user_id, "qwen/qwen3-coder:free")
        )
        await db.commit()
        return "qwen/qwen3-coder:free"

async def set_user_model(user_id: int, model: str) -> bool:
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT OR REPLACE INTO user_settings (user_id, preferred_model) VALUES (?, ?)",
            (user_id, model)
        )
        await db.commit()
    return True

async def save_fact(user_id: int, fact_type: str, content: str):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT INTO facts (user_id, fact_type, fact_content) VALUES (?, ?, ?)",
            (user_id, fact_type, content)
        )
        await db.commit()

async def get_facts(user_id: int) -> List[Dict[str, Any]]:
    async with aiosqlite.connect(DB_PATH) as db:
        cursor = await db.execute(
            "SELECT fact_type, fact_content, timestamp FROM facts WHERE user_id = ? ORDER BY id DESC LIMIT 10",
            (user_id,)
        )
        rows = await cursor.fetchall()
        return [{"type": r, "content": r, "time": r} for r in rows]

async def clear_facts(user_id: int):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("DELETE FROM facts WHERE user_id = ?", (user_id,))
        await db.commit()

async def save_message_to_history(user_id: int, role: str, content: str):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT INTO history (user_id, role, content) VALUES (?, ?, ?)",
            (user_id, role, content)
        )
        # Keep only last 20 messages per user
        await db.execute("""
            DELETE FROM history WHERE id NOT IN (
                SELECT id FROM history WHERE user_id = ? ORDER BY timestamp DESC LIMIT 20
            )
        """, (user_id,))
        await db.commit()

async def get_history(user_id: int) -> List[Dict[str, str]]:
    async with aiosqlite.connect(DB_PATH) as db:
        cursor = await db.execute(
            "SELECT role, content FROM history WHERE user_id = ? ORDER BY timestamp ASC",
            (user_id,)
        )
        rows = await cursor.fetchall()
        return [{"role": r, "content": r} for r in rows]

# --- LLM & Search Clients ---
async def call_llm(messages: List[Dict[str, str]], model: str, temperature: float = 0.3) -> Optional[str]:
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "HTTP-Referer": "https://localhost",
        "X-Title": "VibeBot"
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": 12000,
        "stream": False
    }

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(url, json=payload, headers=headers)
            if resp.status_code != 200:
                logger.error(f"LLM Error {resp.status_code}: {resp.text}")
                return None
            data = resp.json()
            return data.get("choices", [{}]).get("message", {}).get("content")
    except Exception as e:
        logger.exception(f"LLM Request Failed: {e}")
        return None

async def perform_search(query: str) -> Optional[str]:
    url = "https://api.tavily.com/search"
    headers = {"x-api-key": TAVILY_API_KEY}
    payload = {
        "query": query,
        "search_depth": "basic",
        "max_results": 3,
        "include_answer": True
    }

    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.post(url, json=payload, headers=headers)
            if resp.status_code != 200:
                logger.error(f"Search Error {resp.status_code}")
                return None
            data = resp.json()
            if data.get("results"):
                res_text = "🔍 **Search Results:**\n"
                for r in data["results"]:
                    res_text += f"- {r['title']}\n  {r['url']}\n"
                if data.get("answer"):
                    res_text += f"\n📝 **Direct Answer:** {data['answer']}"
                return res_text
            return None
    except Exception as e:
        logger.exception(f"Search Request Failed: {e}")
        return None

async def analyze_image(image_url: str, caption: Optional[str]) -> Optional[str]:
    # Using a generic vision model available on OpenRouter as of 2026 context
    model = "qwen/qwen3-vision:free" 
    messages = [
        {"role": "system", "content": "You are a helpful assistant that describes images."},
        {"role": "user", "content": [
            {"type": "text", "text": f"Describe this image. Caption provided: {caption}" if caption else "Describe this image."},
            {"type": "image_url", "image_url": {"url": image_url}}
        ]}
    ]
    return await call_llm(messages, model, temperature=0.7)

async def process_file(file_path: str) -> Optional[str]:
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read(10000) # Limit to 10k chars
        return content
    except Exception as e:
        logger.error(f"File read error: {e}")
        return None

# --- Message Handling Utilities ---
async def safe_send(message: types.Message, text: str):
    if not text:
        return
    
    # Split by paragraphs first
    chunks = 
    current_chunk = ""
    for paragraph in text.split("\n\n"):
        if len(current_chunk) + len(paragraph) + 2 <= 4000:
            current_chunk += (f"\n\n{paragraph}" if current_chunk else paragraph)
        else:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = paragraph
    if current_chunk:
        chunks.append(current_chunk)

    for chunk in chunks:
        try:
            await message.answer(chunk, parse_mode=ParseMode.MARKDOWN_V2)
        except Exception:
            try:
                await message.answer(chunk) # Fallback to plain text
            except Exception as e:
                logger.error(f"Failed to send message: {e}")

# --- Bot Handlers ---
async def handle_start(message: types.Message):
    await message.answer("👋 Привет! Я твой AI-ассистент. Используй /model для смены модели, /facts для памяти, /search для поиска.")

async def handle_model_command(message: types.Message):
    args = message.text.split(maxsplit=1)
    if len(args) < 2:
        current = await get_user_model(message.from_user.id)
        await message.answer(f"🤖 Текущая модель: `{current}`\nДоступные: `qwen/qwen3-coder:free`, `deepseek/deepseek-chat`")
        return
    
    model_name = args.strip()
    valid_models = ["qwen/qwen3-coder:free", "deepseek/deepseek-chat"]
    if model_name not in valid_models:
        await message.answer("⚠️ Неверная модель. Доступные: `qwen/qwen3-coder:free`, `deepseek/deepseek-chat`")
        return
    
    set_user_model(message.from_user.id, model_name)
    await message.answer(f"✅ Модель изменена на: `{model_name}`")

async def handle_facts_command(message: types.Message):
    facts = await get_facts(message.from_user.id)
    if not facts:
        await message.answer("📜 У тебя пока нет сохраненных фактов.")
        return
    
    text = "🧠 **Твои факты:**\n\n"
    for i, f in enumerate(facts, 1):
        text += f"{i}. [{f['type']}] {f['content']}\n"
    await safe_send(message, text)

async def handle_clear_command(message: types.Message):
    await clear_facts(message.from_user.id)
    await message.answer("🗑 Память очищена.")

async def handle_search_command(message: types.Message):
    args = message.text.split(maxsplit=1)
    if len(args) < 2:
        await message.answer("🔍 Используй: `/search твой запрос`")
        return
    
    query = args
    await message.answer("🔍 Ищу информацию...")
    result = await perform_search(query)
    if result:
        await safe_send(message, result)
    else:
        await message.answer("Ничего не найдено.")

async def handle_text_message(message: types.Message):
    user_text = message.text
    user_id = message.from_user.id
    
    # Check for memory trigger
    if user_text.upper().startswith("ЗАПОМНИ:"):
        fact_content = user_text[8:].strip()
        fact_type = "general"
        await save_fact(user_id, fact_type, fact_content)
        await message.answer("💾 Запомнил!")
        return

    # Check for auto-search trigger
    search_trigger_words = ["новости", "2026", "ошибка", "найди", "тренд", "документация", "как исправить"]
    search_context = ""
    if any(word in user_text.lower() for word in search_trigger_words):
        await message.answer("🔍 Ищу актуальные данные...")
        search_res = await perform_search(user_text)
        if search_res:
            search_context = search_res

    # Get history
    history = await get_history(user_id)
    system_prompt = "Ты полезный ассистент для кодинга и общих задач. Отвечай кратко и по делу."
    if search_context:
        system_prompt += f"\n\nКонтекст поиска:\n{search_context}"
    
    messages = [{"role": "system", "content": system_prompt}] + history
    messages.append({"role": "user", "content": user_text})

    # Get model
    model = await get_user_model(user_id)
    temp = 0.3 if "coder" in model.lower() else 0.7

    await message.answer("🤖 Думаю...")
    response = await call_llm(messages, model, temp)
    
    if response:
        await safe_send(message, response)
        await save_message_to_history(user_id, "user", user_text)
        await save_message_to_history(user_id, "assistant", response)
    else:
        await message.answer("❌ Не удалось получить ответ от AI. Попробуй позже.")

async def handle_photo(message: types.Message):
    if not message.photo:
        return
    
    photo = message.photo[-1] # Get highest resolution
    file = await message.bot.get_file(photo.file_id)
    file_url = f"https://api.telegram.org/file/bot{TELEGRAM_TOKEN}/{file.file_path}"
    caption = message.caption
    
    await message.answer("🖼 Анализирую изображение...")
    description = await analyze_image(file_url, caption)
    
    if description:
        await safe_send(message, f"👁 Описание изображения:\n{description}")
        await save_message_to_history(message.from_user.id, "user", f"[Image: {caption or 'no caption'}]")
        await save_message_to_history(message.from_user.id, "assistant", description)
    else:
        await message.answer("Не удалось проанализировать изображение.")

async def handle_document(message: types.Message):
    doc = message.document
    allowed_exts = [".py", ".js", ".ts", ".json", ".md", ".txt", ".sql", ".html", ".css", ".yaml", ".yml", ".xml", ".csv"]
    
    if not any(doc.file_name.lower().endswith(ext) for ext in allowed_exts):
        return # Ignore unsupported files

    await message.answer("📄 Скачиваю и читаю файл...")
    file = await message.bot.get_file(doc.file_id)
    file_url = f"https://api.telegram.org/file/bot{TELEGRAM_TOKEN}/{file.file_path}"
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(file_url)
            resp.raise_for_status()
            content = resp.text[:10000]
        
        model = await get_user_model(message.from_user.id)
        system_prompt = f"Ты ассистент по коду. Вот содержимое файла:\n\n{content}\n\nОтветь на вопрос пользователя, используя этот контекст."
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": message.caption or "Проанализируй этот файл и скажи, что в нем интересного или есть ли ошибки."}
        ]
        
        await message.answer("🤖 Анализирую код...")
        response = await call_llm(messages, model, 0.3)
        
        if response:
            await safe_send(message, response)
            await save_message_to_history(message.from_user.id, "user", f"[File: {doc.file_name}]")
            await save_message_to_history(message.from_user.id, "assistant", response)
        else:
            await message.answer("Не удалось проанализировать файл.")
            
    except Exception as e:
        logger.exception(f"File processing error: {e}")
        await message.answer("Ошибка при обработке файла.")

# --- Health Check Server ---
async def handle_health(request):
    return web.Response(text="OK")

async def start_health_server():
    app = web.Application()
    app.router.add_get("/", handle_health)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", PORT)
    await site.start()
    logger.info(f"✅ Health server started on port {PORT}")
    return runner

# --- Main Entry Point ---
async def main():
    await init_db()
    bot = Bot(token=TELEGRAM_TOKEN)
    dp = Dispatcher()
    
    dp.message.register(handle_start, Command("start"))
    dp.message.register(handle_model_command, Command("model"))
    dp.message.register(handle_facts_command, Command("facts"))
    dp.message.register(handle_clear_command, Command("clear"))
    dp.message.register(handle_search_command, Command("search"))
    dp.message.register(handle_text_message, F.text)
    dp.message.register(handle_photo, F.photo)
    dp.message.register(handle_document, F.document)
    
    # Start health server in background
    health_runner = await start_health_server()
    
    try:
        logger.info("🚀 Starting bot polling...")
        await dp.start_polling(bot)
    finally:
        await health_runner.cleanup()
        await bot.session.close()

if __name__ == "__main__":
    asyncio.run(main())

