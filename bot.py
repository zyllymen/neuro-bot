# ============================================================
#  VIBE CODING BOT 2026 — Final Production Code
#  aiogram 3.x + streaming + vision + web search + memory
#  Free providers: NVIDIA NIM -> Groq -> OpenRouter (chain fallback)
# ============================================================

import os
import time
import json
import asyncio
import logging
import sqlite3
from typing import Optional

import httpx
import aiosqlite
from ddgs import DDGS
from aiohttp import web
from aiogram import Bot, Dispatcher, Router, F
from aiogram.types import Message, FSInputFile, BufferedInputFile
from aiogram.filters import Command, CommandObject
from aiogram.enums import ParseMode
from aiogram.client.default import DefaultBotProperties
from aiogram.exceptions import TelegramRetryAfter, TelegramBadRequest

# ============================================================
#  ЛОГИРОВАНИЕ
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ============================================================
#  КОНФИГУРАЦИЯ — провайдеры LLM (chain fallback)
# ============================================================
# Все три провайдера бесплатны, без карты, OpenAI-совместимые.
# Источник лимитов на октябрь 2026:
#   NVIDIA NIM: ~40 RPM, 10 000 RPD  -> https://build.nvidia.com
#   Groq:       30 RPM,  1 000 RPD  -> https://console.groq.com
#   OpenRouter: 20 RPM,    200 RPD  -> https://openrouter.ai/keys

PROVIDERS = [
    {
        "name": "nim",
        "base_url": "https://integrate.api.nvidia.com/v1",
        "api_key": os.getenv("NVIDIA_NIM_API_KEY", ""),
        "model": "openai/gpt-oss-20b",
        "max_tokens": 8000,
    },
    {
        "name": "groq",
        "base_url": "https://api.groq.com/openai/v1",
        "api_key": os.getenv("GROQ_API_KEY", ""),
        "model": "qwen/qwen3.8-27b",
        "max_tokens": 8000,
    },
    {
        "name": "openrouter",
        "base_url": "https://openrouter.ai/api/v1",
        "api_key": os.getenv("OPENROUTER_API_KEY", ""),
        "model": "qwen/qwen3.8-27b:free",
        "max_tokens": 8000,
    },
]

# Vision-модель (на OpenRouter, бесплатная, поддерживает изображения)
VISION_MODEL = "qwen/qwen3.8-27b:free"

# Tavily (опционально, для усиленного поиска)
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
TAVILY_URL = "https://api.tavily.com/search"

# Telegram
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")

# ============================================================
#  СИСТЕМНЫЙ ПРОМПТ
# ============================================================
SYSTEM_PROMPT = """Ты — продвинутый AI-ассистент нового поколения для vibe coding. Дата: Октябрь 2026.
Ты пишешь чистый, типизированный, production-ready код (FastAPI, React, Neon, Render по умолчанию).
Ты анализируешь загруженные файлы и фото. Ты используешь предоставленные факты из памяти и результаты поиска в интернете для максимально точных ответов.
Без воды. Без извинений. Только решения, код и архитектурные trade-offs.

Контекстная дата: октябрь 2026. Учитывай актуальные версии библиотек, API, фреймворков.
Мир AI развивается ускоренно: новые модели выходят почти еженедельно.
Пользователь занимается коммерческой разработкой с целью получения прибыли.
Учитывай бизнес-логику: время до запуска, стоимость поддержки, масштабируемость.

ПАМЯТЬ:
Ниже приведены факты о пользователе и его проекте, извлеченные из прошлых диалогов. Используй их как абсолютный контекст:
{USER_FACTS}

Если пользователь просит запомнить что-то важное (стек, архитектуру, баг), ответь: "ЗАПОМНИЛ: [краткая суть]", и система сохранит это автоматически.
"""

# Триггеры для автопоиска в интернете
SEARCH_TRIGGERS = [
    "новости", "2026", "ошибка", "найди", "тренд", "документация",
    "как исправить", "релиз", "обновление", "версия", "latest",
    "актуальн", "последн", "новый", "deprecated", "changelog",
]

# Разрешённые расширения файлов
ALLOWED_EXTENSIONS = {
    ".py", ".js", ".ts", ".tsx", ".jsx", ".json", ".md", ".txt",
    ".sql", ".html", ".css", ".yaml", ".yml", ".go", ".rs",
    ".java", ".cpp", ".c", ".h", ".sh", ".toml", ".env", ".xml",
}

DB_FILE = "memory.db"

# ============================================================
#  БАЗА ДАННЫХ (aiosqlite + WAL)
# ============================================================
db: Optional[aiosqlite.Connection] = None

async def init_db():
    global db
    db = await aiosqlite.connect(DB_FILE)
    # WAL-режим: предотвращает "database is locked" при конкурентных запросах
    await db.execute("PRAGMA journal_mode=WAL")
    await db.execute("PRAGMA synchronous=NORMAL")
    # Таблица фактов
    await db.execute("""
        CREATE TABLE IF NOT EXISTS facts (
            user_id TEXT,
            fact_content TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    # Таблица истории диалога
    await db.execute("""
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            role TEXT,
            content TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    # Таблица настроек (выбор модели)
    await db.execute("""
        CREATE TABLE IF NOT EXISTS settings (
            user_id TEXT PRIMARY KEY,
            provider TEXT DEFAULT 'auto'
        )
    """)
    await db.commit()
    logger.info("База данных инициализирована (WAL-режим)")

async def save_fact(user_id: str, fact: str):
    async with db.execute(
        "INSERT INTO facts (user_id, fact_content) VALUES (?, ?)",
        (str(user_id), fact),
    ):
        pass
    # Ограничиваем 15 фактами на пользователя
    await db.execute("""
        DELETE FROM facts WHERE user_id = ? AND id NOT IN (
            SELECT id FROM facts WHERE user_id = ? ORDER BY timestamp DESC LIMIT 15
        )
    """, (str(user_id), str(user_id)))
    await db.commit()

async def get_facts(user_id: str) -> str:
    async with db.execute(
        "SELECT fact_content FROM facts WHERE user_id = ? ORDER BY timestamp DESC LIMIT 15",
        (str(user_id),),
    ) as cursor:
        rows = await cursor.fetchall()
    if not rows:
        return "Нет сохранённых фактов."
    return "\n".join([f"- {r[0]}" for r in rows])

async def save_message(user_id: str, role: str, content: str):
    await db.execute(
        "INSERT INTO history (user_id, role, content) VALUES (?, ?, ?)",
        (str(user_id), role, content[:3000]),  # Ограничиваем длину
    )
    # Оставляем последние 20 сообщений (10 пар вопрос-ответ)
    await db.execute("""
        DELETE FROM history WHERE user_id = ? AND id NOT IN (
            SELECT id FROM history WHERE user_id = ? ORDER BY timestamp DESC LIMIT 20
        )
    """, (str(user_id), str(user_id)))
    await db.commit()

async def get_history(user_id: str) -> list:
    async with db.execute(
        "SELECT role, content FROM history WHERE user_id = ? ORDER BY timestamp ASC LIMIT 20",
        (str(user_id),),
    ) as cursor:
        rows = await cursor.fetchall()
    return [{"role": r[0], "content": r[1]} for r in rows]

async def clear_memory(user_id: str):
    await db.execute("DELETE FROM facts WHERE user_id = ?", (str(user_id),))
    await db.execute("DELETE FROM history WHERE user_id = ?", (str(user_id),))
    await db.commit()

async def get_provider(user_id: str) -> str:
    async with db.execute(
        "SELECT provider FROM settings WHERE user_id = ?",
        (str(user_id),),
    ) as cursor:
        row = await cursor.fetchone()
    return row[0] if row else "auto"

async def set_provider(user_id: str, provider: str):
    await db.execute(
        "INSERT OR REPLACE INTO settings (user_id, provider) VALUES (?, ?)",
        (str(user_id), provider),
    )
    await db.commit()

# ============================================================
#  ВЕБ-ПОИСК (ddgs — бесплатно, безлимитно, без ключа)
# ============================================================
def search_ddgs(query: str, max_results: int = 3) -> str:
    """Поиск через ddgs (DuckDuckGo metasearch). Безлимитно, без API-ключа."""
    try:
        results = DDGS().text(query, max_results=max_results)
        if not results:
            return ""
        parts = []
        for r in results:
            title = r.get("title", "")
            body = r.get("body", r.get("content", ""))[:300]
            href = r.get("href", r.get("url", ""))
            parts.append(f"Источник: {title}\nСсылка: {href}\nСуть: {body}")
        return "\n---\n".join(parts)
    except Exception as e:
        logger.warning(f"ddgs ошибка: {e}")
        return ""

async def search_tavily(query: str) -> str:
    """Усиленный поиск через Tavily. 1 кредит за запрос (basic)."""
    if not TAVILY_API_KEY:
        return ""
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.post(TAVILY_URL, json={
                "api_key": TAVILY_API_KEY,
                "query": query,
                "search_depth": "basic",
                "max_results": 3,
            })
            if resp.status_code != 200:
                return ""
            data = resp.json()
            results = data.get("results", [])
            parts = []
            for r in results:
                title = r.get("title", "")
                content = r.get("content", "")[:400]
                url = r.get("url", "")
                parts.append(f"Источник: {title}\nСсылка: {url}\nСуть: {content}")
            return "\n---\n".join(parts) if parts else ""
    except Exception as e:
        logger.warning(f"Tavily ошибка: {e}")
        return ""

async def do_search(query: str) -> str:
    """Сначала ddgs (бесплатно, безлимитно), если пусто — Tavily."""
    # ddgs синхронный, запускаем в потоке
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, search_ddgs, query)
    if result:
        return result
    # Fallback на Tavily
    tavily_result = await search_tavily(query)
    return tavily_result or "Поиск не дал результатов."

# ============================================================
#  ЗАПРОС К LLM (chain fallback + streaming)
# ============================================================
async def call_llm_stream(
    messages: list,
    user_id: str,
    bot: Bot,
    chat_id: int,
):
    """
    Streaming-запрос к LLM.
    Перебирает провайдеров по цепочке. При 429/5xx переключается на следующий.
    Текст печатается в Telegram через send_message_draft.
    """
    provider_pref = await get_provider(user_id)

    # Формируем очередь провайдеров
    if provider_pref == "auto":
        queue = list(PROVIDERS)
    else:
        # Выбранный провайдер первым, остальные как fallback
        queue = [p for p in PROVIDERS if p["name"] == provider_pref]
        queue += [p for p in PROVIDERS if p["name"] != provider_pref]

    for provider in queue:
        if not provider["api_key"]:
            continue
        try:
            logger.info(f"Запрос к провайдеру: {provider['name']} ({provider['model']})")
            return await _stream_from_provider(
                provider, messages, bot, chat_id
            )
        except httpx.HTTPStatusError as e:
            status = e.response.status_code
            logger.warning(f"{provider['name']} вернул {status}, переключаюсь...")
            if status in (429, 500, 502, 503, 504):
                continue
            else:
                raise
        except Exception as e:
            logger.warning(f"{provider['name']} ошибка: {e}, переключаюсь...")
            continue

    # Все провайдеры упали
    return None

async def _stream_from_provider(provider, messages, bot, chat_id):
    """Стримит ответ от одного провайдера через send_message_draft."""
    headers = {
        "Authorization": f"Bearer {provider['api_key']}",
        "Content-Type": "application/json",
    }
    # OpenRouter требует HTTP-Referer и X-Title
    if "openrouter" in provider["base_url"]:
        headers["HTTP-Referer"] = "https://github.com/vibe-coding-bot"
        headers["X-Title"] = "VibeCodingBot2026"

    payload = {
        "model": provider["model"],
        "messages": messages,
        "temperature": 0.3,
        "max_tokens": provider["max_tokens"],
        "stream": True,
    }

    draft_id = int(time.time() * 1000) % 2147483647
    full_text = ""
    last_update = 0.0
    update_interval = 0.7  # 700ms между обновлениями черновика
    first_token = True

    async with httpx.AsyncClient(timeout=120) as client:
        async with client.stream(
            "POST",
            provider["base_url"] + "/chat/completions",
            headers=headers,
            json=payload,
        ) as response:
            if response.status_code != 200:
                body = await response.aread()
                raise httpx.HTTPStatusError(
                    f"HTTP {response.status_code}: {body[:200]}",
                    request=response.request,
                    response=response,
                )

            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data.strip() == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                    delta = chunk.get("choices", [{}])[0].get("delta", {})
                    content = delta.get("content", "")
                    if content:
                        full_text += content
                        # Обновляем черновик не чаще, чем раз в 700ms
                        now = time.monotonic()
                        if first_token or (now - last_update >= update_interval):
                            try:
                                await bot.send_message_draft(
                                    draft_id=draft_id,
                                    chat_id=chat_id,
                                    text=full_text,
                                )
                                last_update = now
                                first_token = False
                            except TelegramRetryAfter as e:
                                logger.info(f"FloodWait: спим {e.retry_after} сек")
                                await asyncio.sleep(e.retry_after + 0.5)
                                # Повторяем обновление после ожидания
                                try:
                                    await bot.send_message_draft(
                                        draft_id=draft_id,
                                        chat_id=chat_id,
                                        text=full_text,
                                    )
                                    last_update = time.monotonic()
                                except Exception:
                                    pass
                            except Exception as e:
                                logger.debug(f"Draft update skip: {e}")
                except json.JSONDecodeError:
                    continue

    return full_text

# ============================================================
#  ОТПРАВКА СООБЩЕНИЙ (safe_send с fallback)
# ============================================================
async def safe_send(message: Message, text: str):
    """Отправляет сообщение. Пробует MarkdownV2 -> HTML -> plain text."""
    # Если текст слишком длинный — разбиваем по абзацам
    if len(text) > 4000:
        chunks = split_text(text, 4000)
        for chunk in chunks:
            await safe_send(message, chunk)
        return

    # Пробуем MarkdownV2
    try:
        await message.answer(text, parse_mode=ParseMode.MARKDOWN_V2)
        return
    except TelegramBadRequest:
        pass
    # Пробуем HTML
    try:
        await message.answer(text, parse_mode=ParseMode.HTML)
        return
    except TelegramBadRequest:
        pass
    # Plain text (без форматирования)
    await message.answer(text)

def split_text(text: str, max_len: int = 4000) -> list:
    """Разбивает текст по абзацам (\n\n), не разрывая блоки кода."""
    paragraphs = text.split("\n\n")
    chunks = []
    current = ""
    for para in paragraphs:
        if len(current) + len(para) + 2 > max_len:
            if current:
                chunks.append(current)
            current = para
        else:
            current = current + "\n\n" + para if current else para
    if current:
        chunks.append(current)
    return chunks

# ============================================================
#  VISION (анализ изображений)
# ============================================================
async def analyze_image(bot: Bot, message: Message, caption: str = "") -> str:
    """
    Скачивает фото, отправляет к vision-модели через OpenRouter.
    """
    openrouter_key = os.getenv("OPENROUTER_API_KEY", "")
    if not openrouter_key:
        return "Для анализа изображений нужен OpenRouter API-ключ. Получите бесплатно на openrouter.ai/keys"

    # Берём самое большое фото
    photo = message.photo[-1]
    file = await bot.get_file(photo.file_id)
    # Скачиваем
    photo_bytes = await bot.download_file(file.file_path)

    import base64
    b64 = base64.b64encode(photo_bytes.read()).decode("utf-8")
    image_url = f"data:image/jpeg;base64,{b64}"

    question = caption if caption else "Опиши, что на этом изображении, и найди технические детали."

    headers = {
        "Authorization": f"Bearer {openrouter_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/vibe-coding-bot",
        "X-Title": "VibeCodingBot2026",
    }

    payload = {
        "model": VISION_MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            }
        ],
        "temperature": 0.3,
        "max_tokens": 4000,
    }

    try:
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload,
            )
            if resp.status_code != 200:
                return f"Ошибка Vision API: {resp.status_code}"
            data = resp.json()
            return data["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Ошибка анализа изображения: {e}"

# ============================================================
#  ОБРАБОТКА ФАЙЛОВ
# ============================================================
async def read_document(bot: Bot, message: Message) -> tuple:
    """Читает содержимое загруженного файла. Возвращает (filename, content) или None."""
    doc = message.document
    if not doc:
        return None

    filename = doc.file_name or "unknown"
    ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in ALLOWED_EXTENSIONS:
        return None

    # Проверяем размер (до 1 МБ)
    if doc.file_size and doc.file_size > 1_048_576:
        return (filename, "Файл слишком большой (макс. 1 МБ)")

    try:
        file = await bot.get_file(doc.file_id)
        file_bytes = await bot.download_file(file.file_path)
        content = file_bytes.read().decode("utf-8", errors="replace")[:10000]
        return (filename, content)
    except Exception as e:
        logger.warning(f"Ошибка чтения файла: {e}")
        return (filename, f"Ошибка чтения: {e}")

# ============================================================
#  TELEGRAM BOT — роутер и хендлеры
# ============================================================
router = Router()
bot = Bot(
    token=TELEGRAM_TOKEN,
    default=DefaultBotProperties(parse_mode=ParseMode.HTML),
)

@router.message(Command("start"))
async def cmd_start(message: Message):
    await message.answer(
        "🚀 <b>Vibe Coding Assistant 2026</b>\n\n"
        "Я твой технический партнёр для разработки.\n"
        "Пишу код, анализирую файлы и фото, ищу в интернете, помню контекст проекта.\n\n"
        "<b>Команды:</b>\n"
        "• Просто напиши задачу — решу.\n"
        "• <code>ЗАПОМНИ: [факт]</code> — сохраню в память.\n"
        "• <code>/model</code> — выбор провайдера LLM.\n"
        "• <code>/search [запрос]</code> — поиск в интернете.\n"
        "• <code>/facts</code> — показать память.\n"
        "• <code>/clear</code> — очистить память.\n"
        "• Кинь фото — проанализирую.\n"
        "• Кинь файл (.py, .js, .json и т.д.) — прочитаю и разберу.\n\n"
        "Стек по умолчанию: FastAPI, React, Neon, Render.\n"
        "Провайдеры: NVIDIA NIM → Groq → OpenRouter (бесплатно)."
    )

@router.message(Command("model"))
async def cmd_model(message: Message, command: CommandObject):
    user_id = str(message.from_user.id)
    if not command.args:
        current = await get_provider(user_id)
        await message.answer(
            f"Текущий провайдер: <b>{current}</b>\n\n"
            "Доступные:\n"
            "• <code>/model auto</code> — chain fallback (рекомендуется)\n"
            "• <code>/model nim</code> — NVIDIA NIM (10 000 RPD)\n"
            "• <code>/model groq</code> — Groq (1 000 RPD, самый быстрый)\n"
            "• <code>/model openrouter</code> — OpenRouter (200 RPD, vision)"
        )
        return
    arg = command.args.strip().lower()
    valid = {"auto", "nim", "groq", "openrouter"}
    if arg not in valid:
        await message.answer(f"Неизвестный провайдер. Доступные: {', '.join(valid)}")
        return
    await set_provider(user_id, arg)
    await message.answer(f"✅ Провайдер переключён на <b>{arg}</b>")

@router.message(Command("search"))
async def cmd_search(message: Message, command: CommandObject):
    if not command.args:
        await message.answer("Напиши запрос: <code>/search React 19 новые фичи</code>")
        return
    await message.answer("🔍 Ищю...")
    results = await do_search(command.args)
    await safe_send(message, results)

@router.message(Command("facts"))
async def cmd_facts(message: Message):
    user_id = str(message.from_user.id)
    facts = await get_facts(user_id)
    await message.answer(f"📋 <b>Память:</b>\n{facts}")

@router.message(Command("clear"))
async def cmd_clear(message: Message):
    user_id = str(message.from_user.id)
    await clear_memory(user_id)
    await message.answer("🗑 Память очищена (факты + история).")

@router.message(F.photo)
async def handle_photo(message: Message):
    """Анализ изображений через Vision-модель."""
    caption = message.caption or ""
    await message.answer("👁 Анализирую изображение...")
    result = await analyze_image(bot, message, caption)
    await safe_send(message, result)

@router.message(F.document)
async def handle_document(message: Message):
    """Чтение и анализ загруженных файлов."""
    doc_info = await read_document(bot, message)
    if doc_info is None:
        await message.answer(
            "Поддерживаемые форматы: " + ", ".join(sorted(ALLOWED_EXTENSIONS))
        )
        return
    filename, content = doc_info
    caption = message.caption or "Проанализируй этот файл и предложи улучшения."

    user_id = str(message.from_user.id)
    facts = await get_facts(user_id)
    history = await get_history(user_id)

    # Формируем промпт с содержимым файла
    file_context = f'Пользователь загрузил файл "{filename}". Содержимое:\n```\n{content}\n```'
    user_msg = f"{file_context}\n\nЗапрос: {caption}"

    system = SYSTEM_PROMPT.format(USER_FACTS=facts)
    messages = [{"role": "system", "content": system}]
    messages.extend(history)
    messages.append({"role": "user", "content": user_msg})

    await message.answer("📄 Анализирую файл, стримлю ответ...")
    full_text = await call_llm_stream(messages, user_id, bot, message.chat.id)
    if full_text:
        # Финальное сообщение (черновик эфемерный, нужно отправить настоящее)
        await safe_send(message, full_text)
        await save_message(user_id, "user", user_msg[:2000])
        await save_message(user_id, "assistant", full_text[:2000])
    else:
        await message.answer("❌ Все провайдеры недоступны. Попробуй позже.")

@router.message(F.text)
async def handle_text(message: Message):
    user_text = message.text
    user_id = str(message.from_user.id)

    # Команда ЗАПОМНИ:
    if user_text.upper().startswith("ЗАПОМНИ:"):
        fact = user_text[8:].strip()
        await save_fact(user_id, fact)
        await message.answer(f"✅ Запомнил: {fact}")
        return

    # Сборка контекста
    facts = await get_facts(user_id)
    history = await get_history(user_id)

    # Автопоиск
    search_context = ""
    lower_text = user_text.lower()
    if any(trigger in lower_text for trigger in SEARCH_TRIGGERS):
        search_results = await do_search(user_text)
        if search_results and "не дал результатов" not in search_results:
            search_context = f"\n\n[АКТУАЛЬНЫЕ ДАННЫЕ ИЗ ИНТЕРНЕТА]:\n{search_results}\nИспользуй эти данные для ответа."

    system = SYSTEM_PROMPT.format(USER_FACTS=facts) + search_context
    messages = [{"role": "system", "content": system}]
    messages.extend(history)
    messages.append({"role": "user", "content": user_text})

    # Стриминг ответа
    full_text = await call_llm_stream(messages, user_id, bot, message.chat.id)
    if full_text:
        await safe_send(message, full_text)
        await save_message(user_id, "user", user_text[:2000])
        await save_message(user_id, "assistant", full_text[:2000])
    else:
        await message.answer("❌ Все провайдеры недоступны. Попробуй позже.")

# ============================================================
#  HEALTH CHECK (aiohttp.web в том же event loop)
# ============================================================
async def health_handler(request):
    return web.Response(text="OK", status=200)

async def root_handler(request):
    return web.Response(text="🤖 Vibe Coding Bot 2026 is LIVE", status=200)

async def start_web_server():
    app_web = web.Application()
    app_web.router.add_get("/health", health_handler)
    app_web.router.add_get("/", root_handler)
    runner = web.AppRunner(app_web)
    await runner.setup()
    port = int(os.environ.get("PORT", 10000))
    site = web.TCPSite(runner, "0.0.0.0", port)
    await site.start()
    logger.info(f"Health-check запущен на порту {port}")

# ============================================================
#  MAIN
# ============================================================
async def main():
    # Валидация ключей
    missing = []
    if not TELEGRAM_TOKEN:
        missing.append("TELEGRAM_TOKEN")
    if not any(p["api_key"] for p in PROVIDERS):
        missing.append("Хотя бы один LLM API ключ (NVIDIA_NIM_API_KEY / GROQ_API_KEY / OPENROUTER_API_KEY)")
    if missing:
        for m in missing:
            logger.error(f"ОТСУТСТВУЕТ: {m}")
        logger.error("Задайте переменные окружения и перезапустите.")
        return

    # Инициализация БД
    await init_db()

    # Запуск веб-сервера (health-check)
    await start_web_server()

    # Запуск бота
    dp = Dispatcher()
    dp.include_router(router)
    logger.info("Бот запущен. Ожидание сообщений...")
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
