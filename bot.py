import os
import asyncio
import logging
import sqlite3
import json
import re
from typing import Any

import httpx
from aiogram import Bot, Dispatcher, types, F
from aiogram.filters import Command
from aiogram.exceptions import TelegramBadRequest
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
import aiosqlite
from aiohttp import web

# ============================================================================
# ЛОГИРОВАНИЕ
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger("vibe-bot")

# ============================================================================
# КОНФИГУРАЦИЯ
# ============================================================================
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")

DB_PATH = "memory.db"
MAX_HISTORY = 20          # сколько сообщений истории хранить
MAX_FACTS = 15            # максимум фактов в памяти
MAX_FILE_CHARS = 10000    # лимит содержимого файла
MAX_SEARCH_RESULTS = 3    # результатов поиска ddgs
MAX_TAVILY_RESULTS = 5    # результатов Tavily
MESSAGE_CHUNK_SIZE = 4000 # размер куска сообщения для Telegram

# ============================================================================
# МОДЕЛИ (актуально на октябрь 2026)
# ============================================================================
MODELS: dict[str, dict[str, Any]] = {
    "groq": {
        "label": "Qwen3.8 27B — Groq (быстрый, по умолчанию)",
        "model": "qwen/qwen3.8-27b",
        "base_url": "https://api.groq.com/openai/v1",
        "key": GROQ_API_KEY,
        "max_tokens": 8000,
        "vision": False,
    },
    "nemotron": {
        "label": "Nemotron 3 Super 120B — OpenRouter (тяжёлые задачи)",
        "model": "nvidia/nemotron-3-super-120b-a12b:free",
        "base_url": "https://openrouter.ai/api/v1",
        "key": OPENROUTER_API_KEY,
        "max_tokens": 8000,
        "vision": False,
    },
    "gemma": {
        "label": "Gemma 4 26B Vision — OpenRouter (vision + текст)",
        "model": "google/gemma-4-26b-a4b-it:free",
        "base_url": "https://openrouter.ai/api/v1",
        "key": OPENROUTER_API_KEY,
        "max_tokens": 8000,
        "vision": True,
    },
    "qwen-vision": {
        "label": "Qwen3.8 27B Vision — OpenRouter (vision + код)",
        "model": "qwen/qwen3.8-27b:free",
        "base_url": "https://openrouter.ai/api/v1",
        "key": OPENROUTER_API_KEY,
        "max_tokens": 8000,
        "vision": True,
    },
    "cohere": {
        "label": "North Mini Code — OpenRouter (кодинг-специалист)",
        "model": "cohere/north-mini-code:free",
        "base_url": "https://openrouter.ai/api/v1",
        "key": OPENROUTER_API_KEY,
        "max_tokens": 8000,
        "vision": False,
    },
}

DEFAULT_MODEL = "groq"
VISION_MODEL = "gemma"  # для анализа изображений

# Chain fallback: если основная модель упала (429/5xx), пробуем следующие
FALLBACK_CHAIN = ["groq", "nemotron", "cohere"]

# Триггеры для автопоиска
SEARCH_TRIGGERS = [
    "найди", "поищи", "новости", "2026", "2025", "тренд", "ошибка",
    "документация", "как исправить", "релиз", "обновление", "актуальн",
    "последн", "latest", "changelog", "версия", "выйти", "вышел",
]

# Расширения файлов, которые бот читает
ALLOWED_EXTENSIONS = {
    ".py", ".js", ".ts", ".jsx", ".tsx", ".json", ".md", ".txt",
    ".sql", ".html", ".css", ".yaml", ".yml", ".go", ".rs", ".java",
    ".cpp", ".c", ".h", ".sh", ".toml", ".ini", ".env", ".dockerfile",
    ".gitignore", ".cfg",
}

# ============================================================================
# СИСТЕМНЫЙ ПРОМПТ
# ============================================================================
SYSTEM_PROMPT = """\
Ты — продвинутый AI-ассистент нового поколения для vibe coding. \
Дата: Октябрь 2026.

Ты пишешь чистый, типизированный, production-ready код \
(FastAPI, React, Neon, Render по умолчанию).
Ты анализируешь загруженные файлы и фото.
Ты используешь предоставленные факты из памяти и результаты поиска \
в интернете для максимально точных ответов.

Без воды. Без извинений. Только решения, код и архитектурные trade-offs.

ПРАВИЛА:
1. Выдавай готовый код, не «реализуй здесь сами».
2. Если есть результаты поиска — опирайся на них, не выдумывай факты.
3. Учитывай факты из памяти пользователя как абсолютный контекст.
4. Код — в блоках с указанием языка.
5. Длинные ответы разбивай на разделы через ##.
"""

# ============================================================================
# ВАЛИДАЦИЯ ОКРУЖЕНИЯ
# ============================================================================
def validate_env() -> list[str]:
    missing = []
    if not TELEGRAM_TOKEN:
        missing.append("TELEGRAM_TOKEN — получи у @BotFather")
    if not GROQ_API_KEY and not OPENROUTER_API_KEY:
        missing.append("Нужен хотя бы один: GROQ_API_KEY или OPENROUTER_API_KEY")
    if missing:
        for m in missing:
            log.error("ОТСУТСТВУЕТ: %s", m)
    return missing

# ============================================================================
# БАЗА ДАННЫХ (aiosqlite + WAL)
# ============================================================================
async def init_db() -> aiosqlite.Connection:
    db = await aiosqlite.connect(DB_PATH)
    await db.execute("PRAGMA journal_mode=WAL")
    await db.execute("PRAGMA synchronous=NORMAL")
    await db.execute("PRAGMA foreign_keys=ON")

    await db.execute("""
        CREATE TABLE IF NOT EXISTS facts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            content TEXT NOT NULL,
            created_at TEXT DEFAULT (datetime('now'))
        )
    """)
    await db.execute("""
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TEXT DEFAULT (datetime('now'))
        )
    """)
    await db.execute("""
        CREATE TABLE IF NOT EXISTS user_settings (
            user_id INTEGER PRIMARY KEY,
            model_key TEXT NOT NULL DEFAULT 'groq'
        )
    """)
    await db.commit()
    log.info("База данных инициализирована (WAL mode)")
    return db

async def db_save_fact(db: aiosqlite.Connection, user_id: int, content: str):
    count = await (await db.execute(
        "SELECT COUNT(*) FROM facts WHERE user_id = ?", (user_id,)
    )).fetchone()
    if count[0] >= MAX_FACTS:
        await db.execute(
            "DELETE FROM facts WHERE id = (SELECT MIN(id) FROM facts WHERE user_id = ?)",
            (user_id,)
        )
    await db.execute(
        "INSERT INTO facts (user_id, content) VALUES (?, ?)", (user_id, content)
    )
    await db.commit()

async def db_get_facts(db: aiosqlite.Connection, user_id: int) -> list[str]:
    cursor = await db.execute(
        "SELECT content FROM facts WHERE user_id = ? ORDER BY created_at DESC", (user_id,)
    )
    rows = await cursor.fetchall()
    return [r[0] for r in rows]

async def db_clear_facts(db: aiosqlite.Connection, user_id: int):
    await db.execute("DELETE FROM facts WHERE user_id = ?", (user_id,))
    await db.commit()

async def db_add_message(db: aiosqlite.Connection, user_id: int, role: str, content: str):
    await db.execute(
        "INSERT INTO history (user_id, role, content) VALUES (?, ?, ?)",
        (user_id, role, content[:8000])
    )
    # Удаляем старые, оставляем последние MAX_HISTORY
    await db.execute("""
        DELETE FROM history WHERE id NOT IN (
            SELECT id FROM history WHERE user_id = ? ORDER BY id DESC LIMIT ?
        ) AND user_id = ?
    """, (user_id, MAX_HISTORY, user_id))
    await db.commit()

async def db_get_history(db: aiosqlite.Connection, user_id: int) -> list[dict]:
    cursor = await db.execute(
        "SELECT role, content FROM history WHERE user_id = ? ORDER BY id ASC", (user_id,)
    )
    rows = await cursor.fetchall()
    return [{"role": r[0], "content": r[1]} for r in rows]

async def db_clear_history(db: aiosqlite.Connection, user_id: int):
    await db.execute("DELETE FROM history WHERE user_id = ?", (user_id,))
    await db.commit()

async def db_get_model(db: aiosqlite.Connection, user_id: int) -> str:
    cursor = await db.execute(
        "SELECT model_key FROM user_settings WHERE user_id = ?", (user_id,)
    )
    row = await cursor.fetchone()
    return row[0] if row else DEFAULT_MODEL

async def db_set_model(db: aiosqlite.Connection, user_id: int, model_key: str):
    await db.execute(
        "INSERT OR REPLACE INTO user_settings (user_id, model_key) VALUES (?, ?)",
        (user_id, model_key)
    )
    await db.commit()

# ============================================================================
# ПОИСК (ddgs — бесплатно, безлимитно, без ключа)
# ============================================================================
async def search_ddgs(query: str, max_results: int = MAX_SEARCH_RESULTS) -> str:
    """Поиск через ddgs (бывший duckduckgo-search). Бесплатно, без ключа."""
    def _sync_search():
        try:
            from ddgs import DDGS
            results = DDGS().text(query, max_results=max_results)
            if not results:
                return ""
            lines = []
            for r in results:
                title = r.get("title", "")
                href = r.get("href", r.get("url", ""))
                body = r.get("body", r.get("content", ""))[:300]
                lines.append(f"- {title}\n  {href}\n  {body}")
            return "\n".join(lines)
        except Exception as e:
            log.warning("ddgs поиск не удался: %s", e)
            return ""

    return await asyncio.to_thread(_sync_search)

async def search_tavily(query: str, max_results: int = MAX_TAVILY_RESULTS) -> str:
    """Tavily — структурированный поиск для AI. 1 кредит за запрос (basic)."""
    if not TAVILY_API_KEY:
        return ""
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": TAVILY_API_KEY,
                    "query": query,
                    "search_depth": "basic",
                    "max_results": max_results,
                },
            )
            if resp.status_code != 200:
                log.warning("Tavily вернул %s", resp.status_code)
                return ""
            data = resp.json()
            results = data.get("results", [])
            lines = []
            for r in results:
                title = r.get("title", "")
                url = r.get("url", "")
                content = r.get("content", "")[:400]
                lines.append(f"- {title}\n  {url}\n  {content}")
            return "\n".join(lines)
    except Exception as e:
        log.warning("Tavily ошибка: %s", e)
        return ""

async def do_search(query: str, force_tavily: bool = False) -> str:
    """Сначала ddgs (бесплатно), потом Tavily если нужно."""
    if force_tavily and TAVILY_API_KEY:
        tavily_results = await search_tavily(query)
        if tavily_results:
            return f"[TAVILY]:\n{tavily_results}"
    ddgs_results = await search_ddgs(query)
    if ddgs_results:
        return f"[WEB]:\n{ddgs_results}"
    if TAVILY_API_KEY and not force_tavily:
        tavily_results = await search_tavily(query)
        if tavily_results:
            return f"[TAVILY]:\n{tavily_results}"
    return ""

def needs_search(text: str) -> bool:
    lower = text.lower()
    return any(t in lower for t in SEARCH_TRIGGERS)

# ============================================================================
# LLM ЗАПРОСЫ
# ============================================================================
def _build_headers(model_cfg: dict) -> dict:
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {model_cfg['key']}",
    }
    if "openrouter.ai" in model_cfg["base_url"]:
        headers["HTTP-Referer"] = "https://github.com/vibe-coding-bot"
        headers["X-Title"] = "VibeCodingBot2026"
    return headers

async def call_llm(
    messages: list[dict],
    model_key: str,
    temperature: float = 0.3,
) -> tuple[str, str]:
    """Вызывает LLM. Возвращает (ответ, использованный_ключ_модели)."""
    # Если auto — пробуем chain
    chain = FALLBACK_CHAIN if model_key == "auto" else [model_key]

    last_error = ""
    for key in chain:
        cfg = MODELS.get(key)
        if not cfg or not cfg["key"]:
            continue
        payload = {
            "model": cfg["model"],
            "messages": messages,
            "temperature": temperature,
            "max_tokens": cfg["max_tokens"],
        }
        try:
            async with httpx.AsyncClient(timeout=90.0) as client:
                resp = await client.post(
                    cfg["base_url"] + "/chat/completions",
                    headers=_build_headers(cfg),
                    json=payload,
                )
                if resp.status_code == 429:
                    log.warning("Rate limit на %s, пробуем следующую", key)
                    last_error = f"429 на {key}"
                    continue
                if resp.status_code != 200:
                    log.warning("API %s вернул %s: %s", key, resp.status_code, resp.text[:200])
                    last_error = f"{resp.status_code} на {key}"
                    continue
                data = resp.json()
                content = data["choices"][0]["message"]["content"]
                return content, key
        except Exception as e:
            log.warning("Ошибка запроса к %s: %s", key, e)
            last_error = str(e)
            continue

    return f"⚠️ Все провайдеры недоступны. Последняя ошибка: {last_error}", "error"

async def call_vision(
    image_url: str,
    question: str,
    model_key: str = VISION_MODEL,
) -> str:
    """Анализ изображения через vision-модель."""
    cfg = MODELS.get(model_key, MODELS[VISION_MODEL])
    if not cfg["key"]:
        # Пробуем qwen-vision
        cfg = MODELS.get("qwen-vision", MODELS[VISION_MODEL])
        if not cfg["key"]:
            return "⚠️ Нет API ключа для vision-модели (нужен OPENROUTER_API_KEY)"

    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": question or "Опиши, что на этом изображении, и найди технические детали."},
            {"type": "image_url", "image_url": {"url": image_url}},
        ]
    }]

    payload = {
        "model": cfg["model"],
        "messages": messages,
        "temperature": 0.3,
        "max_tokens": cfg["max_tokens"],
    }

    try:
        async with httpx.AsyncClient(timeout=90.0) as client:
            resp = await client.post(
                cfg["base_url"] + "/chat/completions",
                headers=_build_headers(cfg),
                json=payload,
            )
            if resp.status_code != 200:
                return f"⚠️ Vision API ошибка {resp.status_code}: {resp.text[:200]}"
            data = resp.json()
            return data["choices"][0]["message"]["content"]
    except Exception as e:
        return f"⚠️ Vision ошибка: {e}"

# ============================================================================
# БЕЗОПАСНАЯ ОТПРАВКА
# ============================================================================
async def safe_send(message: types.Message, text: str, reply: bool = True):
    """Пробует MarkdownV2 → HTML → plain text. Режет по абзацам."""
    chunks = split_text(text)
    for chunk in chunks:
        sent = False
        # Попытка 1: MarkdownV2
        if not sent:
            try:
                await message.reply(chunk, parse_mode=ParseMode.MARKDOWN_V2) if reply \
                    else await message.answer(chunk, parse_mode=ParseMode.MARKDOWN_V2)
                sent = True
            except TelegramBadRequest:
                pass
        # Попытка 2: HTML
        if not sent:
            try:
                await message.reply(chunk, parse_mode=ParseMode.HTML) if reply \
                    else await message.answer(chunk, parse_mode=ParseMode.HTML)
                sent = True
            except TelegramBadRequest:
                pass
        # Попытка 3: plain text
        if not sent:
            try:
                await message.reply(chunk) if reply \
                    else await message.answer(chunk)
                sent = True
            except TelegramBadRequest as e:
                log.error("Не удалось отправить сообщение: %s", e)
        # Только первый кусок — reply, остальные — answer
        reply = False
        await asyncio.sleep(0.3)  # мягкий rate limit

def split_text(text: str) -> list[str]:
    """Режет текст по \n\n, не разрывая блоки кода."""
    if len(text) <= MESSAGE_CHUNK_SIZE:
        return [text]
    chunks = []
    current = ""
    parts = text.split("\n\n")
    for part in parts:
        if len(current) + len(part) + 2 > MESSAGE_CHUNK_SIZE:
            if current:
                chunks.append(current)
            # Если один абзац длиннее лимита — режем по строкам
            if len(part) > MESSAGE_CHUNK_SIZE:
                lines = part.split("\n")
                for line in lines:
                    if len(current) + len(line) + 1 > MESSAGE_CHUNK_SIZE:
                        if current:
                            chunks.append(current)
                        current = line + "\n"
                    else:
                        current += line + "\n"
            else:
                current = part + "\n\n"
        else:
            current += part + "\n\n"
    if current:
        chunks.append(current)
    return chunks

# ============================================================================
# TELEGRAM БОТ
# ============================================================================
bot = Bot(
    token=TELEGRAM_TOKEN,
    default=DefaultBotProperties(parse_mode=ParseMode.MARKDOWN_V2),
)
dp = Dispatcher()
db: aiosqlite.Connection = None  # инициализируется в main()

# --- Команды ---

@dp.message(Command("start"))
async def cmd_start(message: types.Message):
    text = (
        "🚀 **Vibe Coding Assistant 2026**\n\n"
        "Я твой технический партнёр. Стек по умолчанию: FastAPI, React, Neon, Render.\n"
        "Я помню контекст, ищу интернет и пишу production-ready код.\n\n"
        "**Команды:**\n"
        "• Напиши задачу или скинь код — решу\n"
        "• `ЗАПОМНИ: [факт]` — сохраню в память\n"
        "• `/model` — покажу доступные модели\n"
        "• `/model [name]` — переключу модель\n"
        "• `/search [запрос]` — принудительный поиск\n"
        "• `/facts` — покажу память\n"
        "• `/clear` — очищу память\n"
        "• Скинь файл `.py/.js/.ts/...` — проанализирую\n"
        "• Скинь фото — опишу и найду детали\n"
    )
    await safe_send(message, text)

@dp.message(Command("model"))
async def cmd_model(message: types.Message):
    args = message.text.split(maxsplit=1)
    if len(args) < 2:
        current = await db_get_model(db, message.from_user.id)
        lines = [f"Текущая модель: `{current}`\n\nДоступные:"]
        for key, cfg in MODELS.items():
            marker = " ← сейчас" if key == current else ""
            vision = " 👁 vision" if cfg["vision"] else ""
            lines.append(f"• `/{key}` — {cfg['label']}{vision}{marker}")
        lines.append(f"\n• `/model auto` — chain fallback (groq→nemotron→cohere)")
        await safe_send(message, "\n".join(lines))
        return
    key = args[1].strip().lower()
    if key not in MODELS and key != "auto":
        await safe_send(message, f"Модель `{key}` не найдена. Используй `/model` для списка.")
        return
    await db_set_model(db, message.from_user.id, key)
    cfg = MODELS.get(key, {"label": "auto (chain fallback)"})
    await safe_send(message, f"✅ Модель переключена: {cfg['label']}")

@dp.message(Command("search"))
async def cmd_search(message: types.Message):
    args = message.text.split(maxsplit=1)
    if len(args) < 2:
        await safe_send(message, "Использование: `/search [запрос]`")
        return
    query = args[1].strip()
    await bot.send_chat_action(message.chat.id, "typing")
    results = await do_search(query, force_tavily=True)
    if not results:
        await safe_send(message, "Поиск не дал результатов.")
        return
    await safe_send(message, f"🔎 Результаты поиска:\n\n{results}")

@dp.message(Command("facts"))
async def cmd_facts(message: types.Message):
    facts = await db_get_facts(db, message.from_user.id)
    if not facts:
        await safe_send(message, "Память пуста. Используй `ЗАПОМНИ: [факт]` для сохранения.")
        return
    lines = ["📋 **Сохранённые факты:**\n"]
    for i, f in enumerate(facts, 1):
        lines.append(f"{i}. {f}")
    await safe_send(message, "\n".join(lines))

@dp.message(Command("clear"))
async def cmd_clear(message: types.Message):
    await db_clear_facts(db, message.from_user.id)
    await db_clear_history(db, message.from_user.id)
    await safe_send(message, "🗑️ Память и история полностью очищены.")

# --- Обработка фото (Vision) ---

@dp.message(F.photo)
async def handle_photo(message: types.Message):
    if not OPENROUTER_API_KEY:
        await safe_send(message, "⚠️ Для анализа фото нужен OPENROUTER_API_KEY")
        return
    await bot.send_chat_action(message.chat.id, "typing")

    # Получаем file_path
    file_id = message.photo[-1].file_id
    file_info = await bot.get_file(file_id)
    file_path = file_info.file_path
    image_url = f"https://api.telegram.org/file/bot{TELEGRAM_TOKEN}/{file_path}"

    question = message.caption or "Опиши, что на этом изображении, и найди технические детали."

    # Факты + история
    facts = await db_get_facts(db, message.from_user.id)
    facts_text = "\n".join([f"- {f}" for f in facts]) if facts else "Нет фактов."
    enriched_question = f"Контекст проекта:\n{facts_text}\n\nВопрос: {question}"

    response = await call_vision(image_url, enriched_question)
    await safe_send(message, response)

    # Сохраняем в историю
    await db_add_message(db, message.from_user.id, "user", f"[фото] {question}")
    await db_add_message(db, message.from_user.id, "assistant", response[:2000])

# --- Обработка файлов ---

@dp.message(F.document)
async def handle_document(message: types.Message):
    doc = message.document
    file_name = doc.file_name or "unknown"
    ext = os.path.splitext(file_name)[1].lower()

    if ext not in ALLOWED_EXTENSIONS:
        await safe_send(message, f"Файл `{ext}` не поддерживается. Допустимо: {', '.join(sorted(ALLOWED_EXTENSIONS))}")
        return

    await bot.send_chat_action(message.chat.id, "typing")

    # Скачиваем файл
    file_info = await bot.get_file(doc.file_id)
    file_path = file_info.file_path
    local_path = f"/tmp/{doc.file_id}_{file_name}"

    try:
        await bot.download_file(file_path, local_path)
        with open(local_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read(MAX_FILE_CHARS)
        if len(content) == MAX_FILE_CHARS:
            content += "\n\n[... файл обрезан ...]"
        os.remove(local_path)
    except Exception as e:
        await safe_send(message, f"Ошибка чтения файла: {e}")
        return

    # Формируем промпт
    caption = message.caption or "Проанализируй этот файл и предложи улучшения."
    user_msg = f"Пользователь загрузил файл `{file_name}`.\nСодержимое:\n```{ext[1:]}\n{content}\n```\n\nЗадача: {caption}"

    await process_message(message, user_msg, save_history=True)

# --- Обычные текстовые сообщения ---

@dp.message(F.text)
async def handle_text(message: types.Message):
    text = message.text

    # Команда ЗАПОМНИ
    if text.upper().startswith("ЗАПОМНИ:"):
        fact = text[8:].strip()
        if not fact:
            await safe_send(message, "Формат: `ЗАПОМНИ: используем FastAPI + Neon`")
            return
        await db_save_fact(db, message.from_user.id, fact)
        await safe_send(message, f"✅ Запомнил: {fact}")
        return

    await process_message(message, text, save_history=True)

# ============================================================================
# ОБРАБОТКА СООБЩЕНИЯ (основная логика)
# ============================================================================
async def process_message(message: types.Message, user_text: str, save_history: bool = True):
    user_id = message.from_user.id
    await bot.send_chat_action(message.chat.id, "typing")

    # 1. Получаем модель пользователя
    model_key = await db_get_model(db, user_id)

    # 2. Получаем факты
    facts = await db_get_facts(db, user_id)
    facts_text = "\n".join([f"- {f}" for f in facts]) if facts else "Нет сохранённых фактов."

    # 3. История диалога
    history = await db_get_history(db, user_id)

    # 4. Поиск (ddgs — бесплатно, при триггерах)
    search_context = ""
    if needs_search(user_text):
        search_results = await search_ddgs(user_text)
        if search_results:
            search_context = f"\n\n[АКТУАЛЬНЫЕ ДАННЫЕ ИЗ ИНТЕРНЕТА]:\n{search_results}\nИспользуй эти данные для ответа."

    # 5. Сборка системного промпта
    system = SYSTEM_PROMPT + f"\n\nФАКТЫ О ПРОЕКТЕ:\n{facts_text}{search_context}"

    # 6. Сборка сообщений
    messages = [{"role": "system", "content": system}]
    # Добавляем историю (последние 20)
    for h in history[-MAX_HISTORY:]:
        messages.append({"role": h["role"], "content": h["content"]})
    messages.append({"role": "user", "content": user_text})

    # 7. Запрос к LLM
    response, used_model = await call_llm(messages, model_key)

    # 8. Отправка
    await safe_send(message, response)

    # 9. Сохраняем в историю
    if save_history:
        await db_add_message(db, user_id, "user", user_text[:8000])
        await db_add_message(db, user_id, "assistant", response[:8000])

# ============================================================================
# HEALTH CHECK (aiohttp.web в том же event loop)
# ============================================================================
async def health_handler(request: web.Request) -> web.Response:
    return web.Response(text="OK")

async def root_handler(request: web.Request) -> web.Response:
    return web.Response(text="🤖 Vibe Coding Bot 2026 is LIVE and ready to build.")

# ============================================================================
# MAIN
# ============================================================================
async def main():
    global db

    # Валидация
    missing = validate_env()
    if missing:
        log.error("Нет обязательных переменных окружения. Завершение.")
        return

    # База данных
    db = await init_db()

    # aiohttp web server для health-check
    web_app = web.Application()
    web_app.router.add_get("/", root_handler)
    web_app.router.add_get("/health", health_handler)
    runner = web.AppRunner(web_app)
    await runner.setup()
    port = int(os.environ.get("PORT", 10000))
    site = web.TCPSite(runner, "0.0.0.0", port)
    await site.start()
    log.info("Health-check сервер запущен на порту %s", port)

    # Удаляем webhook и запускаем polling
    await bot.delete_webhook(drop_pending_updates=True)
    log.info("Бот запущен. Polling started.")

    try:
        await dp.start_polling(bot, allowed_updates=dp.resolve_used_update_types())
    finally:
        await db.close()
        await runner.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
