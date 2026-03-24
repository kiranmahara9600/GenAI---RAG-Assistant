# Telegram RAG Bot

A Telegram chatbot that answers questions from your own private documents using Retrieval-Augmented Generation (RAG). Instead of manually searching through policies, FAQs, and guides, users simply type a question in Telegram and receive an instant, sourced answer powered by GPT.
It also describes an image input sent by user and guides with how to send command around image input.

---

## Table of Contents

- [What It Does](#what-it-does)
- [Models and APIs Used](#models-and-apis-used)
- [System Design](#system-design)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Running Locally](#running-locally)
- [Running with Docker Compose](#running-with-docker-compose)
- [Bot Commands](#bot-commands)
- [Knowledge Base](#knowledge-base)
- [Configuration Reference](#configuration-reference)
---

## What It Does

Large language models like GPT know public internet data — but they know nothing about private company documents. This bot bridges that gap using **RAG (Retrieval-Augmented Generation)**:

1. Your documents are split into small chunks and converted into vectors (embeddings)
2. When a user asks a question, the question is also converted to a vector
3. The most semantically similar document chunks are retrieved
4. Those chunks are passed to GPT as context with query, which generates a grounded answer
5. The answer is sent back to the user along with the source document name
6. When user sends an image input it analyses and describes it using vision model from openai.

This means the bot never hallucinates — it can only answer from what is in your documents.

### Features

| Feature | Description |
|---|---|
| `/ask <question>` | Retrieves top-3 relevant chunks, calls GPT, replies with answer and source file |
| `/summarize` | Summarises recent conversation in bullet points |
| `/clear` | Wipes conversation memory for fresh start |
| Conversation memory | Remembers last 3 exchanges per user across restarts |
| Query caching | Identical questions skip re-embedding - instant response |
| Source attribution | Every answer shows which document it came from |
| Persistent sessions | Conversation history survives bot restarts via SQLite |
| `/image` |  Describe any photo using GPT-4o Vision|
---

## Models and APIs Used

### Embedding Model — `all-MiniLM-L6-v2`

- **What it does:** Converts text chunks and user queries into 384-dimensional vectors
- **Where it runs:** Locally on the machine — no API calls, no cost
- **Library:** `langchain-huggingface`
- **Why this model:** Fast, lightweight (~80MB), high quality for English semantic search. Good balance of speed and accuracy for a local bot.

### Language Model — `gpt-4o-mini` (OpenAI)

- **What it does:** Generates human-readable answers from retrieved context chunks
- **Where it runs:** OpenAI API (cloud)
- **Why this model:** Best cost-to-quality ratio for Q&A tasks. Much cheaper than GPT-4o while still producing accurate, well-structured answers. Trained with RLHF (Reinforcement Learning from Human Feedback) specifically to be helpful, accurate, and honest about the limits of its knowledge. For a RAG bot where users trust the answer, this matters enormously. A wrong answer about a leave policy or expense limit has real consequences.


### Telegram API

- **What it does:** Receives user messages and sends replies via the Telegram messaging platform
- **Library:** `python-telegram-bot`
- **Authentication:** Bot token from @BotFather

### Vector Storage — SQLite

- **What it does:** Stores document chunk embeddings as BLOB columns, query cache, and user session history
- **Why SQLite:** No server needed. Works on a laptop. Sufficient for hundreds of documents and thousands of queries. Zero infrastructure cost.

---

## System Design

```
     
│                         STARTUP (once)                                        │
│                                                                               │
│  Knowledge Base/*.md  ──►  chunk_text()  ──►  model.encode()  ──►  SQLite     │
│  (docs)       400-char chunks      384-dim vectors     rag_store.db           │



│                      /ask QUERY FLOW                                │
│                                                                     │
│  User on Telegram                                                   │
│       │                                                             │
│       │  /ask How many leave days do I get?                         │
│       ▼                                                             │
│  Emerald_bot.py  ──►  cmd_ask()                                     │
│       │                                                             │
│       ├──►  rag_engine.retrieve()                                   │
│       │         │                                                   │
│       │         ├──  Check query_cache (SQLite)                     │
│       │         │         hit  ──────────────────────────►  chunks  │
│       │         │         miss ──►  model.encode(query)             │
│       │         │                       │                           │
│       │         │                       ▼                           │
│       │         └──  cosine_similarity vs all chunk vectors         │
│       │                       │                                     │
│       │                       ▼                                     │
│       │              top-3 chunks returned                          │
│       │                                                             │
│       ├──►  build_context()  ──►  numbered text block from chunks   │
│       │                                                             │
│       ├──►  session.get_history()  ──►  last 3 exchanges            │
│       │                                                             │
│       ├──►  llm_client.answer_with_context()                        │
│       │         │                                                   │
│       │         ▼                                                   │
│       │    OpenAI GPT-4o-mini                                       │
│       │    [system prompt + history + context + question]           │
│       │         │                                                   │
│       │         ▼                                                   │
│       │    "You have 20 days of annual leave per year..."           │
│       │                                                             │
│       ├──►  session.add_turn()  ──►  saves chat istory to SQLite    │
│       │                                                             │
│       └──►  reply_text()  ──►  User sees answer + Sources           │



│                     SQLITE DATABASE                                 │
│                                                                     │
│  rag_store.db                                                       │
│  ├── chunks        document text + 384-dim embedding vectors        │
│  ├── query_cache   cached query results (skip re-embedding)         │
│  └── sessions      per-user conversation history (JSON)             │

```

---

## Project Structure

```
rag_bot/
│
├── Emerald_bot.py          Entry point. Registers all Telegram command
│                           handlers and starts polling for messages.
│
├── requirements.txt        All Python dependencies.
│
├── .env                    environment variable config.
│                           
│
|
│
├
│
├── src/
│   |
│   │
│   ├── rag_engine.py       The search engine.
│   │                       - chunk_text(): splits docs into 400-char chunks
│   │                       - index_documents(): embeds and stores all chunks
│   │                       - retrieve(): finds top-k relevant chunks
│   │                       - build_context(): formats chunks for LLM prompt
│   │
│   ├── llm_client.py       The AI brain.
│   │                       - answer_with_context(): builds RAG prompt, calls GPT
│   │                       - summarise_history(): summarises conversation turns
│   │
│   └── session.py          The memory system.
│                           - SessionManager class
│                           - Stores last 6 messages per user in SQLite
│                           - Persists across bot restarts
│
└── Knowledge Base/         Knowledge base — edit freely.
    ├── company_policy.md   Leave, remote work, expense, conduct policies
    ├── tech_faq.md         VPN, password reset, software, IT contacts
    ├── onboarding.md       First day, first week, tools, key contacts
    └── recipes.md          Standard operating procedures and runbooks
```

---

## Prerequisites

- Python 3.11 or higher (3.12+ supported)
- A Telegram account
- An OpenAI account with API credits
- Git (optional, for cloning)

---

## Running Locally

### Step 1 — Get a Telegram Bot Token

```
1. Open Telegram and search for @BotFather
2. Send /newbot
3. Choose a name: e.g. My RAG Bot
4. Choose a username ending in 'bot': e.g. myragbot_bot
5. Copy the token BotFather gives you:

```

### Step 2 — Get an OpenAI API Key

### Step 3 — Configure environment

Put the telegram token, open AI API token in .env

It should look like below:

TELEGRAM_BOT_TOKEN=110201543:***************************fSs4tVEMT1DRCn
OPENAI_API_KEY=sk-proj-xxxxxxxxxxxxxxxx
OPENAI_MODEL=gpt-4o-mini


**Important:** Never commit  `.env` file to Git. It is already in `.gitignore`.


### Step 4 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 5 — Run the bot

```bash
python Emerald_bot.py
```

Expected output:

```
INFO | Indexing knowledge base documents…
INFO | Processed company_policy.md: 4 chunks
INFO | Processed onboarding.md: 3 chunks
INFO | Processed recipes.md: 5 chunks
INFO | Processed tech_faq.md: 3 chunks
INFO | Indexing complete. New chunks inserted: 15
INFO | Bot is running. Press Ctrl+C to stop.
```

### Step 6 — Talk to bot on Telegram

Search for your bot by username (e.g. `@EemeraldBot`) and start chatting.
## Bot Commands

| Command | Description | Example |
|---|---|---|
| `/start` | Welcome message | `/start` |
| `/help` | Show all commands with examples | `/help` |
| `/ask <question>` | Search documents and get AI answer | `/ask How many leave days do I get?` |
| `/image` | Then send a photo — bot describes it | `/image` → attach photo |
| `/summarize` | Summarise recent conversation | `/summarize` |
| `/clear` | Wipe your conversation history | `/clear` |

---


### Upcoming Additions

## Running with Docker Compose

Docker lets you run the bot without installing Python or any dependencies on your machine.



---

## Knowledge Base

The bot answers questions from the documents in the `Knowledge Base/` folder. To add your own content:

1. Create a `.md` or `.txt` file inside `Knowledge Base/`
2. Write your content in plain text or Markdown
3. Restart the bot - new documents are indexed automatically

The indexer is idempotent — restarting never duplicates existing chunks. Only new or changed content gets re-indexed.

### Included sample documents

| File | Content |
|---|---|
| `company_policy.md` | Remote work, leave, expense, and conduct policies |
| `tech_faq.md` | VPN setup, password reset, software requests, IT contacts |
| `onboarding.md` | First day checklist, key tools, probation process |
| `recipes.md` | Standard operating procedures: onboarding steps, expense submission, software requests, retrospectives |

### Chunking behaviour

Documents are split into 400-character chunks with 80-character overlap. This means:

- A 1000-character section produces approximately 3 chunks
- The 80-character overlap ensures sentences at chunk boundaries are never lost
- Shorter sections (under 400 chars) are kept as a single chunk

---

## Configuration Reference

All configuration lives in `.env` file and in constants at the top of each source file.

| Variable | Location | Default | Description |
|---|---|---|---|
| `TELEGRAM_BOT_TOKEN` | `.env` | — | Required. From @BotFather |
| `OPENAI_API_KEY` | `.env` | — | Required. From platform.openai.com |
| `OPENAI_MODEL` | `.env` | `gpt-4o-mini` | OpenAI model for answer generation |
| `EMBED_MODEL` | `rag_engine.py` | `all-MiniLM-L6-v2` | Local embedding model |
| `CHUNK_SIZE` | `rag_engine.py` | `400` | Characters per chunk |
| `CHUNK_OVERLAP` | `rag_engine.py` | `80` | Overlap between chunks |
| `TOP_K` | `rag_engine.py` | `3` | Number of chunks retrieved per query |
| `DB_PATH` | `rag_engine.py` | `rag_store.db` | SQLite database file path |
| `MAX_TOKENS` | `llm_client.py` | `600` | Max tokens in LLM response |
| `TEMPERATURE` | `llm_client.py` | `0.3` | LLM creativity (0=deterministic, 1=creative) |
| `HISTORY_LIMIT` | `session.py` | `6` | Max conversation turns stored per user |

---

`New added piece of image description uses open ai vision model to provide description of an image based on user input`

**The bot only runs while your terminal is open.**

