"""
Bot Name: Emerald_bot
Bot Desc: Telegram bot with RAG and conversation memory.
File Desc: Telegram bot entry point - creates bot application, registers all commands (/ask, /help, /summarize, /clear),
starts polling telegram for incoming messages then connects them to the RAG engine, LLM client, and session memory as per user ip.

Commands
--------
/start      -  welcome message
/help       -  usage instructions
/ask        -  user query against the knowledge base
/summarize  -  summarise recent conversation
/clear      -  clear session history

Environment variables
---------------------
TELEGRAM_BOT_TOKEN  -  from @BotFather
OPENAI_API_KEY      -  OpenAI key
OPENAI_MODEL        -  optional -->  defaults to gpt-4o-mini

"""


##imported required libraries

#general lib
import os
import sys
import logging
from pathlib import Path
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import (Application,CommandHandler,MessageHandler,ContextTypes,filters)
import asyncio

#defined modules
from src.rag_engine import index_documents, retrieve, build_context
from src.llm_client import answer_with_context, summarise_history
from src.session import sessions


#to get env var in os.environ
load_dotenv()

#to make all the source doc and custom modules available when bot file runs from proj root
sys.path.insert(0, str(Path(__file__).parent / "src"))   #/AVIVO/src
DOCS_DIR = Path(__file__).parent / "Knowledge Base"  #/AVIVO/Knowledge Base



#------------------------------------------------------
#LOGGING - to create a logger obj for this file (Emerald_bot.py)
#------------------------------------------------------

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


#------------------------------------------------------
#COMMANDS
#------------------------------------------------------

#/start -  welcome message
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Hi! I am your RAG assistant Emerald.\n"
        "Type /help to see what I can do."
        "Thanks!"
    )

#/help  -  usage instructions
HELP_TEXT = """
*Available commands*

/ask <question>
   Search the knowledge base and get an AI-generated answer with sources.
   _Example_: `/ask What is the remote work policy?`

/summarize
   Summarise your recent conversation

/clear
   Clear the conversation history and start fresh.

/help
   Show this message.

*Knowledge base covers:*
   Company policies
   Tech FAQs
   Recipes
   Onboarding guide
"""

async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(HELP_TEXT, parse_mode="Markdown")


#/ask  -  user query against the knowledge base
async def cmd_ask(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    query   = " ".join(context.args).strip() if context.args else ""

    if not query:
        await update.message.reply_text(
            "Please provide a question.\n_Example:_ `/ask What is the remote work policy??`",
            parse_mode="Markdown"
        )
        return

    await update.message.reply_text("Searching knowledge base…\n\n Please wait...")

    try:
        #retrieve relevant chunks
        chunks = retrieve(query)

        if not chunks or chunks[0]["score"] < 0.25:
            await update.message.reply_text(
                "I couldn't find relevant information in the knowledge base for the question."
            )
            return

        #build context string and source list
        context_text, sources = build_context(chunks)

        #get conversation history for this user
        history = sessions.get_history(user_id)

        #call chatmodel
        answer = answer_with_context(query, context_text, history)

        #update session
        sessions.add_turn(user_id, "user", query)
        sessions.add_turn(user_id, "assistant", answer)

        #format response with source attribution
        source_line = "*Sources:* " + ", ".join(f"{s}" for s in sources)
        reply = f"{answer}\n\n{source_line}"

        await update.message.reply_text(reply, parse_mode="Markdown")

    except Exception as exc:
        logger.exception("Error in /ask handler")
        await update.message.reply_text(f"Something went wrong: {exc}")


#/summarize  -  summarise recent conversation
async def cmd_summarize(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    history = sessions.get_history(user_id)

    try:
        if not history:
            await update.message.reply_text("Nothing to summarise yet - try /ask first.")
            return

        await update.message.reply_text("Summarising…")

        summary = summarise_history(history)
            

        await update.message.reply_text(f"*Summary:*\n\n{summary}", parse_mode="Markdown")

    except Exception as exc:
        logger.exception("Error in /summarize handler")
        await update.message.reply_text(f"Could not summarise: {exc}")


#/clear  -  clear session history
async def cmd_clear(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    sessions.clear(update.effective_user.id)
    await update.message.reply_text("Session history has been cleared.")


#fallback for unexpected messages
async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Use /ask <question> to query the knowledge base, or /help for all commands."
    )


#------------------------------------------------------
#MAIN FUNCTION
#------------------------------------------------------

async def  main() -> None:
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        logger.error("TELEGRAM_BOT_TOKEN not set. Add it to .env file.")
        sys.exit(1)

    #index documents on startup 
    logger.info("Indexing knowledge base documents…")
    n = index_documents(str(DOCS_DIR))
    logger.info(f"Index ready. ({n} new chunks added this run)")

    # Build application
    app = Application.builder().token(token).build()

    # Register handlers
    app.add_handler(CommandHandler("start",     cmd_start))
    app.add_handler(CommandHandler("help",      cmd_help))
    app.add_handler(CommandHandler("ask",       cmd_ask))
    app.add_handler(CommandHandler("summarize", cmd_summarize))
    app.add_handler(CommandHandler("clear",     cmd_clear))

    logger.info("Bot is running. Press Ctrl+C to stop.")
    #app.run_polling()  #bot connects to Telegram API   --- throwing event loop error
    async with app:
        await app.initialize()
        await app.start()
        await app.updater.start_polling(drop_pending_updates=True)

        try:
            await asyncio.Event().wait()  # blocks until Ctrl+C
        except (KeyboardInterrupt, asyncio.CancelledError):
            logger.info(f"Bye Bye")
        finally:
            await app.updater.stop()
            await app.stop()


if __name__ == "__main__":
    asyncio.run(main())   #creates and manages event loop properly


##_________________________________END____________________________________