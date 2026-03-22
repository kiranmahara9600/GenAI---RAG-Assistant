"""
llm_client.py
-------------

Desc: A Langchain framework which uses openai chat model to generate response from augmented query. 
Supports conversation history injection and RAG-style prompting.
"""




import os
import logging
from typing import Optional
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser


logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# MODEL HELPERS  -- AUGMENTATION - GENERATION - SUMMARIZATION
# ---------------------------------------------------------------------------

MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
API_KEY = os.environ.get("OPENAI_API_KEY")
MAX_TOKENS = 600
TEMPERATURE = 0.3

#Chatmodel initialisation
llm = ChatOpenAI(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        api_key=API_KEY
)



#to generate user response
def answer_with_context(query: str,context: str,history: Optional[list[dict]] = None) -> str:
    """
    Augment(query+context) a RAG prompt using LangChain and return the model's answer.

    Args:
        query:   The user's question
        context: Retrieved document chunks
        history: Up to last-N turns as [{"role": ..., "content": ...}]
    """
    #Prompt template
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=(
            "You are a helpful assistant. Answer the user's question using only the "
            "provided context snippets. If the context does not contain enough information "
            "to answer confidently, say so clearly. Be concise and precise. "
            "Do not make up facts."
        )),
        MessagesPlaceholder(variable_name="history"),   #injects conversation history
        HumanMessage(content=(
            f"Context snippets:\n{context}\n\n Question: {query}"
        ))
    ])

    logger.info("QUERY:", query)
    logger.info("CONTEXT:", context[:200])  
    logger.info("HISTORY:", history)
    logger.info("PROMPT:", prompt)

    #Chain pipeline
    chain = prompt | llm   

    #convert history dicts to LangChain message objects
    lc_history = []
    if history:
        for msg in history:
            if msg["role"] == "user":
                lc_history.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "emerald":
                lc_history.append(AIMessage(content=msg["content"]))

    #invoke the chain
    response = chain.invoke({
        "history": lc_history,
        "context": context,
        "query": query
    })

    logger.info(f"response generated: {response.content.strip()}")

    return response.content.strip()



#to summarise chat history
def summarise_history(history: list[dict]) -> str:
    """Summarise a list of conversation turns using LangChain."""

    #Prompt template
    summarise_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "Summarise the following conversation concisely in 3-5 bullet points."
        ),
        (
            "user",
            "{turns}"       #conversation history injected here
        )
    ])

    logger.info("HISTORY:", history)

    #Chain
    summarise_chain = summarise_prompt | llm | StrOutputParser()


    if not history:
        return "No conversation history to summarise."

    #format history dicts into a readable transcript
    turns = "\n".join(
        f"{m['role'].upper()}: {m['content']}" for m in history
    )

    #invoke the chain - returns a plain string directly
    summary = summarise_chain.invoke({"turns": turns})
    logger.info(f"summary generated: {summary}")

    return summary



#----------------------------------END---------------------------------------