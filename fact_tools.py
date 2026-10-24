import uuid
import os
from datetime import datetime 
import chromadb 
from chromadb.utils import embedding_functions
from langchain_core.tools import tool
from pathlib import Path
#persistent client 
DATA_DIR = Path(os.getenv("MEMORY_DATA_DIR", Path(__file__).parent / "data"))
DATA_DIR.mkdir(exist_ok=True)
client = chromadb.PersistentClient(path=str(DATA_DIR / "memory_db"))
ef=embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

col=client.get_or_create_collection("facts",embedding_function=ef)

@tool
def remember_fact(fact:str)->str:
    """Store a NEW fact about the user in short,self-contained sentences.
    for eg. 'user is allergic to peanust'.before calling this ,call search_memory 
    to check if a related fact exists , if yes then use update_fact instead."""
    if col.count() > 0:
        hit = col.query(query_texts=[fact], n_results=1)
        if hit["distances"][0][0] < 0.3:
            return (f"NOT stored: a very similar fact already exists: "
                    f"[{hit['ids'][0][0]}] {hit['documents'][0][0]}. "
                    "If the new information is identical, do nothing. "
                    "If it changes that fact, call update_fact with that id.")
    fact_id=str(uuid.uuid4())[:8]
    col.add(
        ids=[fact_id],
        documents=[fact],  
        metadatas=[{"created":datetime.now().isoformat()}] 
     )
    return f"Stored fact{fact_id}:{fact}"

@tool 
def search_memory(query:str)->str:
    """Search stored facts by meaning. Returns matching facts with their ids.
    Always call this before update_fact or forget_fact, because they need an id from here."""

    n=min(3,col.count())
    if n==0:
        return "No memories stored."
    res=col.query(query_texts=[query],n_results=n)
    lines=[]
    for fact_id , doc ,dist in zip(res["ids"][0], res["documents"][0], res["distances"][0]):
        lines.append(f"[{fact_id}] {doc} (distance {dist:.2f})")
    return "\n".join(lines)


@tool
def update_fact(fact_id: str, new_fact: str) -> str:
    """Replace an existing fact with new text. Use when new information contradicts
    or refines a stored fact. fact_id must come from search_memory. Never guess one."""
    if not col.get(ids=[fact_id])["ids"]:
        return "ERROR: fact id not found"
    col.update(ids=[fact_id], documents=[new_fact])  # re-embeds the new text automatically
    return f"Updated {fact_id}: {new_fact}"


@tool
def forget_fact(fact_id: str) -> str:
    """Permanently delete a fact. fact_id must come from search_memory. Never guess one."""
    if not col.get(ids=[fact_id])["ids"]:
        return "ERROR: fact id not found"
    col.delete(ids=[fact_id])
    return f"Forgot {fact_id}"


FACT_TOOLS = [remember_fact, search_memory, update_fact, forget_fact]

