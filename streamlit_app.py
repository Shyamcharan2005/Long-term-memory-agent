import json
import os
import uuid
from contextlib import closing

from fact_tools import col
from graph_tools import _conn
import streamlit as st

try:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
except Exception:
    pass  # running locally: the key comes from .env

from langgraph.types import Command

from agent import app
from fact_tools import DATA_DIR

st.set_page_config(page_title="Long-Term Memory Agent")
st.title("Long-Term Memory Agent")
st.caption("Remembers across chats: facts in a vector store, relationships in a graph. Deleting a memory needs your aproval.Demo memory is shared by all visitors and may reset. Please use made-up details.")

if "thread_id" not in st.session_state:
    st.session_state.thread_id = uuid.uuid4().hex[:8]


def cfg(tid):
    return {"configurable": {"thread_id": tid}}


def past_threads(limit=10):
    """Thread ids that have saved checkpoints, most recent first."""
    seen = []
    for cp in app.checkpointer.list(None):
        tid = cp.config["configurable"]["thread_id"]
        if tid not in seen:
            seen.append(tid)
        if len(seen) >= limit:
            break
    return seen

TITLES_FILE = DATA_DIR / "chat_titles.json"


def load_titles():
    try:
        return json.loads(TITLES_FILE.read_text())
    except (FileNotFoundError, ValueError):
        return {}


def save_titles(titles):
    TITLES_FILE.write_text(json.dumps(titles))


def thread_title(tid):
    custom = load_titles().get(tid)
    if custom:
        return custom
    msgs = app.get_state(cfg(tid)).values.get("messages", [])
    return next((m.content for m in msgs if m.type == "human"), None)

config = cfg(st.session_state.thread_id)

# ---- sidebar: new chat + past chats ----
with st.sidebar:
    if st.button("New chat"):
        st.session_state.thread_id = uuid.uuid4().hex[:8]
        st.rerun()

    st.caption("Past chats")
    for tid in past_threads():
        title = thread_title(tid)
        if not title:
            continue

        # Rename mode: only for the chat where Rename was clicked
        if st.session_state.get("renaming") == tid:
            new_name = st.text_input("Rename chat", value=title[:60], key=f"rename-{tid}",
                                     label_visibility="collapsed")
            save_col, cancel_col = st.columns(2)
            if save_col.button("Save", key=f"save-{tid}"):
                titles = load_titles()
                titles[tid] = new_name.strip() or title
                save_titles(titles)
                st.session_state.renaming = None
                st.rerun()
            if cancel_col.button("Cancel", key=f"cancel-{tid}"):
                st.session_state.renaming = None
                st.rerun()
            continue

        row, menu = st.columns([5, 1])
        if row.button(title[:28], key=f"open-{tid}"):
            st.session_state.thread_id = tid
            st.rerun()
        with menu.popover("⋯"):
            if st.button("Rename", key=f"ren-{tid}"):
                st.session_state.renaming = tid
                st.rerun()
            if st.button("Delete", key=f"del-{tid}"):
                app.checkpointer.delete_thread(tid)
                titles = load_titles()
                titles.pop(tid, None)
                save_titles(titles)
                if tid == st.session_state.thread_id:
                    st.session_state.thread_id = uuid.uuid4().hex[:8]
                st.rerun()
    with st.popover("Reset all memory"):
        st.write("Deletes every stored fact and relation. Chats are not affected.")
        if st.button("Yes, erase memory", key="wipe"):
            ids = col.get()["ids"]
            if ids:
                col.delete(ids=ids)
            with closing(_conn()) as c:
                c.execute("DELETE FROM relations")
                c.commit()
            st.success("Memory erased.")

# ---- everything on screen comes from the saved thread state ----
state = app.get_state(config)
messages = state.values.get("messages", [])
pending = state.tasks[0].interrupts[0].value if state.tasks and state.tasks[0].interrupts else None

tools = []
for m in messages:
    if m.type == "human":
        tools = []
        with st.chat_message("user"):
            st.write(m.content)
    elif m.type == "ai":
        tools += [f"{c['name']}({c['args']})" for c in m.tool_calls]
        if m.content and not m.tool_calls:
            with st.chat_message("assistant"):
                st.write(m.content)
                if tools:
                    with st.expander(f"Tool calls ({len(tools)})"):
                        for t in tools:
                            st.code(t)

# ---- approval step, or normal input ----
if pending:
    st.warning(f'Delete this memory?  {pending["description"]}')
    yes_col, no_col = st.columns(2)
    answer = None
    if yes_col.button("Yes, delete"):
        answer = "yes"
    elif no_col.button("No, keep it"):
        answer = "no"
    if answer:
        try:
            app.invoke(Command(resume=answer), config=config)
        except Exception as e:
            st.error(f"Error: {e}")
        else:
            st.rerun()
else:
    prompt = st.chat_input("Tell me something or ask about what I know about you")
    if prompt:
        with st.chat_message("user"):
            st.write(prompt)
        with st.spinner("Thinking..."):
            try:
                app.invoke({"messages": [("user", prompt)]}, config=config)
            except Exception as e:
                st.error(f"Error: {e}")
            else:
                st.rerun()