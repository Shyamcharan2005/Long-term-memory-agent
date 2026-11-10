import time
from contextlib import closing

import pytest
from langchain_core.messages import HumanMessage
from langgraph.types import Command

from agent import app, memory_context
from fact_tools import col
from graph_tools import _conn


# ---------- helpers ----------
def chat(thread, text, approve="yes"):
    """One user message in one session (thread). Auto-answers the approval prompt."""
    config = {"configurable": {"thread_id": thread}}
    try:
        result = app.invoke({"messages": [("user", text)]}, config=config)
        while "__interrupt__" in result:
            result = app.invoke(Command(resume=approve), config=config)
    except Exception as e:
        if "rate_limit_exceeded" in str(e):
            pytest.skip("Groq rate limit hit: not a test result")
        raise
    time.sleep(1)  # be gentle with Groq rate limits


def facts():
    return [d.lower() for d in col.get()["documents"]]


def relations():
    with closing(_conn()) as c:
        return [" ".join(r) for r in c.execute("SELECT subject, relation, object FROM relations")]


def new_session_memory(question):
    """What a brand-new chat would be handed for this question. No LLM, no thread."""
    return memory_context({"messages": [HumanMessage(question)]}).lower()


# ---------- recall: tell it in session 1, check what a NEW session is handed ----------
def test_r1_fact_recall():
    chat("r1-s1", "I'm allergic to peanuts and I prefer morning meetings.")
    assert any("peanut" in x for x in facts())
    assert "peanut" in new_session_memory("What food should I avoid?")


def test_r2_fact_recall():
    chat("r2-s1", "My name is Arjun and I work as a data analyst.")
    assert any("analyst" in x for x in facts())
    assert "analyst" in new_session_memory("What do I do for work?")


def test_r3_relation_recall():
    chat("r3-s1", "My sister Meera works at Wipro.")
    rel = relations()
    assert "user sister meera" in rel
    assert "meera works_at wipro" in rel
    assert "wipro" in new_session_memory("Where does my sister work?")


# ---------- update: the stale value must be gone from the stores ----------
def test_u1_update_city():
    chat("u1-s1", "I live in Hyderabad.")
    chat("u1-s2", "I moved to Pune.")
    all_stored = facts() + relations()
    assert any("pune" in x for x in facts())
    assert not any("hyderabad" in x for x in all_stored)


def test_u2_update_preference():
    chat("u2-s1", "My favorite programming language is Java.")
    chat("u2-s2", "Actually I switched, my favorite is Python now.")
    f = facts()
    assert any("python" in x for x in f)
    assert not any("java" in x for x in f)


def test_u3_no_duplicates():
    chat("u3-s1", "I live in Hyderabad.")
    chat("u3-s2", "I live in Hyderabad.")
    matches = [x for x in facts() + relations() if "hyderabad" in x]
    assert len(matches) == 1, f"matching facts={matches}"


def test_u4_relation_update():
    chat("u4-s1", "My sister Meera works at Wipro.")
    chat("u4-s2", "Meera now works at TCS.")
    rel = relations()
    assert any("tcs" in x for x in rel)
    assert not any("wipro" in x for x in rel)


# ---------- forget: approved delete removes it, denied delete keeps it ----------
def test_f1_forget_approved():
    chat("f1-s1", "I'm allergic to peanuts.")
    chat("f1-s2", "Forget that I'm allergic to peanuts.", approve="yes")
    assert not any("peanut" in x for x in facts())


def test_f2_forget_denied():
    chat("f2-s1", "I'm allergic to peanuts.")
    chat("f2-s2", "Forget that I'm allergic to peanuts.", approve="no")
    assert any("peanut" in x for x in facts())


def test_f3_relation_forget_approved():
    chat("f3-s1", "My sister Meera works at Wipro.")
    chat("f3-s2", "Forget that Meera works at Wipro.", approve="yes")
    assert not any("wipro" in x for x in relations())