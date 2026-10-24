import json
import os
import sys
import tempfile
import time
from contextlib import closing

# Isolated data dir, set BEFORE importing the agent, so your real memory is never touched
os.environ["MEMORY_DATA_DIR"] = tempfile.mkdtemp(prefix="memeval_")

from langgraph.types import Command

from agent import app
from fact_tools import col
from graph_tools import _conn, remember_relation, query_relations


# ---------- helpers ----------
def reset_memory():
    ids = col.get()["ids"]
    if ids:
        col.delete(ids=ids)
    with closing(_conn()) as c:
        c.execute("DELETE FROM relations")
        c.commit()


def chat(thread, text, approve="yes"):
    """One user message in one session (thread). Auto-answers the approval prompt."""
    config = {"configurable": {"thread_id": thread}}
    result = app.invoke({"messages": [("user", text)]}, config=config)
    while "__interrupt__" in result:
        result = app.invoke(Command(resume=approve), config=config)
    time.sleep(1)  # be gentle with Groq rate limits
    return result["messages"][-1].content


def facts():
    return [d.lower() for d in col.get()["documents"]]

def relations():
    with closing(_conn()) as c:
        return [" ".join(r) for r in c.execute("SELECT subject, relation, object FROM relations")]

def has(text, word):
    return word.lower() in text.lower()


# ---------- recall: tell it in session 1, ask in a NEW thread ----------
def r1():
    chat("r1-s1", "I'm allergic to peanuts and I prefer morning meetings.")
    reply = chat("r1-s2", "What food should I avoid?")
    return has(reply, "peanut"), reply


def r2():
    chat("r2-s1", "My name is Arjun and I work as a data analyst.")
    reply = chat("r2-s2", "What do I do for work?")
    return has(reply, "analyst"), reply


def r3():
    chat("r3-s1", "My sister Meera works at Wipro.")
    reply = chat("r3-s2", "Where does my sister work?")
    return has(reply, "wipro"), reply


# ---------- update: the stale fact must be gone from the store ----------
def u1():
    chat("u1-s1", "I live in Hyderabad.")
    chat("u1-s2", "I moved to Pune.")
    reply = chat("u1-s3", "Where do I live?")
    f = facts()
    stale = any("hyderabad" in x for x in facts() + relations())
    ok = has(reply, "pune") and any("pune" in x for x in f) and not stale
    return ok, f"reply={reply!r} facts={f}"


def u2():
    chat("u2-s1", "My favorite programming language is Java.")
    chat("u2-s2", "Actually I switched, my favorite is Python now.")
    reply = chat("u2-s3", "What is my favorite programming language?")
    f = facts()
    ok = has(reply, "python") and any("python" in x for x in f) and not any("java" in x for x in f)
    return ok, f"reply={reply!r} facts={f}"


def u3_no_duplicates():
    chat("u3-s1", "I live in Hyderabad.")
    chat("u3-s2", "I live in Hyderabad.")
    f = [x for x in facts() + relations() if "hyderabad" in x]
    return len(f) == 1, f"matching facts={f}"

def u4_relation_update():
    chat("u4-s1", "My sister Meera works at Wipro.")
    chat("u4-s2", "Meera now works at TCS.")
    reply = chat("u4-s3", "Where does my sister work?")
    rel = relations()
    ok = has(reply, "tcs") and any("tcs" in x for x in rel) and not any("wipro" in x for x in rel)
    return ok, f"reply={reply!r} relations={rel}"


def f3_relation_forget():
    chat("f3-s1", "My sister Meera works at Wipro.")
    chat("f3-s2", "Forget that Meera works at Wipro.", approve="yes")
    rel = relations()
    return not any("wipro" in x for x in rel), f"relations={rel}"
# ---------- forget: approved delete removes it, denied delete keeps it ----------
def f1():
    chat("f1-s1", "I'm allergic to peanuts.")
    chat("f1-s2", "Forget that I'm allergic to peanuts.", approve="yes")
    f = facts()
    return not any("peanut" in x for x in f), f"facts={f}"


def f2():
    chat("f2-s1", "I'm allergic to peanuts.")
    chat("f2-s2", "Forget that I'm allergic to peanuts.", approve="no")
    f = facts()
    return any("peanut" in x for x in f), f"facts={f}"


CASES = [
    ("recall", "R1 fact recall", r1),
    ("recall", "R2 fact recall", r2),
    ("recall", "R3 relation recall", r3),
    ("update", "U1 update city", u1),
    ("update", "U2 update preference", u2),
    ("update", "U3 no duplicate facts", u3_no_duplicates),
    ("forget", "F1 forget approved", f1),
    ("forget", "F2 forget denied", f2),
    ("update", "U4 relation update", u4_relation_update),
    ("forget", "F3 relation forget approved", f3_relation_forget),
]


# ---------- vector-only vs graph demo (no LLM, deterministic) ----------
def graph_vs_vector():
    reset_memory()
    question = "Where does my sister's husband work?"
    jobs = [("Ananya", "Google"), ("Rohan", "Amazon"), ("Karthik", "Flipkart"),
            ("Divya", "Microsoft"), ("Suresh", "Oracle"), ("Neha", "Adobe"),
            ("Vikram", "Zoho"), ("Meena", "Swiggy"), ("Imran", "Paytm")]
    couples = [("Meera", "Sanjay"), ("Sunita", "Mohan"), ("Kavya", "Arun"), ("Isha", "Kabir")]
    chain = ["Priya is my sister", "Karan is Priya's husband", "Karan works at Infosys"]

    # Vector-only: every statement is its own fact (agent's search_memory also returns top 3)
    stored = chain + [f"{p} works at {c}" for p, c in jobs] + [f"{h} is {w}'s husband" for w, h in couples]
    col.add(ids=[f"d{i}" for i in range(len(stored))], documents=stored)
    top3 = col.query(query_texts=[question], n_results=3)["documents"][0]
    vector_ok = all(c in top3 for c in chain)

    # Graph: same information as relations, answered by following links
    reset_memory()
    for s, r, o in [("user", "sister", "Priya"), ("Priya", "husband", "Karan"), ("Karan", "works_at", "Infosys")]:
        remember_relation.invoke({"subject": s, "relation": r, "obj": o})
    for p, c in jobs:
        remember_relation.invoke({"subject": p, "relation": "works_at", "obj": c})
    for w, h in couples:
        remember_relation.invoke({"subject": w, "relation": "husband", "obj": h})
    hop1 = query_relations.invoke({"subject": "user", "relation": "sister"})
    hop2 = query_relations.invoke({"subject": hop1.split("->")[-1].strip(), "relation": "husband"})
    hop3 = query_relations.invoke({"subject": hop2.split("->")[-1].strip(), "relation": "works_at"})
    graph_ok = has(hop3, "infosys")

    return {"question": question, "vector_top3": top3, "vector_answerable": vector_ok,
            "graph_hops": [hop1, hop2, hop3], "graph_answerable": graph_ok}

if __name__ == "__main__":
    runs = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    totals = {}   # category -> [passed, total]
    log = []

    for run in range(runs):
        for cat, name, fn in CASES:
            reset_memory()
            try:
                ok, detail = fn()
            except Exception as e:
                if "rate_limit_exceeded" in str(e):
                    sys.exit(f"Groq rate limit hit during '{name}'. This is not a test result. Rerun after the limit resets.")
                ok, detail = False, f"EXCEPTION: {e}"
            totals.setdefault(cat, [0, 0])
            totals[cat][0] += int(ok)
            totals[cat][1] += 1
            log.append({"run": run + 1, "case": name, "passed": ok, "detail": detail})
            print(f"[run {run + 1}] {'PASS' if ok else 'FAIL'}  {name}\n        {detail}\n")

    demo = graph_vs_vector()
    print("GRAPH vs VECTOR demo")
    print(json.dumps(demo, indent=2))

    print("\nPASS RATES")
    all_p = sum(p for p, _ in totals.values())
    all_t = sum(t for _, t in totals.values())
    for cat, (p, t) in totals.items():
        print(f"  {cat:7s} {p}/{t} = {p / t:.0%}")
    print(f"  overall {all_p}/{all_t} = {all_p / all_t:.0%}")

    with open("eval_results.json", "w") as f:
        json.dump({"runs": runs, "pass_rates": totals, "cases": log, "graph_vs_vector": demo}, f, indent=2)