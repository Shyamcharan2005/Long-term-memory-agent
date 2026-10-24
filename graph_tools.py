import sqlite3
from contextlib import closing
from datetime import datetime
from pathlib import Path
from langchain_core.tools import tool
from fact_tools import DATA_DIR

DB_PATH = str(DATA_DIR / "graph.db")

def _conn():
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """CREATE TABLE IF NOT EXISTS relations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            subject TEXT NOT NULL,
            relation TEXT NOT NULL,
            object TEXT NOT NULL,
            created TEXT NOT NULL
        )"""
    )
    return conn


def _norm(text: str) -> str:
    return text.strip().lower()


@tool
def remember_relation(subject: str, relation: str, obj: str) -> str:
    """Store a link between two named things as (subject, relation, obj),
    e.g. ('user', 'sister', 'Priya') or ('Priya', 'works_at', 'Infosys').
    Use 'user' as the subject when the user is talking about themselves.
    Use this for links between people, places, companies. Use remember_fact for plain facts and preferences.
    Store one relation per call."""
    s, r, o = _norm(subject), _norm(relation).replace(" ", "_"), _norm(obj)
    with closing(_conn()) as conn:
        exists = conn.execute(
            "SELECT 1 FROM relations WHERE subject=? AND relation=? AND object=?",
            (s, r, o),
        ).fetchone()
        if exists:
            return f"Already stored: {s} -{r}-> {o}"
        conn.execute(
            "INSERT INTO relations (subject, relation, object, created) VALUES (?,?,?,?)",
            (s, r, o, datetime.now().isoformat()),
        )
        conn.commit()
    return f"Stored: {s} -{r}-> {o}"


@tool
def query_relations(subject: str = "", relation: str = "", obj: str = "") -> str:
    """Look up stored relations. Give at least one of subject, relation, obj; leave the others empty.
    Examples: subject='user', relation='sister' finds the user's sister.
    subject='priya', relation='works_at' finds where Priya works.
    For multi-step questions ('where does my sister work?') call this once per hop:
    first find the sister's name, then query that name."""
    s, r, o = _norm(subject), _norm(relation).replace(" ", "_"), _norm(obj)
    if not (s or r or o):
        return "ERROR: give at least one of subject, relation, obj"

    clauses, params = [], []
    if s:
        clauses.append("subject=?")
        params.append(s)
    if r:
        clauses.append("relation=?")
        params.append(r)
    if o:
        clauses.append("object=?")
        params.append(o)

    sql = "SELECT subject, relation, object FROM relations WHERE " + " AND ".join(clauses) + " LIMIT 20"
    with closing(_conn()) as conn:
        rows = conn.execute(sql, params).fetchall()
    if not rows:
        return "No matching relations."
    return "\n".join(f"{a} -{b}-> {c}" for a, b, c in rows)

@tool
def update_relation(subject: str, relation: str, old_obj: str, new_obj: str) -> str:
    """Change an existing relation, e.g. Priya works_at Infosys becomes Priya works_at TCS:
    subject='Priya', relation='works_at', old_obj='Infosys', new_obj='TCS'.
    Call query_relations first to see exactly what is stored. Never guess."""
    s, r = _norm(subject), _norm(relation).replace(" ", "_")
    old, new = _norm(old_obj), _norm(new_obj)
    with closing(_conn()) as conn:
        cur = conn.execute(
            "UPDATE relations SET object=?, created=? WHERE subject=? AND relation=? AND object=?",
            (new, datetime.now().isoformat(), s, r, old),
        )
        conn.commit()
        if cur.rowcount == 0:
            return f"ERROR: no stored relation {s} -{r}-> {old}"
    return f"Updated: {s} -{r}-> {old} is now {s} -{r}-> {new}"


@tool
def forget_relation(subject: str, relation: str, obj: str) -> str:
    """Permanently delete one stored relation, e.g. subject='Priya', relation='works_at', obj='Infosys'.
    Call query_relations first to see exactly what is stored. Never guess."""
    s, r, o = _norm(subject), _norm(relation).replace(" ", "_"), _norm(obj)
    with closing(_conn()) as conn:
        cur = conn.execute(
            "DELETE FROM relations WHERE subject=? AND relation=? AND object=?", (s, r, o)
        )
        conn.commit()
        if cur.rowcount == 0:
            return f"ERROR: no stored relation {s} -{r}-> {o}"
    return f"Forgot: {s} -{r}-> {o}"

def graph_context(root="user", max_rows=30):
    """Everything linked to the user, plus one more hop, as plain text."""
    with closing(_conn()) as conn:
        first = conn.execute(
            "SELECT subject, relation, object FROM relations WHERE subject=?", (root,)
        ).fetchall()
        rows = list(first)
        for _, _, obj in first:
            rows += conn.execute(
                "SELECT subject, relation, object FROM relations WHERE subject=?", (obj,)
            ).fetchall()
    return "\n".join(f"{a} -{b}-> {c}" for a, b, c in rows[:max_rows])

GRAPH_TOOLS = [remember_relation, query_relations, update_relation, forget_relation]

