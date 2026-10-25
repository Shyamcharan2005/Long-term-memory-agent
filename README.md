# Long-Term Memory Agent

An LLM agent that remembers things about you across conversations. Plain facts go into a vector store, relationships between people and places go into a small graph, and anything destructive waits for a human to click Yes.

**Live demo:** <https://long-term-memory-agent-e4yqy63ecqzl5kmfhkovz8.streamlit.app/> ·

**Repo:** <https://github.com/Shyamcharan2005/Long-term-memory-agent>

## What this is

Chat models forget everything once a conversation ends. LangGraph's checkpointer fixes this only per thread: the same `thread_id` gets its history back, and a new chat starts empty. This project adds the missing layer. What you tell the agent is distilled into small records and kept in separate stores, so a brand-new chat still knows who you are.

It is deliberately narrow. It does not book anything or browse the web. It remembers, updates and forgets, and I spent the effort on making that part dependable and measurable.

Stack: LangGraph, ChromaDB, SQLite, Streamlit. The model is `openai/gpt-oss-120b` on the Groq free tier, used for the whole project. Embeddings run locally with `all-MiniLM-L6-v2`, so nothing besides the chat model needs an API key.

## Architecture

```
 user message
      |
      v
 +-----------------------------------------------+
 | llm node                                      |
 |   1. retrieve from BOTH stores (local lookup) |
 |        - ChromaDB: top 3 facts for the message|
 |        - SQLite graph: everything linked to   |
 |          "user", plus one more hop            |
 |   2. attach results to the system prompt      |
 |   3. call the model                           |
 +-----------------------------------------------+
      |
      +-- no tool calls --------------> answer
      |
      +-- tool calls --> tools node --> back to llm node
                            |
                            +-- forget_fact / forget_relation:
                                interrupt() -> human approves or denies
```

Three kinds of memory are involved:

| Layer | Where it lives | Scope |
|---|---|---|
| Conversation | `SqliteSaver` checkpoints | One `thread_id` |
| Facts | ChromaDB with local embeddings | All threads |
| Relations | SQLite table `(subject, relation, object)` | All threads |

A fact is one short, self-contained sentence such as "user is allergic to peanuts", with an id and a timestamp. A relation is a row such as `(user, sister, ananya)` or `(ananya, works_at, infosys)`. Names are lowercased on the way in so "Ananya" and "ananya" match.

## Design decisions

**Two stores instead of one.** Vector search is good at fuzzy questions: "what should I avoid eating?" finds "allergic to peanuts" without sharing a keyword. It is poor at following links. "Where does my sister's husband work?" needs three separate statements chained together, and ranking by similarity to the question does not chain anything. A table of relations handles that by querying one hop at a time. The evaluation below has a concrete example where the vector store fails and the graph succeeds.

**SQLite for the graph.** The relations are a single table with three columns and a few dozen rows per user. A graph database would add a service to run and deploy for no gain at this size. The tools only need "find rows where subject is X", which is a normal SQL query.

**Retrieval happens in code, not through a tool.** My first version gave the model `search_memory` and `query_relations` and told it to check them before answering. It did not do this reliably. In one session the model had stored "I study at CMR" as a relation, then answered a later question by searching only the facts store and said it did not know. Telling it to check both stores in the prompt was not enough, so the lookup now runs before every model call and the results are attached to the prompt. This is a local query, so it costs no extra model call. The lookup tools still exist for follow-up hops the attached memory does not cover.

**Rules live in the tools, not only in the prompt.** The model sometimes stored the same fact twice. Rather than adding another prompt instruction, `remember_fact` checks the nearest stored fact first and, if it is almost identical (distance under 0.3), refuses and tells the model to update instead. The check works whatever the model decides to do.

**Approval before anything is deleted.** `forget_fact` and `forget_relation` call `interrupt()`, and the approval box shows what will actually be removed (the fact text or the relation), not an opaque id. One detail matters here: when a LangGraph node resumes after an interrupt, the whole node runs again from the top. If a model emits a store and a delete in the same step, a naive node would run the store twice. The tools node therefore collects every approval first and only then executes the tools.

**Few model calls.** A message that stores something usually takes two model calls (one to emit the tool calls, one to answer). A plain recall question takes one. This matters on a free tier with a daily token cap.

## Tools

| Tool | What it does |
|---|---|
| `remember_fact` | Store a new fact (rejects near-duplicates) |
| `search_memory` | Similarity search over facts, returns ids and distances |
| `update_fact` | Replace a stored fact, by id |
| `forget_fact` | Delete a fact, by id. Needs approval |
| `remember_relation` | Store `(subject, relation, object)` |
| `query_relations` | Look up relations by any combination of subject, relation, object |
| `update_relation` | Change the object of an existing relation |
| `forget_relation` | Delete a relation. Needs approval |

## Demo walkthrough

Use a made-up person if you are trying the hosted version, since memory there is shared.

1. Chat 1: `I'm Ravi. I live in Pune. My sister Ananya works at Infosys in Bengaluru.`
   Open the "Tool calls" expander. You should see calls along the lines of `remember_fact("user's name is Ravi")`, `remember_fact("user lives in Pune")`, `remember_relation(user, sister, Ananya)` and `remember_relation(Ananya, works_at, Infosys)`.
2. Click **New chat**. The conversation is gone, and the memory is not.
3. Chat 2: `Where does my sister work?` The answer comes from stored memory, with no earlier messages in the thread.
4. Say `I moved to Chennai.` The agent should call `update_fact` on the existing location rather than adding a second one. Ask where you live to check.
5. Say `Forget where I live.` The approval box names the fact. Click **No** and ask again to confirm it is still there, then repeat and click **Yes**.
6. Say `Ananya now works at TCS.` This updates the relation instead of adding a conflicting one.

The sidebar keeps your past chats. Each has a menu for renaming or deleting it. Deleting a chat does not delete anything the agent learned, which is the point of keeping the memory separate.

## Evaluation

`python eval.py` runs ten scripted scenarios. A few choices make the results meaningful:

- Every scenario runs against a fresh temporary data folder, and both stores are wiped between scenarios, so real memory is never touched and scenarios cannot influence each other.
- Information is given in one session and asked about in a **new thread**, so a correct answer can only come from long-term memory, not from conversation history.
- Update and forget cases check the stores themselves, not the wording of the reply. A model saying "done, removed" proves nothing.
- The approval prompt is answered automatically, with yes or no depending on the case.

| ID | Scenario | Passes if |
|---|---|---|
| R1 | Tell it an allergy and a meeting preference, then ask what food to avoid in a new thread | Reply mentions peanuts |
| R2 | Tell it a name and job, ask the job in a new thread | Reply mentions the job |
| R3 | Tell it a sister works at Wipro, ask where in a new thread | Reply mentions Wipro (relation recall) |
| U1 | Live in Hyderabad, later "I moved to Pune", then ask | Reply says Pune, and Hyderabad is gone from both stores |
| U2 | Favorite language Java, later Python | Python stored, Java gone |
| U3 | Say the same fact in two sessions | Exactly one stored copy |
| U4 | Sister works at Wipro, later at TCS | Relation table has TCS, no Wipro |
| F1 | Store an allergy, ask to forget it, approve | Fact is gone |
| F2 | Same, but deny the approval | Fact is still stored |
| F3 | Store a relation, ask to forget it, approve | Relation is gone |

| Category | Result |
|---|---|
| Recall | 3/3 |
| Update | 4/4 |
| Forget | 3/3 |

That is 10/10 on one run. Language model output varies between runs and ten scenarios is a small suite, so read this as a sanity check that the mechanics work, not as an accuracy benchmark. Raw results are in `eval_results.json`.

### Graph versus vector

For the question "Where does my sister's husband work?" I stored the same information two ways: as 16 separate facts (three of them forming the chain, thirteen about other people), and as relations. The vector top 3 for the question was:

```
Priya is my sister
Vikram works at Zoho
Rohan works at Amazon
```

The two statements linking Priya to her husband and his employer are not in the results, so the question cannot be answered from them. Following relations in the graph answers it in three hops: `user -sister-> priya`, `priya -husband-> karan`, `karan -works_at-> infosys`. This is deterministic and uses no model. The setup is `graph_vs_vector()` in `eval.py`. A two-step chain (sister, then her employer) did not fail in the same setup, so the graph's advantage shows up on longer chains, not on every multi-step question.

## Problems I ran into

1. **Prompt rules were not enough.** Several of the fixes above (duplicate guard, automatic retrieval) exist because an instruction in the system prompt was followed most of the time and ignored the rest. Anything that must hold is now enforced in code.
2. **The eval gave a false pass.** An early update test checked only the facts store, so it passed while a stale "lives in Hyderabad" relation was still sitting in the graph. The check now covers both stores.
3. **Facts versus relations.** The model sometimes filed statements about the user themselves ("I study at CMR") as relations, which earlier broke recall. Automatic retrieval from both stores fixed the symptom, and the system prompt now describes the split with examples.
4. **Weak similarity margins.** With a small embedding model, the right fact and a wrong one can score close (0.88 against 0.94 in one early test). That is why search returns the top three with distances and the model judges relevance, instead of trusting the top hit.
5. **Resumed interrupts re-run the node.** See the approval design above.
6. **Free-tier limits.** The Groq free tier caps tokens per day. I used it up with repeated eval runs, and the resulting rate-limit errors showed up as failed scenarios. Failures with a 429 error are infrastructure, not test results.

## Limitations

- The model decides whether something is a fact or a relation. Retrieval covers both stores, so a wrong choice no longer loses the information, but the stored form can still be inconsistent.
- Relation names are not enforced. The prompt suggests a vocabulary, and the model can still invent a new name.
- Memory belongs to one user. In the hosted demo, every visitor reads and writes the same store, and it may reset when the app restarts because free hosting has temporary storage. Do not enter real personal information there.
- The agent does not store its own guesses. It stores what you say.
- Ten scenarios, one run.

## Run locally

```
pip install -r requirements.txt
```

Create a `.env` file containing `GROQ_API_KEY=your_key`, then:

```
streamlit run streamlit_app.py
```

Terminal version: `python agent.py`. Evaluation: `python eval.py`. For Streamlit Community Cloud, put the key in the app's secrets as `GROQ_API_KEY`.

Runtime data (Chroma store, relations, checkpoints, chat titles) is written to a `data/` folder next to the code. It is git-ignored.

## Project layout

| File | Purpose |
|---|---|
| `agent.py` | LangGraph agent: nodes, automatic retrieval, approval step, terminal chat loop |
| `fact_tools.py` | ChromaDB fact memory and its tools |
| `graph_tools.py` | SQLite relation graph and its tools |
| `streamlit_app.py` | Chat UI: past chats, tool-call view, approval buttons |
| `eval.py` | Scripted multi-session evaluation and the graph-vs-vector comparison |