import sqlite3
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import ToolMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import interrupt, Command

from fact_tools import FACT_TOOLS, col, DATA_DIR, search_memory
from graph_tools import GRAPH_TOOLS, graph_context
load_dotenv()

ALL_TOOLS = FACT_TOOLS + GRAPH_TOOLS
TOOL_FUNCTIONS = {t.name: t for t in ALL_TOOLS}

RISKY = {"forget_fact", "forget_relation"}

SYSTEM_PROMPT ="""You are a personal assistant with long-term memory in two stores:
- FACTS (remember_fact, search_memory, update_fact, forget_fact): short free-text statements, e.g. "User's name is Shyam", "User studies at CMR", "User is allergic to peanuts".
- RELATIONS (remember_relation, query_relations, update_relation, forget_relation): links between two named things as (subject, relation, obj), e.g. ('user','sister','Meera'), ('Meera','works_at','Wipro'). Use 'user' as the subject for the user.

STORING: When the user tells you something, use your own judgment about which store fits: statements about the user themself are facts; links between named people, places or organisations are relations. Extract everything in the message and make all the tool calls in one step. Use short snake_case relation names. Do not search before storing new information.

CHANGING: If the user changes something already stored, find it first (search_memory for facts, query_relations for relations), then call update_fact or update_relation.
FORGETTING: find it first, then call forget_fact or forget_relation. Never guess ids or values.

INFERENCE: You may combine stored memory with general knowledge to answer (for example,
someone who lives in Hyderabad is in India). Say when an answer is an inference rather than 
something the user told you. Never store an inference. Never infer sensitive traits such as 
gender, religion or orientation;say you don't have that information.

ANSWERING: The memory most relevant to the user's latest message is attached below under STORED 
MEMORY, retrieved from both stores automatically. Answer from it. If the answer needs a further 
hop (e.g. a relative's employer's city), call query_relations for it. Only say you don't know if
 the attached memory and any extra lookups do not contain the answer. Never guess."""

llm = ChatGroq(model="openai/gpt-oss-120b", max_retries=5).bind_tools(ALL_TOOLS)


class State(TypedDict):
    messages: Annotated[list, add_messages]

def memory_context(state):
    """Retrieve from BOTH stores for the user's latest message. No LLM call involved."""
    last_user = next((m.content for m in reversed(state["messages"]) if m.type == "human"), "")
    facts = search_memory.invoke({"query": last_user}) if last_user else "none"
    rels = graph_context() or "none"
    return (
        "\n\nSTORED MEMORY (retrieved automatically for the user's latest message):\n"
        f"Facts (with ids, ranked by distance):\n{facts}\n"
        f"Relations linked to the user, plus one hop further:\n{rels}"
    )


def llm_node(state: State):
    system = SYSTEM_PROMPT + memory_context(state)
    return {"messages": [llm.invoke([SystemMessage(system)] + state["messages"])]}



def tools_node(state: State):
    last = state["messages"][-1]

    # Pass 1: ask for approval on every risky call BEFORE running anything.
    # On resume, this node re-runs from the top, so nothing with side effects may run before the interrupt.
    decisions = {}
    for call in last.tool_calls:
        if call["name"] in RISKY:
            args = call["args"]
            if call["name"] == "forget_fact":
                fact_id = args.get("fact_id", "")
                found = col.get(ids=[fact_id])["documents"] if fact_id else []
                desc = f'fact [{fact_id}] "{found[0] if found else "(id not found)"}"'
            else:
                desc = f'relation {args.get("subject")} -{args.get("relation")}-> {args.get("obj")}'
            decisions[call["id"]] = interrupt({"action": call["name"], "description": desc})

    # Pass 2: run the tools
    results = []
    for call in last.tool_calls:
        if call["name"] in RISKY and decisions[call["id"]] != "yes":
            output = "Human denied this action. Do not retry. Tell the user it was not done."
        else:
            try:
                output = TOOL_FUNCTIONS[call["name"]].invoke(call["args"])
            except Exception as e:
                output = f"ERROR: {e}"
        print(f"  [tool] {call['name']}({call['args']}) -> {output}")
        results.append(ToolMessage(content=output, tool_call_id=call["id"]))
    return {"messages": results}


def should_continue(state: State):
    if state["messages"][-1].tool_calls:
        return "tools"
    return END


graph = StateGraph(State)
graph.add_node("llm", llm_node)
graph.add_node("tools", tools_node)
graph.add_edge(START, "llm")
graph.add_conditional_edges("llm", should_continue, ["tools", END])
graph.add_edge("tools", "llm")

conn = sqlite3.connect(str(DATA_DIR / "checkpoints.db"), check_same_thread=False)
app = graph.compile(checkpointer=SqliteSaver(conn))


if __name__ == "__main__":
    config = {"configurable": {"thread_id": input("thread id: ")}}

    while True:
        user_text = input("you: ")
        if user_text.lower() in ("quit", "exit"):
            break

        result = app.invoke({"messages": [("user", user_text)]}, config=config)

        while "__interrupt__" in result:
            info = result["__interrupt__"][0].value
            print(f"APPROVAL NEEDED: delete {info['description']}")
            answer = input("yes/no: ")
            result = app.invoke(Command(resume=answer), config=config)

        print("agent:", result["messages"][-1].content)
