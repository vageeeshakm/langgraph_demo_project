# graph/main_graph.py

# from langgraph.graph import StateGraph, END
# from langgraph.graph import StateGraph, START

import sys
print(sys.path)

from langgraph.graph import StateGraph, START, END

from typing import TypedDict, NotRequired

from src.sql_agent import get_sql_query_agent

# Step 1: Define your state type
class GraphState(TypedDict):
    query: str
    sql: NotRequired[str]
    result: NotRequired[str]


# Node that runs the SQL agent
def run_sql_agent(state: GraphState) -> GraphState:
    result = get_sql_query_agent().invoke({"input": state["query"]})
    # return {**state, "sql": sql, "result": result}
    return {"input": state["query"], "sql_result": str(result)}


def print_sql_node(state: dict) -> dict:
    sql_q = state.get("sql_result", "No SQL to print")
    print(f"Final SQL Query Generated:---> {sql_q}")
    # Return state unchanged or with a flag
    return state

# Node that runs the SQL agent
def end_agent(state: GraphState) -> GraphState:
    print("Workflow completed.")
    return state


# Build the LangGraph
builder = StateGraph(GraphState)
builder.add_node("sql_agent", run_sql_agent)
builder.add_node("print_sql", print_sql_node)
builder.add_node("end", end_agent)

# Define edges (transitions)
builder.add_edge(START, "sql_agent")
builder.add_edge("sql_agent", "print_sql")
builder.add_edge("print_sql", "end")

# Optional: compile to validate or optimize
compiled_graph = builder.compile()

while True:
    try:
        user_input = input("User: ")
        state: GraphState = {
            "query": user_input,
        }
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break

        updated_state = run_sql_agent(state)
        print(f"I am printing ----> {updated_state.get('sql_result')}")

    except Exception as e:
        print(f"Somethig went wrong {e}")
        break