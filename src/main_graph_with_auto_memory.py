import os
import json
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate, MessagesPlaceholder
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.agents import tool
from langgraph.graph import StateGraph, END, MessagesState

# from src.router_agent import router_agent_node
# from src.api_agent import api_agent_node
# from src.sql_agent import sql_agent, sql_is_done


from pydantic import BaseModel
from typing import List, Dict, Optional
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent



memory = MemorySaver()


class CustomState(MessagesState):
    user_input: str
    route: str | None = None
    ag_output: str | None = None

config = {"configurable": {"thread_id": "test-fourth"}}


def router_agent(state):

    model = ChatOpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.environ["OPENAI_ENDPOINT"],
        model_name=os.environ["MODEL_NAME"],
        temperature=0
    )
    
    user_input = state['user_input']

    system_message = "You are a routing agent. Decide if the query is for SQL or for an external API. Reply with either 'sql' or 'api' based on input"

    @tool
    def decide_next_step(decided_input) -> str:
        """ This tool is used to Decide next step, returns the next step as string"""
        print("Final answer is decided")
        return decided_input

    router_agent_ex = create_react_agent(
        model, tools=[], prompt=system_message, checkpointer=memory
    )

    result = router_agent_ex.invoke({
        'messages': [
            ("user", user_input)
        ]},
        config
    )

    return {
        # "messages": result['messages'][-1], ------------------------------------------------>>>>>>>>>>>>
        "route": result["messages"][-1].content
    }


def api_assistent(state):
    api_user_input = state['user_input']

    @tool
    def fill_timesheet(email: str, date: str, hours: float) -> str:
        """ This tool is used to fill the timesheet and return success """
        import ipdb; ipdb.set_trace();
        return "Timesheet filled successfully"


    @tool
    def apply_vacation(email: str, date: str) -> str:
        """ This tool is used to apply the vacation and return success """
        import ipdb; ipdb.set_trace();
        return f"Vacation aplied successfully for email {email} and date {date}"
    


    api_model = ChatOpenAI(
            api_key=os.environ["OPENAI_API_KEY"],
            base_url=os.environ["OPENAI_ENDPOINT"],
            model_name=os.environ["MODEL_NAME"],
            temperature=0
    )

    system_message = """You are an expert who can decide which of the below APIs to call based on user input."""

    api_tools = [fill_timesheet, apply_vacation]

    api_agent_ex = create_react_agent(
        api_model, api_tools, prompt=system_message, checkpointer=memory
    )


    import ipdb; ipdb.set_trace();

    api_result = api_agent_ex.invoke({
            'messages': api_user_input
        },
        config
    )


    return {
            "ag_output": api_result["messages"][-1].content
        }

    # return {
    #     "messages": api_result['messages'][-1]  ------------------------------------------------>>>>>>>>>>>>
    # }



def sql_assistent(state):
    sql_user_input = state['user_input']

    # sql_user_input = state["messages"][-1].content


    sql_model = ChatOpenAI(
            api_key=os.environ["OPENAI_API_KEY"],
            base_url=os.environ["OPENAI_ENDPOINT"],
            model_name=os.environ["MODEL_NAME"],
            temperature=0
    )

    sql_system_message = """
        You are an expert SQL generator.
        Convert the user's natural language request into a SQL query & execute and send result to the user.
        If you need more information to generate the SQL query, ask the user.

        Below are the tables with schema:
            [
                "table": "api_sfdeal",
                "columns": "project_id, project_name, client_id, client_name, start_date, end_date"
            ],
            [
                "table": "api_user",
                "columns": "user_id, name, email, manager, joining_date, status"
            ]
        """
    
    @tool
    def execute_and_return_result(sql_query) -> str:
        """ This tool is used to execute the SQL and run query and return result out of it """
        print("Executing ----------->>>>> Query -------->>>> Using Tool and Returnig Result!!!")
        print(sql_query)

        return "Result is 'vageesha.manjappa@galepartners.com"


    sql_agent_ex = create_react_agent(
        sql_model, [execute_and_return_result], prompt=sql_system_message, checkpointer=memory
    )


    sql_result = sql_agent_ex.invoke({
        'messages': [
            ("user", sql_user_input)
        ]},
        config
    )

    return {
        "ag_output": sql_result["messages"][-1].content
    }

    # return {
    #     "messages": sql_result['messages'][-1]
    # }



builder = StateGraph(CustomState)
# builder = StateGraph(State)

builder.add_node("router_agent", router_agent)
builder.add_node("sql_agent", sql_assistent)
builder.add_node("api_agent", api_assistent)
builder.set_entry_point("router_agent")

# lambda s: s.get("route"), --> We can also do get
builder.add_conditional_edges("router_agent",
    lambda s: s['route'], {
    "sql": "sql_agent",
    "api": "api_agent"
})

# This will keep calling sql_agent untill it return completed
builder.add_edge("sql_agent", END)
builder.add_conditional_edges("sql_agent", sql_assistent, {
    True: END,
    False: "sql_agent"
})

builder.add_edge("api_agent", END)  
# graph = builder.compile()
graph = builder.compile(checkpointer=memory)


while True:
    # 🧠 Get user input
    user_query = input("🧑 User: ")
    input_message = HumanMessage(content=user_query)

    if user_query == "quit":
        # a = State(input=user_query, route="done", output="Goodbye", sql_is_done=True)
        break
    else:
        result = graph.invoke({"user_input": user_query}, config)
        final_output = result["ag_output"]
        print(final_output)


        # for event in graph.stream({"messages": [input_message]}, config, stream_mode="values"):
        #     event["messages"][-1].pretty_print()


    print("Done!")
