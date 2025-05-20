import os
import json
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate, MessagesPlaceholder
# from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langgraph.prebuilt import create_react_agent
from langchain.agents import tool
from src.redis_utils import RedisMemory


system_message = """
You are an expert who can decide which of the below APIs to call based on user input.
You have access 2 below tools:
    Vacation tool: You can apply vacation using it
        - params:
            - email: user email address 
            - date: date on which vacation to be applied

    fill_timesheet -- Timesheet tool: You can apply timesheet using it
        - params:
            - email: user email address 
            - date: date on which vacation to be applied
            - hours: Number of hours
"""

# human_message = HumanMessagePromptTemplate.from_template(
# """
# User input: {user_input}
# """
# )

# prompt = ChatPromptTemplate.from_messages([
#     system_message,
#     # MessagesPlaceholder(variable_name="history"), ---> Include history, check how to include last 5 and things like that
#     human_message
# ])


@tool
def fill_timesheet(email: str, date: str, hours: float) -> str:
    """ This tool is used to fill the timesheet and return success """
    import ipdb; ipdb.set_trace();
    return "Timesheet filled successfully"


@tool
def apply_vacation(email: str, date: str) -> str:
    """ This tool is used to apply the vacation and return success """
    import ipdb; ipdb.set_trace();
    return "Vacation aplied successfully"



llm = ChatOpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.environ["OPENAI_ENDPOINT"],
        model_name=os.environ["MODEL_NAME"],
        temperature=0
)

tools = [fill_timesheet, apply_vacation]

# system_message=SystemMessagePromptTemplate.from_template(
#     """You are an expert who can decide & call which of the below APIs to call based on user input.
#     Timesheet API : email, date, hours are needed
#     Vacation API: email and date are neeeded
# """
# )


# # Define a ChatPromptTemplate, including agent_scratchpad
# prompt = ChatPromptTemplate.from_messages([
#     ("system", system_message),
#     ("user", "{user_input}"),
#     ("agent_scratchpad", "{agent_scratchpad}")  # Add the agent_scratchpad to the prompt
# ])

# tools = [fill_timesheet, apply_vacation]

# agent = create_openai_functions_agent(
#     llm=llm,
#     tools=tools
# )

# agent_executor = AgentExecutor.from_agent_and_tools(
#     agent=agent,
#     tools=tools,
#     verbose=True  # helpful for debugging
# )


# from pydantic import BaseModel
# from typing import List, Dict, Optional

# # LangGraph state
# class AgentState(BaseModel):
#     messages: List[Dict[str, str]]
#     output: Optional[str] = None


def get_chat_history(state):
    redismemory = RedisMemory(session_id='langgraph_1')
    chat_history = redismemory.get_last_n()

    print("State being sent to LLM:", state)

    full_prompt = system_message + "\n Previous Chat history is" + "\n".join([f'{entry["role"]}: {entry["content"]}' for entry in chat_history])

    
    return {
        **state,
        "input": full_prompt,
        "chat_history": chat_history
    }

# Agent logic

# def api_agent_node(state: MessagesState):
def api_agent_node(state):
    config = {"configurable": {"thread_id": "test-thread"}}
    # history = state.get("history", [])
    user_input = state['user_input']

    redismemory = RedisMemory(session_id='langgraph_1')
    chat_history = redismemory.get_last_n()

    
    # final_string = ""
    # for each_chat in chat_history:
    #     each_chat

    ap_agent = create_react_agent(
        llm, tools,
        prompt="You are API assistant which can help in calling the APIs and respond back to the user in case anything needed."
    )

    import ipdb; ipdb.set_trace();
    result = ap_agent.invoke(
        {"messages": [{"role": "user", "content": user_input}]},
        config
    )
    # invoke(messages)

    print("API calling-------")
    print(result)
    print("API calling END------")
    
    full_messages = result["messages"]
    ai_response = full_messages[-1].content


    return {
            "user_input": user_input,
            "ag_output": ai_response
        }









