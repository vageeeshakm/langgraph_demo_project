import os
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from src.redis_utils import RedisMemory


llm = ChatOpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.environ["OPENAI_ENDPOINT"],
        model_name=os.environ["MODEL_NAME"],
        temperature=0,
        streaming=True,
        verbose=True
)

router_prompt = ChatPromptTemplate.from_messages([
    ("system", """
     You are a routing agent. Decide if the query is for SQL or for an external API.
     """
    ),
    ("human", "User input: {user_input}\n\nReply with either 'sql' or 'api'.")
])

def router_agent_node(state):

    user_input = state['user_input']

    # redismemory = RedisMemory(session_id=state.session_id)
    # chat_history = redismemory.get_last_n()
    # if not chat_history:
    #     chat_history = []

    router_prompts = router_prompt.format_messages(
        user_input=user_input
        # chat_history=chat_history
    )

    print(router_prompts)

    response = llm(router_prompts)

    print("response is ========>>>>>>>>")
    print(response)
    print("response end ========>>>>>>>>")

    route = response.content.strip().lower()

    if route not in ["sql", "api"]:
        route = "sql"  # default fallback

    return {
        "user_input": user_input,
        "route": route
    }
    # "history": state.history + [HumanMessage(content=user_input), response]
