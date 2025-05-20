import os
from langchain.sql_database import SQLDatabase
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.chat_models import ChatOpenAI
from langchain.agents import create_openai_tools_agent, AgentExecutor
from src.redis_utils import RedisMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


# previous_history = SystemMessagePromptTemplate.from_template("""
# Previous conversation history:
# {chat_history}
# """
# )

def get_sql_query_agent() -> AgentExecutor:

    # Everest Application DB Credentials
    DB_NAME=os.environ["DB_NAME"]
    DB_USER=os.environ["DB_USER"]
    DB_PASSWORD=os.environ["DB_PASSWORD"]
    DB_HOST=os.environ["DB_HOST"]
    DB_PORT=os.environ["DB_PORT"]

    # 1. Connect to DB
    allowed_tables = [
        "namely_employee", "api_sfdeal", "api_allocation",
        "namely_domain", "namely_office", "namely_jobtier",
        "namely_jobtitle", "namely_discipline", "timesheets_ettimeentry",
        "timesheets_ettimesheet"
    ]
    db = SQLDatabase.from_uri(
        f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}",
        include_tables=allowed_tables
    )

    # db.table_info = """  # 👈 Your custom schema string here
    # ### Table: users
    # - id: Primary key
    # - name: Full name
    # ...
    # """

    # 2. Create LLM and toolkit
    llm = ChatOpenAI(
            api_key=os.environ["OPENAI_API_KEY"],
            base_url=os.environ["OPENAI_ENDPOINT"],
            model_name=os.environ["MODEL_NAME"],
            temperature=0,
            streaming=True,
            verbose=True
    )
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)

    # 3. Filter tools (optional but safe)
    # tools = [tool for tool in toolkit.get_tools() if tool.name in ["query_sql_db"]]

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system", (
                "You are an expert SQL assistant."
                "Your job is to convert user questions into accurate and syntactically correct SQL queries"
                )
            ),
            MessagesPlaceholder("chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ]
    )

    # 4. Create agent
    agent = create_openai_tools_agent(
        llm=llm,
        tools=toolkit.get_tools(),
        prompt=prompt
    )
    agent_executor = AgentExecutor(agent=agent, tools=toolkit.get_tools(), verbose=True)
    return agent_executor
