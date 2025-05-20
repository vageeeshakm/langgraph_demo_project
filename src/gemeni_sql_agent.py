# Here's how to modify the LangGraph SQL agent to work with a limited set of tables:1.  Modify the SQLDatabase InitializationInstead of giving the SQLDatabase access to all tables, specify the tables you want to include.from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_openai import ChatOpenAI

# 1. Set up the database connection with limited tables
db = SQLDatabase.from_uri(
    "sqlite:///Chinook.db",  # Or your database URI
    include_tables=["Artist", "Album", "Track"]  # Specify the tables here
)

# 2. Initialize the LLM
llm = ChatOpenAI(model="gpt-4", temperature=0)  # Or your preferred LLM

# 3. Set up the SQL Toolkit.
toolkit = SQLDatabaseToolkit(db=db, llm=llm)
sql_tools = toolkit.get_tools()


# ======================================== ======================================== Second Approach ======================================== ======================================== 
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_openai import ChatOpenAI
from typing import List, Dict, Union
from typing_extensions import TypedDict

class State(TypedDict):
    messages: List[Union[Dict, str]]
    query: str
    result: str

# 4. Define the prompt template for generating the SQL query
sql_prompt = PromptTemplate.from_template(
    """
    You are a SQL expert. Given the user question, generate a SQL query to answer that question
    based on the schema of the following tables:

    {schema}

    Question: {question}
    """
)

# 5. Define the schema manually.  Crucially, move this *outside* the function.
CUSTOM_SCHEMA = {
    "Artist": {
        "description": "Stores information about artists.",
        "columns": {
            "ArtistId": "Unique identifier for the artist (INTEGER, Primary Key)",
            "Name": "The name of the artist (TEXT)",
        },
    },
    "Album": {
        "description": "Stores information about albums.",
        "columns": {
            "AlbumId": "Unique identifier for the album (INTEGER, Primary Key)",
            "Title": "The title of the album (TEXT)",
            "ArtistId": "The ID of the artist who created the album (INTEGER, Foreign Key referencing Artist.ArtistId)",
        },
    },
    "Track": {
        "description": "Stores information about songs.",
        "columns": {
            "TrackId": "Unique identifier for the track (INTEGER, Primary Key)",
            "Name": "The name of the song (TEXT)",
            "AlbumId": "The ID of the album the song is on (INTEGER, Foreign Key referencing Album.AlbumId)",
            "MediaTypeId": "The ID of the media type (INTEGER)",
            "GenreId": "The ID of the genre (INTEGER)",
            "Composer": "The composer of the song (TEXT)",
            "Milliseconds": "The length of the song in milliseconds (INTEGER)",
            "UnitPrice": "The price of the track (NUMERIC)",
        },
    },
    "Customer": {
        "description": "Stores information about customers",
        "columns": {
            "CustomerId": "Unique identifier for the customer (INTEGER, Primary Key)",
            "FirstName": "First name of the customer (TEXT)",
            "LastName": "Last name of the customer (TEXT)",
            "Company": "Company the customer works for (TEXT)",
            "Address": "Address of the customer (TEXT)",
            "City": "City of the customer (TEXT)",
            "State": "State of the customer (TEXT)",
            "Country": "Country of the customer (TEXT)",
            "PostalCode": "Postal code of the customer (TEXT)",
            "Phone": "Phone number of the customer (TEXT)",
            "Fax": "Fax number of the customer (TEXT)",
            "Email": "Email address of the customer (TEXT)",
            "SupportRepId": "The ID of the employee who supports the customer (INTEGER)",
        },
    },
    "Invoice": {
        "description": "Stores information about invoices",
        "columns": {
            "InvoiceId": "Unique identifier for the invoice (INTEGER, Primary Key)",
            "CustomerId": "The customer id for the invoice (INTEGER)",
            "BillingAddress": "Billing address for the invoice (TEXT)",
            "BillingCity": "Billing city for the invoice (TEXT)",
            "BillingState": "Billing state for the invoice (TEXT)",
            "BillingCountry": "Billing country for the invoice (TEXT)",
            "BillingPostalCode": "Billing postal code for the invoice (TEXT)",
            "InvoiceDate": "Date the invoice was created (TEXT)",
            "Total": "Total amount of the invoice (NUMERIC)",
        },
    },
    "Employee": {
        "description": "Stores information about employees",
        "columns": {
            "EmployeeId": "Unique identifier for the employee (INTEGER, Primary Key)",
            "LastName": "Last name of the employee (TEXT)",
            "FirstName": "First name of the employee (TEXT)",
            "Title": "Title of the employee (TEXT)",
            "ReportsTo": "The employee id that the employee reports to (INTEGER)",
            "BirthDate": "Birth date of the employee (TEXT)",
            "HireDate": "Hire date of the employee (TEXT)",
            "Address": "Address of the employee (TEXT)",
            "City": "City of the employee (TEXT)",
            "State": "State of the employee (TEXT)",
            "Country": "Country of the employee (TEXT)",
            "PostalCode": "Postal code of the employee (TEXT)",
            "Phone": "Phone number of the employee (TEXT)",
            "Fax": "Fax number of the employee (TEXT)",
            "Email": "Email address of the employee (TEXT)",
        },
    },
}


def get_table_schema_with_descriptions(table_names: List[str]) -> str:
    """
    Generates a detailed schema description for the specified tables using the predefined CUSTOM_SCHEMA.
    """
    schema_string = ""
    for table_name in table_names:
        if table_name in CUSTOM_SCHEMA:
            table_description = CUSTOM_SCHEMA[table_name]["description"]
            schema_string += f"Table: {table_name} - {table_description}\n"
            schema_string += "Columns:\n"
            for column_name, column_description in CUSTOM_SCHEMA[table_name]["columns"].items():
                schema_string += f"    {column_name} ({column_description})\n"
        else:
            raise ValueError(f"Table '{table_name}' not found in CUSTOM_SCHEMA")
    return schema_string



def generate_sql(state: State):
    messages = state["messages"]
    user_question = messages[-1]["content"]

    # 5. Dynamically select schema based on user question.
    if "customer" in user_question.lower():
        relevant_tables = ["Customer", "Invoice"]
    elif "employee" in user_question.lower():
        relevant_tables = ["Employee"]
    else:
        relevant_tables = ["Artist", "Album", "Track"]  # default

    schema = get_table_schema_with_descriptions(relevant_tables)  # No db parameter needed here

    sql_query = sql_prompt.invoke({"question": user_question, "schema": schema}) | llm
    return {"messages": messages + [HumanMessage(content=user_question), AIMessage(content=sql_query.content)], "query": sql_query.content}



from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain_core.runnables import RunnablePassthrough



def execute_sql(state: State):
    messages = state["messages"]
    query = state["query"]
    execute_query_tool = QuerySQLDataBaseTool(db=db)  # Use the db object
    result = execute_query_tool.invoke({"query": query})
    return {"messages": messages + [HumanMessage(content=query), AIMessage(content=result)], "result": result}


response_prompt = PromptTemplate.from_template(
    """
    You are a helpful assistant.  Here is the query I ran:
    {query}

    Here is the result of the query:
    {result}
    Now, provide the answer to the user.
    """
)


def respond_to_user(state):
    messages = state["messages"]
    result = state["result"]
    query = state["query"]
    response = response_prompt.invoke({"result": result, "query": query}) | llm
    return {"messages": messages + [AIMessage(content=response.content)]}


from langgraph.graph import StateGraph

# 1. Set up the database connection with limited tables
db = SQLDatabase.from_uri(
    "sqlite:///Chinook.db",  # Or your database URI
    include_tables=["Artist", "Album", "Track", "Customer", "Invoice", "Employee"],  # Specify the tables here
)

# 2. Initialize the LLM
llm = ChatOpenAI(model="gpt-4", temperature=0)  # Or your preferred LLM

# 3. Build the graph
graph_builder = StateGraph(State)
graph_builder.add_node("generate_sql", generate_sql)
graph_builder.add_node("execute_sql", execute_sql)
graph_builder.add_node("respond_to_user", respond_to_user)
graph_builder.add_edge("generate_sql", "execute_sql")
graph_builder.add_edge("execute_sql", "respond_to_user")
graph_builder.set_entry_point("generate_sql")
graph = graph_builder.compile()

# 8. Run the graph
inputs = {"messages": [{"role": "user", "content": "How many artists are there?"}]}
for output in graph.stream(inputs):
    for key, value in output.items():
        print(f"[{key.upper()}]")
        print(value)
    print("-" * 40)


# ===================================Approach to decide what tables and schemas to use ==============================

from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_openai import ChatOpenAI
from typing import List, Dict, Union
from typing_extensions import TypedDict
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field

class State(TypedDict):
    messages: List[Union[Dict, str]]
    query: str
    result: str

# 4. Define the prompt template for generating the SQL query
sql_prompt = PromptTemplate.from_template(
    """
    You are a SQL expert. Given the user question, generate a SQL query to answer that question
    based on the schema of the following tables:

    {schema}

    Question: {question}
    """
)

# 5. Define the simplified schema with table relations.
TABLE_RELATIONS = {
    "Artist": ["Album"],
    "Album": ["Artist", "Track"],
    "Track": ["Album"],
    "Customer": ["Invoice"],
    "Invoice": ["Customer"],
    "Employee": []
}

# 5.1 Define a simple class for table and description.  No longer used, but kept for potential future use.
class TableDescription(BaseModel):
    table_name: str = Field(description="Name of the table")
    description: str = Field(description="Description of the table")

def get_relevant_tables(user_question: str, llm: ChatOpenAI) -> List[str]:
    """
    Use an LLM to identify relevant tables based on the user question, using only table relations.
    """
    # 5.2. Create an LLM chain to identify relevant tables
    prompt = PromptTemplate.from_template(
        """
        You are a helpful assistant that can identify the relevant tables
        from a database schema based on the user's question.
        Here are the tables and their related tables:
        {table_relations}
        
        Here is the user's question:
        {question}
        
        Return a list of the table names that are relevant to the user's question.
        """
    )
    parser = JsonOutputParser(pydantic_object=List[str])
    chain = prompt | llm | parser
    # Format table relations for the prompt
    table_relations_for_prompt = "\n".join(
        [f"{table}: {related}" for table, related in TABLE_RELATIONS.items()]
    )
    result: List[str] = chain.invoke(
        {"question": user_question, "table_relations": table_relations_for_prompt}
    )
    return result

# 5.3  Define the schema manually.  Crucially, move this *outside* the function.  Adapt this to not include FK info.
CUSTOM_SCHEMA = {
    "Artist": {
        "description": "Stores information about artists.",
        "columns": {
            "ArtistId": "Unique identifier for the artist (INTEGER, Primary Key)",
            "Name": "The name of the artist (TEXT)",
        },
    },
    "Album": {
        "description": "Stores information about albums.",
        "columns": {
            "AlbumId": "Unique identifier for the album (INTEGER, Primary Key)",
            "Title": "The title of the album (TEXT)",
            "ArtistId": "The ID of the artist who created the album (INTEGER)", # Removed FK
        },
    },
    "Track": {
        "description": "Stores information about songs.",
        "columns": {
            "TrackId": "Unique identifier for the track (INTEGER, Primary Key)",
            "Name": "The name of the song (TEXT)",
            "AlbumId": "The ID of the album the song is on (INTEGER)", # Removed FK
            "MediaTypeId": "The ID of the media type (INTEGER)",
            "GenreId": "The ID of the genre (INTEGER)",
            "Composer": "The composer of the song (TEXT)",
            "Milliseconds": "The length of the song in milliseconds (INTEGER)",
            "UnitPrice": "The price of the track (NUMERIC)",
        },
    },
    "Customer": {
        "description": "Stores information about customers",
        "columns": {
            "CustomerId": "Unique identifier for the customer (INTEGER, Primary Key)",
            "FirstName": "First name of the customer (TEXT)",
            "LastName": "Last name of the customer (TEXT)",
            "Company": "Company the customer works for (TEXT)",
            "Address": "Address of the customer (TEXT)",
            "City": "City of the customer (TEXT)",
            "State": "State of the customer (TEXT)",
            "Country": "Country of the customer (TEXT)",
            "PostalCode": "Postal code of the customer (TEXT)",
            "Phone": "Phone number of the customer (TEXT)",
            "Fax": "Fax number of the customer (TEXT)",
            "Email": "Email address of the customer (TEXT)",
            "SupportRepId": "The ID of the employee who supports the customer (INTEGER)",
        },
    },
    "Invoice": {
        "description": "Stores information about invoices",
        "columns": {
            "InvoiceId": "Unique identifier for the invoice (INTEGER, Primary Key)",
            "CustomerId": "The customer id for the invoice (INTEGER)", # Removed FK
            "BillingAddress": "Billing address for the invoice (TEXT)",
            "BillingCity": "Billing city for the invoice (TEXT)",
            "BillingState": "Billing state for the invoice (TEXT)",
            "BillingCountry": "Billing country for the invoice (TEXT)",
            "BillingPostalCode": "Billing postal code for the invoice (TEXT)",
            "InvoiceDate": "Date the invoice was created (TEXT)",
            "Total": "Total amount of the invoice (NUMERIC)",
        },
    },
    "Employee": {
        "description": "Stores information about employees",
        "columns": {
            "EmployeeId": "Unique identifier for the employee (INTEGER, Primary Key)",
            "LastName": "Last name of the employee (TEXT)",
            "FirstName": "First name of the employee (TEXT)",
            "Title": "Title of the employee (TEXT)",
            "ReportsTo": "The employee id that the employee reports to (INTEGER)", # Removed FK
            "BirthDate": "Birth date of the employee (TEXT)",
            "HireDate": "Hire date of the employee (TEXT)",
            "Address": "Address of the employee (TEXT)",
            "City": "City of the employee (TEXT)",
            "State": "State of the employee (TEXT)",
            "Country": "Country of the employee (TEXT)",
            "PostalCode": "Postal code of the employee (TEXT)",
            "Phone": "Phone number of the employee (TEXT)",
            "Fax": "Fax number of the employee (TEXT)",
            "Email": "Email address of the employee (TEXT)",
        },
    },
}


def get_table_schema_with_descriptions(table_names: List[str]) -> str:
    """
    Generates a detailed schema description for the specified tables using the predefined CUSTOM_SCHEMA.
    """
    schema_string = ""
    for table_name in table_names:
        if table_name in CUSTOM_SCHEMA:
            table_description = CUSTOM_SCHEMA[table_name]["description"]
            schema_string += f"Table: {table_name} - {table_description}\n"
            schema_string += "Columns:\n"
            for column_name, column_description in CUSTOM_SCHEMA[table_name]["columns"].items():
                schema_string += f"    {column_name} ({column_description})\n"
        else:
            raise ValueError(f"Table '{table_name}' not found in CUSTOM_SCHEMA")
    return schema_string



def generate_sql(state: State):
    messages = state["messages"]
    user_question = messages[-1]["content"]

    # 5. Dynamically select schema based on user question.
    # relevant_tables = ["Customer", "Invoice"]
    relevant_tables = get_relevant_tables(user_question, llm) # Get relevant tables using LLM

    schema = get_table_schema_with_descriptions(relevant_tables)

    sql_query = sql_prompt.invoke({"question": user_question, "schema": schema}) | llm
    return {"messages": messages + [HumanMessage(content=user_question), AIMessage(content=sql_query.content)], "query": sql_query.content}



from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain_core.runnables import RunnablePassthrough



def execute_sql(state: State):
    messages = state["messages"]
    query = state["query"]
    execute_query_tool = QuerySQLDataBaseTool(db=db)  # Use the db object
    result = execute_query_tool.invoke({"query": query})
    return {"messages": messages + [HumanMessage(content=query), AIMessage(content=result)], "result": result}


response_prompt = PromptTemplate.from_template(
    """
    You are a helpful assistant.  Here is the query I ran:
    {query}

    Here is the result of the query:
    {result}
    Now, provide the answer to the user.
    """
)


def respond_to_user(state: State):
    messages = state["messages"]
    result = state["result"]
    query = state["query"]
    response = response_prompt.invoke({"result": result, "query": query}) | llm
    return {"messages": messages + [AIMessage(content=response.content)]}


from langgraph.graph import StateGraph

# 1. Set up the database connection with limited tables
db = SQLDatabase.from_uri(
    "sqlite:///Chinook.db",  # Or your database URI
    include_tables=["Artist", "Album", "Track", "Customer", "Invoice", "Employee"],  # Specify the tables here
)

# 2. Initialize the LLM
llm = ChatOpenAI(model="gpt-4", temperature=0)  # Or your preferred LLM

# 3. Build the graph
graph_builder = StateGraph(State)
graph_builder.add_node("generate_sql", generate_sql)
graph_builder.add_node("execute_sql", execute_sql)
graph_builder.add_node("respond_to_user", respond_to_user)
graph_builder.add_edge("generate_sql", "execute_sql")
graph_builder.add_edge("execute_sql", "respond_to_user")
graph_builder.set_entry_point("generate_sql")
graph = graph_builder.compile()

# 8. Run the graph
inputs = {"messages": [{"role": "user", "content": "show users and their invoices"}]}
for output in graph.stream(inputs):
    for key, value in output.items():
        print(f"[{key.upper()}]")
        print(value)
    print("-" * 40)

