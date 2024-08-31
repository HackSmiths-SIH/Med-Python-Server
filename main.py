from crew import crew
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
import os
import getpass

from langchain.agents import Tool, AgentType, initialize_agent
from langchain.memory import ConversationBufferMemory
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    HarmBlockThreshold,
    HarmCategory,
)
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain.agents import AgentExecutor

from langchain.agents.format_scratchpad import format_log_to_str
from langchain.agents.output_parsers import ReActSingleInputOutputParser
from langchain.tools.render import render_text_description

from dotenv import load_dotenv
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Initialize Google Generative AI model
llm = ChatGoogleGenerativeAI(
    model="gemini-pro",
    convert_system_message_to_human=True,
    handle_parsing_errors=True,
    temperature=0.6,
    max_tokens= 200,
    safety_settings = {
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    },
)


# Create Arxiv tool

from langchain_community.utilities import ArxivAPIWrapper
from langchain_community.tools import ArxivQueryRun



arxiv_wrapper = ArxivAPIWrapper(top_k_results=1)
arxiv_tool = ArxivQueryRun(api_wrapper=arxiv_wrapper)

#arxiv tool ends

#create google scholar tool


# from langchain_community.tools.google_scholar import GoogleScholarQueryRun
# from langchain_community.utilities.google_scholar import GoogleScholarAPIWrapper

# gscholartool = GoogleScholarQueryRun(api_wrapper=GoogleScholarAPIWrapper())

# google scholar tool ends

#pubmed tool starts

from langchain_community.tools.pubmed.tool import PubmedQueryRun
pubmedtool = PubmedQueryRun()

#pubmed tool ends

#semantic scholar tool starts

from langchain_community.tools.semanticscholar.tool import SemanticScholarQueryRun

semantictool = SemanticScholarQueryRun()

#semanticscholar ends

# duck duck go tool!

from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_community.tools import DuckDuckGoSearchResults

duckduckWrapper= DuckDuckGoSearchAPIWrapper(max_results=1)
duckTool = DuckDuckGoSearchResults(api_wrapper=duckduckWrapper)

#duck duck go tool ends

# Create tools list
tools = [arxiv_tool,semantictool,duckTool,pubmedtool]




# Prompt
from langchain import hub
agent_prompt = hub.pull("mikechan/gemini")
agent_prompt.template

prompt = agent_prompt.partial(
    tools=render_text_description(tools),
    tool_names=", ".join([t.name for t in tools]),
)
llm_with_stop = llm.bind(stop=["\nObservation"])


from langchain.agents import create_openai_tools_agent

# Create agent
agent = (
    {
        "input": lambda x: x["input"],
        "agent_scratchpad": lambda x: format_log_to_str(x["intermediate_steps"]),
        "chat_history": lambda x: x["chat_history"],
    }
    | prompt
    | llm_with_stop
    | ReActSingleInputOutputParser()
)
memory = ConversationBufferMemory(memory_key="chat_history")
from langchain.agents import AgentExecutor

# Create agent executor
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True,memory=memory)

# Invoke the agent with a query

from fastapi import FastAPI,HTTPException
from fastapi.params import Body


app=FastAPI()
from pydantic import BaseModel



class Question(BaseModel):
    question:str
# Set your desired port number, or get it from an environment variable


@app.get("/")
def server_started():
    return f"Server running on port {PORT}"

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("SERVER_PORT", 8000)))

@app.post('/api/researchBot')
def return_response(text:Question):
    try:
        result=crew.kickoff(inputs={'topic':text.question})
        return {"response":result,"status":200}
    except Exception as e:
        # Handle any errors that occur during the process
        raise HTTPException(status_code=500, detail=str(e))

