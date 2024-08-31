from dotenv import load_dotenv
load_dotenv()
import os
os.environ['SERPER_API_KEY']=os.getenv('SERPER_API_KEY')
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")
from crewai_tools import SerperDevTool
from crewai_tools import tool
serper_search=SerperDevTool()


#### Tool 1
# Create Tavily tool
from langchain_community.tools import TavilySearchResults
tavily_tool = TavilySearchResults(
    max_results=1,
    search_depth="advanced",
    include_answer=True,
    include_raw_content=True,
    include_images=True,
)
@tool('TavilySearch')
def tavily_search(search_query: str):
    """Search the web for information on a given topic"""
    return tavily_tool.run(search_query)
#Tavily tool ends

#### Tool 2
# Create Arxiv tool
from langchain_community.utilities import ArxivAPIWrapper
from langchain_community.tools import ArxivQueryRun
arxiv_wrapper = ArxivAPIWrapper(top_k_results=1)
arxiv_tool = ArxivQueryRun(api_wrapper=arxiv_wrapper)
@tool('ArxivSearch')
def arxiv_search(search_query: str):
    """Search the web for information on a given topic"""
    return arxiv_tool.run(search_query)
#arxiv tool ends




#### Tool 3
#create google scholar tool
# from langchain_community.tools.google_scholar import GoogleScholarQueryRun
# from langchain_community.utilities.google_scholar import GoogleScholarAPIWrapper
# gscholartool = GoogleScholarQueryRun(api_wrapper=GoogleScholarAPIWrapper())
# google scholar tool ends




#### Tool 4
# Pubmed Tavily tool
from langchain_community.tools.pubmed.tool import PubmedQueryRun
pubmedtool = PubmedQueryRun()
@tool('PubmedSearch')
def pubmed_search(search_query: str):
    """Search the web for information on a given topic"""
    return pubmedtool.run(search_query)
#pubmed tool ends


#### Tool 5
#semantic scholar tool starts
from langchain_community.tools.semanticscholar.tool import SemanticScholarQueryRun
semantictool = SemanticScholarQueryRun()
@tool('SemanticScholarSearch')
def semantic_search(search_query: str):
    """Search the web for information on a given topic"""
    return semantictool.run(search_query)
#semanticscholar ends


#### Tool 6
# duck duck go tool!
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_community.tools import DuckDuckGoSearchResults
duckduckWrapper= DuckDuckGoSearchAPIWrapper(max_results=1)
duckTool = DuckDuckGoSearchResults(api_wrapper=duckduckWrapper)
@tool('DuckDuckGoSearch')
def duck_search(search_query: str):
    """Search the web for information on a given topic"""
    return duckTool.run(search_query)
#duck duck go tool ends



# from langchain_community.tools.google_scholar import GoogleScholarQueryRun
# from langchain_community.utilities.google_scholar import GoogleScholarAPIWrapper

# os.environ["SERP_API_KEY"] = os.getenv("SERP_API_KEY")

# google_tool = GoogleScholarQueryRun(api_wrapper=GoogleScholarAPIWrapper())

# @tool('GoogleSearch')
# def google_search(search_query: str):
#     """Search the web for information on a given topic"""
#     return google_tool.run(search_query)

# Create tools list







