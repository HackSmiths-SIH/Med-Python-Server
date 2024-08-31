# from crewai import Agent
# from dotenv import load_dotenv
# from tools import tool
# from tools import arxiv_search,pubmed_search,semantic_search,serper_search
# load_dotenv()

# from langchain_google_genai import ChatGoogleGenerativeAI
# import os

# # Initialize the LLM with proper API key
# llm = ChatGoogleGenerativeAI(
#     model="gemini-1.5-flash",
#     verbose=True,
#     temperature=0.6,
#     max_tokens=200,
#     google_api_key=os.getenv("GOOGLE_API_KEY")
# )

# # List of all tools
# all_tools = [arxiv_search,pubmed_search,semantic_search,serper_search]

# # Create a Senior Researcher agent
# researcher = Agent(
#     role="Senior Researcher",
#     goal=(
#         "Give a thorough in-depth analysis and all the latest groundbreaking research about the idea/ideas, "
#         "so that it can help the user massively in their research. If the user asks for idea suggestions, suggest "
#         "the best research ideas. If the user asks about a particular suggestion or question about a research idea, "
#         "provide an in-depth answer with relevant references and links attached so the user can refer to them."
#     ),
#     verbose=True,
#     memory=True,
#     backstory=(
#         "Driven by passion for research and curiosity, you're at the forefront of modern innovative research, "
#         "eager to explore and share knowledge that could make a huge impact on the world or solve a specific research use case."
#     ),
#     tools=all_tools,
#     llm=llm,
#     allow_delegation=True
# )

# # Create a Writer agent
# writer = Agent(
#     role="Writer",
#     goal=(
#         "Express the ideas in text given to you in a simplified manner, so that the text is not complex to comprehend. "
#         "You must include relevant references and links so that the user can expand their knowledge and read more about the topics."
#     ),
#     verbose=True,
#     memory=True,
#     backstory=(
#         "With a flair for simplifying complex topics, you craft texts that explain topics in a professional yet easy-to-comprehend way. "
#         "Ensure that relevant references and links are included so the user can further explore and understand the subject."
#     ),
#     tools=[serper_search],
#     llm=llm,
#     allow_delegation=False
# )


from typing import List
from crewai import Agent
from crewai_tools import SerperDevTool
from dotenv import load_dotenv
from tools import tool
from tools import arxiv_search, pubmed_search, semantic_search, serper_search,duck_search,tavily_search
from langchain_google_genai import ChatGoogleGenerativeAI
import os
load_dotenv()


class MedicalResearchAgents:

    def __init__(self):
        self.searchInternetTool = serper_search
        self.pubmedSearchTool = pubmed_search
        self.semanticSearchTool=semantic_search
        self.arxivSearchTool=arxiv_search
        self.duckSearchTool=duck_search
        self.tavilySearchTool=tavily_search
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            verbose=True,
            temperature=0.6,
            max_tokens=200,
            google_api_key=os.getenv("GOOGLE_API_KEY"),
        )

    def research_manager(self, question: str) -> Agent:
        return Agent(
            role="Medical Head Assistant",
            goal=f"""Generate a JSON object containing the answer to the question(s) asked by the user. JSON Object
                    should contain a "answer" section in which the answer to the {question} is there, and a "references" section
                    in which all the links through which you have retrieved the answer are mentioned.
                It is your job to make the answer better and also ensure that the JSON object is in the format as asked above.

                Important:
                - "response tone should be that of a nurse, or a friendly doctor"
                - The final JSON object must include the answer in "answer" section, and the relevant references in the "references" section for user to refer to. Ensure that it is in this format.
                - The references in the references section should be a list of named URLs
                - If the final JSON object does not has an "answer" section or a "references" section, regenerate the JSON object
                - If you can't find information for a specific question, just reply back as "Sorry, I'll not be able to assist with that".
                - Do not generate fake information. Only return the information you find. Nothing else!
                - Do not stop searching until you find the requested information for the question, ensure that the answer is professional, appropriate, well-detailed, easy to understand.
                - The answer to the information asked in the question exists so keep researching until you find the information.
                - Make sure you attach all the relevant links and references 
                - Start the references section by "Here are relevant references for further reading: "
                """,
            backstory="""As a Medical Head Assistant, you are responsible for aggregating all the searched information
                into a well formed response/answer with the relevant information and references present.""",
            llm=self.llm,
            tools=[self.searchInternetTool,self.tavilySearchTool, self.pubmedSearchTool,self.semanticSearchTool,
        self.arxivSearchTool,
        self.duckSearchTool],
            verbose=True,
            allow_delegation=True,
        )

    def medical_research_agent(self) -> Agent:
        return Agent(
            role="Medical Assistant",
            goal="""Look up for the answer to user's question and return the most appropriate answer to the questions/questions,
            make the user feel as if you are a medical assistant who is there to assist the user, help them in the best possible way with their
            queries asked related to medical terms and terminologies, or general guidelines and advices, plan of action,etc
            It is your job to return this collected 
            information in a JSON object""",
            backstory="""As a medical assistant, you are responsible for looking up for the questions/questions asked by the user
            and return the most detailed, appropriate, easy to comprehend answer or response to that question.
                
                Important:
                - Once you've found the information, immediately stop searching for additional information.
                - Only return the requested information. NOTHING ELSE!
                - Make sure you find the most appropriate and detailed, simple to understand answer to the question.
                - Do not generate fake information. Only return the information you find. Nothing else!
                - Make sure that the links and references from where you formed your response/answer are also properly mentioned.
                """,
             tools=[self.searchInternetTool, self.tavilySearchTool, self.pubmedSearchTool,self.semanticSearchTool,
        self.arxivSearchTool,
        self.duckSearchTool],
            llm=self.llm,
            verbose=True,
        )
