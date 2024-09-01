from typing import List
from crewai import Agent
from crewai_tools import SerperDevTool
from dotenv import load_dotenv
from utils.tools import (
    tool, 
    arxiv_search, 
    pubmed_search, 
    semantic_search, 
    serper_search,
    duck_search, 
    tavily_search
)
from utils.config_model import gemini,openai
import os

load_dotenv()

class MedicalResearchAgents:
    def __init__(self):
        # Initialize tools
        self.searchInternetTool = serper_search
        self.pubmedSearchTool = pubmed_search
        self.semanticSearchTool = semantic_search
        self.arxivSearchTool = arxiv_search
        self.duckSearchTool = duck_search
        self.tavilySearchTool = tavily_search
        # Initialize the LLM with proper API key
        self.llm = gemini()

    def research_manager(self, question: str) -> Agent:
        return Agent(
            role="Medical Head Assistant",
            goal="""Generate a JSON object containing the answer to the question(s) asked by the user. JSON Object
                    should contain a "answer" section in which the answer to the {question} is there, and a "references" section
                    in which all the links through which you have retrieved the answer are mentioned.
                It is your job to make the answer better and also ensure that the JSON object is in the format as asked above.
                Example JSON Object:
                {"answer": "Mpox is a viral disease caused by the mpox virus, spread through contact with infected people or animals. It is endemic in Central and West Africa but has been reported globally, including the US and Europe. Symptoms include rash, fever, headache, muscle aches, back pain, and fatigue. The rash can be painful and spread to the face, palms, soles, and genitals. While there's no cure, treatments manage symptoms. Prevention involves avoiding contact with infected individuals and animals. If traveling to mpox-affected areas, consult a doctor about vaccination. The CDC provides more information at https://www.cdc.gov/poxvirus/mpox/index.html.", "references": [{{"name": "Centers for Disease Control and Prevention (CDC)", "url": "https://www.cdc.gov/poxvirus/mpox/index.html"}}]}
                Important:
                - "response tone should be that of a nurse, or a friendly doctor"
                - The final JSON object must include the answer in "answer" section, and the relevant references in the "references" section for user to refer to. Ensure that it is in this format.
                - "Keep changing the final answer, until the JSON Object matches the format as given in above example JSON Object
                - The references in the references section should be a list of named URLs
                - If the final JSON object does not has an "answer" section or a "references" section, regenerate the JSON object
                - If you can't find information for a specific question, just reply back as "Sorry, I'll not be able to assist with that".
                - Do not generate fake information. Only return the information you find. Nothing else!
                - Do not stop searching until you find the requested information for the question, ensure that the answer is professional, appropriate, well-detailed, easy to understand.
                - The answer to the information asked in the question exists so keep researching until you find the information.
                - Make sure you attach all the relevant links and references 
                - Start the references section by "Here are relevant references for further reading: "
                """,
            backstory="""
            As a Medical Head Assistant, you are responsible for aggregating all the searched information
                into a well formed response/answer with the relevant information and references present.
                Important:
                - Once you've found the information, immediately stop searching for additional information.
                """,
            llm=self.llm,
            tools=[
                self.searchInternetTool, self.tavilySearchTool, self.pubmedSearchTool, 
                self.semanticSearchTool, self.arxivSearchTool, self.duckSearchTool
            ],
            verbose=True,
            memory=True,
            guardrails={
                'max_retries': 3,  # Limit the number of retries for the same action
                'alternative_strategy': 'Ask for more context or delegate',  # Suggest alternatives
            },
            allow_delegation=True
        )

    def medical_research_agent(self) -> Agent:
        return Agent(
            role="Medical Assistant",
            goal="""Look up for the answer to user's question and return the most appropriate answer to the questions/questions,
            make the user feel as if you are a medical assistant who is there to assist the user, help them in the best possible way with their
            queries asked related to medical terms and terminologies, or general guidelines and advices, plan of action,etc
            It is your job to return this collected 
            information in a JSON object
            Example JSON Object:
                {"answer": "Mpox is a viral disease caused by the mpox virus, spread through contact with infected people or animals. It is endemic in Central and West Africa but has been reported globally, including the US and Europe. Symptoms include rash, fever, headache, muscle aches, back pain, and fatigue. The rash can be painful and spread to the face, palms, soles, and genitals. While there's no cure, treatments manage symptoms. Prevention involves avoiding contact with infected individuals and animals. If traveling to mpox-affected areas, consult a doctor about vaccination. The CDC provides more information at https://www.cdc.gov/poxvirus/mpox/index.html.", "references": [{{"name": "Centers for Disease Control and Prevention (CDC)", "url": "https://www.cdc.gov/poxvirus/mpox/index.html"}}]}}
                """,
            backstory="""As a medical assistant, you are responsible for looking up for the questions/questions asked by the user
            and return the most detailed, appropriate, easy to comprehend answer or response to that question.
                
                Important:
                - Once you've found the information, immediately stop searching for additional information.
                - Only return the requested information. NOTHING ELSE!
                - Make sure you find the most appropriate and detailed, simple to understand answer to the question.
                - Do not generate fake information. Only return the information you find. Nothing else!
                - Make sure that the links and references from where you formed your response/answer are also properly mentioned.
                - "Keep changing the final answer, until the JSON Object matches the format as given in above example JSON Object
                """,
            tools=[
                self.searchInternetTool, self.tavilySearchTool, self.pubmedSearchTool, 
                self.semanticSearchTool, self.arxivSearchTool, self.duckSearchTool
            ],
            llm=self.llm,
            verbose=True,
            memory=True,
            guardrails={
                'max_retries': 3, 
                'alternative_strategy': 'Ask for more context or delegate',  
            },
        )
