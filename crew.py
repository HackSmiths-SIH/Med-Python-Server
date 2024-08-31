from datetime import datetime
from typing import Callable
from agents import MedicalResearchAgents
from job_manager import append_event
from tasks import MedicalResearchTasks
from crewai import Crew
from crewai import Agent
from dotenv import load_dotenv
from tools import tool
from tools import arxiv_search,pubmed_search,semantic_search,serper_search
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
import os



class MedicalResearchCrew:
    def __init__(self, job_id: str):
        self.job_id = job_id
        self.crew = None
        self.llm =ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    verbose=True,
    temperature=0.6,
    max_tokens=200,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

    def setup_crew(self, question:str):
        print(f"Setting up crew for {self.job_id} with question: {question}")

        #setup agents
        agents = MedicalResearchAgents()
        tasks = MedicalResearchTasks(
            job_id=self.job_id)

        research_manager = agents.research_manager(
            question)
        medical_research_agent = agents.medical_research_agent()

        #setup tasks
        medical_research_tasks = [
            tasks.medical_research(medical_research_agent, question)
        ]

        manage_research_task = tasks.manage_research(
            research_manager, question, medical_research_tasks)

        # CREATE CREW
        self.crew = Crew(
            agents=[research_manager, medical_research_agent],
            tasks=[*medical_research_tasks, manage_research_task],
            verbose=True,
        )

    def kickoff(self):
        if not self.crew:
            append_event(self.job_id, "Crew not set up")
            return "Crew not set up"

        append_event(self.job_id, "Task Started")
        try:
            results = self.crew.kickoff()
            append_event(self.job_id, "Task Complete")
            return results
        except Exception as e:
            append_event(self.job_id, f"An error occurred: {e}")
            return str(e)