from Controller.crew_items.agents import MedicalResearchAgents
from Controller.crew_items.tasks import MedicalResearchTasks
from crewai import Crew
from utils.config_model import gemini,openai

from utils.job_manager import append_event



class MedicalResearchCrew:
    def __init__(self, job_id: str):
        self.job_id = job_id
        self.crew = None
        self.llm = gemini()

    def setup_crew(self, question:str):
        print(f"Setting up crew for {self.job_id} with question: {question}")
        #setup agents
        agents = MedicalResearchAgents()
        research_manager = agents.research_manager(question)#return an AGENT loaded with TOOLS and PROMPT
        medical_research_agent = agents.medical_research_agent()#return an AGENT loaded with TOOLS and PROMPT

        #setup tasks
        tasks = MedicalResearchTasks(job_id=self.job_id)
        medical_research_tasks = [
            tasks.medical_research(medical_research_agent, question)
        ]

        manage_research_task = tasks.manage_research(research_manager, question, medical_research_tasks)

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