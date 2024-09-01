import logging
from datetime import datetime

from .crew import MedicalResearchCrew
from utils.job_manager import Event, jobs,jobs_lock,append_event

def kickoff_crew(job_id: str, question: str):
    logging.info(f"Crew for job {job_id} is starting")

    results = None
    try:
        medical_research_crew = MedicalResearchCrew(job_id)
        medical_research_crew.setup_crew(question)#Generate Crew of Agents
        results = medical_research_crew.kickoff()
        logging.info(f"Crew for job {job_id} is complete")

    except Exception as e:
        logging.error(f"Error in kickoff_crew for job {job_id}: {e}")
        append_event(job_id, f"An error occurred: {e}")
        with jobs_lock:
            jobs[job_id].status = 'ERROR'
            jobs[job_id].result = str(e)

    with jobs_lock:
        jobs[job_id].status = 'COMPLETE'
        jobs[job_id].result = results
        append_event(job_id, f"Crew complete")
