# Standard library imports
from datetime import datetime
import json
from threading import Thread
from uuid import uuid4

# Related third-party imports
from flask import Flask, jsonify, request, abort
from flask_cors import CORS
from dotenv import load_dotenv

# Local application/library specific imports
from crew import MedicalResearchCrew
from job_manager import append_event, jobs, jobs_lock, Event
import logging


load_dotenv()

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})


def kickoff_crew(job_id:str, question: str):
    logging.info(f"Crew for job {job_id} is starting")

    results = None
    try:
        medical_research_crew = MedicalResearchCrew(job_id)
        medical_research_crew.setup_crew(
            question)
        results =medical_research_crew.kickoff()
        logging.info(f"Crew for job {job_id} is complete", results)

    except Exception as e:
        logging.error(f"Error in kickoff_crew for job {job_id}: {e}")
        append_event(job_id, f"An error occurred: {e}")
        with jobs_lock:
            jobs[job_id].status = 'ERROR'
            jobs[job_id].result = str(e)

    with jobs_lock:
        jobs[job_id].status = 'COMPLETE'
        jobs[job_id].result = results
        jobs[job_id].events.append(
            Event(timestamp=datetime.now(), data="Crew complete"))


@app.route('/api/crew', methods=['POST'])
def run_crew():
    logging.info("Received request to run crew")
    # Validation
    data = request.json
    if not data or 'question' not in data:
        abort(400, description="Invalid input data provided.")

    job_id = str(uuid4())
    question = data['question']

    thread = Thread(target=kickoff_crew, args=(
        job_id, question))
    thread.start()

    return jsonify({"job_id": job_id}), 202


@app.route('/api/crew/<job_id>', methods=['GET'])
def get_status(job_id):
    with jobs_lock:
        job = jobs.get(job_id)
        if job is None:
            abort(404, description="Job not found")

     # Parse the job.result string into a JSON object
    try:
        result_json = json.loads(job.result)
    except json.JSONDecodeError:
        # If parsing fails, set result_json to the original job.result string
        result_json = job.result

    return jsonify({
        "job_id": job_id,
        "status": job.status,
        "result": result_json,
        "events": [{"timestamp": event.timestamp.isoformat(), "data": event.data} for event in job.events]
    })


if __name__ == '__main__':
    app.run(debug=True, port=3001)