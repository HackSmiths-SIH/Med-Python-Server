# Standard library imports
import json
from threading import Thread
from crewai.crews.crew_output import CrewOutput
from uuid import uuid4
# Related third-party imports
from flask import Flask, jsonify, request, abort
from flask_cors import CORS
from dotenv import load_dotenv
# Local application/library specific imports
from utils.job_manager import  jobs, jobs_lock
from Controller.crew_control.kickoff_crew import kickoff_crew
import logging
load_dotenv()

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

@app.route('/api/crew', methods=['POST'])
def run_crew():
    logging.info("Received request to run crew")
    # Validation
    data = request.json
    if not data or 'question' not in data:
        abort(400, description="Invalid input data provided.")

    job_id = str(uuid4())
    question = data['question']

    thread = Thread(target=kickoff_crew, args=(job_id, question))
    thread.start()

    return jsonify({"job_id": job_id}), 202

#Returns Status/Results of each subprocess (Job)
@app.route('/api/crew/<job_id>', methods=['GET'])
def get_status(job_id):
    with jobs_lock:
        job = jobs.get(job_id)
        if job is None:
            abort(404, description="Job not found")

        if isinstance(job.result, CrewOutput):
            # Use json_dict if available
            if job.result.json_dict:
                result_json = job.result.json_dict
            # Use json string if json_dict is not available
            elif job.result.json:
                result_json = json.loads(job.result.json)
            else:
                result_json = {"error": "No JSON result available"}
        else:
            try:
                result_json = json.loads(job.result)
            except (TypeError, json.JSONDecodeError):
                result_json = job.result

    return jsonify({
        "job_id": job_id,
        "status": job.status,
        "result": result_json,
        "events": [{"timestamp": event.timestamp.isoformat(), "data": event.data} for event in job.events]
    })

if __name__ == '__main__':
    app.run(debug=True, port=3001)
