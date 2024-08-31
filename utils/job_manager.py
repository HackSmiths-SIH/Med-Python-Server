from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict
from threading import Lock
import logging

jobs_lock = Lock()
jobs: Dict[str, "Job"] = {}


@dataclass
class Event:
    timestamp: datetime
    data: str


@dataclass
class Job:
    status: str
    events: List[Event]
    result: str

#Managing a directory containing information of all the subprocesses (Job) 
def append_event(job_id: str, event_data: str):
    with jobs_lock:
        if job_id not in jobs:
            logging.info("Job %s started", job_id)
            jobs[job_id] = Job(
                status='STARTED',
                events=[],
                result='')
            logging.info("Appending event for job %s: %s", job_id, event_data)
        #always append event to a job id
        jobs[job_id].events.append(
            Event(timestamp=datetime.now(), data=event_data))