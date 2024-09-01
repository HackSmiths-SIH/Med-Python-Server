from crewai import Task, Agent
from crewai.tasks.task_output import TaskOutput
from crewai.crews.crew_output import CrewOutput
from textwrap import dedent
from utils.job_manager import append_event
from utils.models import PositionInfo
import logging


class MedicalResearchTasks:

    def __init__(self, job_id: str):
        self.job_id = job_id

    def append_event_callback(self, task_output: CrewOutput):
        print(f"Appending event for {self.job_id} with output {task_output}")
        logging.info("Callback called: %s", task_output)
    
    # Convert the CrewOutput object to a JSON string
        if isinstance(task_output, CrewOutput):
            logging.info("yes it is")
            task_output_str = task_output.json()  # Convert to JSON string
        else:
            task_output_str = task_output.raw
            append_event(self.job_id, task_output_str)
        append_event(self.job_id, task_output_str)

    def manage_research(self, agent: Agent, question: str, tasks: list[Task]):
        return Task(
             description = dedent(
            f"Based on the {question}, use the results from the Medical Research Agent to search about the question: {question}, "
            "and combine your results with the results of the Medical Research Agent, and ensure that the tone of the response is that of a friendly but professional nurse/doctor. "
            "Keep the answer neither too short nor too big unless specifically asked to do so. Ensure that the JSON object has an 'answer' section where the final answer "
            "is mentioned and a 'references' section where the links are mentioned for relevant references from where the information is taken.\n\n"
            "Example JSON Object:\n"
            """{"answer": "Mpox is a viral disease caused by the mpox virus, spread through contact with infected people or animals. It is endemic in Central and West Africa but has been reported globally, including the US and Europe. Symptoms include rash, fever, headache, muscle aches, back pain, and fatigue. The rash can be painful and spread to the face, palms, soles, and genitals. While there's no cure, treatments manage symptoms. Prevention involves avoiding contact with infected individuals and animals. If traveling to mpox-affected areas, consult a doctor about vaccination. The CDC provides more information at https://www.cdc.gov/poxvirus/mpox/index.html.", "references": [{"name": "Centers for Disease Control and Prevention (CDC)", 'url': "https://www.cdc.gov/poxvirus/mpox/index.html"}]}\n\n"""
            "Important:\n"
            "- Once you've found the information, immediately stop searching for additional information."
            "- 'Response tone should be that of a nurse, or a friendly doctor'\n"
            "- The final JSON object must include the answer in 'answer' section, and the relevant references in the 'references' section for user to refer to. Ensure that it is in this format.\n"
            "- The references in the 'references' section should be a list of named URLs\n"
            "- If the final JSON object does not have an 'answer' section or a 'references' section, regenerate the JSON object\n"
            "- If you can't find information for a specific question, just reply back as 'Sorry, I'll not be able to assist with that'.\n"
            "- Do not generate fake information. Only return the information you find. Nothing else!\n"
            "- Do not stop searching until you find the requested information for the question, ensure that the answer is professional, appropriate, well-detailed, easy to understand.\n"
            "- Make sure you attach all the relevant links and references.\n"
            "- Start the references section by 'Here are relevant references for further reading: '"
             " - Keep changing the final answer, until the JSON Object matches the format as given in above example JSON Object"
        ),
            agent=agent,
            expected_output=(
                "A JSON object containing an 'answer' section where the final answer is mentioned and a 'references' section where the links are attached for relevant references from where the information is taken"
            ),
            callback=self.append_event_callback,
            context=tasks,
            output_json=PositionInfo
        )

    def medical_research(self, agent: Agent, question: str): 
        return Task(
             description = dedent(
            f"Search about the question {question}, and use sources like Google, PubMed, articles, and blogs, to form a response for the question. "
            "Remember to maintain a list of links and references that you are using to refer for the information, so that you can mention the references "
            "from where you have referred to form the answer/response.\n\n"
            "Example JSON Object:\n"
            """{"answer": "Mpox is a viral disease caused by the mpox virus, spread through contact with infected people or animals. It is endemic in Central and West Africa but has been reported globally, including the US and Europe. Symptoms include rash, fever, headache, muscle aches, back pain, and fatigue. The rash can be painful and spread to the face, palms, soles, and genitals. While there's no cure, treatments manage symptoms. Prevention involves avoiding contact with infected individuals and animals. If traveling to mpox-affected areas, consult a doctor about vaccination. The CDC provides more information at https://www.cdc.gov/poxvirus/mpox/index.html.", "references": [{"name": "Centers for Disease Control and Prevention (CDC)", "url": "https://www.cdc.gov/poxvirus/mpox/index.html"}]}\n\n"""
            "Helpful Tips:\n"
            "- To find the blog articles names and URLs, perform searches on Google such as:\n"
            "   - '{question} blog articles'\n"
            "- To find the relevant information for the question, use sources like Blogs, websites, PubMed if necessary, Arxiv if necessary, Semantic Scholar if necessary, DuckDuckGo:\n"
            "   - 'Can Search for the important parts about the {question}, breaking it into logical parts to form the final answer'\n\n"
            "Important:\n"
            "- Once you've found the information, immediately stop searching for additional information.\n"
            "- Only return the requested information. NOTHING ELSE!\n"
            "- Do not generate fake information. Only return the information you find. Nothing else!\n"
            "- Do not stop searching until you find the requested information for the question."
            " - Keep changing the final answer, until the JSON Object matches the format as given in above example JSON Object"
        ),
            agent=agent,
            expected_output="A JSON object containing the searched information for the question.",
            callback=self.append_event_callback,
            output_json=PositionInfo,
            async_execution=True
        )
