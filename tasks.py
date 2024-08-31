# from crewai import Task
# from tools import arxiv_tool, semantictool, duckTool, pubmedtool
# from tools import tool
# from agents import researcher, writer

# # List of all tools
# all_tools = [semantictool,pubmedtool,tool,arxiv_tool, duckTool]

# # Create a research task
# research_task = Task(
#     description=(
#         "Identify the most relevant results related to the {topic}. "
#         "If the user asks for idea suggestions or a particular question, give a thorough answer. "
#         "Focus on providing a simple-to-comprehend, yet thorough overview of the research papers and their content, "
#         "which are relevant to the {topic}. Include all the references and links so that the user can see them for themselves."
#     ),
#     expected_output=(
#         "A comprehensive, thorough report on the topic asked, including all explanations, references, links, and other relevant content."
#     ),
#     agent=researcher,
# )

# # Create a writing task
# write_task = Task(
#     description=(
#         "Give a thorough in-depth analysis and all the latest groundbreaking research about the idea/ideas, "
#         "so that it can help the user massively in their research. If the user asks for idea suggestions, suggest the best research ideas. "
#         "If the user asks about a particular suggestion or question about a research idea, give an in-depth answer with relevant references and links. "
#         "Focus on the latest trends, information, and knowledge. Content should be professional, easy to understand, and positive."
#     ),
#     expected_output=(
#         "A comprehensive, thorough report on the topic asked, including all explanations, references, links, and other relevant content, "
#         "formatted as a markdown."
#     ),
#     agent=writer,
#     async_execution=False,
#     output_file='new-blog-post.md'
# )

from crewai import Task, Agent
from textwrap import dedent
from job_manager import append_event
from models import PositionInfo
import logging
# from utils.logging import logger


class MedicalResearchTasks():

    def __init__(self, job_id:str):
        self.job_id = job_id

    def append_event_callback(self, task_output):
        print(f"Appending event for {self.job_id} with output {task_output}")
        logging.info("Callback called: %s", task_output)
        append_event(self.job_id, task_output.exported_output)

    def manage_research(self, agent: Agent, question: str, tasks: list[Task]):
        return Task(
            description=dedent(f"""Based on the {question},
                use the results from the Medical Research Agent to search about the question: {question}, and combine your results 
                with the results of the Medical Research Agent, and ensure that the tone of the response is that of a friendly but professional nurse/doctor.          
                Keep the answer neither too short nor too big unless specifically asked to do so. Ensure that the JSON object has an "answer" section where the final answer
                is mentioned and a "references" section where the links are mentioned for relevant references from where the information is taken.
                 Important:
                - "response tone should be that of a nurse, or a friendly doctor"
                - The final JSON object must include the answer in "answer" section, and the relevant references in the "references" section for user to refer to. Ensure that it is in this format.
                - The references in the "references" section should be a list of named URLs
                - If the final JSON object does not has an "answer" section or a "references" section, regenerate the JSON object
                - If you can't find information for a specific question, just reply back as "Sorry, I'll not be able to assist with that".
                - Do not generate fake information. Only return the information you find. Nothing else!
                - Do not stop searching until you find the requested information for the question, ensure that the answer is professional, appropriate, well-detailed, easy to understand.
                - The answer to the information asked in the question exists so keep researching until you find the information.
                - Make sure you attach all the relevant links and references 
                - Start the references section by "Here are relevant references for further reading: "
                """),
            agent=agent,
            expected_output=dedent(
                """A json object containing an "answer" section where the final answer is mentioned and a "references" section where the links are attached for relevant references from where the information is taken"""),
            callback=self.append_event_callback,
            context=tasks,
            output_json=PositionInfo
        )

    def medical_research(self, agent: Agent, question:str):
        return Task(
            description=dedent(f"""search about the question {question}, and use sources like google, pubmed, 
                               articles and blogs, to form a response for the question. Remember to mantain a list of links and 
                               references that you are using to refer for the information, so that you can mention the references 
                               from where you have referred to form the answer/response.
                               
                Helpful Tips:
                - To find the blog articles names and URLs, perform searches on Google such like the following:
                    - "{question} blog articles"
                - To find the blog articles names and URLs, perform searches on Google such like the following:
                   - "{question}"
                - To find the relevant information for question, use sources like Blogs, websites, pubmed if necessary, arxiv if necessary, semantic scholar if necessary, duck duck go, tavily:
                   - "Can Search for the important parts about the {question}, breaking it into logical parts to form the final answer"
                               
                Important:
                - Once you've found the information, immediately stop searching for additional information.
                - Only return the requested information. NOTHING ELSE!
                - Do not generate fake information. Only return the information you find. Nothing else!
                - Do not stop searching until you find the requested information for the question.
                """),
            agent=agent,
            expected_output="""A JSON object containing the searched information for question.""",
            callback=self.append_event_callback,
            output_json=PositionInfo,
            async_execution=True #for gpt only maybe
        )