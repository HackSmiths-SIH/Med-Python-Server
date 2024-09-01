
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
import os
load_dotenv()

def gemini():
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        verbose=True,
        temperature=0.6,
        max_tokens=200,
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )

from langchain_openai import OpenAI
def openai():
    return OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-3.5-turbo",  
        verbose=True,
        temperature=0.6,
        max_tokens=200,
    )