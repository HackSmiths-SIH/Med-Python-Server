import os
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
from langchain_core.documents import Document
from langchain_community.document_loaders import S3FileLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
load_dotenv()
# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
bucket_name = os.getenv('BUCKET_NAME')

async def fetch_all(file_keys):
    logger.info('Starting fetch_all')
    logger.info(f"Starting to fetch {len(file_keys)} Files")
    documents = []
    async for doc in load_documents(file_keys):
        if isinstance(doc, str):
            yield doc  # This is a progress update
        else:
            documents.extend(doc)  # This is the actual document list
    yield f"Finished fetching {len(documents)} Files. Starting chunking."
    async for update in chunk_documents(documents):
        yield update
    logger.info('Finished fetch_all')
    yield documents  # Yield the final document list

async def load_documents(file_keys):
    documents = []
    total_files = len(file_keys)
    with ThreadPoolExecutor() as executor:
        tasks = []
        loop = asyncio.get_event_loop()
        for i, key in enumerate(file_keys):
            task = loop.run_in_executor(executor, customS3LoaderFetch, key)
            tasks.append(task)
        
        # Run tasks concurrently
        for i, future in enumerate(asyncio.as_completed(tasks)):
            document_list = await future
            documents.extend(document_list)
            if (i + 1) % 10 == 0 or i + 1 == total_files:
                 logger.info(f"Fetched document {i + 1}/{total_files}")
    
    yield documents  # Yield the final list of documents

def customS3LoaderFetch(key):
    logger.info(f'Custom Loading document with key: {key}')
    
    loader = S3FileLoader(
        bucket=bucket_name, 
        key=key,
        aws_access_key_id=access_key_id,
        aws_secret_access_key=secret_access_key,
        mode="paged"
    )
    
    documents = list(loader.lazy_load())
    for doc in documents:
        simplified_metadata = {
            'page_number': doc.metadata.get('page_number', None),
            'filename': key
        }
        doc.metadata = simplified_metadata
    
    logger.info(f'Finished loading document with key: {key}')
    return documents

async def chunk_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2500,
        chunk_overlap=300
    )
    
    chunked_documents = []
    total_docs = len(documents)
    for i, doc in enumerate(documents):
        chunks = text_splitter.split_text(doc.page_content)
        for j, chunk in enumerate(chunks):
            new_doc = Document(
                page_content=chunk,
                metadata={
                    **doc.metadata,
                    "chunk": j,
                    "chunk_size": len(chunk),
                }
            )
            print(f"New chunk metadata: {new_doc.metadata}")
            chunked_documents.append(new_doc)
        if (i + 1) % 10 == 0 or i + 1 == total_docs:
            yield f"Chunked {i + 1}/{total_docs} chunks"
    
    yield f"Finished chunking. Total chunks: {len(chunked_documents)}"
    yield chunked_documents  # Yield the final list of chunked documents