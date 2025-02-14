import os
import sys
import logging
import splunklib.client as client
import splunklib.results as results
from collections import OrderedDict
import google.generativeai as genai
from haystack import Pipeline, Document
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack_integrations.components.embedders.ollama import OllamaDocumentEmbedder
from haystack.components.retrievers import InMemoryEmbeddingRetriever
from haystack_integrations.components.generators.google_ai import GoogleAIGeminiGenerator
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack_integrations.components.generators.ollama import OllamaGenerator
from haystack.components.builders import PromptBuilder
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.document_stores.in_memory import InMemoryDocumentStore

logging.basicConfig(
    filename='chatlogs.log',
    filemode='a',
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Splunk connection settings
#HOST = "10.10.1.22"
#PORT = 8089
#USERNAME = "administrator"
#PASSWORD = "administrator"

# Set up API keys and model selection
def configure_models(use_gemini=True):
    os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
    ngrock_url = os.getenv("ngrock_ollama_url")
    
    if use_gemini:
        genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
        return genai.GenerativeModel('gemini-1.5-flash-latest')
    else:
        return None  # Placeholder for local model setup

# Fetch logs from Splunk
def get_splunk_logs(HOST, PORT, USERNAME, PASSWORD, EARLIEST_TIME, LATEST_TIME):
    service = client.connect(host=HOST, port=PORT, username=USERNAME, password=PASSWORD)

    kwargs_oneshot = {
        "earliest_time": EARLIEST_TIME,  # Accepting custom earliest time
        "latest_time": LATEST_TIME  # Accepting custom latest time
    }

    searchquery_oneshot = "search *"
    oneshotsearch_results = service.jobs.oneshot(searchquery_oneshot, **kwargs_oneshot)
    
    reader = results.ResultsReader(oneshotsearch_results)
    keys_to_keep = ["_raw", "_sourcetype", "host", "_source", "_time"]

    filtered_data = [OrderedDict((key, item[key]) for key in keys_to_keep if key in item) for item in reader]
    logs = "\n".join(f"{key}: {value}" for item in filtered_data for key, value in item.items())

    return logs

# Load logs from a file
def get_security_logs(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as log_file:
            return log_file.read()
    except FileNotFoundError:
        logging.error(f"File '{file_path}' not found.")
        return ""
    except Exception as e:
        logging.error(f"Error reading file: {e}")
        return ""

# Generate prompt for AI analysis
def make_prompt(query, logs):
    return f"""
    You are a security analyst with deep knowledge of system logs and cybersecurity.
    Carefully inspect the lines that follow.
    QUERY: '{query}'
    LOGS: '{logs}'
    ANSWER:
    """

# Perform log analysis
def analyze_logs(logs, use_gemini=True):
    ngrock_url=os.getenv("ngrock_ollama_url")

    # Define the full prompt structure
    query = "Identify any suspicious, abnormal, or malicious behavior. Summarize potential threats, if there are any threats explain what attack techniques are used based on mitre attack technique as well as the affected devices and accounts.Do not output anything related to recommendations. Just answer the analysis."
    template = """
    You are a security analyst with deep knowledge of system logs and
    cybersecurity, given the following logs, answer the query 
    LOGS: 
    {% for document in documents %}
        {{ document.content }}
    {% endfor %}

    QUERY: {{ query }}
    ANSWER:
    """
    documentstore = InMemoryDocumentStore()
    documents = [Document(content=logs)]
    document_embedder = SentenceTransformersDocumentEmbedder()
    document_embedder.warm_up()

    documents_with_embeddings = document_embedder.run(documents)["documents"]
    documentstore.write_documents(documents_with_embeddings)
    pipe = Pipeline()    

    pipe.add_component("retriever", InMemoryBM25Retriever(document_store=documentstore))

    pipe.add_component("prompt_builder", PromptBuilder(template=template))

    if use_gemini:
        pipe.add_component('model',GoogleAIGeminiGenerator(model="gemini-1.5-flash-latest"))

    else:
        if ngrock_url is None:
            raise ValueError("ngrock_url must be provided when using Ollama.")
        pipe.add_component('model',OllamaGenerator(model="deepseek-r1:14b", url=ngrock_url))
    pipe.connect("retriever", "prompt_builder.documents")
    pipe.connect("prompt_builder", "model")

    # Run the query and pass the prompt to the model
    response = pipe.run({
        "prompt_builder": {
            "query": query
        },
        "retriever": {
            "query": query
        }
    })
    # Return the processed result
    return response["model"]["replies"][0]

# Main function for execution
#def main():
#    logs = get_splunk_logs() or get_security_logs("T1003_windows_security.log")
#    result = analyze_logs(logs, use_gemini=True)
#    logging.info(f"Analysis Result: {result}")
#    print(result)

#if __name__ == "__main__":
#    main()
