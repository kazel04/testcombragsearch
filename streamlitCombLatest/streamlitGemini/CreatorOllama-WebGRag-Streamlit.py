import os
import subprocess
import streamlit as st
import re
#from vector_store import make_prompt, auto_search
import os
import gdown
from dotenv import load_dotenv

load_dotenv()

os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
ngrock_url = os.getenv("ngrock_ollama_url")

def create_operator(name, tasks,googlekey, use_gemini=False):
    """Create a new operator with specified tasks."""
    # Configure the GDrive & preprocessing pipeline
    #runs well on colab but not on local comp? see if still have issues after dockerising.. 
    url = googlekey
    output_dir = "recipe_files"

    gdown.download_folder(url, quiet=True, output=output_dir)
        
    #GDRIVE preprocessing pipeline
    from haystack.components.writers import DocumentWriter
    from haystack.components.converters import MarkdownToDocument, PyPDFToDocument, TextFileToDocument
    from haystack.components.preprocessors import DocumentSplitter, DocumentCleaner
    from haystack.components.routers import FileTypeRouter
    from haystack.components.joiners import DocumentJoiner
    from haystack import Pipeline
    from haystack.document_stores.in_memory import InMemoryDocumentStore
    from haystack_integrations.components.embedders.ollama import OllamaDocumentEmbedder
    from haystack_integrations.components.generators.google_ai import GoogleAIGeminiGenerator

    document_store = InMemoryDocumentStore()
    file_type_router = FileTypeRouter(mime_types=["text/plain", "application/pdf", "text/markdown"])
    text_file_converter = TextFileToDocument()
    markdown_converter = MarkdownToDocument()
    pdf_converter = PyPDFToDocument()
    document_joiner = DocumentJoiner()

    document_cleaner = DocumentCleaner()
    document_splitter = DocumentSplitter(split_by="word", split_length=150, split_overlap=50)

    document_embedder = OllamaDocumentEmbedder(model="mxbai-embed-large", url=ngrock_url) # This is the default model and URL
    document_writer = DocumentWriter(document_store)

    preprocessing_pipeline = Pipeline()
    preprocessing_pipeline.add_component(instance=file_type_router, name="file_type_router")
    preprocessing_pipeline.add_component(instance=text_file_converter, name="text_file_converter")
    preprocessing_pipeline.add_component(instance=markdown_converter, name="markdown_converter")
    preprocessing_pipeline.add_component(instance=pdf_converter, name="pypdf_converter")
    preprocessing_pipeline.add_component(instance=document_joiner, name="document_joiner")
    preprocessing_pipeline.add_component(instance=document_cleaner, name="document_cleaner")
    preprocessing_pipeline.add_component(instance=document_splitter, name="document_splitter")
    preprocessing_pipeline.add_component(instance=document_embedder, name="document_embedder")
    preprocessing_pipeline.add_component(instance=document_writer, name="document_writer")

    preprocessing_pipeline.connect("file_type_router.text/plain", "text_file_converter.sources")
    preprocessing_pipeline.connect("file_type_router.application/pdf", "pypdf_converter.sources")
    preprocessing_pipeline.connect("file_type_router.text/markdown", "markdown_converter.sources")
    preprocessing_pipeline.connect("text_file_converter", "document_joiner")
    preprocessing_pipeline.connect("pypdf_converter", "document_joiner")
    preprocessing_pipeline.connect("markdown_converter", "document_joiner")
    preprocessing_pipeline.connect("document_joiner", "document_cleaner")
    preprocessing_pipeline.connect("document_cleaner", "document_splitter")
    preprocessing_pipeline.connect("document_splitter", "document_embedder")
    preprocessing_pipeline.connect("document_embedder", "document_writer")

    from pathlib import Path

    preprocessing_pipeline.run({"file_type_router": {"sources": list(Path(output_dir).glob("**/*"))}})

    os.environ["SERPERDEV_API_KEY"] = "95bd031b508dfdecbc673b81d74f9bbefc82d825" #update free api key here

    from haystack.components.routers import ConditionalRouter

    main_routes = [
        {
            "condition": "{{'N0_ANSWER' in replies[0].replace('\n', '')}}",
            "output" :"{{query}}",
            "output_name": "go_web",
            "output_type": str,
        },
        {
            "condition": "{{'NO_ANSWER' not in replies[0].replace('\n', '')}}",
            "output": "{{replies[0]}}",
            "output_name": "answer",
            "output_type": str,
        },
    ]

    agent_prompt_template = """<start_of_turn>user
{% if web_documents %}
    You were asked to answer the following query given the documents retrieved from Haystack's documentation but the context was not enough.
    Here is the user question: {{ query }}
    Context:
    {% for document in documents %}
        {{document.content}}
    {% endfor %}
    {% for document in web_documents %}
    URL: {{document.meta.link}}
    TEXT: {{document.content}}
    ---
    {% endfor %}
    Answer the question based on the given context.
    Return your answer with the used links..
{% else %}
Answer the following query based on the documents retrieved

Documents:
{% for document in documents %}
  {{document.content}}
{% endfor %}

Query: {{query}}
{% endif %}

<end_of_turn>
<start_of_turn>model
"""

    from haystack import Pipeline
    from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
    from haystack.components.websearch import SerperDevWebSearch
    from haystack.components.builders import PromptBuilder
    from haystack_integrations.components.generators.ollama import OllamaGenerator

    self_reflecting_agent = Pipeline(max_runs_per_component=1) 
    self_reflecting_agent.add_component("retriever", InMemoryBM25Retriever(document_store=document_store, top_k=3))
    self_reflecting_agent.add_component("prompt_builder_for_agent", PromptBuilder(template=agent_prompt_template))
    #self_reflecting_agent.add_component("llm_for_agent", OllamaGenerator(model="deepseek-r1:14b", url = "https://accepted-briefly-stallion.ngrok-free.app"))
    if use_gemini:
        self_reflecting_agent.add_component("llm_for_agent", GoogleAIGeminiGenerator(model="gemini-1.5-flash-latest"))
    else:
        self_reflecting_agent.add_component("llm_for_agent", OllamaGenerator(model="deepseek-r1:14b", url=ngrock_url))

    self_reflecting_agent.add_component("web_search", SerperDevWebSearch())
    self_reflecting_agent.add_component("router", ConditionalRouter(main_routes))

    self_reflecting_agent.connect("retriever.documents", "prompt_builder_for_agent.documents")
    self_reflecting_agent.connect("prompt_builder_for_agent", "llm_for_agent")
    self_reflecting_agent.connect("llm_for_agent.replies", "router.replies")
    self_reflecting_agent.connect("router.go_web", "web_search.query")
    self_reflecting_agent.connect("web_search.documents", "prompt_builder_for_agent.web_documents")

    #show pipeline graph
    #self_reflecting_agent.show()


    try:
        # Create the directory
        os.mkdir("operators/"+name)
        # Define the activation script content
        script_content = """import os
import subprocess

def run_all_python_scripts():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    files = os.listdir(current_dir)
    python_files = [f for f in files if f.endswith(".py") and f != os.path.basename(__file__)]
    for python_file in python_files:
        print(f"Running: {python_file}")
        try:
            subprocess.run(["python", python_file], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error running {python_file}: {e}")
        print("-" * 40)

if __name__ == "__main__":
    run_all_python_scripts()
"""
        # Save the script content to a file
        file_name = "activationscript.py"
        file_path = os.path.join("operators/"+name, file_name)
        with open(file_path, "w") as file:
            file.write(script_content)
    except FileExistsError:
        st.warning(f"Directory '{name}' already exists.")
    except PermissionError:
        st.error(f"Permission denied: Unable to create '{name}'.")
    except Exception as e:
        st.error(f"An error occurred: {e}")

    # Generate tasks using Ollama API
    #USE THIS PROMPT!!!
    query = f"Create a detailed list of ONLY Python code scripts to fully automate each task here without any human interference, DO NOT WRITE ANYTHING THAT IS NOT CODE: {tasks}"
    #model = genai.GenerativeModel('models/gemini-1.5-pro-002', generation_config=model_config)
    result = self_reflecting_agent.run({"retriever":{"query":query}, "prompt_builder_for_agent":{"query":query}, "router":{"query":query}}, include_outputs_from={"retriever", "router", "llm_for_agent", "web_search", "prompt_builder_for_agent"})


    #response = model.generate_content(contents=prompt, tools='google_search_retrieval')

    task_list = result["router"]["answer"]

   #Generate code script based on the task_list, ready for activation
    file_name = "genscripts.py"
    #if alr have genscripts, generate genscripts 2 etc (Multiple task lists feature TBD!)
    file_path = os.path.join("operators",name, file_name)
    with open(file_path, "w") as file:
        file.write(task_list)

    #clean the file
    # Read the file content
    with open(file_path, "r") as file:
        lines = file.readlines()
    
    # Ensure the file has more than two lines to modify
    if len(lines) > 2:
        # Remove the first and last lines
        modified_lines = lines[2:-2]
        
        # Write the modified content back to the file
        with open(file_path, "w") as file:
            file.writelines(modified_lines)
        
        print(f"First and last lines removed from {file_path}.")
    else:
        print(f"{file_path} has too few lines to modify.")

#WORKS NOW!!! (must remove first and last 2 lines)

def main():
    st.set_page_config(page_icon="📈")
    st.title("Operator Creation Tool")
    st.sidebar.success("Sequential Modules")
    

    # User input for operator name and tasks
    googlekey = st.text_input("Google Drive Folder Link:")
    operator_name = st.text_input("Enter the name of the new Operator:")
    tasks_input = st.text_area("List the tasks you want this Operator to perform:")
    use_gemini = st.checkbox("Gemini (Oh no I ran out of google colab and for testing)")

    if st.button("Create Operator"):
        if operator_name and tasks_input:
            create_operator(operator_name, tasks_input, googlekey,use_gemini)

        else:
            st.error("Please provide both the operator name and tasks.")

if __name__ == "__main__":
    main()
