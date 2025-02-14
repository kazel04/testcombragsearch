import os
import subprocess
import streamlit as st
import google.generativeai as genai
import re
from dotenv import load_dotenv
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
def create_operator(name, tasks,googlekey):
    """Create a new operator with specified tasks."""
    # Configure the Gemini API

    genai.configure(api_key=GOOGLE_API_KEY)

    # Model Configuration, test with it to see
    model_config = {
    "temperature": 0,
    #"top_p": 0.99,
    #"top_k": 10,
    #"max_output_tokens": 4096,
    }

    try:
        # Create the directory
        os.mkdir(name)
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
        file_path = os.path.join(name, file_name)
        with open(file_path, "w") as file:
            file.write(script_content)
    except FileExistsError:
        st.warning(f"Directory '{name}' already exists.")
    except PermissionError:
        st.error(f"Permission denied: Unable to create '{name}'.")
    except Exception as e:
        st.error(f"An error occurred: {e}")

    # Generate tasks using Gemini API
    #USE THIS PROMPT!!!
    prompt = f"Create a detailed list of ONLY Python code scripts to fully automate each task here without any human interference, DO NOT WRITE ANYTHING THAT IS NOT CODE: {tasks}"
    #model = genai.GenerativeModel('models/gemini-1.5-pro-002', generation_config=model_config)

    model = genai.GenerativeModel('gemini-1.5-flash-latest', generation_config=model_config)
    response = model.generate_content(contents=prompt, tools='google_search_retrieval')
    print(response)
    task_list = response.candidates[0].content.parts[0].text

   #Generate code script based on the task_list, ready for activation
    file_name = "genscripts.py"
    #if alr have genscripts, generate genscripts 2 etc (Multiple task lists feature TBD!)
    file_path = os.path.join(name, file_name)
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
    googlekey = st.text_input("Google Gemini API Key:")
    operator_name = st.text_input("Enter the name of the new Operator:")
    tasks_input = st.text_area("List the tasks you want this Operator to perform:")

    if st.button("Create Operator"):
        if operator_name and tasks_input:
            create_operator(operator_name, tasks_input, googlekey)

        else:
            st.error("Please provide both the operator name and tasks.")

if __name__ == "__main__":
    main()
