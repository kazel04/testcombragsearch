# 1. Install Ollama: Follow instructions on https://ollama.com/
# 2. Install the Ollama Python library: `pip install ollama`
# 3. Pull the llama3.2-vision model: `ollama pull llama3.2-vision`  (or `ollama pull llama3.2-vision:90b` for the larger model)
# 4. Install Pillow library for image processing: `pip install Pillow`
# 5.  Have your Symantec Endpoint logs ready (likely in CSV format).  You may need to preprocess them for easier analysis.


# Code to run setup commands (optional, requires subprocess module):
import subprocess

try:
    subprocess.run(['pip', 'install', 'ollama'], check=True)
    subprocess.run(['ollama', 'pull', 'llama3.2-vision'], check=True)
    subprocess.run(['pip', 'install', 'Pillow'], check=True)
    print("Setup complete!")
except subprocess.CalledProcessError as e:
    print(f"Error during setup: {e}")



from ollama import chat
from PIL import Image
import csv
import re


def analyze_image_with_llama(image_path, prompt):
    try:
        with open(image_path, "rb") as image_file:
            image_data = image_file.read()
            
        response = chat(
            model="llama3.2-vision",
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                    "images": [image_data],
                }
            ],
        )
        return response.message.content
    except Exception as e:
        return f"Error analyzing image: {e}"


def analyze_sep_logs(log_file_path, pattern):
    alarms = []
    try:
        with open(log_file_path, 'r', encoding='utf-8') as file:
            reader = csv.reader(file)  # Assuming CSV format, adjust if needed
            header = next(reader) #skip header row if present

            for row in reader:
                log_entry = ','.join(row) #reconstruct log entry
                if re.search(pattern, log_entry):
                    alarms.append(log_entry)
        return alarms
    except FileNotFoundError:
        return ["Log file not found."]
    except Exception as e:
        return [f"Error reading logs: {e}"]


# Example Usage:
if __name__ == "__main__":

    image_path = "path/to/your/image.jpg"  # Replace with your image path
    prompt = "What is in this image?"

    sep_log_file = "path/to/your/sep_logs.csv" # Replace with your log file path
    alarm_pattern = r"critical|error|malware|intrusion" # Customize your alarm pattern

    sep_alarms = analyze_sep_logs(sep_log_file, alarm_pattern)
    print("\nSymantec Endpoint Alarms:\n", sep_alarms)

    image_analysis = analyze_image_with_llama(image_path, prompt)
    print("Image Analysis:\n", image_analysis)

    
