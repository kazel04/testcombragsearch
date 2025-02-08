import os
import sys
import splunklib.client as client
import splunklib.results as results
from collections import OrderedDict
import ollama

# Define Splunk connection parameters
HOST = "192.168.9.129"
PORT = 8089
USERNAME = "administrator"
PASSWORD = "administrator"

# Connect to the Splunk instance
def get_splunk_logs():
    service = client.connect(
        host=HOST,
        port=PORT,
        username=USERNAME,
        password=PASSWORD)

    #searches for previous day
    kwargs_oneshot = {"earliest_time":"-500d@d"}
    #search being run
    searchquery_oneshot = "search *"
    #running and storing results
    oneshotsearch_results = service.jobs.oneshot(searchquery_oneshot, **kwargs_oneshot)
    # Get the results and display them using the ResultsReader
    reader = results.ResultsReader(oneshotsearch_results)
    keys_to_keep = ["_raw", "_sourcetype", "host", "_source", "_time"]
    filtered_data = []
    for item in reader:
        filtered_item = OrderedDict((key, item[key]) for key in keys_to_keep if key in item)
        filtered_data.append(filtered_item)

    logs = ""
    for orderedDict in filtered_data:
        for key, value in orderedDict.items():
            logs += f"{key} {value}"
    return logs

# Send combined _raw data to Ollama for analysis
def analyze_with_ollama(combined_raw_data):
    if combined_raw_data:  # Ensure that combined raw data is not empty
        try:
            # Send the combined raw data to Ollama for analysis
            response = ollama.chat(model="llama3.2", messages=[{"role": "user", "content": combined_raw_data}])
            
            # Extract and print the analysis
            analysis = response.get('text', '')
            print(f"Combined Raw Data:\n{combined_raw_data}\nAnalysis: {analysis}\n{'='*50}")
        except Exception as e:
            print(f"Error while analyzing with Ollama: {e}")

# Analyze the combined _raw data
if __name__ == "__main__":
    combined_raw_data = get_splunk_logs()
    analyze_with_ollama(combined_raw_data)