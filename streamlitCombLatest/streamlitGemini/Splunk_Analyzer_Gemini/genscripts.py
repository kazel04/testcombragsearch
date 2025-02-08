import os 
import sys 

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lib"))
import splunklib.client as client
import splunklib.results as results
#from __future__ import print_function
from collections import OrderedDict
import google.generativeai as genai

HOST = "192.168.9.129"
PORT = 8089
USERNAME = "administrator" 
PASSWORD = "administrator"


# Create a Service instance and log in
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

os.environ['GOOGLE_API_KEY'] = "AIzaSyAhodJxMYK5Q8jOzfCRVfUk08wclfsAa-E"
genai.configure(api_key=os.environ['GOOGLE_API_KEY'])


def make_prompt(query, logs):
  #escaped = relevant_passage.replace("'", "").replace('"', "").replace("\n", " ")
  prompt = ("""You are a helpful security operations analyst that understands logs \
  Analyze the logs provided and answer based on what is in the logs
  QUERY: '{query}'
  LOGS: '{logs}'

    ANSWER:
  """).format(query=query,logs=logs)

  return prompt

if __name__ == "__main__":
    model = genai.GenerativeModel('gemini-1.5-flash-latest')
    logs = get_splunk_logs()
    prompt = make_prompt("Highlight potential malicious or alarming activty", logs)
    response = model.generate_content(prompt)
    print(response.candidates[0].content.parts[0].text)
