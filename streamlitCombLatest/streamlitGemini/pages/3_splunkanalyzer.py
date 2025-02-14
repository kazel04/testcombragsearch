import streamlit as st
import datetime
from operators.Splunk_Analyzer.genscripts import analyze_logs, get_splunk_logs

def main():
    st.title("Splunk Log Analyzer")

    # Initialize session state for file upload tracking
    if "file_uploaded" not in st.session_state:
        st.session_state.file_uploaded = False

    # Log File Selection
    uploaded_file = st.file_uploader("Upload a log file (optional)", type=["log", "txt"])

    # Update session state based on file upload
    st.session_state.file_uploaded = uploaded_file is not None

    # Splunk Connection Details (Disable if file uploaded)
    HOST = st.text_input("Splunk Host", "10.10.1.22", disabled=st.session_state.file_uploaded)
    PORT = st.number_input("Splunk Port", 8089, step=1, disabled=st.session_state.file_uploaded)
    USERNAME = st.text_input("Username", "administrator", disabled=st.session_state.file_uploaded)
    PASSWORD = st.text_input("Password", "administrator", type="password", disabled=st.session_state.file_uploaded)

    # Choose AI Model
    use_gemini = st.checkbox("Gemini (Oh no I ran out of Google Colab and for testing)")

    # Date and Time Selection
    col1, col2 = st.columns(2)

    with col1:
        earliest_date = st.date_input("Earliest Date", datetime.date.today(), disabled=st.session_state.file_uploaded)
        earliest_time = st.time_input("Earliest Time", datetime.datetime.now().time(), disabled=st.session_state.file_uploaded)

    with col2:
        latest_date = st.date_input("Latest Date", datetime.date.today(), disabled=st.session_state.file_uploaded)
        latest_time = st.time_input("Latest Time", datetime.datetime.now().time(), disabled=st.session_state.file_uploaded)

    # Combine selected date and time
    earliest_datetime = datetime.datetime.combine(earliest_date, earliest_time)
    latest_datetime = datetime.datetime.combine(latest_date, latest_time)

    if st.button("Analyze Logs"):
        logs = ""
        if uploaded_file is not None:
            logs = uploaded_file.read().decode("utf-8")
        else:
            logs = get_splunk_logs(
                HOST, PORT, USERNAME, PASSWORD,
                earliest_datetime.strftime('%Y-%m-%dT%H:%M:%S'),
                latest_datetime.strftime('%Y-%m-%dT%H:%M:%S')
            )

        response = analyze_logs(logs, use_gemini)
        st.subheader("Analysis Result")
        st.write(response)

if __name__ == "__main__":
    main()
