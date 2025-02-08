# 1. Install necessary libraries: pip install openpyxl pywin32 cbapi
# 2. Configure Carbon Black EDR API credentials.  Replace placeholders below.
# 3. Ensure you have the correct permissions to block processes via the Carbon Black API.


# Code to run setup commands (optional, for automated installation):
# import subprocess
# subprocess.check_call(['pip', 'install', 'openpyxl', 'pywin32', 'cbapi'])


import openpyxl
import cbapi
from cbapi.errors import ApiError
from cbapi.response import CbResponseAPI, Process, Binary, Sensor

# Carbon Black API credentials
CARBONBLACK_URL = "YOUR_CARBONBLACK_URL"
CARBONBLACK_TOKEN = "YOUR_CARBONBLACK_TOKEN"

# Excel file path
EXCEL_FILE = "path/to/your/excel/file.xlsx"

def get_hashes_from_excel(excel_file):
    """Reads hashes from an Excel file."""
    workbook = openpyxl.load_workbook(excel_file, data_only=True)
    sheet = workbook.active  # Assumes hashes are in the first sheet
    hashes = []
    for row in sheet.iter_rows(min_row=2, values_only=True): # Assumes header row at index 1
        hashes.append(row[0]) # Assumes hash is in the first column
    return hashes

def block_hash_on_carbonblack(cb, hash_value):
    """Blocks a hash on Carbon Black EDR."""
    try:
        process = cb.select(Process).where(process_hash=hash_value).first()
        if process:
            process.block()
            print(f"Blocked process with hash: {hash_value}")
        else:
            print(f"No process found with hash: {hash_value}")
    except ApiError as e:
        print(f"Error blocking hash {hash_value}: {e}")

if __name__ == "__main__":
    hashes = get_hashes_from_excel(EXCEL_FILE)
    cb = cbapi.CbApi(CARBONBLACK_TOKEN, url=CARBONBLACK_URL)
    for hash_value in hashes:
        block_hash_on_carbonblack(cb, hash_value)
