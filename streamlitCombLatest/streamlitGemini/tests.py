 #Generate code script based on the task_list, ready for activation
import os
print("Current working directory:", os.getcwd())
file_name = "genscripts.py"
file_path = "/test1/genscripts.py"
file_path = os.path.join('test1', 'genscripts.py')

print(file_path)
#clean the file
# Read the file content
with open(file_path, "r") as file:
    lines = file.readlines()

# Ensure the file has more than two lines to modify
if len(lines) > 2:
    # Remove the first and last lines
    modified_lines = lines[1:-1]
    
    # Write the modified content back to the file
    with open(file_path, "w") as file:
        file.writelines(modified_lines)
    
    print(f"First and last lines removed from {file_path}.")
else:
    print(f"{file_path} has too few lines to modify.")