import os
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
