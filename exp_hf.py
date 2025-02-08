import subprocess

# List of Python scripts to run
scripts = [
    "exp_math_concurrent_chn_hf_nonre.py",
    "exp_math_concurrent_chn_stp_hf_nonre.py",
    "exp_math_concurrent_eng_stp_hf_nonre.py",
]

# Start all scripts
processes = []
for script in scripts:
    process = subprocess.Popen(["python", script])
    processes.append(process)
    print(f"Started {script} with PID {process.pid}")

# Optionally, wait for all scripts to finish
for process in processes:
    process.wait()

print("All scripts have completed.")