import os
import subprocess
import shutil
from datetime import datetime
import socket

# --- Path Logic ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
BASIC_TESTS_DIR = os.path.join(ROOT_DIR, "basic_tests")
RESULTS_OUT_DIR = os.path.join(ROOT_DIR, "results")

# Base hostname (e.g., 'frozone2')
SHORT_HOSTNAME = socket.gethostname().split('.')[0]
APP_NAME = "spatter"

def run_cmd(cmd, workspace_path):
    """Utility to run shell commands with consistent workspace targeting."""
    # 1. Clean the environment of existing workspace activations
    # This prevents the "invalid ramble workspace" error
    env = os.environ.copy()
    env.pop('RAMBLE_WORKSPACE', None)
    
    # 2. Construct the full command
    # We use -D {path} and ensure the workspace path is quoted
    if "workspace create"  in cmd:
        full_cmd = f"ramble {cmd}"
    else:
        full_cmd = f"ramble -D \"{workspace_path}\" {cmd}"
    
    print(f"Executing: {full_cmd}")
    
    # Use the cleaned environment (env) for the subprocess
    subprocess.run(full_cmd, shell=True, check=True, env=env)

def setup_spatter_matrix():
    # Detect Hardware
    try:
        allocated_threads = os.cpu_count()
    except AttributeError:
        allocated_threads = os.cpu_count() or 1
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Dynamic Workspace Naming
    workspace_name = f"{SHORT_HOSTNAME}_t{allocated_threads}_{timestamp}"
    # Create the ramble workspace from whereever this script is run!
    workspace_path = os.path.join(os.getcwd(), workspace_name)

    

    print(f"--- Node: {SHORT_HOSTNAME} | Allocation: {allocated_threads} ---")

    # 3. Create Workspace properly
    if os.path.exists(workspace_path):
        shutil.rmtree(workspace_path)
    
    # IMPORTANT: Use the full path in the -d flag during creation
    run_cmd(f"workspace create -d \"{workspace_path}\"", workspace_path)

    # 4. Define Workloads
    workloads = {"UStride": "cpu-ustride.json", "Stream": "cpu-stream.json"}

    print("--- Mapping Experiment Matrix ---")
    for wl_name, wl_file in workloads.items():
        json_path = os.path.join(workspace_path, 'inputs/spatter/spatter', wl_file)
        # print(json_path)
        # if not os.path.exists(json_path): continue
        exp_name = f"{wl_name}_t{allocated_threads}"
        # Ensure we are calling the correct ramble command structure
        run_cmd(f"workspace manage experiments {APP_NAME} --overwrite "
                        f"-e {exp_name} "
                        f"-v f={json_path} "
                        f"-v t={allocated_threads} " # Use 't' as defined in your class for threads
                        f"-v r=100 "
                        f"-v a=1", 
                        workspace_path)

    # 5. Remaining configs and execution...
    configs = [
        "variants:package_manager:spack",
        "software:packages:spatter:pkg_spec:'spatter@develop backend=openmp'",
        "software:environments:spatter:packages:[spatter]"
    ]

    for cfg in configs:
        run_cmd(f"config add \"{cfg}\"", workspace_path)

    run_cmd("workspace setup", workspace_path)
    run_cmd("workspace concretize", workspace_path)
    run_cmd("on", workspace_path)
    run_cmd("workspace analyze --format json", workspace_path)

    # Create results directory if it doesn't exist
    os.makedirs(RESULTS_OUT_DIR, exist_ok=True)

    source_results_txt = os.path.join(workspace_path, "results.latest.txt")
    source_results_json = os.path.join(workspace_path, "results.latest.json")
    target_results_txt = os.path.join(RESULTS_OUT_DIR, f"{workspace_name}.txt")
    target_results_json = os.path.join(RESULTS_OUT_DIR, f"{workspace_name}.json")

    if os.path.exists(source_results_txt):
        shutil.copy2(source_results_txt, target_results_txt)
    if os.path.exists(source_results_json):
        shutil.copy2(source_results_json, target_results_json)



if __name__ == "__main__":
    setup_spatter_matrix()
