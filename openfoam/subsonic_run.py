import os
import subprocess
import shutil
import re
import csv
import glob
from PyFoam.RunDictionary.SolutionDirectory import SolutionDirectory
from PyFoam.RunDictionary.ParsedParameterFile import ParsedParameterFile
from PyFoam.Execution.BasicRunner import BasicRunner

GEOMETRY_STL = "uav.stl"

def main():
    processors_per_job = 64
    num_iterations = 1000
    
    # Parameters
    cg = 1.25          # Center of gravity x-coordinate
    u = 50.0           # Freestream velocity (MagUInf)
    c = 0.23           # Reference chord (lRef)
    S = 0.100          # Reference area (Aref)
    
    alphas = [15]
    initialize_results_csv()

    for alpha in alphas:
        job_id = f"run_alpha_{int(alpha)}"
        job_directory = f"./{job_id}"
        
        print(f"\n{'='*40}")
        print(f"Starting Simulation: {job_id} | Alpha: {alpha}°")
        print(f"{'='*40}")

        # 1. Prepare Case Directory
        print("[1/5] Preparing case from template...")
        if not prepare(job_directory, processors_per_job, cg, u, c, S, num_iterations):
            print(f"Error: Failed to prepare case for {job_id}. Skipping...")
            continue
            
        # 2. Setup Geometry
        geom_target_path = os.path.join(job_directory, f"{job_id}.stl")
        shutil.copy(GEOMETRY_STL, geom_target_path)

        # 3. Mesh Generation
        print("[2/5] Generating mesh (snappyHexMesh)...")
        try:
            if not mesh(job_directory, alpha, processors_per_job):
                print(f"Error: Meshing failed to produce polyMesh for {job_id}. Skipping...")
                continue
        except Exception as e:
            print(f"Exception during meshing: {e}")
            continue

        # 4. Solve
        print("[3/5] Solving (foamRun)...")
        try:
            if not solve(job_directory, processors_per_job, num_iterations):
                print(f"Error: Solver failed to complete for {job_id}. Skipping...")
                continue
        except Exception as e:
            print(f"Exception during solving: {e}")
            continue

        # 5. Post-Process
        # print("[4/5] Running ParaView post-processing...")
        # try:
        #     if not postprocess(job_directory, job_id):
        #         print(f"Warning: Post-processing failed or images not generated for {job_id}.")
        # except Exception as e:
        #     print(f"Exception during post-processing: {e}")

        # # 6. Clean Up
        # print("[5/5] Cleaning up heavy mesh/processor files...")
        # cleanup(job_directory)
        
        print(f"Successfully completed {job_id}!")
        
def initialize_results_csv():
    """Deletes old results.csv if it exists and initializes a fresh one with headers."""
    csv_file_path = "results.csv"
    if os.path.exists(csv_file_path):
        os.remove(csv_file_path)
        print(f"[*] Deleted existing {csv_file_path}")
    with open(csv_file_path, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([
            "Alpha", "CL", "CD", "L/D", "CM", 
            "Lift_N", "Drag_N", "Avg_yPlus", "Max_yPlus"
        ])
    print(f"[*] Initialized fresh {csv_file_path} for new data sweep.")

def prepare(job_directory, processors_per_job, cg, u, c, S, num_iterations):
    try:
        case_template = SolutionDirectory("case_template_subsonic")
        case_template.cloneCase(job_directory)
        
        # decomposParDict
        decompose_par_dict_filepath = f"{job_directory}/system/decomposeParDict"
        decompose_par_dict = ParsedParameterFile(decompose_par_dict_filepath)
        decompose_par_dict["numberOfSubdomains"] = processors_per_job
        decompose_par_dict.writeFile()
        
        # controlDict
        control_dict_filepath = f"{job_directory}/system/controlDict"
        control_dict_file = ParsedParameterFile(control_dict_filepath)
        control_dict_file["endTime"] = num_iterations
        control_dict_file["functions"]["forceCoeffsWing"]["lRef"] = c
        control_dict_file["functions"]["forceCoeffsWing"]["Aref"] = S
        control_dict_file["functions"]["forceCoeffsWing"]["MagUInf"] = u
        CofR = f"({cg} 0 0)"
        control_dict_file["functions"]["forceCoeffsWing"]["CofR"] = CofR
        control_dict_file["functions"]["forcesWing"]["CofR"] = CofR
        control_dict_file.writeFile()
        
        # initialConditions
        initial_cond_filepath = f"{job_directory}/0/include/initialConditions"
        with open(initial_cond_filepath, 'r') as file:
            file_data = file.read()
        file_data = re.sub(r'flowVelocity\s+.*;', f'flowVelocity      ({u} 0 0);', file_data)

        with open(initial_cond_filepath, 'w') as file:
            file.write(file_data)
        
        run_ok = True
    except Exception as e:
        print(f"Exception during preparation: {e}")
        run_ok = False
        
    return run_ok

def mesh(job_directory, alpha, processors_per_job):
    COMMANDS = [
        f"surfaceTransformPoints -rotate-angle '((0 1 0) {alpha})' Wing.stl {job_directory}/constant/triSurface/Wing.stl",
        f"surfaceFeatureExtract -case {job_directory}",
        f"blockMesh -case {job_directory}",
        f"decomposePar -case {job_directory}",
        f"mpirun -np {processors_per_job} snappyHexMesh -parallel -overwrite -case {job_directory}",
        f"reconstructParMesh -constant -case {job_directory}",
        f"rm -rf {job_directory}/processor*"
    ]
    
    os.makedirs(f"{job_directory}/constant/triSurface", exist_ok=True)
    for command in COMMANDS:
        print(f"  -> Executing: {command}")
        result = subprocess.run(command, shell=True, executable='/bin/bash')
        if result.returncode != 0:
            print(f"Meshing step failed on command: {command}")
            return False

    run_ok = True
    if not os.path.isdir(f"{job_directory}/constant/polyMesh"):
        run_ok = False
    return run_ok

def solve(job_directory, processors_per_job, num_iterations):
    COMMANDS = [
        f"mpirun -np {processors_per_job} redistributePar -parallel -decompose -overwrite -case {job_directory}",
        f"mpirun -np {processors_per_job} simpleFoam -parallel -case {job_directory}",
        f"mpirun -np {processors_per_job} redistributePar -parallel -reconstruct -latestTime -case {job_directory}",
    ]

    for command in COMMANDS:
        runner = BasicRunner(argv=command.split(" "))
        runner.start()
        if not runner.runOK():
            raise Exception(f"{command} failed")
        
    subprocess.run(f"rm -rf {job_directory}/processor*", shell=True)

    run_ok = True
    if not os.path.isdir(f"{job_directory}/{num_iterations}"):
        run_ok = False
    return run_ok

def postprocess(job_directory, job_id):
    os.makedirs(f"{job_directory}/images", exist_ok=True)
    command = (
        f"LIBGL_ALWAYS_SOFTWARE=1 pvbatch --force-offscreen-rendering post_process.py {job_directory}/{job_id}.foam {job_directory}/images"
    )
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    
    if result.stdout:
        print(result.stdout)
    
    if result.returncode != 0:
        print(f"ParaView failed with error:\n{result.stderr}")
        return False
        
    images = glob.glob(f"{job_directory}/images/*.png")
    return len(images) > 0

def cleanup(job_directory):
    COMMANDS = [
        f"rm -rf {job_directory}/constant/polyMesh",
        f"rm -rf {job_directory}/PyFoam*",
        f"rm -rf {job_directory}/processors*",
        "rm -rf PyFoam*"
    ]

    for command in COMMANDS:
        subprocess.run(command, shell=True, capture_output=True, text=True)

if __name__ == "__main__":
    main()