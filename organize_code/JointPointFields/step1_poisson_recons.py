import subprocess
import os
import time
import argparse

def run_poisson_recon(input_points, output_mesh, output_grid, exe_path="../weights/PoissonRecon/Bin/Linux/PoissonRecon", depth="7"):
    if not os.path.exists(exe_path):
        print(f"Executable not found: {exe_path}")
        return

    cmd = [
        exe_path,
        '--in', input_points,
        '--out', output_mesh,
        '--grid', output_grid,
        '--depth', depth
    ]

    try:
        start_time = time.time()
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        end_time = time.time()

        print(f"[✓] Finished in {end_time - start_time:.2f}s")
        if result.returncode != 0:
            print(f"[!] Error while running PoissonRecon:\n{result.stderr}")
        else:
            print(result.stdout)
    except Exception as e:
        print(f"[!] Failed to run PoissonRecon: {e}")

def main(obj_name):
    folder_path = os.path.join("../exp_res", obj_name, "dense_poisson_scripts")
    result_folder = os.path.join(folder_path, f"{obj_name}_results")
    os.makedirs(result_folder, exist_ok=True)

    scene_input = os.path.join("../GaussianSplattingLightning/results_here", obj_name, "scene_point_cloud.ply")
    scene_mesh = os.path.join(result_folder, "scene_mesh.ply")
    scene_grid = os.path.join(result_folder, "scene_mesh.bin")
    run_poisson_recon(scene_input, scene_mesh, scene_grid)

    obj_input = os.path.join("../GaussianSplattingLightning/results_here", obj_name, "obj_extract_pc_0.ply")
    obj_mesh = os.path.join(result_folder, "obj_mesh_0.ply")
    obj_grid = os.path.join(result_folder, "obj_mesh_0.bin")
    run_poisson_recon(obj_input, obj_mesh, obj_grid)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Poisson Reconstruction")
    parser.add_argument("obj_name", type=str, help="Name of the object (e.g., bear)")
    args = parser.parse_args()
    main(args.obj_name)
