import glob
import os
import sys
import numpy as np

from paraview.simple import * # type: ignore
paraview.simple._DisableFirstRenderCameraReset()

# Run this script with: export PYTHONPATH="/usr/lib/python3/dist-packages:$PYTHONPATH"
# Then execute: LIBGL_ALWAYS_SOFTWARE=1 pvbatch --force-offscreen-rendering post_process.py run_alpha_0.foam images

# Camera view configuration
FOCAL_POINT = [1.4, 0.0, 0.0]
VIEW_SIZE_HIRES = [3840, 2160]
VIEW_SIZE_STANDARD = [3840, 2160]

VIEWS_3D = {
    "diagonal": {"position": [-5.13, -4.81, 3.09], "view_up": [0.24, 0.27, 0.93]},
    "bottom":    {"position": [1.4, 0.0, -8.56],  "view_up": [0, 1, 0]},
    "top":      {"position": [1.4, 0.0, 10.36],   "view_up": [0, 1, 0]}
}

VIEW_2D_SLICE = {
    "position": [2.5, -12.0, 0.0],
    "focal_point": [2.5, 0.0, 0.0],
    "view_up": [0, 0, 1],
    "slice-origin": [0.0, 0.01, 0.0],
    "slice-normal": [0, 1, 0]
}

# Validate inputs
if len(sys.argv) < 3:
    print("Usage: pvbatch script_name.py <path_to_your_openfoam_file> <folder_to_save_images>")
    sys.exit(1)

# Parse inputs
input_filepath = sys.argv[1]
job_directory = sys.argv[2]
base_case_dir = os.path.dirname(input_filepath)

def get_latest_time(reader):
    reader.UpdatePipelineInformation()
    times = reader.TimestepValues
    return times[-1] if len(times) > 0 else 0.0

def save_all_views(renderView, prefix):
    for view_name, cam in VIEWS_3D.items():
        renderView.CameraPosition = cam["position"]
        renderView.CameraFocalPoint = FOCAL_POINT
        renderView.CameraViewUp = cam["view_up"]
        Render()
        SaveScreenshot(f"{job_directory}/{prefix}-{view_name}.png", renderView, ImageResolution=renderView.ViewSize)

def geometry():
    reader = OpenDataFile(input_filepath)
    latest_time = get_latest_time(reader)
    renderView = CreateView("RenderView")
    renderView.ViewSize = VIEW_SIZE_HIRES
    renderView.ViewTime = latest_time
    reader.MeshRegions = ["patch/uav"]
    reader.UpdatePipeline(latest_time)
    display = Show(reader, renderView)
    ColorBy(display, ("CELLS", ""))
    save_all_views(renderView, "geometry")
    ResetSession()
    
def mesh():
    reader1 = OpenDataFile(input_filepath)
    reader2 = OpenDataFile(input_filepath)
    latest_time = get_latest_time(reader1)
    renderView = CreateView("RenderView")
    renderView.ViewSize = VIEW_SIZE_STANDARD
    renderView.ViewTime = latest_time
    renderView.CameraPosition = VIEW_2D_SLICE["position"]
    renderView.CameraFocalPoint = VIEW_2D_SLICE["focal_point"]
    renderView.CameraViewUp = VIEW_2D_SLICE["view_up"]
    renderView.OrientationAxesVisibility = 0
    slice = Slice(Input=reader1)
    slice.SliceType = "Plane"
    slice.SliceType.Origin = [0.4, 0.6, 0.25]
    slice.SliceType.Normal = [0, 1, 0]
    slice.UpdatePipeline(latest_time)
    display1 = Show(slice, renderView)
    display1.Representation = "Surface With Edges"
    ColorBy(display1, ("CELLS", ""))
    reader2.MeshRegions = ["patch/uav"]
    reader2.UpdatePipeline(latest_time)
    display2 = Show(reader2, renderView)
    display2.Representation = "Surface With Edges"
    ColorBy(display2, ("CELLS", ""))
    Render()
    SaveScreenshot(f"{job_directory}/mesh.png", renderView, ImageResolution=renderView.ViewSize)
    ResetSession()
    
def cp_countour():
    reader = OpenDataFile(input_filepath)
    latest_time = get_latest_time(reader)
    reader.MeshRegions = ["patch/uav"]
    reader.UpdatePipeline(latest_time)
    renderView = CreateView("RenderView")
    renderView.ViewSize = VIEW_SIZE_STANDARD
    renderView.ViewTime = latest_time
    pLUT = GetColorTransferFunction("p")
    HideScalarBarIfNotNeeded(pLUT, renderView)
    calculator1 = Calculator(registrationName="Calculator1", Input=reader)
    calculator1.ResultArrayName = "Cp"
    calculator1.Function = "(p - 0)/(0.5*1.225*100)"
    calculator1.AttributeType = "Cell Data"
    calculator1.UpdatePipeline(latest_time)
    cpLUT = GetColorTransferFunction("Cp")
    cpPWF = GetOpacityTransferFunction("Cp")
    display1 = Show(calculator1, renderView)
    display1.RescaleTransferFunctionToDataRange(True, False)
    display1.SetScalarBarVisibility(renderView, True)
    ColorBy(display1, ("CELLS", "Cp"))
    save_all_views(renderView, "cp-contour")
    ResetSession()
    
def pressure_slice():
    reader = OpenDataFile(input_filepath)
    latest_time = get_latest_time(reader)
    renderView = CreateView("RenderView")
    renderView.ViewSize = VIEW_SIZE_STANDARD
    renderView.ViewTime = latest_time
    renderView.CameraPosition = VIEW_2D_SLICE["position"]
    renderView.CameraFocalPoint = VIEW_2D_SLICE["focal_point"]
    renderView.CameraViewUp = VIEW_2D_SLICE["view_up"]
    slice = Slice(Input=reader)
    slice.SliceType = "Plane"
    slice.SliceType.Origin = VIEW_2D_SLICE["slice-origin"]
    slice.SliceType.Normal = VIEW_2D_SLICE["slice-normal"]
    slice.UpdatePipeline(latest_time)
    pLUT = GetColorTransferFunction("p")
    HideScalarBarIfNotNeeded(pLUT, renderView)
    display1 = Show(slice, renderView)
    ColorBy(display1, ("CELLS", "p", "Magnitude"))
    display1.RescaleTransferFunctionToDataRange(True, False)
    display1.SetScalarBarVisibility(renderView, True)
    Render()
    SaveScreenshot(f"{job_directory}/slice-pressure.png", renderView, ImageResolution=renderView.ViewSize)
    ResetSession()

def velocity_slice():
    reader = OpenDataFile(input_filepath)
    latest_time = get_latest_time(reader)
    renderView = CreateView("RenderView")
    renderView.ViewSize = VIEW_SIZE_STANDARD
    renderView.ViewTime = latest_time
    renderView.CameraPosition = VIEW_2D_SLICE["position"]
    renderView.CameraFocalPoint = VIEW_2D_SLICE["focal_point"]
    renderView.CameraViewUp = VIEW_2D_SLICE["view_up"]
    slice = Slice(Input=reader)
    slice.SliceType = "Plane"
    slice.SliceType.Origin = VIEW_2D_SLICE["slice-origin"]
    slice.SliceType.Normal = VIEW_2D_SLICE["slice-normal"]
    slice.UpdatePipeline(latest_time)
    display1 = Show(slice, renderView)
    ColorBy(display1, ("CELLS", "U", "Magnitude"))
    uLUT = GetColorTransferFunction("U")
    uLUT.RescaleTransferFunction(50.0, 550.0)
    display1.SetScalarBarVisibility(renderView, True)
    Render()
    SaveScreenshot(f"{job_directory}/slice-velocity.png", renderView, ImageResolution=renderView.ViewSize)
    ResetSession()

def wall_shear():
    reader = OpenDataFile(input_filepath)
    latest_time = get_latest_time(reader)
    renderView = CreateView("RenderView")
    renderView.ViewSize = VIEW_SIZE_STANDARD
    renderView.ViewTime = latest_time
    reader.MeshRegions = ["patch/uav"]
    reader.UpdatePipeline(latest_time)
    wallShearStressLUT = GetColorTransferFunction("wallShearStress")
    HideScalarBarIfNotNeeded(wallShearStressLUT, renderView)
    display1 = Show(reader, renderView)
    UpdateScalarBarsComponentTitle(wallShearStressLUT, display1)
    ColorBy(display1, ("CELLS", "wallShearStress", "X"))
    display1.RescaleTransferFunctionToDataRange(True, False)
    display1.SetScalarBarVisibility(renderView, True)
    save_all_views(renderView, "wall-shear")
    ResetSession()

def yplus():
    reader = OpenDataFile(input_filepath)
    latest_time = get_latest_time(reader)
    renderView = CreateView("RenderView")
    renderView.ViewSize = VIEW_SIZE_STANDARD
    renderView.ViewTime = latest_time
    reader.MeshRegions = ["patch/uav"]
    reader.UpdatePipeline(latest_time)
    display1 = Show(reader, renderView)
    ColorBy(display1, ("CELLS", "yPlus"))
    yPlusLUT = GetColorTransferFunction("yPlus")
    yPlusLUT.RescaleTransferFunction(30.0, 700.0)
    display1.SetScalarBarVisibility(renderView, True)
    save_all_views(renderView, "yplus")
    ResetSession()
    
def print_and_plot_stats():
    import csv
    import json

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from metrics import (extract_metrics, format_summary, window_stats,
                         _read_dat, SYMMETRY_FACTOR, TAIL_FRACTION)

    m = extract_metrics(base_case_dir)
    print(format_summary(m))

    with open(os.path.join(base_case_dir, "metrics.json"), "w") as fh:
        json.dump(m, fh, indent=2)

    # Convergence plots
    try:
        import matplotlib.pyplot as plt

        panels = []
        tc, C = _read_dat(base_case_dir, "forceCoeffs", "coefficient.dat", 8)
        if tc is not None:
            panels += [("Cl", tc, C[:, 3] * SYMMETRY_FACTOR, m.get("Cl"),
                        "Lift Coefficient (Cl)", "#1f77b4", "convergence_Cl.png"),
                       ("Cd", tc, C[:, 0] * SYMMETRY_FACTOR, m.get("Cd"),
                        "Drag Coefficient (Cd)", "#d62728", "convergence_Cd.png")]

        tf, F = _read_dat(base_case_dir, "forces", "force.dat", 4)
        if tf is not None:
            panels += [("Fz", tf, F[:, 2] * SYMMETRY_FACTOR, m.get("lift_N"),
                        "Lift Force (N)", "#1f77b4", "convergence_lift.png"),
                       ("Fx", tf, F[:, 0] * SYMMETRY_FACTOR, m.get("drag_N"),
                        "Drag Force (N)", "#d62728", "convergence_drag.png")]

        for _, t, y, mean, ylabel, colour, fname in panels:
            t_cut = t[0] + (1.0 - TAIL_FRACTION) * (t[-1] - t[0])
            plt.figure(figsize=(8, 5))
            plt.plot(t, y, color=colour, linewidth=1.2)
            plt.axvspan(t_cut, t[-1], color="0.6", alpha=0.25,
                        label=f"averaging window (last {100*TAIL_FRACTION:.0f}%)")
            if mean is not None:
                plt.axhline(mean, color="k", linestyle="--", linewidth=1.2,
                            label=f"mean = {mean:.5g}")
            plt.xlabel("Time (s)")
            plt.ylabel(ylabel)
            plt.title(f"{ylabel} Convergence")
            plt.grid(True, linestyle="--", alpha=0.7)
            # Clip the y-range to the settled part; the startup spike is 4x the converged value and would flatten everything else.
            tail = y[t >= t_cut]
            if len(tail) > 1:
                pad = max(6.0 * (tail.max() - tail.min()), 0.02 * abs(tail.mean()))
                plt.ylim(tail.min() - pad, tail.max() + pad)
            plt.legend(fontsize=8)
            plt.savefig(f"{job_directory}/{fname}", dpi=200, bbox_inches="tight")
            plt.close()
    except Exception as e:
        print(f"Warning: Could not plot convergence histories. Error: {e}")

def plot_residuals():
    try:
        import matplotlib.pyplot as plt
        def _start_time(path):
            try:
                return float(os.path.basename(os.path.dirname(path)))
            except ValueError:
                return 0.0
 
        solver_files = sorted(
            glob.glob(f"{base_case_dir}/postProcessing/solverInfo/*/solverInfo.dat"),
            key=_start_time
        )
        if not solver_files:
            print("Warning: No solverInfo.dat files found.")
            return
 
        residuals = {}  # field_name -> ([times], [initial_residuals])
        for solver_file in solver_files:
            with open(solver_file, 'r') as f:
                lines = f.readlines()
 
            header_line = next((l for l in lines if l.startswith('# Time')), None)
            if header_line is None:
                continue
            columns = header_line.lstrip('#').split()
            # Map field name -> column index of its "_initial" residual
            field_cols = {
                name[:-len('_initial')]: idx
                for idx, name in enumerate(columns)
                if name.endswith('_initial')
            }
 
            for line in lines:
                if line.startswith('#'):
                    continue
                parts = line.split()
                if len(parts) != len(columns):
                    continue
                time = float(parts[0])
                for field, idx in field_cols.items():
                    t_list, v_list = residuals.setdefault(field, ([], []))
                    # Skip overlapping time entries from restarts
                    if t_list and time <= t_list[-1]:
                        continue
                    t_list.append(time)
                    v_list.append(float(parts[idx]))
 
        if not residuals:
            print("Warning: No residual fields found in solverInfo.dat.")
            return
 
        plt.figure(figsize=(9, 6))
        for field, (times, values) in sorted(residuals.items()):
            plt.semilogy(times, values, label=field, linewidth=1.0)
 
        plt.xlabel("Time (s)")
        plt.ylabel("Initial Residual")
        plt.title("Residual Convergence")
        plt.grid(True, which="both", linestyle="--", alpha=0.5)
        plt.legend()
        plt.savefig(f"{job_directory}/residuals.png", dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved residuals plot to {job_directory}/residuals.png")
 
    except Exception as e:
        print(f"Warning: Could not process/plot residuals. Error: {e}")
    
if __name__ == "__main__":
    input_filepath = sys.argv[1]
    job_directory = sys.argv[2]
    base_case_dir = os.path.dirname(input_filepath)
    os.makedirs(job_directory, exist_ok=True)
    
    geometry()
    mesh()
    cp_countour()
    pressure_slice()
    velocity_slice()
    wall_shear()
    yplus()
    print_and_plot_stats()
    plot_residuals()