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

# Wide views, used for the Mach isosurfaces which reach well beyond the aircraft
VIEWS_3D = {
    "diagonal": {"position": [-5.13, -4.81, 3.09], "view_up": [0.24, 0.27, 0.93]},
    "bottom":    {"position": [1.4, 0.0, -8.56],  "view_up": [0, 1, 0]},
    "top":      {"position": [1.4, 0.0, 10.36],   "view_up": [0, 1, 0]}
}

# Close views of the aircraft itself (fuselage x -0.25 to 2.05, semi-span 0.8)
FOCAL_POINT_CLOSE = [0.95, 0.0, 0.0]
VIEWS_CLOSE = {
    "diagonal": {"position": [-1.45, -2.25, 1.45], "view_up": [0.24, 0.27, 0.93]},
    "bottom":   {"position": [0.95, 0.0, -4.0],    "view_up": [0, 1, 0]},
    "top":      {"position": [0.95, 0.0, 4.0],     "view_up": [0, 1, 0]}
}

# Wide side view for the mesh image, shows the refinement regions around the aircraft
VIEW_MESH = {
    "position": [2.5, -12.0, 0.0],
    "focal_point": [2.5, 0.0, 0.0],
    "view_up": [0, 0, 1],
}

VIEW_2D_SLICE = {
    "position": [1.0, -5.0, 0.0],
    "focal_point": [1.0, 0.0, 0.0],
    "view_up": [0, 0, 1],
    "slice-origin": [0.0, 0.01, 0.0],
    "slice-normal": [0, 1, 0]
}

# Colour bar filling the right-hand side of the image
SCALAR_BAR = {"position": [0.90, 0.05], "length": 0.90, "thickness": 64,
              "title_size": 56, "label_size": 44, "n_labels": 6}

# Freestream for Cp: fixed Mach 1.5 at 15,000 m ISA
MACH_INF = 1.5
P_INF = 12111.8                              # [Pa] ISA static pressure at 15 km
Q_INF = 0.5 * 1.4 * P_INF * MACH_INF**2      # [Pa] dynamic pressure, 0.5*gamma*p*M^2
CP_RANGE = [-0.4, 1.0]

# Cross-flow Mach slices (x-normal)
MACH_STATIONS = {
    "forebody": 0.4,   # inboard wing, ahead of the break (break LE at x = 0.82)
    "kink":     1.0,   # just behind the leading-edge break
    "midwing":  1.45,  # outer wing, between tip LE (1.38) and tip TE (1.54)
    "wake":     2.5,   # 0.45 m behind the tail (x = 2.05)
}
MACH_SLICE_SCALE = 0.75    # parallel-projection half height [m]
MACH_RANGE = [1.25, 1.75]

# Isosurface values
MACH_ISO = [1.45, 1.55]   # shock compression / expansion surfaces
MACH_BOX = {"position": [-0.5, -3.0, -2.0], "length": [7.5, 6.0, 4.0]}
VORT_ISO = [100.0, 1000.0, 10000.0]   # Vorticity magnitude isovalues [1/s], nested shells
# Box around the UAV for the vorticity gradient
VORT_BOX = {"position": [-0.5, 0.0, -0.5], "length": [3.5, 1.2, 1.0]}

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

def reflect(source, time):
    source.UpdatePipeline(time)
    if source.GetDataInformation().GetBounds()[2] < -1e-6:
        return source
    reflected = Reflect(Input=source)
    reflected.Plane = "Y"
    reflected.Center = 0.0
    reflected.CopyInput = 1
    reflected.UpdatePipeline(time)
    return reflected

def show_scalar_bar(display, renderView, array_name, title=None):
    # Call after the LUT has been rescaled, the labels are spread over its final range
    display.SetScalarBarVisibility(renderView, True)
    lut = GetColorTransferFunction(array_name)
    bar = GetScalarBar(lut, renderView)
    bar.WindowLocation = "Any Location"
    bar.Orientation = "Vertical"
    bar.Position = SCALAR_BAR["position"]
    bar.ScalarBarLength = SCALAR_BAR["length"]
    bar.ScalarBarThickness = SCALAR_BAR["thickness"]
    bar.TitleFontSize = SCALAR_BAR["title_size"]
    bar.LabelFontSize = SCALAR_BAR["label_size"]
    bar.TitleBold = 1
    # A fixed number of evenly spaced labels, the automatic ones crowd a full-height bar
    lo, hi = lut.RGBPoints[0], lut.RGBPoints[-4]
    n = SCALAR_BAR["n_labels"]
    bar.UseCustomLabels = 1
    bar.CustomLabels = [lo + (hi - lo) * i / (n - 1) for i in range(1, n - 1)]
    bar.AutomaticLabelFormat = 1
    bar.RangeLabelFormat = "%-#.3g"
    if title:
        bar.Title = title
        bar.ComponentTitle = ""
    return bar

def save_all_views(renderView, prefix, views=VIEWS_CLOSE, focal_point=FOCAL_POINT_CLOSE):
    for view_name, cam in views.items():
        renderView.CameraPosition = cam["position"]
        renderView.CameraFocalPoint = focal_point
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
    display = Show(reflect(reader, latest_time), renderView)
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
    renderView.CameraPosition = VIEW_MESH["position"]
    renderView.CameraFocalPoint = VIEW_MESH["focal_point"]
    renderView.CameraViewUp = VIEW_MESH["view_up"]
    renderView.OrientationAxesVisibility = 0
    slice = Slice(Input=reflect(reader1, latest_time))
    slice.SliceType = "Plane"
    slice.SliceType.Origin = [0.4, 0.15, 0.0]   # through wing segment 1
    slice.SliceType.Normal = [0, 1, 0]
    slice.UpdatePipeline(latest_time)
    display1 = Show(slice, renderView)
    display1.Representation = "Surface With Edges"
    ColorBy(display1, ("CELLS", ""))
    reader2.MeshRegions = ["patch/uav"]
    reader2.UpdatePipeline(latest_time)
    display2 = Show(reflect(reader2, latest_time), renderView)
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
    calculator1 = Calculator(registrationName="Calculator1", Input=reflect(reader, latest_time))
    calculator1.ResultArrayName = "Cp"
    calculator1.Function = f"(p - {P_INF})/{Q_INF}"
    calculator1.AttributeType = "Cell Data"
    calculator1.UpdatePipeline(latest_time)
    cpLUT = GetColorTransferFunction("Cp")
    cpPWF = GetOpacityTransferFunction("Cp")
    display1 = Show(calculator1, renderView)
    # Show() auto-colours by p, so switch to Cp before rescaling and adding the bar
    ColorBy(display1, ("CELLS", "Cp"))
    HideScalarBarIfNotNeeded(pLUT, renderView)
    cpLUT.RescaleTransferFunction(*CP_RANGE)
    show_scalar_bar(display1, renderView, "Cp")
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
    slice = Slice(Input=reflect(reader, latest_time))
    slice.SliceType = "Plane"
    slice.SliceType.Origin = VIEW_2D_SLICE["slice-origin"]
    slice.SliceType.Normal = VIEW_2D_SLICE["slice-normal"]
    slice.UpdatePipeline(latest_time)
    pLUT = GetColorTransferFunction("p")
    HideScalarBarIfNotNeeded(pLUT, renderView)
    display1 = Show(slice, renderView)
    ColorBy(display1, ("CELLS", "p", "Magnitude"))
    display1.RescaleTransferFunctionToDataRange(True, False)
    show_scalar_bar(display1, renderView, "p", "p [Pa]")
    Render()
    SaveScreenshot(f"{job_directory}/slice-pressure.png", renderView, ImageResolution=renderView.ViewSize)
    ResetSession()

def mach_slices():
    reader = OpenDataFile(input_filepath)
    latest_time = get_latest_time(reader)
    reflected = reflect(reader, latest_time)
    for name, x in MACH_STATIONS.items():
        renderView = CreateView("RenderView")
        renderView.ViewSize = VIEW_SIZE_STANDARD
        renderView.ViewTime = latest_time
        # Look upstream from behind the station
        renderView.CameraPosition = [x + 10.0, 0.0, 0.0]
        renderView.CameraFocalPoint = [x, 0.0, 0.0]
        renderView.CameraViewUp = [0, 0, 1]
        renderView.CameraParallelProjection = 1
        renderView.CameraParallelScale = MACH_SLICE_SCALE
        slice = Slice(Input=reflected)
        slice.SliceType = "Plane"
        slice.SliceType.Origin = [x, 0.0, 0.0]
        slice.SliceType.Normal = [1, 0, 0]
        slice.UpdatePipeline(latest_time)
        display1 = Show(slice, renderView)
        ColorBy(display1, ("CELLS", "Ma"))
        maLUT = GetColorTransferFunction("Ma")
        maLUT.RescaleTransferFunction(*MACH_RANGE)
        show_scalar_bar(display1, renderView, "Ma", "Mach")
        Render()
        SaveScreenshot(f"{job_directory}/slice-mach-{name}.png", renderView, ImageResolution=renderView.ViewSize)
        Delete(slice)
        Delete(renderView)
    ResetSession()

def show_body(time, renderView):
    body = OpenDataFile(input_filepath)
    body.MeshRegions = ["patch/uav"]
    body.UpdatePipeline(time)
    display = Show(reflect(body, time), renderView)
    ColorBy(display, None)
    return display

def mach_isosurface():
    reader = OpenDataFile(input_filepath)
    latest_time = get_latest_time(reader)
    reader.CellArrays = ["Ma"]
    renderView = CreateView("RenderView")
    renderView.ViewSize = VIEW_SIZE_STANDARD
    renderView.ViewTime = latest_time
    box = Clip(Input=reader)
    box.ClipType = "Box"
    box.ClipType.Position = MACH_BOX["position"]
    box.ClipType.Length = MACH_BOX["length"]
    box.Crinkleclip = 1
    contour = Contour(Input=box)
    contour.ContourBy = ["POINTS", "Ma"]
    contour.Isosurfaces = MACH_ISO
    contour.ComputeScalars = 1
    contour.UpdatePipeline(latest_time)
    show_body(latest_time, renderView)
    display1 = Show(reflect(contour, latest_time), renderView)
    ColorBy(display1, ("POINTS", "Ma"))
    maLUT = GetColorTransferFunction("Ma")
    maLUT.RescaleTransferFunction(*MACH_ISO)
    show_scalar_bar(display1, renderView, "Ma", "Mach")
    display1.Opacity = 0.4
    save_all_views(renderView, "iso-mach", VIEWS_3D, FOCAL_POINT)
    ResetSession()

def vorticity_isosurface():
    reader = OpenDataFile(input_filepath)
    latest_time = get_latest_time(reader)
    reader.CellArrays = ["U"]
    renderView = CreateView("RenderView")
    renderView.ViewSize = VIEW_SIZE_STANDARD
    renderView.ViewTime = latest_time
    box = Clip(Input=reader)
    box.ClipType = "Box"
    box.ClipType.Position = VORT_BOX["position"]
    box.ClipType.Length = VORT_BOX["length"]
    box.Crinkleclip = 1
    # Cell-based gradient, the point-based one fails on degenerate snappy cells (NaN / 1e9 spikes)
    gradient = Gradient(Input=box)
    gradient.ScalarArray = ["CELLS", "U"]
    gradient.ComputeGradient = 0
    gradient.ComputeVorticity = 1
    gradient.VorticityArrayName = "Vorticity"
    vort_mag = Calculator(Input=gradient)
    vort_mag.AttributeType = "Cell Data"
    vort_mag.ResultArrayName = "VorticityMag"
    vort_mag.Function = "mag(Vorticity)"
    points = CellDatatoPointData(Input=vort_mag)
    contour = Contour(Input=points)
    contour.ContourBy = ["POINTS", "VorticityMag"]
    contour.Isosurfaces = VORT_ISO
    contour.ComputeScalars = 1
    contour.UpdatePipeline(latest_time)
    show_body(latest_time, renderView)
    display1 = Show(reflect(contour, latest_time), renderView)
    ColorBy(display1, ("POINTS", "VorticityMag"))
    vortLUT = GetColorTransferFunction("VorticityMag")
    vortLUT.RescaleTransferFunction(VORT_ISO[0], VORT_ISO[-1])
    # Spread the colour points evenly in log space, otherwise the low decades all map to one colour
    vortLUT.MapControlPointsToLogSpace()
    vortLUT.UseLogScale = 1
    bar = show_scalar_bar(display1, renderView, "VorticityMag", "Vorticity magnitude [1/s]")
    # Log-spaced labels, the default linear ones bunch at the top of a log bar
    bar.CustomLabels = [300.0, 1000.0, 3000.0]
    display1.Opacity = 0.4
    save_all_views(renderView, "iso-vorticity")
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
    display1 = Show(reflect(reader, latest_time), renderView)
    UpdateScalarBarsComponentTitle(wallShearStressLUT, display1)
    ColorBy(display1, ("CELLS", "wallShearStress", "X"))
    display1.RescaleTransferFunctionToDataRange(True, False)
    show_scalar_bar(display1, renderView, "wallShearStress", "Wall shear stress X")
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
    display1 = Show(reflect(reader, latest_time), renderView)
    ColorBy(display1, ("CELLS", "yPlus"))
    yPlusLUT = GetColorTransferFunction("yPlus")
    yPlusLUT.RescaleTransferFunction(30.0, 300.0)
    show_scalar_bar(display1, renderView, "yPlus", "y+")
    save_all_views(renderView, "yplus")
    ResetSession()
    
def print_and_plot_stats():
    import csv
    import json

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from metrics import (extract_metrics, format_summary, window_stats,
                         _read_dat, SYMMETRY_FACTOR, TAIL_FRACTION)

    m = extract_metrics(base_case_dir)

    # Keep extra values written by supersonic_run.py (reference values, timings)
    metrics_path = os.path.join(base_case_dir, "metrics.json")
    try:
        with open(metrics_path) as fh:
            prev = json.load(fh)
        m = {**prev, **m}
    except (OSError, ValueError):
        pass

    print(format_summary(m))

    with open(metrics_path, "w") as fh:
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
            # Any function object name (solverInfo, residuals, ...) writing solverInfo.dat
            glob.glob(f"{base_case_dir}/postProcessing/*/*/solverInfo.dat"),
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
    mach_slices()
    mach_isosurface()
    vorticity_isosurface()
    wall_shear()
    yplus()
    print_and_plot_stats()
    plot_residuals()