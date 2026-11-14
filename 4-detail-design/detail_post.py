"""LIBGL_ALWAYS_SOFTWARE=1 pvbatch --force-offscreen-rendering detail_post.py <case.foam> <images_dir>"""
import json
import os
import sys

from paraview import servermanager
from paraview.simple import (OpenFOAMReader, Calculator, Slice, Clip, Reflect, Threshold, IntegrateVariables,
                             Text, CreateView, Show, ColorBy, GetColorTransferFunction, GetScalarBar, Render,
                             SaveScreenshot, Delete)

IMAGE_SIZE = [1920, 1080]
COLORMAP = "Turbo"                   # ParaView preset for the scalar scenes
GREY = [0.55, 0.57, 0.60]            # vehicle surface for context
MESH_COLOUR = [0.93, 0.93, 0.90]
Y_SLICE = 1e-4                       # centreline cut just off the y = 0 symmetry plane
LOW_P_LIMIT = 2000.0                 # [Pa] cells below this are reported (uav_review)
WALLS = ["airframe", "inlet_lip", "inlet_duct"]

UPZ = [0, 0, 1]
# camera presets: focal point, camera position, view up, parallel scale (half height, m)
SIDE = ([0.9, 0, 0.1], [0.9, -8, 0.1], UPZ, 1.3)
SIDE_FAR = ([1.5, 0, 0], [1.5, -12, 0], UPZ, 3.0)
TOP = ([0.9, 0, 0], [0.9, 0, 8], [-1, 0, 0], 1.4)
INLET = ([0.84, 0, 0.06], [0.84, -8, 0.06], UPZ, 0.17)
INLET_TOP = ([0.75, 0, 0.1034], [0.75, 0, 8], [-1, 0, 0], 0.14)
TAIL = ([1.65, 0, 0], [1.65, -8, 0], UPZ, 0.30)
NOSE = ([-0.15, 0, 0], [-0.15, -8, 0], UPZ, 0.14)
ISO_TOP = ([0.6, 0, 0.05], [-2.6, -3.4, 2.6], UPZ, 0.75)
ISO_BOTTOM = ([0.6, 0, 0.05], [-2.6, -3.4, -2.6], UPZ, 0.75)
ISO_AFT = ([0.9, 0, 0.05], [4.0, -3.2, 2.4], UPZ, 0.85)
FRONT = ([1.0345, 0, 0], [-6, 0, 0], UPZ, 0.055)
ISO_STN = ([0.9, 0, 0.08], [-1.0, -4.6, 1.5], UPZ, 0.62)
STATIONS = [0.45, 0.85, 1.30, 1.70]
NEAR_BOX = {"y": (-0.62, 0.62), "z": (-0.15, 0.35)}   # station cuts trimmed to the near field

foam, images = sys.argv[1], sys.argv[2]
case = os.path.dirname(os.path.abspath(foam))
os.makedirs(images, exist_ok=True)
with open(os.path.join(case, "metrics.json")) as fh:
    M = json.load(fh)
F = M["flow"]
G, R = F["gamma"], F["R"]
HALF = F["half_model"]


# ------------------------------------------------------------------------------------------ pipeline
def reader(regions):
    r = OpenFOAMReader(FileName=foam)
    r.MeshRegions = regions
    r.UpdatePipelineInformation()
    return r


def latest(r):
    times = r.TimestepValues
    return times[-1] if hasattr(times, "__len__") and len(times) else (times or 0.0)


def derived(src):
    # STAR-style derived fields as point data (the reader interpolates cell data to the points)
    src.UpdatePipeline(T_LATEST)
    names = set(src.PointData.keys())
    defs = [("Mach", f"mag(U)/sqrt({G}*{R}*T)"),
            ("Cp", f"(p-{F['p_inf']})/{F['q_inf']}"),
            ("PressureRatio", f"p/{F['p_inf']}"),
            ("p0", f"p*(1+{0.5 * (G - 1)}*Mach^2)^{G / (G - 1)}"),
            ("PressureRecovery", f"p0/{F['p0_inf']}")]
    if "rho" in names and "nut" in names:
        defs.append(("TurbulentViscosityRatio",
                     f"nut*rho*(T+{F['sutherland_Ts']})/({F['sutherland_As']}*T^1.5)"))
    if "grad(rho)" in names:
        defs.append(("Schlieren", 'mag("grad(rho)")'))
    if "wallShearStress" in names:
        defs.append(("Cf", f"mag(wallShearStress)/{F['q_inf']}"))
    for name, fn in defs:
        src = Calculator(Input=src)
        src.AttributeType = "Point Data"
        src.ResultArrayName = name
        src.Function = fn
    src.UpdatePipeline(T_LATEST)
    return src


def mirror(src):
    # Complete aircraft from a half model (sources that do not touch y = 0 only are left alone)
    if not HALF:
        return src
    r = Reflect(Input=src)
    r.Plane = "Y"
    r.Center = 0.0
    r.CopyInput = 1
    r.UpdatePipeline(T_LATEST)
    return r


def plane(src, origin, normal, triangulate=True):
    s = Slice(Input=src)
    s.SliceType = "Plane"
    s.SliceType.Origin = origin
    s.SliceType.Normal = normal
    try:
        s.Triangulatetheslice = 1 if triangulate else 0
    except AttributeError:
        pass
    s.UpdatePipeline(T_LATEST)
    return s


def near_field(src, x):
    c = Clip(Input=src)
    c.ClipType = "Box"
    c.ClipType.Position = [x - 0.05, NEAR_BOX["y"][0], NEAR_BOX["z"][0]]
    c.ClipType.Length = [0.1, NEAR_BOX["y"][1] - NEAR_BOX["y"][0], NEAR_BOX["z"][1] - NEAR_BOX["z"][0]]
    c.Invert = 1                     # keep the inside of the box
    c.UpdatePipeline(T_LATEST)
    return c


# ------------------------------------------------------------------------------------------ scenes
def view(cam):
    rv = CreateView("RenderView")
    rv.ViewSize = IMAGE_SIZE
    rv.ViewTime = T_LATEST
    rv.OrientationAxesVisibility = 0
    try:
        rv.UseColorPaletteForBackground = 0
    except AttributeError:
        pass
    rv.Background = [1, 1, 1]
    rv.CameraFocalPoint, rv.CameraPosition, rv.CameraViewUp = cam[0], cam[1], cam[2]
    rv.CameraParallelProjection = 1
    rv.CameraParallelScale = cam[3]
    return rv


def solid(d, colour, edges=False):
    ColorBy(d, None)
    d.AmbientColor = colour
    d.DiffuseColor = colour
    if edges:
        d.Representation = "Surface With Edges"
        d.EdgeColor = [0.1, 0.1, 0.12]


def colour(d, rv, array, lo, hi, cmap=None, title=None):
    ColorBy(d, ("POINTS", array))
    lut = GetColorTransferFunction(array)
    lut.ApplyPreset(cmap or COLORMAP, True)
    lut.AutomaticRescaleRangeMode = "Never"
    lut.RescaleTransferFunction(lo, hi)
    lut.NumberOfTableValues = 64
    lut.Discretize = 1
    d.SetScalarBarVisibility(rv, True)
    bar = GetScalarBar(lut, rv)
    bar.Title = title or array
    bar.ComponentTitle = ""
    bar.WindowLocation = "Any Location"
    bar.Orientation = "Vertical"
    bar.Position = [0.92, 0.1]
    bar.ScalarBarLength = 0.8
    bar.TitleColor = [0, 0, 0]
    bar.LabelColor = [0, 0, 0]
    bar.TitleFontSize = 20
    bar.LabelFontSize = 18
    bar.RangeLabelFormat = "%-#.4g"


def save(rv, name):
    t = Show(Text(Text=name), rv)
    t.WindowLocation = "Upper Left Corner"
    t.FontSize = 22
    t.Color = [0, 0, 0]
    Render(rv)
    fname = "scene_" + "".join(c if c.isalnum() or c in "+-" else "_" for c in name) + ".png"
    SaveScreenshot(os.path.join(images, fname), rv, ImageResolution=IMAGE_SIZE)
    Delete(rv)
    print("  wrote", fname, flush=True)


def mesh_scene(name, parts, cam):
    rv = view(cam)
    if parts:
        for p in parts:
            solid(Show(p, rv), MESH_COLOUR, edges=True)
    else:
        solid(Show(VEHICLE, rv), GREY, edges=True)
    save(rv, name)


def scalar_scene(name, array, lo, hi, parts, cam, cmap=None, vehicle=False, title=None):
    rv = view(cam)
    for p in parts:
        if array not in p.PointData.keys():
            print(f"  skipped {name}: no {array}")
            Delete(rv)
            return
        colour(Show(p, rv), rv, array, lo, hi, cmap, title)
    if vehicle:
        solid(Show(VEHICLE, rv), GREY)
    save(rv, name)


def lic_scene(name, cam):
    # Wall shear stress line integral convolution: surface streamlines, separation and attachment lines
    rv = view(cam)
    d = Show(SURF, rv)
    try:
        d.Representation = "Surface LIC"
        d.SelectInputVectors = ["POINTS", "wallShearStress"]
        colour(d, rv, "Cf", 0.0, 0.006, title="Cf")
        d.EnhanceContrast = "Color Only"
    except Exception as e:                       # no Surface LIC in this ParaView build
        print(f"  {name}: Surface LIC unavailable ({e}), showing Cf only")
        colour(d, rv, "Cf", 0.0, 0.006, title="Cf")
    save(rv, name)


# ------------------------------------------------------------------------------------------ checks
def integrate(src):
    i = IntegrateVariables(Input=src)
    i.DivideCellDataByVolume = 0
    i.UpdatePipeline(T_LATEST)
    cd = servermanager.Fetch(i).GetCellData()
    out = {cd.GetArrayName(k): cd.GetArray(k).GetValue(0) for k in range(cd.GetNumberOfArrays())}
    Delete(i)
    return out


def checks():
    post = {"time": T_LATEST, "yplus": {}, "field_range": {}}
    for patch in WALLS + ["walls"]:
        r = reader([f"patch/{p}" for p in WALLS] if patch == "walls" else [f"patch/{patch}"])
        r.UpdatePipeline(T_LATEST)
        if "yPlus" not in r.CellData.keys():
            continue
        I = integrate(r)
        post["yplus"][patch] = {"avg": I["yPlus"] / I["Area"], "max": r.CellData["yPlus"].GetRange()[1],
                                "area_m2": I["Area"]}
    for name in ("p", "T", "Mach", "TurbulentViscosityRatio", "rho", "PressureRecovery"):
        if name in VOL.PointData.keys():
            post["field_range"][name] = list(VOL.PointData[name].GetRange())
    try:
        th = Threshold(Input=VOL_RAW)
        th.Scalars = ["CELLS", "p"]
        try:
            th.LowerThreshold, th.UpperThreshold = 0.0, LOW_P_LIMIT
            th.ThresholdMethod = "Between"
        except AttributeError:
            th.ThresholdRange = [0.0, LOW_P_LIMIT]
        th.UpdatePipeline(T_LATEST)
        info = th.GetDataInformation()
        post["low_pressure_cells"] = {"limit": LOW_P_LIMIT, "count": info.GetNumberOfCells(),
                                      "bounds": list(info.GetBounds())}
    except Exception as e:
        print("  low-pressure check failed:", e)
    return post


# ------------------------------------------------------------------------------------------ main
VOL_RAW = reader(["internalMesh"])
T_LATEST = latest(VOL_RAW)
VOL = derived(VOL_RAW)
SURF = mirror(derived(reader([f"patch/{p}" for p in WALLS])))
ENGINE = mirror(derived(reader(["patch/engine_face"])))
VEHICLE = mirror(reader([f"patch/{p}" for p in WALLS] + ["patch/engine_face", "patch/exhaust"]))

print("checks ...", flush=True)
M = json.load(open(os.path.join(case, "metrics.json")))
M["post"] = checks()
with open(os.path.join(case, "metrics.json"), "w") as fh:
    json.dump(M, fh, indent=2)

print("derived parts ...", flush=True)
pY0 = plane(VOL, [0, Y_SLICE, 0], [0, 1, 0])
pY0_mesh = plane(VOL, [0, Y_SLICE, 0], [0, 1, 0], triangulate=False)
pZ0 = mirror(plane(VOL, [0, 0, 0.013], [0, 0, 1]))          # through the wing (z = 0.006 - 0.023 m)
pZ0_mesh = mirror(plane(VOL, [0, 0, 0.013], [0, 0, 1], triangulate=False))
pZin = mirror(plane(VOL, [0, 0, 0.1034], [0, 0, 1]))        # inlet lip height
stations = [near_field(mirror(plane(VOL, [x, 0, 0], [1, 0, 0])), x) for x in STATIONS]

print("scenes ...", flush=True)
# mesh
mesh_scene("01 Mesh - centreline (whole)", [pY0_mesh], SIDE_FAR)
mesh_scene("02 Mesh - centreline (near body)", [pY0_mesh], SIDE)
mesh_scene("03 Mesh - inlet and duct", [pY0_mesh], INLET)
mesh_scene("04 Mesh - nose prism layers", [pY0_mesh], NOSE)
mesh_scene("05 Mesh - base and plume", [pY0_mesh], TAIL)
mesh_scene("06 Mesh - wing plane z=0.013", [pZ0_mesh], TOP)
mesh_scene("07 Mesh - surface", [], ISO_TOP)

# centreline
scalar_scene("10 Mach - centreline", "Mach", 0.0, 2.0, [pY0], SIDE)
scalar_scene("11 Mach - centreline (far)", "Mach", 0.0, 2.0, [pY0], SIDE_FAR)
scalar_scene("12 Schlieren - centreline", "Schlieren", 0.0, 5.0, [pY0], SIDE, cmap="Grayscale")
scalar_scene("13 Pressure ratio - centreline", "PressureRatio", 0.0, 3.5, [pY0], SIDE, title="p / p_inf")
scalar_scene("14 Temperature - centreline", "T", 200.0, 340.0, [pY0], SIDE, title="T [K]")
scalar_scene("15 Total pressure recovery - centreline", "PressureRecovery", 0.3, 1.0, [pY0], SIDE,
             title="p0 / p0_inf")
scalar_scene("16 Turbulent viscosity ratio - centreline", "TurbulentViscosityRatio", 0.0, 500.0, [pY0], SIDE,
             title="mu_t / mu")

# planform
scalar_scene("20 Mach - wing plane z=0.013", "Mach", 0.0, 2.0, [pZ0], TOP)
scalar_scene("21 Schlieren - wing plane z=0.013", "Schlieren", 0.0, 5.0, [pZ0], TOP, cmap="Grayscale")
scalar_scene("22 Pressure ratio - wing plane z=0.013", "PressureRatio", 0.0, 3.5, [pZ0], TOP, title="p / p_inf")

# cross-sections
scalar_scene("30 Mach - x stations", "Mach", 0.0, 2.0, stations, ISO_STN, vehicle=True)
scalar_scene("31 Total pressure recovery - x stations", "PressureRecovery", 0.3, 1.0, stations, ISO_STN,
             vehicle=True, title="p0 / p0_inf")
for i, x in enumerate(STATIONS):     # each station head-on, looking forward from behind
    aft = ([x, 0, 0.1], [x + 6.0, 0, 0.1], UPZ, 0.30)
    scalar_scene(f"3{i + 2} Station x={x:.2f} - total pressure recovery", "PressureRecovery", 0.3, 1.0,
                 [stations[i]], aft, title="p0 / p0_inf")

# inlet and engine face
scalar_scene("40 Inlet - Mach", "Mach", 0.0, 2.0, [pY0], INLET)
scalar_scene("41 Inlet - total pressure recovery", "PressureRecovery", 0.5, 1.0, [pY0], INLET, title="p0 / p0_inf")
scalar_scene("42 Inlet - pressure ratio", "PressureRatio", 0.0, 3.5, [pY0], INLET, title="p / p_inf")
scalar_scene("43 Inlet - Schlieren", "Schlieren", 0.0, 20.0, [pY0], INLET, cmap="Grayscale")
scalar_scene("44 Inlet - Mach at lip height (top view)", "Mach", 0.0, 2.0, [pZin], INLET_TOP)
scalar_scene("45 Engine face - total pressure recovery", "PressureRecovery", 0.5, 1.0, [ENGINE], FRONT,
             title="p0 / p0_inf")
scalar_scene("46 Engine face - Mach", "Mach", 0.0, 1.0, [ENGINE], FRONT)

# exhaust plume
scalar_scene("50 Plume - Mach", "Mach", 0.0, 2.5, [pY0], TAIL)
scalar_scene("51 Plume - temperature", "T", 200.0, 2300.0, [pY0], TAIL, title="T [K]")
scalar_scene("52 Plume - pressure ratio", "PressureRatio", 0.0, 2.0, [pY0], TAIL, title="p / p_inf")
scalar_scene("53 Plume - Schlieren", "Schlieren", 0.0, 5.0, [pY0], TAIL, cmap="Grayscale")

# surface quantities
scalar_scene("60 Surface - Cp (top)", "Cp", -0.4, 1.0, [SURF], ISO_TOP)
scalar_scene("61 Surface - Cp (bottom)", "Cp", -0.4, 1.0, [SURF], ISO_BOTTOM)
scalar_scene("62 Surface - Cp (aft)", "Cp", -0.4, 1.0, [SURF], ISO_AFT)
scalar_scene("63 Surface - wall y+ (top)", "yPlus", 0.0, 150.0, [SURF], ISO_TOP, title="y+")
scalar_scene("64 Surface - wall y+ (bottom)", "yPlus", 0.0, 150.0, [SURF], ISO_BOTTOM, title="y+")
scalar_scene("65 Surface - skin friction Cf (top)", "Cf", 0.0, 0.006, [SURF], ISO_TOP)
scalar_scene("66 Surface - skin friction Cf (bottom)", "Cf", 0.0, 0.006, [SURF], ISO_BOTTOM)
scalar_scene("67 Surface - temperature", "T", 200.0, 340.0, [SURF], ISO_TOP, title="T [K]")
lic_scene("68 Surface - shear lines (top)", ISO_TOP)
lic_scene("69 Surface - shear lines (bottom)", ISO_BOTTOM)
print("detail_post finished OK", flush=True)
