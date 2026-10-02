import subprocess
import os
import tempfile

FUSELAGE_LENGTH = 5.0
FUSELAGE_X_LOCATION = -0.2459016393442627901

def create_uav_model(
    # Wing placement
    x_location=0.0,
    z_location=0.08,
    y_rotation=1.5,
    # Segment 1 (root to break)
    seg1root_chord=4.0,
    seg2root_chord=1.6,  # break chord (seg 1 tip / seg 2 root)
    seg1_span=0.75,
    seg1_sweep=73.0,
    seg1_twist=0.0,
    seg1_dihedral=0.0,
    # Segment 2 (break to tip)
    seg2tip_chord=0.8,
    seg2_span=1.0,
    seg2_sweep=48.0,
    seg2_twist=-2.5,
    seg2_dihedral=0.0,
    # Airfoils (double wedge)
    tc_root=0.025,   # XSecCurve_0
    tc_break=0.035,  # XSecCurve_1
    tc_tip=0.050,    # XSecCurve_2
    thick_loc=0.5,   # max-thickness location, 0.5 = symmetric diamond
    # Wing tessellation
    wing_tess_w=101,
    seg1_tess_u=35,
    seg2_tess_u=35,
    stl_path="recreated_uav.stl"
):

    # Absolute path with forward slashes so it is safe inside the vspscript string
    stl_path = os.path.abspath(stl_path)
    os.makedirs(os.path.dirname(stl_path), exist_ok=True)
    stl_path_vsp = stl_path.replace("\\", "/")

    output_name = os.path.splitext(os.path.basename(stl_path))[0]
    script_filename = f"{output_name}_script.vspscript"
    vspscript_content = f"""
void main() {{
    VSPCheckSetup();
    ClearVSPModel();

    // 1. WING GEOMETRY
    string wing_id = AddGeom("WING", "");
    SetGeomName(wing_id, "WingGeom");

    // Placement, pitch (Y Rotation) and planar symmetry (XZ plane)
    SetParmVal(wing_id, "X_Rel_Location", "XForm", {x_location});
    SetParmVal(wing_id, "Z_Rel_Location", "XForm", {z_location});
    SetParmVal(wing_id, "Y_Rel_Rotation", "XForm", {y_rotation});
    SetParmVal(wing_id, "Sym_Planar_Flag", "Sym", 2.0);

    // Insert a cross-section to create 2 segments
    InsertXSec(wing_id, 1, XS_WEDGE);
    Update();

    // --- Wing Segment 1 (Root to Break) ---
    SetParmVal(wing_id, "Root_Chord", "XSec_1", {seg1root_chord});
    SetParmVal(wing_id, "Tip_Chord", "XSec_1", {seg2root_chord});
    SetParmVal(wing_id, "Span", "XSec_1", {seg1_span});
    SetParmVal(wing_id, "Sweep", "XSec_1", {seg1_sweep});
    SetParmVal(wing_id, "Sweep_Location", "XSec_1", 0.0); // Sweep Location kept static
    SetParmVal(wing_id, "Twist", "XSec_1", {seg1_twist});
    SetParmVal(wing_id, "Dihedral", "XSec_1", {seg1_dihedral});
    SetParmVal(wing_id, "SectTess_U", "XSec_1", {seg1_tess_u});
    Update();

    // --- Wing Segment 2 (Break to Tip) ---
    SetParmVal(wing_id, "Root_Chord", "XSec_2", {seg2root_chord});
    SetParmVal(wing_id, "Tip_Chord", "XSec_2", {seg2tip_chord});
    SetParmVal(wing_id, "Span", "XSec_2", {seg2_span});
    SetParmVal(wing_id, "Sweep", "XSec_2", {seg2_sweep});
    SetParmVal(wing_id, "Sweep_Location", "XSec_2", 0.0); // Sweep Location kept static
    SetParmVal(wing_id, "Twist", "XSec_2", {seg2_twist});
    SetParmVal(wing_id, "Dihedral", "XSec_2", {seg2_dihedral});
    SetParmVal(wing_id, "SectTess_U", "XSec_2", {seg2_tess_u});

    SetParmVal(wing_id, "Tess_W", "Shape", {wing_tess_w});

    // --- Wing Airfoils (Double Wedge) ---
    string wing_xsec_surf = GetXSecSurf(wing_id, 0);

    // Iterate through all 3 XSecs to ensure all shapes are Wedge
    for (int i = 0; i < 3; i++) {{
        ChangeXSecShape(wing_xsec_surf, i, XS_WEDGE);
    }}
    Update();

    // Set Thickness-to-Chord ratio and thickness location for each curve
    SetParmVal(wing_id, "ThickChord", "XSecCurve_0", {tc_root});
    SetParmVal(wing_id, "ThickChord", "XSecCurve_1", {tc_break});
    SetParmVal(wing_id, "ThickChord", "XSecCurve_2", {tc_tip});
    for (int i = 0; i < 3; i++) {{
        SetParmVal(wing_id, "ThickLoc", "XSecCurve_" + i, {thick_loc});
    }}

    // 2. FUSELAGE GEOMETRY (fixed)
    string fuse_id = AddGeom("FUSELAGE", "");
    SetGeomName(fuse_id, "FuselageGeom");

    SetParmVal(fuse_id, "Length", "Design", {FUSELAGE_LENGTH});
    SetParmVal(fuse_id, "X_Rel_Location", "XForm", {FUSELAGE_X_LOCATION});
    SetParmVal(fuse_id, "Tess_W", "Shape", 25);
    Update();

    string fuse_xsec_surf = GetXSecSurf(fuse_id, 0);
    if (GetNumXSec(fuse_xsec_surf) != 5) {{
        Print("ERROR: expected 5 fuselage XSecs, got " + GetNumXSec(fuse_xsec_surf));
    }}

    // Set cross-sections: pointed nose, three circles, pointed tail
    ChangeXSecShape(fuse_xsec_surf, 0, XS_POINT);
    ChangeXSecShape(fuse_xsec_surf, 1, XS_CIRCLE);
    ChangeXSecShape(fuse_xsec_surf, 2, XS_CIRCLE);
    ChangeXSecShape(fuse_xsec_surf, 3, XS_CIRCLE);
    ChangeXSecShape(fuse_xsec_surf, 4, XS_POINT);
    Update();

    // Set Locations & Diameters using exact internal parameters
    SetParmVal(fuse_id, "XLocPercent", "XSec_1", 0.15);
    SetParmVal(fuse_id, "Circle_Diameter", "XSecCurve_1", 0.30);

    SetParmVal(fuse_id, "XLocPercent", "XSec_2", 0.5);
    SetParmVal(fuse_id, "Circle_Diameter", "XSecCurve_2", 0.35);

    SetParmVal(fuse_id, "XLocPercent", "XSec_3", 0.8422131147540983243);
    SetParmVal(fuse_id, "Circle_Diameter", "XSecCurve_3", 0.30);

    SetParmVal(fuse_id, "XLocPercent", "XSec_4", 1.0);

    // Section tessellation
    SetParmVal(fuse_id, "SectTess_U", "XSec_1", 20);
    SetParmVal(fuse_id, "SectTess_U", "XSec_2", 50);
    SetParmVal(fuse_id, "SectTess_U", "XSec_3", 50);
    SetParmVal(fuse_id, "SectTess_U", "XSec_4", 70);
    Update();

    // Skinning: nose 45 deg (left strength 0.75), body 0 deg, tail -20 deg
    SetXSecTanAngles(GetXSec(fuse_xsec_surf, 0), XSEC_BOTH_SIDES, 45.0);
    SetXSecTanStrengths(GetXSec(fuse_xsec_surf, 0), XSEC_LEFT_SIDE, 0.75);
    SetXSecTanStrengths(GetXSec(fuse_xsec_surf, 0), XSEC_RIGHT_SIDE, 1.0);
    for (int i = 1; i < 4; i++) {{
        SetXSecTanAngles(GetXSec(fuse_xsec_surf, i), XSEC_BOTH_SIDES, 0.0);
        SetXSecTanStrengths(GetXSec(fuse_xsec_surf, i), XSEC_BOTH_SIDES, 1.0);
    }}
    SetXSecTanAngles(GetXSec(fuse_xsec_surf, 4), XSEC_BOTH_SIDES, -20.0);
    SetXSecTanStrengths(GetXSec(fuse_xsec_surf, 4), XSEC_BOTH_SIDES, 1.0);
    for (int i = 0; i < 5; i++) {{
        SetParmVal(fuse_id, "RLSym", "XSec_" + i, 1.0);
        SetParmVal(fuse_id, "TBSym", "XSec_" + i, 1.0);
    }}
    SetParmVal(fuse_id, "CapUMaxOption", "EndCap", 1.0); // flat tail cap

    // 3. FINALIZE AND EXPORT
    Update();

    // Active XSec indices (GUI selection state only, kept to match uav.vsp3)
    SetParmVal(wing_id, "ActiveXSec", "Index", 2.0);
    SetParmVal(wing_id, "ActiveAirfoil", "Index", 2.0);
    SetParmVal(fuse_id, "ActiveXSec", "Index", 3.0);

    // Export STL with default settings
    ExportFile("{stl_path_vsp}", SET_ALL, EXPORT_STL);
}}
    """

    # Remove any stale STL so success can be checked by the file existing
    if os.path.exists(stl_path):
        os.remove(stl_path)

    # Script lives in a temp dir so nothing is left behind and parallel runs don't collide
    with tempfile.TemporaryDirectory() as tmp_dir:
        script_path = os.path.join(tmp_dir, script_filename)
        with open(script_path, "w") as f:
            f.write(vspscript_content)

        try:
            vsp_command = (
                "module swap gcc/12.3 gcc/13.3 ; "
                f"module use \"$HOME/modulefiles\" ; "
                "module load openvsp/3.51.0-headless ; "
                f"vspscript -script {script_path}"
            )
            subprocess.run(vsp_command, shell=True, executable='/bin/bash', check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error: OpenVSP script execution failed with code {e.returncode}")
            return False

    if not os.path.isfile(stl_path):
        print(f"Error: OpenVSP finished but {stl_path} was not created")
        return False

    print(f"Model created. Exported {stl_path}")
    return True

if __name__ == "__main__":
    create_uav_model()
