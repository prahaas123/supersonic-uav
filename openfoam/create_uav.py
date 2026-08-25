import subprocess
import os

def create_uav_model(
    y_rotation=5.0, 
    seg1_root_chord=4.0, 
    seg1_span=0.75, 
    seg1_tip_chord=1.6, 
    seg1_sweep=73.0,
    seg2_span=1.0, 
    seg2_tip_chord=0.8, 
    seg2_sweep=45.0,
    tc_root=0.025,   # XSecCurve_0
    tc_break=0.035,  # XSecCurve_1
    tc_tip=0.050,    # XSecCurve_2
    fuselage_length=4.0,
    output_name="recreated_uav"
):
    
    script_filename = f"{output_name}_script.vspscript"
    vspscript_content = f"""
void main() {{
    VSPCheckSetup();
    ClearVSPModel();

    // 1. WING GEOMETRY
    string wing_id = AddGeom("WING", "");
    SetGeomName(wing_id, "WingGeom");

    // Set pitch (Y Rotation) and planar symmetry (XZ plane)
    SetParmVal(wing_id, "Y_Rotation", "XForm", {y_rotation});
    SetParmVal(wing_id, "Sym_Planar_Flag", "Sym", 2.0); 

    // Insert a cross-section to create 2 segments
    InsertXSec(wing_id, 1, XS_BICONVEX);
    Update(); 

    // --- Wing Segment 1 (Root to Break) ---
    SetParmVal(wing_id, "Root_Chord", "XSec_1", {seg1_root_chord}); 
    SetParmVal(wing_id, "Span", "XSec_1", {seg1_span});
    SetParmVal(wing_id, "Tip_Chord", "XSec_1", {seg1_tip_chord});
    SetParmVal(wing_id, "Sweep", "XSec_1", {seg1_sweep});
    SetParmVal(wing_id, "Sweep_Location", "XSec_1", 0.0); // Sweep Location kept static

    // --- Wing Segment 2 (Break to Tip) ---
    SetParmVal(wing_id, "Span", "XSec_2", {seg2_span});
    SetParmVal(wing_id, "Tip_Chord", "XSec_2", {seg2_tip_chord});
    SetParmVal(wing_id, "Sweep", "XSec_2", {seg2_sweep});
    SetParmVal(wing_id, "Sweep_Location", "XSec_2", 0.0); // Sweep Location kept static

    // --- Wing Airfoils (Biconvex) ---
    string wing_xsec_surf = GetXSecSurf(wing_id, 0);
    
    // Iterate through all 3 XSecs to ensure all shapes are Biconvex
    for (int i = 0; i < 3; i++) {{
        ChangeXSecShape(wing_xsec_surf, i, XS_BICONVEX);
    }}
    Update(); 

    // Set Thickness-to-Chord ratio for each curve
    SetParmVal(wing_id, "ThickChord", "XSecCurve_0", {tc_root}); 
    SetParmVal(wing_id, "ThickChord", "XSecCurve_1", {tc_break}); 
    SetParmVal(wing_id, "ThickChord", "XSecCurve_2", {tc_tip}); 

    // 2. FUSELAGE GEOMETRY
    string fuse_id = AddGeom("FUSELAGE", "");
    SetGeomName(fuse_id, "FuselageGeom");
    
    // Set overall length
    SetParmVal(fuse_id, "Length", "Design", {fuselage_length});
    Update();

    string fuse_xsec_surf = GetXSecSurf(fuse_id, 0);

    // Set cross-sections
    ChangeXSecShape(fuse_xsec_surf, 0, XS_POINT);
    ChangeXSecShape(fuse_xsec_surf, 1, XS_CIRCLE);
    ChangeXSecShape(fuse_xsec_surf, 2, XS_CIRCLE);
    ChangeXSecShape(fuse_xsec_surf, 3, XS_CIRCLE);
    Update(); 

    // Set Locations & Diameters using exact internal parameters 
    SetParmVal(fuse_id, "XLocPercent", "XSec_1", 0.15);
    SetParmVal(fuse_id, "Circle_Diameter", "XSecCurve_1", 0.30);

    SetParmVal(fuse_id, "XLocPercent", "XSec_2", 0.474385);
    SetParmVal(fuse_id, "Circle_Diameter", "XSecCurve_2", 0.35);

    SetParmVal(fuse_id, "XLocPercent", "XSec_3", 1.0);
    SetParmVal(fuse_id, "Circle_Diameter", "XSecCurve_3", 0.142);

    // 3. FINALIZE AND EXPORT
    Update();
    
    string base_filename = "{output_name}";
    
    // Save OpenVSP Model
    WriteVSPFile(base_filename + ".vsp3", SET_ALL);
    
    // Export STL with default settings
    ExportFile(base_filename + ".stl", SET_ALL, EXPORT_STL); 
}}
    """

    with open(script_filename, "w") as f:
        f.write(vspscript_content)

    try:        
        vsp_command = (
            "module swap gcc/12.3 gcc/13.3 ; "
            f"module use \"$HOME/modulefiles\" ; "
            "module load openvsp/3.51.0-headless ; "
            f"vspscript -script {script_filename}"
        )
        subprocess.run(vsp_command, shell=True, executable='/bin/bash', check=True)
        print(f"Model created. Exported {output_name}.stl") 
    except subprocess.CalledProcessError as e:
        print(f"Error: OpenVSP script execution failed with code {e.returncode}")

if __name__ == "__main__":
    create_uav_model()