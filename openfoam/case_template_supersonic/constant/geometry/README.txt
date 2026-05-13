Place your UAV geometry file here as:

    uav.stl

Requirements:
  - Watertight, manifold surface (no holes, no self-intersections)
  - ASCII or binary STL format
  - Units: metres
  - Nose pointing in the +X direction
  - Fuselage centreline along X-axis at Y=0, Z=0
  - Named surface region: uav_wall
      (In CAD tool, export with region name = "uav_wall")

Recommended pre-processing:
  - surfaceCheck uav.stl             # Check for problems
  - surfaceClean uav.stl uav_clean.stl  # Remove degenerate triangles
  - surfaceFeatureExtract            # Run from case root to generate uav.eMesh

For a half-body (symmetry) simulation:
  - Clip geometry at Z=0
  - Ensure the cut face has a separate named region for the symmetryPlane patch
  - Domain extends from Z=0 to Z=+10 m only
