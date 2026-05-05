# Aerodynamic Design of a Small-Scale Supersonic UAV Powered by a Rotating Detonation Engine

**Research Project — Aerodynamic Design Focus**  
*Status: In Progress*

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Motivation for an RDE-Powered Supersonic UAV](#2-motivation-for-an-rde-powered-supersonic-uav)
3. [Design Plan](#3-design-plan)
4. [Conceptual Design Phase](#4-conceptual-design-phase)
5. [CFD Validation & Inlet Design](#5-cfd-validation-and-inlet-design)

---

## 1. Introduction

Unmanned aerial vehicles (UAVs) have historically operated in the subsonic regime, where conventional piston/jet engines are well understood, mature, and readily available at small scales. However, there is growing strategic and scientific interest in small-scale supersonic UAVs capable of exceeding Mach 1, for applications such as high-speed reconnaissance or time-critical payload delivery.

Achieving supersonic flight at small scale presents a dual challenge: the aerodynamic design must manage the fundamentally different flow physics that arise above the speed of sound, while the propulsion system must deliver sufficient specific thrust efficiently enough to be packaged into a compact, lightweight airframe. Traditional miniaturised jet engines suffer from dramatically reduced efficiency and power density at small scales.

This project investigates the aerodynamic design of a small-scale supersonic UAV that uses a **Rotating Detonation Engine (RDE)** as its propulsion source. RDEs are an emerging class of pressure-gain combustion devices that exploit self-sustaining detonation waves rather than conventional deflagration to release chemical energy. Their near constant-volume thermodynamic cycle offers a theoretical efficiency advantage over the Brayton cycle used in conventional gas turbines, and their geometric simplicity makes them inherently more scalable to small sizes than conventional jet engines.

The scope of this research is confined to the **aerodynamic design** of the vehicle. Propulsion performance (thrust, specific impulse, mass flow) is treated as a set of fixed boundary conditions derived from literature on RDE weight and thrust estimation, with the aerodynamic framework designed to accommodate those constraints.

---

## 2. Motivation for an RDE-Powered Supersonic UAV

The combination of RDE propulsion with a small-scale supersonic UAV platform is motivated by the following factors:

**Propulsion efficiency**: The thermodynamic advantage of detonation over deflagration (Humphrey vs. Brayton cycle) is most impactful in applications where fuel efficiency and range are critical.

**Scalability**: RDEs lack the rotating turbomachinery that severely limits the efficiency of small gas turbines. In theory, an RDE be designed at a scale where a turbojet would be impractically inefficient, making it uniquely suited to small supersonic platforms.

**Simplicity and mass**: An RDE has fewer components than a turbojet, leading to lower mass and higher reliability, both valuable in small-scale UAVs.

**Research novelty**: The integration of RDE propulsion with a purpose-designed supersonic airframe at small scale is an open research area with little published examples. There is scientific value in demonstrating and characterising such an integrated system, even at the design/simulation stage only.

---

## 3. Design Plan

The aerodynamic design research is structured as a hierarchy of progressively integrated tasks, organised across three major frameworks.

### Phase 1 — Conceptual Design Phase

The central body of aerodynamic design work is organised as an iterative design loop:

1. **Constraint analysis**: A classical constraint diagram mapping thrust-to-weight ratio against wing loading, overlaying constraints such as cruise speed, climb rate, stall speed, ceiling altitude and max speed. An approximate T/W ratio and wing loading is selected from here.
2. **Propulsion sizing**: BAsed on existing research/experimental setups about RDEs, an appropriate engine is selected, making reasonable estimates regarding the engine weight, thrust, and fuel flow rate.
3. **Aircraft total weight estimation**: An iterative weight build-up for a supersonic UAV, including structural weight fractions, propulsion system mass (from previous step), fuel mass from range equations (adapted Breguet equation for supersonic cruise), and payload/avionics mass.
4. **Validate OpenVSP CL and CD results**: The baseline configuration (wing planform, fuselage, tail) is modelled in OpenVSP (which uses VLM/panel methods). Lift and drag coefficient predictions are validated against published experimental data for similar configurations.
5. **OpenVSP optimisation to minimise CD**: Parametric studies in OpenVSP are used to optimise the aerodynamic configuration — wing sweep, taper ratio, thickness distribution, fuselage fineness ratio, and area ruling, to minimise total drag coefficient at the cruise conditions.
6. **Centre of gravity (CG) envelope**: Determination of the CG range across the flight envelope (fuel burn, payload configurations) and its relationship to the aerodynamic centre to ensure longitudinal static stability margins are maintained throughout the mission.
7. **Stability analysis and trim conditions**: Static and dynamic stability derivatives are computed for the optimised configuration. Longitudinal, lateral, and directional stability are assessed across the flight envelope. Trim states are established at cruise and manoeuvre conditions.
   - **Tail sizing**: If the baseline configuration is found to be unstable or insufficiently controllable, horizontal and/or vertical tail surfaces are sized to restore adequate stability margins and control authority.

### Phase 2 — CFD Validation & Supersonic Inlet Design

The optimized design from the first phase of the design process is then tested using a RANS/LES solver in OpenFOAM. the inlet is also optimized for maximized thrust from the RDE. The steps for this phase include:

1. **Finalise component and engine placement**: The physical layout of the RDE, inlet duct, fuel system, and payload within the airframe is finalised, establishing the geometric constraints for the inlet.
2. **OpenFOAM test case and settings setup**: Baseline CFD simulations of the overall UAV & the inlet geometry are established in OpenFOAM, including appropriate boundary conditions and solver settings. Mesh sensitivity studies and solver settings are validated against known benchmarks.
3. **OpenVSP vs. OpenFOAM validation cases**: Selected configurations are analysed in OpenFOAM to validate the panel/vortex-lattice/wave drag results from OpenVSP against higher-fidelity RANS/LES solutions. Discrepancies are identified and used to correct or calibrate the lower-fidelity model.
4. **OpenFOAM inlet optimisation framework**: A parametric optimisation framework is applied to the inlet geometry in OpenFOAM to maximise total pressure recovery and flow uniformity at the combustor face, while minimising external drag and maintaining stable operation across the mission Mach number range.

---

## 4. Conceptual Design Phase

## 5. CFD Validation and Inlet Design
