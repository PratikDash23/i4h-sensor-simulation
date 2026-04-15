# Session Report: Raytracing Ultrasound Simulator — Exploration & Project Planning

**Date:** 2026-04-14
**System:** RTX 4090, vast.ai remote GPU instance (SSH access)
**Repository:** `isaac-for-healthcare/i4h-sensor-simulation` (ultrasound-raytracing module)

---

## 1. What We Accomplished

### 1.1 First Successful Run of the Simulator

- Verified system readiness: RTX 4090 (Driver 580.105.08, CUDA 13.0), Docker 29.0.3, NVIDIA Container Toolkit 1.18.0, Python 3.10.12
- Fixed Docker permissions (`sudo usermod -aG docker $USER`)
- Ran the sphere sweep demo successfully via the `./i4h` CLI (Option 1 — Docker-based):
  ```bash
  sudo ./i4h run ultrasound-raytracing sphere_sweep
  ```
- Result: 10 B-mode frames generated in ~2.8 seconds (~2ms/frame, ~3.56 it/s)
- Output location: `ultrasound-raytracing/ultrasound_sweep/frame_000.png` through `frame_009.png`

### 1.2 Interactive Web Server

- Launched the interactive simulator:
  ```bash
  sudo ./i4h run ultrasound-raytracing
  ```
- Accessed via SSH tunnel: `ssh -p 59290 root@213.181.122.2 -L 8080:localhost:8080 -L 9000:localhost:8000`
- Opened in local browser at `http://localhost:9000`
- The web UI provides:
  - Real-time B-mode image of a full abdominal phantom (13 organ meshes)
  - Probe type switching (curvilinear, linear array, phased array)
  - Keyboard navigation (arrow keys for XY, W/S for depth, Q/E/R/F/T/G for rotation)
  - Simulation parameter controls (speckle filter, CUDA timing, debug images, contact epsilon)
  - Manual position/rotation input with "Update Position" button

### 1.3 Added Frequency Control to the Web UI

Modified two files to add a live frequency slider:

**`examples/server.py`** — added two REST endpoints:
- `GET /get_frequency` — returns current probe frequency
- `POST /set_frequency` — sets frequency in MHz (uses existing `set_frequency()` on probe, no rebuild needed)

**`examples/templates/index.html`** — added:
- Slider control (1.0-15.0 MHz, 0.5 MHz steps)
- Auto-updates when switching probe types
- Triggers re-simulation on change

---

## 2. Simulator Architecture — Key Findings

### 2.1 Available Modes

| Command | Description |
|---|---|
| `./i4h run ultrasound-raytracing` | Interactive web server (Flask, port 8000) with full abdominal phantom |
| `./i4h run ultrasound-raytracing sphere_sweep` | Headless batch: 2 spheres, 10-frame sweep, saves PNGs |
| `./i4h run ultrasound-raytracing liver_sweep` | Headless batch: liver mesh sweep |
| `./i4h run ultrasound-raytracing benchmark` | 200-frame performance benchmark |
| `./i4h modes ultrasound-raytracing` | Lists available modes |
| `./i4h run-container ultrasound-raytracing` | Interactive bash shell inside the container |

### 2.2 Simulation Parameters

**Exposed in web UI:**
- **Median Clip Speckle Filter** (default: off) — reduces speckle noise
- **CUDA Timing** (default: on) — prints per-step timing to console
- **Write Debug Images** (default: off) — writes intermediate images to `debug_images/`
- **Contact Epsilon** (default: 0.0 mm) — max distance for element activation

**Code-only parameters:**
- `t_far` (180.0 mm) — maximum imaging depth
- `buffer_size` (4096) — samples per scanline
- `max_depth` (15) — max reflection/refraction bounces
- `min_intensity` (0.001) — ray termination threshold
- `use_scattering` (true) — volumetric tissue scattering
- `conv_psf` (true) — Point Spread Function convolution
- `b_mode_size` (500, 500) — output image resolution

### 2.3 Probe Parameters (All Configurable)

| Parameter | Setter | Unit |
|---|---|---|
| `num_elements_x` | `set_num_elements_x()` | count |
| `frequency` | `set_frequency()` | MHz |
| `width` | constructor | mm |
| `elevational_height` | `set_elevational_height()` | mm |
| `f_num` | `set_f_num()` | unitless |
| `speed_of_sound` | `set_speed_of_sound()` | mm/us |
| `pulse_duration` | `set_pulse_duration()` | cycles |
| `num_el_samples` | `set_num_el_samples()` | count |
| `sector_angle` | constructor (curvilinear/phased) | degrees |
| `radius` | constructor (curvilinear) | mm |

### 2.4 Material Model

Pre-defined materials (hardcoded in `material.cpp`):

| Material | Impedance (MRayl) | Attenuation (dB/cm/MHz) | Speed of Sound (m/s) |
|---|---|---|---|
| water | 1.48 | 0.0022 | 1480 |
| blood | 1.61 | 0.18 | 1570 |
| fat | 1.38 | 0.63 | 1450 |
| liver | 1.65 | 0.7 | 1550 |
| muscle | 1.70 | 1.09 | 1580 |
| bone | 7.80 | 5.0 | 4080 |

Each material also has scattering parameters (mu0, mu1, sigma) and specularity — these control speckle texture and are not derivable from CT.

Full material constructor: `Material(impedance, attenuation, speed_of_sound, mu0, mu1, sigma, specularity)`

### 2.5 PSF (Point Spread Function) Model

- **Axial:** Gaussian envelope x cosine modulation at probe center frequency
- **Lateral:** Gaussian with width = wavelength x f_number
- **Elevational:** Gaussian blur (when num_el_samples > 1)
- Auto-recalculates when frequency, element spacing, or elevational height change
- Generated in `create_gaussian_psf()` at `raytracing_ultrasound_simulator.cpp:51-86`
- **Custom PSF is feasible:** the convolution engine accepts any 1D float array — replace the kernel with a measured impulse response or arbitrary spectral shape

### 2.6 Key Source Files

| File | Purpose |
|---|---|
| `csrc/cuda/optix_trace.cu` | OptiX ray tracing kernels (reflection, refraction, scattering) |
| `csrc/cuda/cuda_algorithms.cu` | RF-to-B-mode: PSF convolution, envelope detection, scan conversion |
| `csrc/core/raytracing_ultrasound_simulator.cpp` | Main pipeline orchestrator, PSF generation |
| `csrc/core/material.cpp` | Material definitions and GPU upload |
| `csrc/python/raysim_bindings.cpp` | pybind11 Python interface |
| `include/raysim/core/probe.hpp` | Base probe class with all parameters |
| `include/raysim/cuda/optix_trace.hpp` | Payload, HitGroupData, LaunchParams structs |
| `examples/server.py` | Flask web server (interactive mode) |
| `examples/templates/index.html` | Web UI |
| `examples/sphere_sweep.py` | Sphere demo script |
| `utils/phantom_maker.py` | Synthetic mesh generation (checker, sphere_in_oval) |

---

## 3. Key Technical Discussions

### 3.1 Ray Tracing vs. k-Wave for RL Training

| | k-Wave | Raytracing Simulator |
|---|---|---|
| Method | k-space pseudospectral (wave equation) | Deterministic ray tracing (geometric optics) |
| Speed | Minutes to hours per 3D frame | ~2ms per frame (~500 FPS) |
| Speed ratio | 1x | ~100,000x faster |
| Physics | Full wave (diffraction, interference, phase) | Geometric (reflection, refraction, attenuation, texture-based scattering) |
| Speckle | Physically accurate (coherent interference) | Texture-based approximation + PSF convolution |

**Conclusion:** For RL training where the agent needs millions of environment steps, the raytracing simulator is the only viable option. k-Wave is appropriate for small validation test sets.

### 3.2 Segmentation Label Passthrough (Paired Data Generation)

The ray tracing `Payload` struct (optix_trace.hpp:61) already tracks per-ray:
```c++
struct Payload {
  float intensity;
  uint32_t depth;
  float t_ancestors;
  uint16_t current_obj_id;      // which mesh the ray is inside
  uint16_t outter_obj_id;
  uint16_t current_material_id; // which material
  uint16_t outter_material_id;
};
```

**Implication:** Pixel-aligned segmentation maps can be generated alongside B-mode images by writing object IDs to a parallel buffer during ray tracing. The information already exists in the pipeline; it just needs to be written to an output buffer and scan-converted with nearest-neighbor interpolation.

This is fundamentally different from k-Wave, where wave phenomena make it impossible to trace pixels back to source labels.

### 3.3 CT Integration — Architecture Constraint

The simulator is **surface-based** (traces rays against triangle meshes), NOT **volumetric** (no per-voxel property grid). This means:
- Cannot directly ingest 3D maps of density/speed-of-sound/attenuation
- Required pipeline: CT -> organ segmentation -> surface meshes (OBJ/STL) -> one discrete material per mesh
- Per-organ average acoustic properties derived from HU values + literature lookup tables for scattering parameters

**Material properties needed beyond density and speed of sound:**
- Attenuation (dB/cm/MHz) — empirical tables exist (Duck 1990, ICRU Report 61), partially derivable from CT
- Scattering (mu0, mu1, sigma) — ultrasound-specific, no CT correlate, assign by tissue type from literature
- Specularity — assign by tissue type

**Alternative approaches for HU-to-acoustic mapping:**
- Tissue-type lookup tables (most common in literature)
- Empirical polynomial fits (Mast 2000, Aubry et al. 2003)
- Porosity models for bone (well-studied for transcranial FUS)

### 3.4 Custom Transducer Spectral Response

Current PSF: Gaussian x cosine (monochromatic, no bandwidth modeling).

Real probes have: center frequency, bandwidth (e.g., C5-2 = 2-5 MHz), transmit/receive spectrum shapes, element directivity.

**Modification path:** The PSF convolution kernel is a generic 1D float array. To use a custom spectrum:
1. Compute impulse response in Python (from measured data, spec sheet, or analytical model)
2. Pass as numpy array to replace default kernel
3. Requires adding `set_axial_psf()` binding in C++ (small change)

The convolution engine doesn't know or care how the kernel was generated.

---

## 4. Project Context & Plan

### 4.1 Goal

Fully autonomous ultrasound scanning of the thyroid using a robot arm.

### 4.2 Architecture (Dual-Model Approach)

1. **Navigation policy** — trained on simulated images (raytracing simulator + patient-specific CT-derived anatomy). Learns probe positioning, spatial reasoning, scan plane optimization.
2. **Image comprehension model** — trained on billions of real patient ultrasound images. Learns anatomy interpretation, quality assessment, pathology detection.

### 4.3 Reference Architecture

The `isaac-for-healthcare/i4h-workflows` robotic ultrasound workflow:
- **Isaac Sim/Lab** for robot simulation + physics (7-DOF Franka arm)
- **Raytracing ultrasound simulator** for image generation
- **Pi0 or GR00T N1** foundation models for policy learning
- **RTI Connext DDS** for real-time inter-process communication
- **Holoscan SDK** for real hardware integration (Clarius probes, RealSense cameras)
- Three operational modes: autonomous, teleoperation, automated state machines
- Data pipeline: teleoperation -> HDF5 -> LeRobot format -> policy training
- Training requires >= 48GB VRAM (RTX 4090's 24GB sufficient for inference/data generation only)

### 4.4 Key Resource: TotalSegmentator

- Segments 117 anatomical classes from CT
- **Thyroid included** — `thyroid_gland` (class 17)
- Also includes carotid arteries, trachea, vertebrae C1-C7, muscles
- Outputs NIfTI masks (NOT meshes — conversion needed)
- 1228 CT subjects available on Zenodo
- Usage: `TotalSegmentator -i ct.nii.gz -o segs/ --roi_subset thyroid_gland carotid_artery_left ...`

### 4.5 Prioritized Next Steps

1. **CT-to-mesh pipeline for thyroid/neck anatomy** (CRITICAL PATH)
   - Download neck CTs from TotalSegmentator Zenodo dataset
   - Run TotalSegmentator for thyroid + surrounding structures
   - Convert NIfTI masks to OBJ meshes (marching cubes + smoothing)
   - Load into simulator with appropriate acoustic materials
   - Verify visually

2. **Visual validation** — compare simulated thyroid images against real thyroid ultrasound

3. **Study i4h robotic ultrasound workflow** — Isaac Sim/Lab integration, DDS, data collection pipeline, policy training

4. **Future work (not yet):**
   - RL environment wrapper (Gymnasium interface)
   - Segmentation label buffer in CUDA kernels
   - Custom PSF for specific transducers
   - Robot arm integration

---

## 5. Files Modified During This Session

| File | Change |
|---|---|
| `ultrasound-raytracing/examples/server.py` | Added `GET /get_frequency` and `POST /set_frequency` endpoints |
| `ultrasound-raytracing/examples/templates/index.html` | Added frequency slider UI (1.0-15.0 MHz), JS wiring, auto-update on probe switch |

---

## 6. Quick Reference Commands

```bash
# SSH with port forwarding for simulator web UI
ssh -p 59290 root@213.181.122.2 -L 8080:localhost:8080 -L 9000:localhost:8000

# Run interactive simulator (access at http://localhost:9000)
sudo ./i4h run ultrasound-raytracing

# Run sphere sweep demo (no mesh needed)
sudo ./i4h run ultrasound-raytracing sphere_sweep

# Run liver sweep demo
sudo ./i4h run ultrasound-raytracing liver_sweep

# Run performance benchmark
sudo ./i4h run ultrasound-raytracing benchmark

# List available modes
sudo ./i4h modes ultrasound-raytracing

# Interactive shell in container (for debugging/experimentation)
sudo ./i4h run-container ultrasound-raytracing
```

---

## 7. VRAM Usage

Only ~3.2 GB of 24 GB used during simulation. Significant headroom for larger/denser meshes or higher resolution output.
