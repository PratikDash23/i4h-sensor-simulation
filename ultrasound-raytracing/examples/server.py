# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import io
import math
import os
import sys

# Add the root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import raysim.cuda as rs
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
from matplotlib import colormaps
from PIL import Image, ImageDraw, ImageFont

# Sentinel value emitted by the C++ scan converter for pixels outside the
# imaging region (no organ / no material).
ID_BACKGROUND = np.uint32(0xFFFFFFFF)
# Render-time alpha for the organ overlay (0=B-mode only, 1=labels only).
# Mutable so /set_sim_params can change it live from the UI slider.
overlay_alpha = 0.5

# Dynamic range (dB) for B-mode log-compression normalization.
# min_db = noise floor (clipped to black), max_db = peak (clipped to white).
min_db = -60.0
max_db = 0.0

app = Flask(__name__)
CORS(app)
mesh_dir = os.environ.get("ULTRASOUND_MESH_DIR", "mesh")


def mesh_path(filename):
    return os.path.join(mesh_dir, filename)

# Create materials and world
materials = rs.Materials()
world = rs.World("water")

material_idx = materials.get_index("fat")
liver_tumor = rs.Mesh(mesh_path("Tumor1.obj"), material_idx)
world.add(liver_tumor)
material_idx = materials.get_index("water")
liver_cyst = rs.Mesh(mesh_path("Tumor2.obj"), material_idx)
world.add(liver_cyst)
# Add liver mesh to world
material_idx = materials.get_index("liver")
liver_mesh = rs.Mesh(mesh_path("Liver.obj"), material_idx)
world.add(liver_mesh)
material_idx = materials.get_index("fat")
skin_mesh = rs.Mesh(mesh_path("Skin.obj"), material_idx)
world.add(skin_mesh)
material_idx = materials.get_index("bone")
bone_mesh = rs.Mesh(mesh_path("Bone.obj"), material_idx)
world.add(bone_mesh)
material_idx = materials.get_index("water")
vessels_mesh = rs.Mesh(mesh_path("Vessels.obj"), material_idx)
world.add(vessels_mesh)
material_idx = materials.get_index("water")
galbladder_mesh = rs.Mesh(mesh_path("Gallbladder.obj"), material_idx)
world.add(galbladder_mesh)
material_idx = materials.get_index("liver")
spleen_mesh = rs.Mesh(mesh_path("Spleen.obj"), material_idx)
world.add(spleen_mesh)
material_idx = materials.get_index("liver")
heart_mesh = rs.Mesh(mesh_path("Heart.obj"), material_idx)
world.add(heart_mesh)
material_idx = materials.get_index("water")
stomach_mesh = rs.Mesh(mesh_path("Stomach.obj"), material_idx)
world.add(stomach_mesh)
material_idx = materials.get_index("liver")
pancreas_mesh = rs.Mesh(mesh_path("Pancreas.obj"), material_idx)
world.add(pancreas_mesh)
material_idx = materials.get_index("water")
small_intestine_mesh = rs.Mesh(mesh_path("Small_bowel.obj"), material_idx)
world.add(small_intestine_mesh)
material_idx = materials.get_index("water")
large_intestine_mesh = rs.Mesh(mesh_path("Colon.obj"), material_idx)
world.add(large_intestine_mesh)

# Display names indexed by world.add() order — must mirror the calls above.
# The simulator returns per-pixel `organ_ids` whose values are GAS indices
# assigned by World::add(), so this list maps id -> human-readable label.
organ_names = [
    "Tumor1", "Tumor2", "Liver", "Skin", "Bone", "Vessels", "Gallbladder",
    "Spleen", "Heart", "Stomach", "Pancreas", "Small_bowel", "Colon",
]


def _build_organ_palette(n):
    """Return an (n, 3) uint8 RGB palette that stays discernible as N grows.

    Up to 20 organs uses matplotlib's `tab20` (a hand-tuned categorical map).
    Beyond that we fall back to evenly-spaced HSV hues which scales without
    repeating colors.
    """
    if n <= 20:
        cmap = colormaps["tab20"]
        rgba = np.asarray([cmap(i / max(19, 1)) for i in range(n)])
    else:
        cmap = colormaps["hsv"]
        rgba = np.asarray([cmap(i / n) for i in range(n)])
    return (rgba[:, :3] * 255).astype(np.uint8)


organ_palette = _build_organ_palette(len(organ_names))

# Cached drawing context just to call textbbox without allocating a new
# ImageDraw per measurement.
_LEGEND_FONT = ImageFont.load_default()
_LEGEND_MEASURE_DRAW = ImageDraw.Draw(Image.new("RGB", (1, 1)))


def _measure_text(text):
    bbox = _LEGEND_MEASURE_DRAW.textbbox((0, 0), text, font=_LEGEND_FONT)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def _compose_overlay(b_mode_uint8, organ_ids, palette, alpha=0.5):
    """Alpha-blend per-pixel organ colors on top of the grayscale B-mode image.

    Background pixels (id == UINT32_MAX) and any ids past the palette length
    pass through as plain grayscale.
    """
    gray_rgb = np.repeat(b_mode_uint8[:, :, None], 3, axis=2).astype(np.float32)
    valid = (organ_ids != ID_BACKGROUND) & (organ_ids < len(palette))
    if not valid.any():
        return gray_rgb.astype(np.uint8)
    organ_rgb = palette[organ_ids.clip(max=len(palette) - 1)].astype(np.float32)
    blended = alpha * organ_rgb + (1.0 - alpha) * gray_rgb
    out = gray_rgb.copy()
    out[valid] = blended[valid]
    return out.astype(np.uint8)


def _render_legend_bottom(width, names, palette):
    """Multi-row legend strip sized to a given image width."""
    swatch = 12
    inner_gap = 4
    entry_gap = 12
    row_h = 18
    pad = 6

    entry_widths = [
        swatch + inner_gap + _measure_text(name)[0] + entry_gap
        for name in names
    ]
    rows, current, current_w = [], [], 0
    for i, ew in enumerate(entry_widths):
        if current and current_w + ew > width - 2 * pad:
            rows.append(current)
            current, current_w = [], 0
        current.append(i)
        current_w += ew
    if current:
        rows.append(current)

    legend_h = pad * 2 + row_h * len(rows)
    img = Image.new("RGB", (width, legend_h), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    y = pad
    for row in rows:
        x = pad
        for i in row:
            r, g, b = (int(c) for c in palette[i])
            draw.rectangle([x, y + 2, x + swatch, y + 2 + swatch],
                           fill=(r, g, b), outline=(0, 0, 0))
            draw.text((x + swatch + inner_gap, y + 1), names[i],
                      fill=(0, 0, 0), font=_LEGEND_FONT)
            x += swatch + inner_gap + _measure_text(names[i])[0] + entry_gap
        y += row_h
    return img


def _render_legend_right(height, names, palette, width=150):
    """Vertical legend strip of fixed width (default 150 px)."""
    swatch = 12
    inner_gap = 6
    row_h = 18
    pad = 6
    img = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    y = pad
    for i, name in enumerate(names):
        if y + row_h > height - pad:
            break  # truncate silently if image is too short for all entries
        r, g, b = (int(c) for c in palette[i])
        draw.rectangle([pad, y + 2, pad + swatch, y + 2 + swatch],
                       fill=(r, g, b), outline=(0, 0, 0))
        draw.text((pad + swatch + inner_gap, y + 1), name,
                  fill=(0, 0, 0), font=_LEGEND_FONT)
        y += row_h
    return img


def _attach_legend(image_rgb, names, palette, layout="bottom"):
    h, w = image_rgb.shape[:2]
    base = Image.fromarray(image_rgb, mode="RGB")
    if layout == "right":
        legend = _render_legend_right(h, names, palette)
        canvas = Image.new("RGB", (w + legend.width, h), (255, 255, 255))
        canvas.paste(base, (0, 0))
        canvas.paste(legend, (w, 0))
    else:
        legend = _render_legend_bottom(w, names, palette)
        canvas = Image.new("RGB", (w, h + legend.height), (255, 255, 255))
        canvas.paste(base, (0, 0))
        canvas.paste(legend, (0, h))
    return canvas


# ---------------------------------------------------------------------------
# Depth scale (vertical cm strip on the LEFT of the cone)
# ---------------------------------------------------------------------------
# The mapping below mirrors the scan-converter math in
# csrc/cuda/cuda_algorithms.cu :: scan_convert_curvilinear_kernel.
# For a curvilinear probe with inner radius `near_mm` and full sector angle a:
#     far_mm   = t_far_mm + near_mm           (outer radius of the sector)
#     offset_z = cos(a/2) * (near_mm / far_mm)
# A pixel on the image's center column (x=0) at output-pixel y_idx in [0, H-1]
# maps to a normalized radial coord:
#     coord_y(y_idx) = offset_z + (y_idx / H) * (1 - offset_z)
# and to a depth-from-probe-surface in mm:
#     depth_mm = coord_y * far_mm - near_mm
# Inverting: pixel y for a target depth d_mm:
#     y_idx = ((d_mm + near_mm) / far_mm - offset_z) / (1 - offset_z) * H
# When near_mm == 0 (linear / phased), this reduces to a linear
# y_idx = d_mm / t_far_mm * H.
def _depth_to_pixel_y(d_mm, cone_h_px, t_far_mm, near_mm, sector_deg):
    if near_mm <= 0.0:
        return int(round(d_mm / t_far_mm * cone_h_px))
    far_mm = t_far_mm + near_mm
    offset_z = math.cos(math.radians(sector_deg) / 2.0) * (near_mm / far_mm)
    coord_y = (d_mm + near_mm) / far_mm
    return int(round((coord_y - offset_z) / (1.0 - offset_z) * cone_h_px))


def _render_depth_scale_left(cone_h_px, total_h_px, t_far_mm, near_mm, sector_deg,
                             width=44):
    """Vertical cm-tick strip aligned with the cone's center-column depth axis.

    White ticks/labels on a black background, matching the cone's outside-FOV color.
    """
    img = Image.new("RGB", (width, total_h_px), (0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Right-edge baseline (touches the image's left edge once pasted).
    draw.line([(width - 1, 0), (width - 1, cone_h_px - 1)],
              fill=(255, 255, 255), width=1)

    t_far_cm = int(t_far_mm // 10)  # number of full cm marks (e.g. 18 for 180 mm)
    for cm in range(0, t_far_cm + 1):
        y = _depth_to_pixel_y(cm * 10.0, cone_h_px, t_far_mm, near_mm, sector_deg)
        if not (0 <= y < cone_h_px):
            continue
        # Major tick every 5 cm; minor every 1 cm.
        major = (cm % 5 == 0)
        tick_len = 12 if major else 6
        line_w = 2 if major else 1
        draw.line([(width - 1 - tick_len, y), (width - 1, y)],
                  fill=(255, 255, 255), width=line_w)
        if major:
            label = str(cm)
            tw, th = _measure_text(label)
            # Right-justified label, vertically centered on the tick.
            tx = max(2, width - 1 - tick_len - 2 - tw)
            ty = max(0, min(cone_h_px - th - 1, y - th // 2))
            draw.text((tx, ty), label, fill=(255, 255, 255), font=_LEGEND_FONT)

    # Units label sits in the legend strip area below the cone.
    if total_h_px > cone_h_px + 4:
        draw.text((4, cone_h_px + 4), "cm", fill=(200, 200, 200), font=_LEGEND_FONT)
    return img


def _attach_depth_scale(canvas, cone_h_px, t_far_mm, near_mm, sector_deg):
    """Prepend the depth-scale strip to the left of an already-composed canvas."""
    scale = _render_depth_scale_left(cone_h_px, canvas.height, t_far_mm,
                                     near_mm, sector_deg)
    out = Image.new("RGB", (scale.width + canvas.width, canvas.height),
                    (255, 255, 255))
    out.paste(scale, (0, 0))
    out.paste(canvas, (scale.width, 0))
    return out


# Initial poses for different probe types
initial_poses = {
    "curvilinear": rs.Pose(
        np.array([-14, -122, 72], dtype=np.float32),  # position (x, y, z)
        np.array([np.deg2rad(-90), np.deg2rad(180), np.deg2rad(0)], dtype=np.float32),
    ),
    "linear": rs.Pose(
        np.array([-14, -122, 72], dtype=np.float32),  # position (x, y, z)
        np.array([np.deg2rad(-90), np.deg2rad(180), np.deg2rad(0)], dtype=np.float32),
    ),
    "phased": rs.Pose(
        np.array([-14, -122, 72], dtype=np.float32),  # position (x, y, z)
        np.array([np.deg2rad(-90), np.deg2rad(180), np.deg2rad(0)], dtype=np.float32),
    ),
}

# Create probes with different geometries
probes = {
    "curvilinear": rs.CurvilinearProbe(  # Original curvilinear probe
        initial_poses["curvilinear"],
        num_elements_x=256,  # Number rays which represent elements
        sector_angle=73.0,  # Field of view in degrees
        radius=45.0,  # probe radius in mm
        frequency=5.0,  # probe frequency in MHz
        elevational_height=7.0,  # probe elevational aperture height in mm
        num_el_samples=10,  # number of samples in elevational direction (default is 1)
    ),
    "linear": rs.LinearArrayProbe(  # Linear array probe
        initial_poses["linear"],
        num_elements_x=256,  # Number of elements
        width=50.0,  # Width of the array in mm
        frequency=7.5,  # probe frequency in MHz
        elevational_height=5.0,  # probe elevational aperture height in mm
        num_el_samples=10,  # number of samples in elevational direction
    ),
    "phased": rs.PhasedArrayProbe(  # Phased array probe
        initial_poses["phased"],
        num_elements_x=128,  # Number of elements
        width=20.0,  # Width of the array in mm
        sector_angle=90.0,  # Full sector angle in degrees
        frequency=3.5,  # probe frequency in MHz
        elevational_height=5.0,  # probe elevational aperture height in mm
        num_el_samples=10,  # number of samples in elevational direction
    ),
}

# Probe geometry mirror (Python doesn't have getters for radius / sector_angle).
# Used by the depth-scale renderer to compute the depth-to-pixel mapping that
# matches the scan-converter (csrc/cuda/cuda_algorithms.cu :: scan_convert_*).
# 'near_mm'   : 0 for linear/phased; probe radius for curvilinear.
# 'sector_deg': 0 for linear; sector full angle in degrees for curvilinear/phased.
PROBE_GEOM = {
    "curvilinear": {"near_mm": 45.0, "sector_deg": 73.0},
    "linear":      {"near_mm":  0.0, "sector_deg":  0.0},
    "phased":      {"near_mm":  0.0, "sector_deg": 90.0},
}

# Current active probe
active_probe = "curvilinear"

# Create simulator
simulator = rs.RaytracingUltrasoundSimulator(world, materials)

# Configure simulation parameters
sim_params = rs.SimParams()
sim_params.conv_psf = True
sim_params.buffer_size = 4096
sim_params.t_far = 180.0
sim_params.enable_cuda_timing = True
sim_params.median_clip_filter = False
# Speed-of-sound–aware echo placement. Default OFF preserves NVIDIA's original
# geometric-binning behaviour (image lines up perfectly with mesh geometry).
# Toggle ON via the UI checkbox to reproduce real-scanner SoS aberration.
# See `simulated_US/sos_aware_echo_placement.md` for the full design.
sim_params.sos_aware = False
sim_params.assumed_sos = 1540.0  # m/s — the scanner's TOF→displayed-depth assumption

# Render-time toggle: when False, /simulate returns the plain grayscale B-mode
# without the organ color overlay or legend. Not part of the C++ SimParams since
# it only affects PNG composition in this server.
show_organ_overlay = True


@app.route("/")
def home():
    return send_file("templates/index.html")


@app.route("/get_probe_types", methods=["GET"])
def get_probe_types():
    # Return list of available probe types
    return jsonify(list(probes.keys()))


@app.route("/set_probe_type", methods=["POST"])
def set_probe_type():
    global active_probe
    probe_type = request.json["probe_type"]
    if probe_type in probes:
        active_probe = probe_type
        return {"status": "success", "probe_type": active_probe}
    else:
        return {"status": "error", "message": f"Unknown probe type: {probe_type}"}, 400


@app.route("/get_initial_pose", methods=["GET"])
def get_initial_pose():
    current_pose = probes[active_probe].get_pose()
    # Convert the pose to a list for JSON serialization
    pose_list = current_pose.position.tolist() + current_pose.rotation.tolist()
    return {"pose": pose_list, "probe_type": active_probe}


@app.route("/get_sim_params", methods=["GET"])
def get_sim_params():
    """Get current simulation parameters"""
    return {
        "median_clip_filter": sim_params.median_clip_filter,
        "enable_cuda_timing": sim_params.enable_cuda_timing,
        "write_debug_images": sim_params.write_debug_images,
        "contact_epsilon": sim_params.contact_epsilon,
        "t_far": sim_params.t_far,
        "show_organ_overlay": show_organ_overlay,
        "overlay_alpha": overlay_alpha,
        "min_db": min_db,
        "max_db": max_db,
        "sos_aware": sim_params.sos_aware,
        "assumed_sos": sim_params.assumed_sos,
    }


@app.route("/set_sim_params", methods=["POST"])
def set_sim_params():
    """Update simulation parameters"""
    global show_organ_overlay, overlay_alpha, min_db, max_db
    try:
        params = request.json

        if "median_clip_filter" in params:
            sim_params.median_clip_filter = bool(params["median_clip_filter"])

        if "enable_cuda_timing" in params:
            sim_params.enable_cuda_timing = bool(params["enable_cuda_timing"])

        if "write_debug_images" in params:
            sim_params.write_debug_images = bool(params["write_debug_images"])

        if "contact_epsilon" in params:
            sim_params.contact_epsilon = float(params["contact_epsilon"])
        if "t_far" in params:
            # Clamp to [50, 300] mm as a defence-in-depth check beyond the UI input.
            sim_params.t_far = float(np.clip(float(params["t_far"]), 50.0, 300.0))

        if "show_organ_overlay" in params:
            show_organ_overlay = bool(params["show_organ_overlay"])

        if "overlay_alpha" in params:
            overlay_alpha = float(np.clip(params["overlay_alpha"], 0.0, 1.0))

        if "min_db" in params:
            min_db = float(params["min_db"])

        if "max_db" in params:
            max_db = float(params["max_db"])

        # SoS-aware echo placement (see SimParams.sos_aware in
        # raytracing_ultrasound_simulator.hpp). Toggling at runtime is safe — the
        # next call to simulate() will copy the updated SimParams into the OptiX
        # `Params` struct via raytracing_ultrasound_simulator.cpp. No rebuild needed.
        if "sos_aware" in params:
            sim_params.sos_aware = bool(params["sos_aware"])

        if "assumed_sos" in params:
            # Clamp to a physically sensible band; UI slider already enforces this
            # but we don't want a bad request to push a nonsense value into OptiX.
            sim_params.assumed_sos = float(np.clip(params["assumed_sos"], 1300.0, 1700.0))

        return {
            "status": "success",
            "params": {
                "median_clip_filter": sim_params.median_clip_filter,
                "enable_cuda_timing": sim_params.enable_cuda_timing,
                "write_debug_images": sim_params.write_debug_images,
                "contact_epsilon": sim_params.contact_epsilon,
        "t_far": sim_params.t_far,
                "show_organ_overlay": show_organ_overlay,
                "overlay_alpha": overlay_alpha,
                "min_db": min_db,
                "max_db": max_db,
                "sos_aware": sim_params.sos_aware,
                "assumed_sos": sim_params.assumed_sos,
            }
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}, 400


@app.route("/get_frequency", methods=["GET"])
def get_frequency():
    """Get the current probe frequency in MHz"""
    return {"frequency": probes[active_probe].get_frequency()}


@app.route("/set_frequency", methods=["POST"])
def set_frequency():
    """Set the probe frequency in MHz"""
    try:
        freq = float(request.json["frequency"])
        if freq <= 0:
            return {"status": "error", "message": "Frequency must be positive"}, 400
        probes[active_probe].set_frequency(freq)
        return {"status": "success", "frequency": probes[active_probe].get_frequency()}
    except Exception as e:
        return {"status": "error", "message": str(e)}, 400


@app.route("/simulate", methods=["POST"])
def simulate():
    pose_delta = request.json["pose_delta"]
    # `?layout=right` lays the legend to the right (+150 px) for save/download;
    # default `bottom` keeps the image width the same for the interactive UI.
    layout = request.args.get("layout", "bottom")
    print(f"Applying delta: {pose_delta}")

    # Get current pose
    current_pose = probes[active_probe].get_pose()

    # Apply deltas to current pose
    position_delta = np.array(pose_delta[0:3], dtype=np.float32)
    rotation_delta = np.array(pose_delta[3:6], dtype=np.float32)

    # Calculate new pose by adding deltas
    new_position = current_pose.position + position_delta
    new_rotation = current_pose.rotation + rotation_delta

    # Set new pose
    new_pose = rs.Pose(new_position, new_rotation)
    probes[active_probe].set_pose(new_pose)

    # The simulator returns three (H, W) arrays: the float32 dB-clipped B-mode
    # image and two uint32 categorical id maps (organ + material). We use the
    # organ ids for the visible overlay; material ids are kept unused here but
    # available for downstream consumers.
    b_mode_image, organ_ids, _material_ids = simulator.simulate(
        probes[active_probe], sim_params)

    # Apply dB-window normalization. Bounds come from the live sliders so the
    # user can widen the floor (more shadow detail) or compress the window
    # (more contrast) without restarting the server.
    normalized_image = np.clip((b_mode_image - min_db) / (max_db - min_db), 0, 1)

    # Convert to 8-bit grayscale. With the overlay enabled, alpha-blend the
    # organ map on top and append the legend; otherwise return plain grayscale.
    img_uint8 = (normalized_image * 255).astype(np.uint8)
    cone_h_px = img_uint8.shape[0]
    if show_organ_overlay:
        overlay_rgb = _compose_overlay(img_uint8, organ_ids, organ_palette, alpha=overlay_alpha)
        composite = _attach_legend(overlay_rgb, organ_names, organ_palette, layout=layout)
    else:
        gray_rgb = np.repeat(img_uint8[:, :, None], 3, axis=2)
        composite = Image.fromarray(gray_rgb, mode="RGB")

    # Vertical cm depth scale on the left, aligned to the cone's center column.
    geom = PROBE_GEOM.get(active_probe, {"near_mm": 0.0, "sector_deg": 0.0})
    composite = _attach_depth_scale(composite, cone_h_px,
                                    sim_params.t_far,
                                    geom["near_mm"], geom["sector_deg"])

    img_io = io.BytesIO()
    composite.save(img_io, "PNG")
    img_io.seek(0)

    # Return the image and the current probe type
    return send_file(img_io, mimetype="image/png")


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000, debug=True)
