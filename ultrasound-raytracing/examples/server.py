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
OVERLAY_ALPHA = 0.5

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


def _compose_overlay(b_mode_uint8, organ_ids, palette, alpha=OVERLAY_ALPHA):
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
        "show_organ_overlay": show_organ_overlay,
    }


@app.route("/set_sim_params", methods=["POST"])
def set_sim_params():
    """Update simulation parameters"""
    global show_organ_overlay
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

        if "show_organ_overlay" in params:
            show_organ_overlay = bool(params["show_organ_overlay"])

        return {
            "status": "success",
            "params": {
                "median_clip_filter": sim_params.median_clip_filter,
                "enable_cuda_timing": sim_params.enable_cuda_timing,
                "write_debug_images": sim_params.write_debug_images,
                "contact_epsilon": sim_params.contact_epsilon,
                "show_organ_overlay": show_organ_overlay,
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

    # Apply normalization as in C++ code
    min_val = -60.0  # Matching C++ min_max.x
    max_val = 0.0  # Matching C++ min_max.y
    normalized_image = np.clip((b_mode_image - min_val) / (max_val - min_val), 0, 1)

    # Convert to 8-bit grayscale. With the overlay enabled, alpha-blend the
    # organ map on top and append the legend; otherwise return plain grayscale.
    img_uint8 = (normalized_image * 255).astype(np.uint8)
    if show_organ_overlay:
        overlay_rgb = _compose_overlay(img_uint8, organ_ids, organ_palette)
        composite = _attach_legend(overlay_rgb, organ_names, organ_palette, layout=layout)
    else:
        gray_rgb = np.repeat(img_uint8[:, :, None], 3, axis=2)
        composite = Image.fromarray(gray_rgb, mode="RGB")

    img_io = io.BytesIO()
    composite.save(img_io, "PNG")
    img_io.seek(0)

    # Return the image and the current probe type
    return send_file(img_io, mimetype="image/png")


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000, debug=True)
