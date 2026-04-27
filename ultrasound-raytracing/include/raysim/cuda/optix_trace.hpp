/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef CPP_OPTIX_TRACE
#define CPP_OPTIX_TRACE

#include <optix.h>
#include <cstdint>

#include "raysim/core/material.hpp"
#include "raysim/core/probe_types.hpp"
#include "raysim/cuda/matrix.hpp"

namespace raysim {
struct Params {
  float* scanlines;
  uint32_t* organ_ids;     // Parallel buffer to `scanlines` (uint32 per depth-bin), sentinel UINT32_MAX
  uint32_t* material_ids;  // Parallel buffer to `scanlines` (uint32 per depth-bin), sentinel UINT32_MAX
  uint32_t buffer_size;
  float t_far;
  float min_intensity;
  uint32_t max_depth;
  Material* materials;
  uint32_t background_material_id;
  cudaTextureObject_t scattering_texture;
  OptixTraversableHandle handle;
  float source_frequency;
  float contact_epsilon;
  // -- Speed-of-sound–aware echo placement -------------------------------------------------------
  // When `sos_aware == false` (default), echoes are binned by *geometric* ray distance — the
  // legacy/NVIDIA behaviour, perfectly aligned with mesh geometry, ideal for ML ground-truth.
  // When `sos_aware == true`, echoes are binned by *displayed depth* derived from time-of-flight:
  //     displayed_depth_mm = assumed_sos[m/s] * tof[s] / 2
  //                        = assumed_sos      * tof_us * 5e-4
  // This reproduces the speed-of-sound aberration a real B-mode scanner shows when the actual
  // tissue speed of sound differs from `assumed_sos` (the scanner's hardcoded TOF→depth assumption,
  // conventionally 1540 m/s).
  bool sos_aware;
  float assumed_sos;  // m/s
  // ----------------------------------------------------------------------------------------------
};

struct RayGenData {
  int probe_type;            // Type of probe (curvilinear, linear, phased)
  float sector_angle;        // Field of view in degrees (sector angle for phased array)
  float elevational_height;  // Height in elevational direction in mm
  float radius;              // Radius of curvature in mm (for curvilinear)
  float width;               // Width of linear/phased array in mm
  float3 position;           // Probe position in world coordinates
  float33 rotation_matrix;   // Probe orientation in world coordinates
};

struct MissData {};

struct HitGroupData {
  uint32_t material_id;
  uint32_t* indices;
  float3* normals;
};

struct Payload {
  float intensity;
  uint32_t depth;
  // Geometric path length accumulated by all ancestor segments [mm]. Used by:
  //   (a) Beer-Lambert attenuation (depends on physical distance through tissue), and
  //   (b) the OptiX `tmax` cap when spawning child rays (the scene is bounded in mm,
  //       not in TOF — see `params.t_far - ray.t_ancestors` in optix_trace.cu).
  // Always populated, irrespective of `params.sos_aware`.
  float t_ancestors;
  // Time of flight accumulated by all ancestor segments [microseconds]. Only meaningful
  // when `params.sos_aware == true` — used to compute the displayed-depth bin via
  //     displayed_depth_mm = params.assumed_sos * tof_ancestors_us * 5e-4
  // Each segment contributes (segment_distance_mm * 1e-3 / segment_c_mps) * 1e6 us.
  float tof_ancestors;
  // use 16 bit for object and material ID to safe space
  uint16_t current_obj_id;
  uint16_t outter_obj_id;
  uint16_t current_material_id;
  uint16_t outter_material_id;
};

}  // namespace raysim

#endif /* CPP_OPTIX_TRACE */
