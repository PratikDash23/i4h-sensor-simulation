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

#include <optix.h>

#include <cuda/helpers.h>
#include <sutil/vec_math.h>

#include "raysim/cuda/optix_trace.hpp"

#include <OptiXToolkit/ShaderUtil/OptixSelfIntersectionAvoidance.h>

namespace raysim {

extern "C" {
__constant__ Params params;
}

static __forceinline__ __device__ Payload get_payload() {
  Payload payload;
  // Payload grew from 5→6 uint32 words when SoS-aware echo placement added `tof_ancestors`
  // (see optix_trace.hpp). Every optixTrace site below + the pipeline's auto-derived
  // `numPayloadValues` (optix_helper.cpp:100) must agree on this count.
  static_assert(sizeof(Payload) / sizeof(uint32_t) == 6);
  reinterpret_cast<uint32_t*>(&payload)[0] = optixGetPayload_0();
  reinterpret_cast<uint32_t*>(&payload)[1] = optixGetPayload_1();
  reinterpret_cast<uint32_t*>(&payload)[2] = optixGetPayload_2();
  reinterpret_cast<uint32_t*>(&payload)[3] = optixGetPayload_3();
  reinterpret_cast<uint32_t*>(&payload)[4] = optixGetPayload_4();
  reinterpret_cast<uint32_t*>(&payload)[5] = optixGetPayload_5();
  return payload;
}

static __device__ float get_scattering_value(float3 pos, const Material* material,
                                             float resolution_mm = 50.f) {
  // Convert point to texture coordinates
  pos /= resolution_mm;

  const float2 scatter_val = tex3D<float2>(params.scattering_texture, pos.x, pos.y, pos.z);

  // Apply material properties
  if (scatter_val.x <= material->mu0_) { return scatter_val.y * material->sigma_; }
  return 0.f;
}

/**
 * Return the scanline-buffer bin index for an echo at *geometric* ray distance `t` [mm].
 *
 * This is the legacy NVIDIA placement: bin = (t / t_far) * (buffer_size - 1).
 * Used unconditionally when `params.sos_aware == false`. With SoS-aware mode on, the bin
 * is computed by `offset_from_tof` instead (below) and this helper is bypassed.
 *
 * @return offset to intensity buffer at ray distance t
 */
static __device__ uint32_t get_intensity_offset(float t) {
  uint32_t offset = uint32_t((t / params.t_far) * (params.buffer_size - 1) + 0.5f);
  assert(offset < params.buffer_size);
  return offset;
}

/**
 * Return the scanline-buffer bin index for an echo at accumulated time-of-flight `tof_us` [μs].
 *
 * Mirrors what a real B-mode scanner does: it has no access to ground-truth geometry, only to
 * arrival times, so it converts time → "displayed depth" by assuming ONE speed of sound for
 * everything (`params.assumed_sos`, conventionally 1540 m/s).
 *
 * The real scanner sees ROUND-TRIP TOF and halves it: d_displayed = c_assumed * t_round_trip / 2.
 * Our `tof_ancestors` is already ONE-WAY (probe → scatter), so the /2 is already absorbed.
 * For a scatter at one-way distance d through material c_real:
 *     t_oneway       = d / c_real
 *     d_displayed_m  = c_assumed * t_oneway       (matches what the scanner shows)
 *
 * Unit conversion (m/s, μs → mm):
 *     displayed_mm = assumed_sos[m/s] * tof_us[μs] * 1e-6[s/μs] * 1e3[mm/m]
 *                  = assumed_sos      * tof_us      * 1e-3
 *
 * Sanity check: if c_real == c_assumed, t_oneway = d/c, so displayed = c * (d/c) = d. Identity.
 *
 * Slow tissue (e.g. fat at c=1450) can push `displayed_depth_mm` past `t_far` for the deepest
 * echoes — that is the correct displayed-depth math, NOT a bug, so we clip to the last bin
 * rather than asserting (assert would fire on perfectly valid inputs).
 *
 * Only used when `params.sos_aware == true`.
 */
static __device__ uint32_t offset_from_tof(float tof_us) {
  const float displayed_mm = params.assumed_sos * tof_us * 1e-3f;
  if (displayed_mm >= params.t_far) { return params.buffer_size - 1; }
  return uint32_t((displayed_mm / params.t_far) * (params.buffer_size - 1) + 0.5f);
}

/**
 * Convert a geometric distance increment to an additional time-of-flight [μs].
 *
 *     tof_us = (distance_mm * 1e-3 / c_mps) * 1e6
 *            = distance_mm / c_mps * 1e3
 *
 * Pulled into a helper to keep the unit conversions in exactly one place — every callsite
 * that adds to `tof_ancestors` should go through this so we never accidentally use the
 * wrong unit (mm vs cm, m/s vs mm/μs, etc).
 */
static __device__ float distance_to_tof_us(float distance_mm, float c_mps) {
  return distance_mm * 1e3f / c_mps;
}

/// Calculate intensity using Beer-Lambert Law
static __device__ float get_intensity_at_distance(float distance, float medium_attenuation) {
  // I = I₀ * 10^(-αfd/20)
  // where α is attenuation coefficient in dB/(cm⋅MHz)
  // f is frequency in MHz
  // d is distance in cm
  const float source_freq = params.source_frequency;                            // MHz
  const float distance_cm = distance * 0.1f;                                    // Convert to cm
  const float attenuation_db = medium_attenuation * source_freq * distance_cm;  // dB
  return __powf(10.f, -attenuation_db * 0.05f);
}

/**
 * Sample intensities along the current ray segment, depositing speckle (and per-pixel
 * organ/material labels) into the scanline buffer.
 *
 * Two binning modes:
 *
 *   `params.sos_aware == false` (legacy): bin by *geometric distance*.
 *     - One `base_offset = (t_ancestors + t_min) → bin` computed once.
 *     - Step n lands at `base_offset + n` since the loop's `t_step` is sized so that
 *       one step exactly equals one bin width in the geometric metric.
 *
 *   `params.sos_aware == true`: bin by *displayed depth* derived from time-of-flight.
 *     - One `base_tof_us = tof_ancestors + tof(t_min in this material)` computed once.
 *     - Each step advances TOF by the constant `tof_step_us = tof(t_step in this material)`,
 *       and the bin is `offset_from_tof(base_tof_us + step * tof_step_us)`.
 *     - We do NOT pre-advance the buffer pointer because consecutive TOF steps generally
 *       map to non-consecutive bins (the bin width varies with the local SoS).
 *
 * The Beer-Lambert intensity calculation is *unchanged* — it depends on physical distance
 * through the medium, not on the binning convention. Only *where* the echo is written moves.
 *
 * @param origin ray origin in world space
 * @param dir ray direction in world space
 * @param t_ancestors accumulated geometric ray length from prior segments [mm]
 * @param tof_ancestors accumulated time-of-flight from prior segments [μs] (only consulted
 *                      when `params.sos_aware == true`; pass 0 in legacy mode)
 * @param t_min start of the current segment along the ray [mm]
 * @param t_max end of the current segment along the ray [mm]
 * @param intensity ray intensity carried into this segment
 * @param material material the ray is currently traveling through
 * @param organ_id current organ id (= GAS index in world add() order); stamped
 *                 into every depth bin this loop touches so speckle pixels
 *                 carry a label
 * @param material_id current material id; stamped into the parallel buffer
 * @param intensities scanline scattering buffer (NOT pre-advanced — this fn writes via
 *                    `intensities[bin]` so the same code works for both binning modes)
 * @param organ_ids_out parallel uint32 buffer for organ labels (same layout
 *                      as `intensities`). Boundary writes from the caller
 *                      naturally overwrite the boundary's bin afterwards.
 * @param material_ids_out parallel uint32 buffer for material labels
 */
static __device__ void sample_intensities(float3 origin, float3 dir, float t_ancestors,
                                          float tof_ancestors, float t_min, float t_max,
                                          float intensity, const Material* material,
                                          uint32_t organ_id, uint32_t material_id,
                                          float* intensities, uint32_t* organ_ids_out,
                                          uint32_t* material_ids_out) {
  // Materials with zero scattering produce no echo, but we still want every depth
  // bin in this segment to carry a label (otherwise water/blood interior would
  // read as background sentinel). Fall through to a label-only loop in that case.
  const bool skip_scatter = (material->mu0_ <= 0.f) || (material->sigma_ == 0.f);

  // In SoS-aware mode, one geometric step maps to (assumed_sos / c_material) bins
  // in displayed depth. For slow materials (c < assumed_sos) that ratio > 1, so
  // the OFF-mode step density would skip ~1 bin in (ratio) — visible as black
  // "no-echo" stripes inside fat-like regions. Inflate the step count by that
  // ratio so every displayed-depth bin gets at least one write. The per-step
  // speckle contribution is then divided by the same factor to keep the
  // accumulated bin intensity comparable to OFF mode.
  const float c_mps = material->speed_of_sound_;
  const float density_factor = (params.sos_aware && params.assumed_sos > c_mps)
                               ? (params.assumed_sos / c_mps) : 1.0f;
  const uint32_t steps = ((t_max - t_min) / params.t_far) * params.buffer_size
                         * density_factor + 0.5f;
  if (steps == 0) { return; }
  const float t_step = (t_max - t_min) / steps;
  const float3 start = origin + t_min * dir;

  // Pre-compute per-mode binning constants once outside the loop.
  // Legacy (geometric): bin index of the segment's first sample.
  const uint32_t base_offset_geom = get_intensity_offset(t_ancestors + t_min);
  // SoS-aware (TOF): TOF at the segment's first sample, plus per-step TOF increment.
  const float base_tof_us = tof_ancestors + distance_to_tof_us(t_min, c_mps);
  const float tof_step_us = distance_to_tof_us(t_step, c_mps);

  // Resolve a step index → scanline bin index. We hoist the SoS-aware branch out of
  // the inner loop so the loop body itself stays branch-free; the two loops below
  // share their compute except for this one line.
  // (Originally written as a __device__ lambda; the project does not enable
  // nvcc --extended-lambda, so we expand the two cases inline.)

  if (skip_scatter) {
    if (params.sos_aware) {
      for (uint32_t step = 0; step < steps; ++step) {
        const uint32_t bin = offset_from_tof(base_tof_us + step * tof_step_us);
        organ_ids_out[bin] = organ_id;
        material_ids_out[bin] = material_id;
      }
    } else {
      for (uint32_t step = 0; step < steps; ++step) {
        const uint32_t bin = base_offset_geom + step;
        organ_ids_out[bin] = organ_id;
        material_ids_out[bin] = material_id;
      }
    }
    return;
  }

  if (params.sos_aware) {
    // density_factor > 1 in slow materials; per-step contribution scaled down
    // by the same factor so the per-bin sum matches what OFF mode produces.
    const float inv_density = 1.0f / density_factor;
    for (uint32_t step = 0; step < steps; ++step) {
      const float distance = (step * t_step);  // geometric mm into this segment
      const float3 pos = start + distance * dir;
      const uint32_t bin = offset_from_tof(base_tof_us + step * tof_step_us);
      // Beer-Lambert is distance-based (physics doesn't change with binning convention).
      intensities[bin] += get_scattering_value(pos, material) * intensity *
                          get_intensity_at_distance(distance, material->attenuation_)
                          * inv_density;
      organ_ids_out[bin] = organ_id;
      material_ids_out[bin] = material_id;
    }
  } else {
    for (uint32_t step = 0; step < steps; ++step) {
      const float distance = (step * t_step);  // geometric mm into this segment
      const float3 pos = start + distance * dir;
      const uint32_t bin = base_offset_geom + step;
      // Beer-Lambert is distance-based (physics doesn't change with binning convention).
      intensities[bin] += get_scattering_value(pos, material) * intensity *
                          get_intensity_at_distance(distance, material->attenuation_);
      organ_ids_out[bin] = organ_id;
      material_ids_out[bin] = material_id;
    }
  }
}

template <OptixPrimitiveType PRIM_TYPE>
static __device__ float3 get_normal(float3 ray_orig, float3 ray_dir, float t_hit, uint32_t hit_id,
                                    const HitGroupData* hit_group_data) {
  const unsigned int prim_idx = optixGetPrimitiveIndex();

  float3 normal;
  if constexpr (PRIM_TYPE == OptixPrimitiveType::OPTIX_PRIMITIVE_TYPE_TRIANGLE) {
    uint32_t* const tri = &hit_group_data->indices[prim_idx * 3];
    const float3 N0 = hit_group_data->normals[tri[0]];
    const float3 N1 = hit_group_data->normals[tri[1]];
    const float3 N2 = hit_group_data->normals[tri[2]];

    const float2 barys = optixGetTriangleBarycentrics();

    normal = (1.f - barys.x - barys.y) * N0 + barys.x * N1 + barys.y * N2;
  } else {
    assert(PRIM_TYPE == OptixPrimitiveType::OPTIX_PRIMITIVE_TYPE_SPHERE);
    const OptixTraversableHandle gas = optixGetGASTraversableHandle();
    float4 q;
    // sphere center (q.x, q.y, q.z), sphere radius q.w
    optixGetSphereData(gas, prim_idx, hit_id, 0.f, &q);

    const float3 raypos = ray_orig + t_hit * ray_dir;
    normal = (raypos - make_float3(q)) / q.w;
  }
  return normal;
}

/**
 * Calculate reflected direction vector
 */
static __device__ float3 calc_reflected_dir(float3 incident_dir, float3 normal) {
  // Ensure normal points against incident direction
  if (dot(incident_dir, normal) > 0.f) { normal = -normal; }
  return incident_dir - 2.f * dot(incident_dir, normal) * normal;
}

/**
 * Calculate reflection coefficient using acoustic impedance
 */
static __device__ float calculate_reflection_coefficient(float incident_angle,
                                                         const Material* material1,
                                                         const Material* material2) {
  float Z1 = material1->impedance_;
  float Z2 = material2->impedance_;
  float cos_theta = fabsf(__cosf(incident_angle));
  float R = ((Z2 * cos_theta - Z1) / (Z2 * cos_theta + Z1));
  return R * R;
}

/**
 * Calculate refracted direction vector using Snell's law
 *
 * @returns true for total internal reflection
 */
static __device__ bool calc_refracted_dir(float3 incident_dir, float3 normal, float v1, float v2,
                                          float3* refracted_dir) {
  // Ensure normal points against incident direction
  if (dot(incident_dir, normal) > 0.f) { normal = -normal; }

  float cos_i = -dot(normal, incident_dir);  // Incidence
  float sin_i = sqrtf(1.f - cos_i * cos_i);  // Incidence
  float sin_t = (v1 / v2) * sin_i;           // Transmission

  // Check for total internal reflection
  if (sin_t >= 1) { return true; }

  float cos_t = sqrtf(1 - sin_t * sin_t);
  *refracted_dir = (v1 / v2) * incident_dir + ((v1 / v2) * cos_i - cos_t) * normal;
  return false;
}

/**
 * Calculate the reflection intensity Ir for ultrasound RF image
 *
 * Eq. 5 Mattausch2016Monte-Carlo
 *
 * @param V_r reflected ray direction vector
 * @param V_i refracted ray direction vector
 * @param total_internal_reflection
 * @param D vector from intersection point to transducer origin
 * @param n surface specularity parameter
 * @return reflection intensity Ir
 */
static __device__ float calculate_specular_intensity(float3 V_r, float3 V_i,
                                                     bool total_internal_reflection, float3 D,
                                                     float n) {
  // Calculate angles using dot product
  float cos_reflected = dot(V_r, D) / (length(V_r) * length(D));
  float cos_refracted;
  if (!total_internal_reflection) {
    cos_refracted = dot(V_i, D) / (length(V_i) * length(D));
  } else {
    cos_refracted = 0.f;
  }

  // Calculate the two terms (reflection and refraction)
  float reflected_term = max(0.f, __powf(cos_reflected, n));
  float refracted_term = max(0.f, __powf(cos_refracted, n));

  // Total intensity is sum of both terms
  float Ir = reflected_term + refracted_term;

  return Ir;
}

/**
 * Self-intersection avoidance. Get the save front and back start points.
 * See https://github.com/NVIDIA/optix-toolkit/tree/master/ShaderUtil#self-intersection-avoidance.
 *
 * @param out_front_start [out] offset spawn point on the front of the surface, safe from self
 * intersection
 * @param out_back_start [out] offset spawn point on the back of the surface, safe from self
 * intersection
 * @param out_wld_norm [out] unit length spawn point normal in world space
 */
static __device__ void get_save_start_point(float3& out_front_start, float3& out_back_start,
                                            float3& out_wld_norm) {
  // Compute a surface point, normal and conservative offset in object-space.
  float3 obj_pos, obj_norm;
  float obj_offset;
  SelfIntersectionAvoidance::getSafeTriangleSpawnOffset(obj_pos, obj_norm, obj_offset);
  // Transform the object-space position, normal and offset into world-space. The output world-space
  // offset includes the input object-space offset and accounts for the transformation.
  float3 wld_pos;
  float wld_offset;
  SelfIntersectionAvoidance::transformSafeSpawnOffset(
      wld_pos, out_wld_norm, wld_offset, obj_pos, obj_norm, obj_offset);

  // The offset is used to compute safe spawn points on the front and back of the surface.
  SelfIntersectionAvoidance::offsetSpawnPoint(
      out_front_start, out_back_start, wld_pos, out_wld_norm, wld_offset);
}

// Helper function to generate ray for curvilinear probe in local coordinates
static __forceinline__ __device__ void generate_curvilinear_probe_ray_local(
    const RayGenData* ray_gen_data, float d_x, float3& out_origin, float3& out_direction) {
  // Convert normalized coordinates to lateral angle in radians
  const float lateral_angle = (ray_gen_data->sector_angle * d_x) * (M_PI / 180.f);

  // Calculate element position on probe surface in probe's local coordinate system
  // where (0,0,0) is at the probe face center
  out_origin =
      make_float3(ray_gen_data->radius * __sinf(lateral_angle),         // x = r * sin(θ)
                  0.f,                                                  // y (elevation added later)
                  ray_gen_data->radius * (__cosf(lateral_angle) - 1.f)  // z = r * (cos(θ) - 1)
      );

  // Calculate ray direction away from center of curvature
  // Center of curvature is at (0,0,-radius) in probe's local coordinate system
  out_direction = normalize(out_origin - make_float3(0.f, 0.f, -ray_gen_data->radius));
}

// Helper function to generate ray for linear array probe in local coordinates
static __forceinline__ __device__ void generate_linear_array_probe_ray_local(
    const RayGenData* ray_gen_data, float d_x, float3& out_origin, float3& out_direction) {
  // For linear arrays, elements are positioned along a straight line
  // Map normalized coordinate to position along the width
  const float element_width = ray_gen_data->width;
  const float element_pos = element_width * d_x;

  // Element position in local coordinates
  out_origin = make_float3(element_pos,  // x position along array
                           0.f,          // y (elevation added later)
                           0.f           // z at surface (probe face)
  );

  // For linear arrays, rays travel perpendicular to the array
  out_direction = make_float3(0.f, 0.f, 1.f);
}

// Helper function to generate ray for phased array probe in local coordinates
static __forceinline__ __device__ void generate_phased_array_probe_ray_local(
    const RayGenData* ray_gen_data, float d_x, float3& out_origin, float3& out_direction) {
  // Use full sector angle range
  // Map d_x from [-0.5, 0.5] directly to [-half_angle_rad, half_angle_rad]
  const float steering_angle = d_x * ray_gen_data->sector_angle;     // in degrees
  const float steering_angle_rad = steering_angle * (M_PI / 180.f);  // convert to radians

  // For phased arrays, all rays originate from a single virtual point (0,0,0)
  // This is the center of the transducer array face
  out_origin = make_float3(0.0f,  // Center of the array
                           0.f,   // y (elevation added later)
                           0.f    // z at the surface of the probe
  );

  // Direction determined by steering angle
  out_direction = make_float3(sinf(steering_angle_rad),  // x component based on steering angle
                              0.f,                       // y component (no elevation steering)
                              cosf(steering_angle_rad)   // z component (along central axis)
  );

  // Normalize direction to ensure unit length vector
  out_direction = normalize(out_direction);
}

extern "C" __global__ void __raygen__rg() {
  const uint3 idx = optixGetLaunchIndex();
  const uint3 dim = optixGetLaunchDimensions();

  const RayGenData* ray_gen_data = reinterpret_cast<RayGenData*>(optixGetSbtDataPointer());

  const float d_x = (static_cast<float>(idx.x) / static_cast<float>(dim.x)) - 0.5f;

  float3 origin;
  float3 direction;

  // Different ray generation based on probe type
  switch (ray_gen_data->probe_type) {
    case PROBE_TYPE_CURVILINEAR: {
      generate_curvilinear_probe_ray_local(ray_gen_data, d_x, origin, direction);
      break;
    }

    case PROBE_TYPE_LINEAR_ARRAY: {
      generate_linear_array_probe_ray_local(ray_gen_data, d_x, origin, direction);
      break;
    }

    case PROBE_TYPE_PHASED_ARRAY: {
      generate_phased_array_probe_ray_local(ray_gen_data, d_x, origin, direction);
      break;
    }
  }

  // Add elevation in probe's local coordinate system (common for all probes)
  const float d_y = (static_cast<float>(idx.y) / static_cast<float>(dim.y)) - 0.5f;
  const float elevation = ray_gen_data->elevational_height * d_y;
  origin.y = elevation;

  // Transform from probe's local coordinate system to global coordinate system
  origin = ray_gen_data->rotation_matrix * origin;
  origin += ray_gen_data->position;

  direction = ray_gen_data->rotation_matrix * direction;

  Payload ray{};
  ray.intensity = 1.f;
  ray.depth = 0;
  ray.current_material_id = params.background_material_id;
  ray.outter_material_id = 0;
  ray.current_obj_id = static_cast<uint16_t>(-1);
  ray.outter_obj_id = static_cast<uint16_t>(-1);

  // `Payload ray{}` zero-initialises everything, so ray.t_ancestors and ray.tof_ancestors
  // (the new SoS-aware accumulator) are both implicitly 0 for primary rays. No explicit
  // init line needed, but mentioning here so the assumption is auditable.
  optixTrace(params.handle,
             origin,
             direction,
             0.f,           // tmin
             params.t_far,  // tmax
             0.f,           // rayTime
             OptixVisibilityMask(1),
             OPTIX_RAY_FLAG_NONE,
             0,  // SBT offset
             1,  // SBT stride
             0,  // missSBTIndex
             reinterpret_cast<uint32_t*>(&ray)[0],
             reinterpret_cast<uint32_t*>(&ray)[1],
             reinterpret_cast<uint32_t*>(&ray)[2],
             reinterpret_cast<uint32_t*>(&ray)[3],
             reinterpret_cast<uint32_t*>(&ray)[4],
             reinterpret_cast<uint32_t*>(&ray)[5]);
  static_assert(sizeof(Payload) / sizeof(uint32_t) == 6);
}

extern "C" __global__ void __miss__ms() {
  const uint3 idx = optixGetLaunchIndex();
  const Payload ray = get_payload();

  // In contact check mode (epsilon > 0), if the initial ray (depth 0) misses all geometry,
  // terminate its path immediately. This blacks out elements that are not pointing towards the
  // phantom.
  if (params.contact_epsilon > 0.f && ray.depth == 0) { return; }

  // no hits, just do scattering up to t_far
  const uint32_t scanline_offset =
      (idx.y * optixGetLaunchDimensions().x + idx.x) * params.buffer_size;
  sample_intensities(
      optixGetWorldRayOrigin(),
      optixGetWorldRayDirection(),
      ray.t_ancestors,
      ray.tof_ancestors,  // SoS-aware TOF accumulator (ignored when params.sos_aware == false)
      optixGetRayTmin(),
      optixGetRayTmax(),
      ray.intensity,
      &params.materials[ray.current_material_id],
      static_cast<uint32_t>(ray.current_obj_id),
      static_cast<uint32_t>(ray.current_material_id),
      &params.scanlines[scanline_offset],
      &params.organ_ids[scanline_offset],
      &params.material_ids[scanline_offset]);
}

template <OptixPrimitiveType PRIM_TYPE>
static __device__ void closest_hit() {
  const float3 ray_orig = optixGetWorldRayOrigin();
  const float3 ray_dir = optixGetWorldRayDirection();
  const float t_min = optixGetRayTmin();
  const float t = optixGetRayTmax();

  const Payload ray = get_payload();

  // In contact check mode (epsilon > 0), if the initial ray (depth 0) hits geometry
  // but the hit distance `t` is greater than the allowed epsilon, terminate the path.
  // This blacks out elements that are too far from the phantom to be considered in contact.
  if (params.contact_epsilon > 0.f && ray.depth == 0 && t > params.contact_epsilon) { return; }

  const uint32_t current_material_id = ray.current_material_id;
  const Material* current_material = &params.materials[current_material_id];
  const uint3 idx = optixGetLaunchIndex();
  const uint32_t scanline_offset =
      (idx.y * optixGetLaunchDimensions().x + idx.x) * params.buffer_size;
  float* const scanline = &params.scanlines[scanline_offset];
  uint32_t* const organ_ids_line = &params.organ_ids[scanline_offset];
  uint32_t* const material_ids_line = &params.material_ids[scanline_offset];

  // add scattering contribution up to hit
  sample_intensities(ray_orig,
                     ray_dir,
                     ray.t_ancestors,
                     ray.tof_ancestors,  // SoS-aware TOF accumulator (no-op when sos_aware == false)
                     t_min,
                     t,
                     ray.intensity,
                     current_material,
                     static_cast<uint32_t>(ray.current_obj_id),
                     static_cast<uint32_t>(ray.current_material_id),
                     scanline,
                     organ_ids_line,
                     material_ids_line);

  // Don't generate secondary rays is max depth is reached
  if (ray.depth + 1 >= params.max_depth) { return; }

  const uint32_t hit_id = optixGetSbtGASIndex();
  const HitGroupData* hit_group_data = reinterpret_cast<HitGroupData*>(optixGetSbtDataPointer());

  uint32_t next_material_id, next_obj_id;
  if (ray.current_obj_id == hit_id) {
    // Exiting an object
    next_material_id = ray.outter_material_id;
    next_obj_id = ray.outter_obj_id;
  } else {
    next_material_id = hit_group_data->material_id;
    next_obj_id = hit_id;
  }

  // Calculate final intensity
  const float final_intensity =
      ray.intensity * get_intensity_at_distance(t - t_min, current_material->attenuation_);

  // Add hit reflection contribution
  const Material* next_material = &params.materials[next_material_id];
  const float3 normal = get_normal<PRIM_TYPE>(ray_orig, ray_dir, t, hit_id, hit_group_data);
  const float incident_angle = acosf(fabsf(dot(ray_dir, normal)));
  const float R = calculate_reflection_coefficient(incident_angle, current_material, next_material);

  float3 refracted_dir;
  const bool total_internal_reflection = calc_refracted_dir(ray_dir,
                                                            normal,
                                                            current_material->speed_of_sound_,
                                                            next_material->speed_of_sound_,
                                                            &refracted_dir);
  const float3 reflected_dir = calc_reflected_dir(ray_dir, normal);
  const float reflected_intensity = final_intensity * R;
  const float refracted_intensity = final_intensity * (1 - R);

  // Add specular reflection from transmitted energy
  const float ray_coherence_attenuation = __powf(0.3f, ray.depth);
  const float specular_reflection = calculate_specular_intensity(reflected_dir,
                                                                 refracted_dir,
                                                                 total_internal_reflection,
                                                                 ray_dir,
                                                                 next_material->specularity_) *
                                    ray_coherence_attenuation;
  // Bin the boundary echo. In legacy (geometric) mode this is just `t_ancestors + t` mm
  // through `get_intensity_offset`. In SoS-aware mode we instead place it at the displayed
  // depth implied by the total time-of-flight up to the hit point — the parent's
  // `tof_ancestors` plus the time spent crossing this segment at the current material's SoS.
  // This is the headline behaviour change: where the boundary echo lands shifts when actual
  // tissue SoS differs from `params.assumed_sos`.
  const uint32_t boundary_offset = params.sos_aware
      ? offset_from_tof(ray.tof_ancestors
                        + distance_to_tof_us(t, current_material->speed_of_sound_))
      : get_intensity_offset(ray.t_ancestors + t);
  scanline[boundary_offset] = 2.f * specular_reflection;
  // Boundary pixel is attributed to the organ/material the ray is refracting INTO
  // (the bright echo comes from crossing into that medium). This overwrites
  // whatever the speckle loop wrote at this exact bin.
  organ_ids_line[boundary_offset] = static_cast<uint32_t>(next_obj_id);
  material_ids_line[boundary_offset] = static_cast<uint32_t>(next_material_id);

  // Self-intersection avoidance
  float3 front_start, back_start, wld_norm;
  if (PRIM_TYPE == OptixPrimitiveType::OPTIX_PRIMITIVE_TYPE_TRIANGLE) {
    get_save_start_point(front_start, back_start, wld_norm);
  } else {
    wld_norm = normal;
    const float epsilon = 1e-5f;
    front_start = ray_orig + ray_dir * (t - epsilon);
    back_start = ray_orig + ray_dir * (t + epsilon);
  }

  bool reflection_on = false;
  // Create reflected ray
  if (reflection_on && (reflected_intensity > params.min_intensity)) {
    Payload reflected_ray{};
    reflected_ray.intensity = reflected_intensity;
    reflected_ray.depth = ray.depth + 1;
    reflected_ray.t_ancestors = ray.t_ancestors + t;
    // SoS-aware TOF accumulator: parent's TOF plus the time spent crossing this segment
    // at the current material's speed of sound (we are reflecting back, so we have not yet
    // entered `next_material`). Always populated — when `params.sos_aware == false` it costs
    // a few extra arithmetic ops but is otherwise unused. tmax stays geometric (see below).
    reflected_ray.tof_ancestors =
        ray.tof_ancestors + distance_to_tof_us(t, current_material->speed_of_sound_);
    reflected_ray.current_material_id = ray.current_material_id;
    reflected_ray.outter_material_id = ray.outter_material_id;
    reflected_ray.current_obj_id = ray.current_obj_id;
    reflected_ray.outter_obj_id = ray.outter_obj_id;

    // Secondary rays along the surface normal should use the generated front point as origin, while
    // rays pointing away from the normal should use the back point as origin.
    const float3 start = (dot(reflected_dir, wld_norm) > 0.f) ? front_start : back_start;
    // tmax is the geometric reach into the scene that OptiX should search for the next mesh
    // intersection — a property of the scene volume in mm, NOT of the displayed-depth axis.
    // It must stay geometric even in SoS-aware mode; using a TOF-derived bound here would
    // truncate rays through bone (display fast → "out of volume" early but geometrically
    // still inside) and over-extend rays through fat. See sos_aware_echo_placement.md.
    optixTrace(params.handle,
               start,
               reflected_dir,
               0.f,                                       // tmin
               params.t_far - reflected_ray.t_ancestors,  // tmax (geometric, intentional)
               0.f,                                       // rayTime
               OptixVisibilityMask(1),
               OPTIX_RAY_FLAG_NONE,
               0,  // SBT offset
               1,  // SBT stride
               0,  // missSBTIndex
               reinterpret_cast<uint32_t*>(&reflected_ray)[0],
               reinterpret_cast<uint32_t*>(&reflected_ray)[1],
               reinterpret_cast<uint32_t*>(&reflected_ray)[2],
               reinterpret_cast<uint32_t*>(&reflected_ray)[3],
               reinterpret_cast<uint32_t*>(&reflected_ray)[4],
               reinterpret_cast<uint32_t*>(&reflected_ray)[5]);
    static_assert(sizeof(Payload) / sizeof(uint32_t) == 6);
  }

  // Create refracted ray
  if ((refracted_intensity > params.min_intensity) && !total_internal_reflection) {
    Payload refracted_ray{};
    refracted_ray.intensity = refracted_intensity;
    refracted_ray.depth = ray.depth + 1;
    refracted_ray.t_ancestors = ray.t_ancestors + t;
    // SoS-aware TOF accumulator: time spent traversing the parent's `current_material`
    // (which the ray was IN for the segment of length `t` ending at this boundary). The
    // child ray will then propagate through `next_material` from here on. Always populated
    // for the same reasons as the reflected case.
    refracted_ray.tof_ancestors =
        ray.tof_ancestors + distance_to_tof_us(t, current_material->speed_of_sound_);
    refracted_ray.current_material_id = next_material_id;
    refracted_ray.outter_material_id = ray.current_material_id;
    refracted_ray.current_obj_id = next_obj_id;
    refracted_ray.outter_obj_id = ray.current_obj_id;

    // Secondary rays along the surface normal should use the generated front point as origin, while
    // rays pointing away from the normal should use the back point as origin.
    const float3 start = (dot(refracted_dir, wld_norm) > 0.f) ? front_start : back_start;
    // tmax stays geometric — see the comment on the reflected branch above for the rationale.
    optixTrace(params.handle,
               start,
               refracted_dir,
               0.f,                                       // tmin
               params.t_far - refracted_ray.t_ancestors,  // tmax (geometric, intentional)
               0.f,                                       // rayTime
               OptixVisibilityMask(1),
               OPTIX_RAY_FLAG_NONE,
               0,  // SBT offset
               1,  // SBT stride
               0,  // missSBTIndex
               reinterpret_cast<uint32_t*>(&refracted_ray)[0],
               reinterpret_cast<uint32_t*>(&refracted_ray)[1],
               reinterpret_cast<uint32_t*>(&refracted_ray)[2],
               reinterpret_cast<uint32_t*>(&refracted_ray)[3],
               reinterpret_cast<uint32_t*>(&refracted_ray)[4],
               reinterpret_cast<uint32_t*>(&refracted_ray)[5]);
    static_assert(sizeof(Payload) / sizeof(uint32_t) == 6);
  }
}

extern "C" __global__ void __closesthit__sphere() {
  closest_hit<OptixPrimitiveType::OPTIX_PRIMITIVE_TYPE_SPHERE>();
}

extern "C" __global__ void __closesthit__triangle() {
  closest_hit<OptixPrimitiveType::OPTIX_PRIMITIVE_TYPE_TRIANGLE>();
}

}  // namespace raysim
