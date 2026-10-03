// GRIMOIRE
// Copyright (C) 2026 Ian Ernst
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include <sycl/sycl.hpp>

#include <cstdio>
#include <cstdlib>

#include "xe2_gdn_profile_header.hpp"

// Bare-metal entry point for the Xe2 chunked gated-delta kernel.  Every
// allocation is owned by Grimoire and every submission uses its queue.
extern "C" __attribute__((visibility("default"))) void
grimoire_xe2_chunk_gdn_raw_bf16(
    sycl::queue* queue, void* output, const void* query, const void* key,
    const void* value, void* scratch_a, void* scratch_w, void* scratch_u,
    const float* beta, float* gate_a, const float* a_log,
    const void* dt_bias, float* state, int state_stride,
    const int* cu_seqlens, const int* cache_index,
    const bool* has_initial_state, int tokens, int k_heads, int k_dim,
    int v_heads, int v_dim) {
  static bool reported = false;
  if (!reported && std::getenv("GRIMOIRE_GDN_PROFILE")) {
    std::fprintf(stderr, "GDN raw profiling bridge active\n");
    reported = true;
  }
  using bf16 = cutlass::bfloat16_t;
  gdn::kernel_launcher<bf16, float>(
      *queue, static_cast<bf16*>(output),
      static_cast<const bf16*>(query), static_cast<const bf16*>(key),
      static_cast<const bf16*>(value), static_cast<bf16*>(scratch_a),
      static_cast<bf16*>(scratch_w), static_cast<bf16*>(scratch_u), beta,
      gate_a, a_log, static_cast<const bf16*>(dt_bias), state, state_stride,
      cu_seqlens, cache_index, has_initial_state, nullptr, 1, tokens, k_heads,
      k_dim, v_heads, v_dim);
}
