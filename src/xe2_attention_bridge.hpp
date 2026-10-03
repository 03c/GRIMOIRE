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

#pragma once
#include <sycl/sycl.hpp>

extern "C" void grimoire_xe2_chunk_prefill_bf16(
    sycl::queue* queue, const void* query, const void* key, const void* value,
    void* output, int q_tokens, int kv_tokens, int q_heads, int kv_heads,
    int head_dim, const int* cu_q, const int* cu_k, float softmax_scale,
    bool causal);

// Exact vLLM FA2 paged-attention contract used by the Muse DFlash assistant.
// Q/O and the paged K/V cache are IEEE FP16.  The cache layout is
// [num_blocks, block_size, kv_heads, head_dim], with a one-row block table.
extern "C" int grimoire_xe2_dflash_paged_f16(
    sycl::queue* queue, const void* query, const void* key_cache,
    const void* value_cache, void* output, int q_tokens, int kv_tokens,
    int q_heads, int kv_heads, int head_dim, int block_size, int num_blocks,
    const int* block_table, const int* cu_q, const int* dummy_cu_k,
    const int* seqused_k, float softmax_scale, int window_left,
    int window_right, bool causal);

extern "C" void grimoire_xe2_chunk_gdn_bf16(
    sycl::queue* queue, void* output, const void* query, const void* key,
    const void* value, const float* beta, const float* gate_a,
    const float* a_log, const void* dt_bias, float* state, int tokens,
    int k_heads, int k_dim, int v_heads, int v_dim, const int* cu_seqlens,
    const int* cache_index, const bool* has_initial_state);
