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

extern "C" void grimoire_xe2_grouped_w4a16(
    sycl::queue* queue, const void* activations_bf16,
    const unsigned char* weights_int4, const void* scales_bf16,
    void* outputs_bf16, int n, int k, const int* rows_per_expert,
    const int* expert_ids, int num_experts, int group_size,
    int* atomic_buffer);
