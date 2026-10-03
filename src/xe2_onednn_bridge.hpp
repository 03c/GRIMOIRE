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
#include <cstddef>
#include <sycl/sycl.hpp>

extern "C" {
void* grimoire_onednn_w4a16_create(sycl::queue* queue, int m, int n, int k,
                                    int group_size, int use_bf16);
size_t grimoire_onednn_w4a16_scratch_size(void* plan);
void grimoire_onednn_w4a16_execute(void* plan, const void* activations,
                                   const void* packed_weight,
                                   const void* scales, const int8_t* zero_point,
                                   void* output, void* scratch);
void grimoire_onednn_w4a16_destroy(void* plan);
void* grimoire_onednn_mxfp4_w4a16_create(sycl::queue* queue, int m, int n,
                                          int k);
size_t grimoire_onednn_mxfp4_w4a16_scratch_size(void* plan);
void grimoire_onednn_mxfp4_w4a16_execute(void* plan,
                                         const void* activations,
                                         const void* packed_weight,
                                         const void* e8m0_scales,
                                         void* output, void* scratch);
void grimoire_onednn_mxfp4_w4a16_destroy(void* plan);
void* grimoire_onednn_bf16_f32_create(sycl::queue* queue, int m, int n,
                                       int k);
size_t grimoire_onednn_bf16_f32_scratch_size(void* plan);
void grimoire_onednn_bf16_f32_execute(void* plan, const void* activations,
                                      const void* weight, void* output,
                                      void* scratch);
void grimoire_onednn_bf16_f32_destroy(void* plan);
void* grimoire_onednn_f16_create(sycl::queue* queue, int m, int n, int k);
size_t grimoire_onednn_f16_scratch_size(void* plan);
void grimoire_onednn_f16_execute(void* plan, const void* activations,
                                 const void* weight, void* output,
                                 void* scratch);
void grimoire_onednn_f16_destroy(void* plan);
}
