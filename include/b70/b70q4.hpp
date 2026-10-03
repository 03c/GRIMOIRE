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

#ifndef B70_B70Q4_HPP
#define B70_B70Q4_HPP

#include "formats.hpp"
#include <cstddef>
#include <cstdint>
#include <vector>

namespace b70 {

// B70Q4 is an execution layout, not an interchange quant format.
//
// A 64x128 [N,K] tile is stored in the byte order consumed by Xe2's
// int8-DPAS B operand after nibble expansion:
//   [k/4][n][two bytes containing k%4 = {0,1} and {2,3}]
// Consequently a subgroup can read consecutive bytes, expand four signed
// nibbles, and write one already-VNNI-ordered int8 vector without a transpose.
// N and K are padded to complete tiles in the file.
constexpr int kB70Q4N = 64;
constexpr int kB70Q4K = 128;

struct B70Q4Weight {
    int N = 0, K = 0;
    int padded_n = 0, padded_k = 0;
    std::vector<uint8_t> payload;
    // One symmetric BF16 scale per real/padded output and 128-wide K tile,
    // laid out [n_tile][k_tile][n_in_tile].
    std::vector<bf16_t> scales;

    size_t payload_offset(int n, int k) const;
    size_t scale_offset(int n, int k) const;
    int8_t code(int n, int k) const;
    float at(int n, int k) const;
};

struct B70Q4View {
    const uint8_t* payload = nullptr;
    const bf16_t* scales = nullptr;
    int N = 0, K = 0, padded_n = 0, padded_k = 0;
};

inline B70Q4View view(const B70Q4Weight& w) {
    return {w.payload.data(), w.scales.data(), w.N, w.K, w.padded_n, w.padded_k};
}

B70Q4Weight quantize_b70q4(const float* src, int N, int K);

} // namespace b70
#endif
