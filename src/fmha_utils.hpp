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
// Torch-free subset of Intel's Xe2 FMHA policy declarations.  Keeping this
// local makes the production bridge a raw SYCL/Level Zero library.
#include <cute/tensor.hpp>

#define HEAD_SIZE_LIMIT_0 64
#define HEAD_SIZE_LIMIT_1 96
#define HEAD_SIZE_LIMIT_2 128
#define HEAD_SIZE_LIMIT_3 192
#define HEAD_SIZE_LIMIT_4 256
#define HEAD_SIZE_LIMIT_5 512

enum class CutlassDType { half, bfloat16, float8_e4m3, float8_e5m2 };
struct CutlassQKType {
  CutlassDType q_type, k_type;
  explicit CutlassQKType(CutlassDType t):q_type(t),k_type(t){}
  CutlassQKType(CutlassDType q,CutlassDType k):q_type(q),k_type(k){}
};

using namespace cute;
#define GRIMOIRE_CHUNK_POLICY(NAME, M, O, SG) \
struct NAME { using ShapeQK=Shape<M,_32,_32>; using ShapePV=Shape<M,_32,_32>; \
  using ShapeOut=Shape<M,O>; using SubgroupLayoutQK=Layout<Shape<SG,_1,_1>>; };
GRIMOIRE_CHUNK_POLICY(chunk_policy_head64,  _128, _64,  _8)
GRIMOIRE_CHUNK_POLICY(chunk_policy_head96,  _128, _96,  _8)
GRIMOIRE_CHUNK_POLICY(chunk_policy_head128, _256, _128, _16)
GRIMOIRE_CHUNK_POLICY(chunk_policy_head192, _256, _192, _32)
GRIMOIRE_CHUNK_POLICY(chunk_policy_head256, _256, _256, _32)
GRIMOIRE_CHUNK_POLICY(chunk_policy_head512, _256, _256, _32)
#undef GRIMOIRE_CHUNK_POLICY
