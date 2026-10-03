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

// =====================================================================
//  count_launches_probe.cpp -- generate exactly N tokens, nothing else.
//
//  The probe half of tools/count_launches.sh: run it at two values of N
//  under SYCL_UR_TRACE=2 and subtract, so model load and prompt prefill
//  cancel and what remains is the decode steps.  See that script for why
//  a launch COUNT is a number worth taking off the card when a timing is
//  not.
//
//  Deliberately minimal: no tokenizer, no sampling, no server.  Anything
//  else in here would land in the difference too.
// =====================================================================
#include "b70/grimoire_api.hpp"
#include "b70/formats.hpp"
#include <cstdio>
#include <string>
#include <vector>
using namespace b70;
int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    const std::string dir = argv[1];
    const int n = argc > 2 ? std::atoi(argv[2]) : 1;
    std::string err;
    Grimoire* e = grimoire_new();
    if (!grimoire_load(*e, dir, Fmt::BF16, 256, err)) {
        std::printf("load failed: %s\n", err.c_str()); return 1;
    }
    std::vector<int32_t> prompt{7,11,3,42,5,90,1,64}, out;
    FinishReason r{};
    std::fprintf(stderr, "@@MARK generate %d\n", n);
    grimoire_serve_generate(*e, prompt, n, -1, out, -1, {}, &r);
    std::fprintf(stderr, "@@MARK done\n");
    grimoire_delete(e);
    std::printf("generated %zu\n", out.size());
    return 0;
}
