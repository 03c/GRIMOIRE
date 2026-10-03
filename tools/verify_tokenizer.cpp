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

#include "b70/tokenizer.hpp"
#include <cstdio>
using namespace b70;
int main(int argc, char** argv) {
    Tokenizer tk; std::string err;
    if (!tk.load(argv[1], err)) { std::printf("load: %s\n", err.c_str()); return 1; }
    std::printf("vocab %zu  merges %zu  malformed %zu\n",
                tk.vocab_size(), tk.merge_count(), tk.bad_merges());
    std::printf("sample merges present:  \"u s\"=%d  \"Ġ Ġ\"=%d  \"Ġ t\"=%d\n",
                tk.has_merge("u","s"), tk.has_merge("\xc4\xa0","\xc4\xa0"),
                tk.has_merge("\xc4\xa0","t"));
    const std::string p = tk.apply_chat_template("Write a haiku about the sea.");
    std::printf("TEMPLATE: %s\n", p.c_str());
    auto ids = tk.encode(p);
    std::printf("NTOK: %zu\nIDS:", ids.size());
    for (int32_t i : ids) std::printf(" %d", i);
    std::printf("\n");
    const int32_t ref[] = {248045,846,198,7734,264,6185,36974,883,279,9117,13,
                           248046,198,248045,74455,198,248068,198};
    const size_t nref = sizeof(ref)/sizeof(ref[0]);
    bool ok = ids.size() == nref;
    for (size_t i = 0; ok && i < nref; ++i) ok = (ids[i] == ref[i]);
    std::printf("MATCHES REFERENCE: %s\n", ok ? "YES" : "NO");
    if (!ok) {
        std::printf("expected %zu:", nref);
        for (int32_t i : ref) std::printf(" %d", i);
        std::printf("\n");
    }
    return ok ? 0 : 1;
}
