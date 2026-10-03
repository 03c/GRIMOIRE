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
    for (int i = 2; i < argc; ++i)
        std::printf("%zu\t%s\n", tk.encode(argv[i]).size(), argv[i]);
    return 0;
}
