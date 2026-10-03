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

#include "b70/native_model.hpp"
#include <cstdio>
int main(int argc,char**argv){
 if(argc<2||argc>3||(argc==3 && std::string(argv[2])!="--list")){std::fprintf(stderr,"usage: %s model.b70 [--list]\n",argv[0]);return 2;}
 b70::NativeModel m;std::string e;if(!m.open(argv[1],e)){std::fprintf(stderr,"%s\n",e.c_str());return 1;}
 auto&h=m.header();std::printf("GRIMOIRE native v%u target bmg_g31 tensors %llu size %.2f GiB\n",
 h.version,(unsigned long long)h.tensor_count,double(h.file_size)/1073741824.0);
 if(argc==3) for(uint64_t i=0;i<h.tensor_count;++i){const auto&r=m.records()[i];
   std::printf("%s encoding=%s rank=%u shape=",r.name,b70::native_encoding_name(r.encoding),r.rank);
   for(uint32_t d=0;d<r.rank;++d)std::printf("%s%lld",d?"x":"",(long long)r.shape[d]);
   std::printf("\n");}
 return 0;
}
