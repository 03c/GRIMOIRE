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
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>

extern "C" void grimoire_xe2_grouped_mxfp4_bf16(
    sycl::queue*,const void*,const unsigned char*,const unsigned char*,void*,
    int,int,const int*,const int*,int,int*);

int main(int argc,char**argv){
  const int E=256,M=argc>1?std::atoi(argv[1]):32768;
  const int N=argc>2?std::atoi(argv[2]):1024,K=argc>3?std::atoi(argv[3]):2048;
  using bf16=sycl::ext::oneapi::bfloat16;
  sycl::queue q{sycl::gpu_selector_v,{sycl::property::queue::in_order{}}};
  auto* rows=sycl::malloc_shared<int>(E,q);auto* atomic=sycl::malloc_device<int>(1,q);
  for(int e=0;e<E;++e)rows[e]=M/E;
  auto*x=sycl::malloc_device<bf16>(size_t(M)*K,q);
  auto*w=sycl::malloc_device<unsigned char>(size_t(E)*N*K/2,q);
  auto*s=sycl::malloc_device<unsigned char>(size_t(E)*N*K/32,q);
  auto*y=sycl::malloc_device<bf16>(size_t(M)*N,q);
  q.memset(x,0,size_t(M)*K*2);q.memset(w,0,size_t(E)*N*K/2);
  q.memset(s,127,size_t(E)*N*K/32);q.wait();
  auto run=[&]{q.memset(atomic,0,4);grimoire_xe2_grouped_mxfp4_bf16(
      &q,x,w,s,y,N,K,rows,nullptr,E,atomic);q.wait();};
  run();double best=1e9;
  for(int i=0;i<5;++i){auto t=std::chrono::steady_clock::now();run();
    best=std::min(best,std::chrono::duration<double,std::milli>(
      std::chrono::steady_clock::now()-t).count());}
  std::printf("grouped MXFP4 E=%d rows=%d N=%d K=%d %.3f ms %.1f TOPS\n",
    E,M,N,K,best,2.0*M*N*K/(best*1e9));
}
