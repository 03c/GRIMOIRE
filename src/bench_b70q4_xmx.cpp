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

#include "kernels.hpp"
#include "b70/b70q4.hpp"
#include <cmath>
#include <cstdio>
#include <vector>

int main(){
  constexpr int M=4096,N=8192,K=2048;
  sycl::queue q{sycl::gpu_selector_v,
    sycl::property_list{sycl::property::queue::enable_profiling{}}};
  std::vector<float> wh(size_t(N)*K);
  for(size_t i=0;i<wh.size();++i)wh[i]=0.02f*std::sin(float(i%10007)*0.013f);
  auto pw=b70::quantize_b70q4(wh.data(),N,K);
  std::vector<int8_t> xh(size_t(M)*K);
  for(size_t i=0;i<xh.size();++i)xh[i]=int8_t(int(i*17%31)-15);
  std::vector<float> sh(M,1.0f);
  auto* dp=sycl::malloc_device<uint8_t>(pw.payload.size(),q);
  auto* ds=sycl::malloc_device<b70::bf16_t>(pw.scales.size(),q);
  auto* dx=sycl::malloc_device<int8_t>(xh.size(),q);
  auto* xs=sycl::malloc_device<float>(M,q);
  auto* dy=sycl::malloc_device<float>(size_t(M)*N,q);
  q.memcpy(dp,pw.payload.data(),pw.payload.size());
  q.memcpy(ds,pw.scales.data(),pw.scales.size()*sizeof(b70::bf16_t));
  q.memcpy(dx,xh.data(),xh.size());q.memcpy(xs,sh.data(),M*sizeof(float)).wait();
  b70::B70Q4View v{dp,ds,N,K,pw.padded_n,pw.padded_k};
  b70::launch_gemm_b70q4(q,v,dx,xs,dy,M).wait();
  auto e=b70::launch_gemm_b70q4(q,v,dx,xs,dy,M);e.wait();
  const uint64_t t0=e.get_profiling_info<sycl::info::event_profiling::command_start>();
  const uint64_t t1=e.get_profiling_info<sycl::info::event_profiling::command_end>();
  const double ms=double(t1-t0)*1e-6;
  std::vector<float> sample(N);q.memcpy(sample.data(),dy,N*sizeof(float)).wait();
  double maxerr=0.0,refnorm=0.0;
  for(int n=0;n<N;n+=257){double ref=0;for(int k=0;k<K;++k)ref+=pw.at(n,k)*xh[k];
    maxerr=std::max(maxerr,std::fabs(double(sample[n])-ref));refnorm=std::max(refnorm,std::fabs(ref));}
  const double tops=2.0*double(M)*N*K/(ms*1e9);
  std::printf("B70Q4 XMX %dx%dx%d: %.3f ms, %.1f TOPS, sampled rel-max %.3e\n",
              M,N,K,ms,tops,maxerr/(refnorm+1e-30));
  sycl::free(dp,q);sycl::free(ds,q);sycl::free(dx,q);sycl::free(xs,q);sycl::free(dy,q);
  return maxerr/(refnorm+1e-30)>1e-4;
}
