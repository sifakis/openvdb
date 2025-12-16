/***************************************************************************************************
 * Copyright (c) 2024 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
#pragma once

#include "cute/tensor.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cute/atom/copy_atom.hpp"
#include <random>

#include "cutlass/util/print_error.hpp"

#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_mma_decl.hpp"

#include "dispatch_policy_custom.hpp"
#include "sm80_mma_multistage_custom.hpp"

using namespace cute;


template<class SettingsT>
struct AmperePredicatedFprop {
  //
  // Static config for conv problem shape
  //
  using D = Int<SettingsT::D>;
  using H = Int<SettingsT::H>;
  using W = Int<SettingsT::W>;

  using T = Int<SettingsT::T>;
  using R = Int<SettingsT::R>;
  using S = Int<SettingsT::S>;

  using Z = Int<SettingsT::Z>;
  using P = Int<SettingsT::P>;
  using Q = Int<SettingsT::Q>;

  using C = Int<SettingsT::C>;
  using K = Int<SettingsT::K>;

  // Tiler config
  using Tiler_K = decltype(cute::min(K{}, _32{}));
  using Tiler_C = decltype(cute::min(C{}, _32{}));
  using Tiler_N = _4;
  using TileM = Tiler_K;
  using TileN = Shape<Tiler_N, Z, P, Q>;
  using TileK = Shape<Tiler_C,_1,_1,_1>;
  using TileP = Shape<Tiler_N, D, H, W>; // Including halo in spatial dimensions
  using PIPE  = _3;
  using TilerFlt = Shape<TileM, TileK>;
  using TilerAct = Shape<TileN, TileK>;
  using TilerGIx = Shape<TileP, TileK>;
  using TilerOut = Shape<TileM, TileN>;

  using TileSizeM = Int<size(TileM{})>;
  using TileSizeN = Int<size(TileN{})>;
  using TileSizeK = Int<size(TileK{})>;
  static constexpr int Stages = PIPE::value;

  using ElementFlt = tfloat32_t;
  using ElementAct = tfloat32_t;
  using ElementOut = float;

  using TiledMma = TiledMMA<
    MMA_Atom<SM80_16x8x8_F32TF32TF32F32_TN>,
    Layout<Shape<_2,_2,_1>>,
    Tile<_32,_32,Underscore>>;

  static constexpr int MaxThreadsPerBlock = size(TiledMma{});
  static constexpr int MinBlocksPerMultiprocessor = 1;

  union SharedStorage {
    struct {
      ElementFlt sAMatrix[size(TileM{}) * size(TileK{}) * size(PIPE{})];
      ElementAct sBMatrix[size(TileN{}) * size(TileK{}) * size(PIPE{})];
      bool       sGpMatrix[size(TileP{})]; // Gather predicate
    } mainloop;

    struct {
      ElementOut sCMatrix[size(TileM{}) * size(TileN{})];
    } epilogue;
  };

  //
  // Stencil tensor
  //

  using GmemLayoutFlt = decltype(make_ordered_layout(
    Shape< K, Shape< C, T, R, S>>{},
    tuple<_1, tuple<_0,_4,_3,_2>>{}));

  // We have 64 elements * 32b each in the major mode that we can vectorize
  // Max vector size is 128b, so lay 16 threads along the major mode with a vector size of 4
  // Rest along the minor mode
  using GmemTiledCopyFlt = decltype(make_tiled_copy(
    Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, ElementFlt>{},
    Layout<Shape <_16, _8>,
           Stride< _8, _1>>{},
    Layout<Shape < _1, _4>>{}));

  // Following layout is also correct, but trades off dynamic strides in the slice for bank conflict free accesses
  // using SmemLayoutFlt = decltype(
  //     composition(Swizzle<3,2,3>{},
  //                 make_ordered_layout(
  //                     Shape<TileSizeM,TileSizeK,PIPE>{},
  //                     tuple<       _1,       _0,  _2>{})));

  using SmemLayoutAtomFlt = decltype(
    composition(Swizzle<1,2,3>{},
                Layout<Shape <_8,Shape <_4, _2>>,
                       Stride<_4,Stride<_1,_32>>>{}));

  using SmemCopyAtomFlt = Copy_Atom<SM75_U32x4_LDSM_N, ElementFlt>;

  //
  // Activation tensor
  //

  // Activation tensor is major in the contraction mode, so vectorize that mode first
  // Then lay out the rest of the threads along the other mode
  using GmemTiledCopyAct = decltype(make_tiled_copy(
    Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, ElementAct>{},
    Layout<Shape <_16, _8>,
           Stride< _8, _1>>{},
    Layout<Shape < _1, _4>>{}));

  // Following layout is also correct, but trades off dynamic strides in the slice for bank conflict free accesses
  // using SmemLayoutAct = decltype(
  //     composition(Swizzle<3,2,3>{},
  //                 make_ordered_layout(
  //                     Shape<TileSizeN,TileSizeK,PIPE>{},
  //                     tuple<       _1,       _0,  _2>{})));

  using SmemLayoutAtomAct = decltype(
    composition(Swizzle<1,2,3>{},
                Layout<Shape <_8,Shape <_4, _2>>,
                       Stride<_4,Stride<_1,_32>>>{}));

  using SmemCopyAtomAct = Copy_Atom<SM75_U32x4_LDSM_N, ElementAct>;

  //
  // Output tensor
  //

  using GmemTiledCopyOut = decltype(make_tiled_copy(
    Copy_Atom<UniversalCopy<uint128_t>, ElementAct>{},
    Layout<Shape <_8, _16>,
           Stride<_1,  _8>>{},
    Layout<Shape <_4,  _1>>{}));

  using SmemCopyAtomOut = Copy_Atom<UniversalCopy<uint32_t>, ElementOut>;

  // This can be optimized to make accesses BCF, but we use a col-major layout here to show off composability
  using SmemLayoutOut = Layout<Shape<TileSizeM, TileSizeN>>;

  //
  // Conv functor (predicated IGEMM)
  //
  template <class EngineFlt, class TensorActivation, class TensorGatherIndex, class TensorOutput>
  void __device__
  operator()(cute::Tensor<EngineFlt, GmemLayoutFlt> mFlt, // ( K,        (C,T,R,S))
             TensorActivation                       mAct, // ((N,Z,P,Q), (C,T,R,S))
             TensorGatherIndex                      mGIx, // ((N,D,H,W), (C,1,1,1))
             TensorOutput                           mOut, // ( K,        (N,Z,P,Q))
             char* smem_buf) const {
    using namespace cute;
    using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveMma<
        cutlass::gemm::MainloopSm80CpAsyncUnpredicatedCustom<PIPE::value>,
        Shape<TileM,TileN,TileK>,
        ElementFlt,
        Underscore, // Ignore the stride, we are passing full cute::Tensor to operator()
        ElementAct,
        Underscore, // Ignore the stride, we are passing full cute::Tensor to operator()
        TiledMma,
        GmemTiledCopyFlt,
        SmemLayoutAtomFlt,
        SmemCopyAtomFlt,
        cute::identity,
        GmemTiledCopyAct,
        SmemLayoutAtomAct,
        SmemCopyAtomAct,
        cute::identity>;

    TiledMma tiled_mma;
    Tensor accum = partition_fragment_C(tiled_mma, TilerOut{});
    clear(accum);

#if 1
    if ((threadIdx.x) == 0 && (blockIdx.x == 0) && (blockIdx.y == 0)) {        
        print("mAct.shape()=");print(mAct.shape());print("\n");
        print("mGIx.shape()=");print(mGIx.shape());print("\n");
        for (int n = 0; n < size<0,0>(mAct); ++n)
            for (int z = 0; z < Z::value; ++z)
            for (int p = 0; p < P::value; ++p)
            for (int q = 0; q < Q::value; ++q)
                for (int c = 0; c < C::value; ++c)
                    for (int t = 0; t < T::value; ++t)
                    for (int r = 0; r < R::value; ++r)
                    for (int s = 0; s < S::value; ++s)
                    {
                        if (&mAct(make_tuple(n,z,p,q),make_tuple(c,t,r,s)) !=
                            mAct.data()+C::value*mGIx(make_tuple(n,z+t,p+r,q+s),make_tuple(c,0,0,0))+c)
                        {
                            print("Pointer discrepancy ((n,z,p,q),(c,t,r,s))=(");
                            print("(");print(n);print(",");print(z);print(",");print(p);print(",");print(q);print("),");
                            print("(");print(c);print(",");print(t);print(",");print(r);print(",");print(s);print(")");
                            print(")\n");
                        }
                        if ((mGIx(make_tuple(n,z+t,p+r,q+s),make_tuple(c,0,0,0)) == 0) &&
                            (mAct(make_tuple(n,z,p,q),make_tuple(c,t,r,s)) != 0.f))
                        {
                            print("Null value discrepancy ((n,z,p,q),(c,t,r,s))=(");
                            print("(");print(n);print(",");print(z);print(",");print(p);print(",");print(q);print("),");
                            print("(");print(c);print(",");print(t);print(",");print(r);print(",");print(s);print(")");
                            print(")\n");
                        }
                    }
    }
    __syncthreads();
#endif

    // Set up tensors
    // NOTE: blockIdx.x projects onto act-NDHW mode, y along the flt-K mode for the sake of higher dynamic range in NDHW
    Tensor gA_mk = local_tile(mFlt, TilerFlt{}, make_coord(_,_));                            // (BLK_M,BLK_K,m',k')
    Tensor gB_nk = local_tile(mAct, TilerAct{}, make_coord(_,_));                            // (BLK_N,BLK_K,n',_1)
    Tensor gG_nk = local_tile(mGIx, TilerGIx{}, make_coord(_,_));                            // (BLK_N,BLK_K,n',_1)
    Tensor gC_mn = local_tile(mOut, TilerOut{}, make_coord(_,_));                            // (BLK_M,BLK_N,m',n')

    // Compute m_coord and n_coord with their post-tiled shapes
    auto m_coord = idx2crd(int(blockIdx.y), shape<2>(gA_mk));
    auto n_coord = idx2crd(int(blockIdx.x), shape<2>(gB_nk));
    Tensor gA = gA_mk(_,_,m_coord,_);                                                        // (BLK_M,BLK_K,k')
    Tensor gB = gB_nk(_,_,n_coord,_);                                                        // (BLK_N,BLK_K,_1)
    Tensor gG = gG_nk(_,_,n_coord,_);                                                        // (BLK_N,BLK_K,_1)
    Tensor gC = gC_mn(_,_,m_coord,n_coord);                                                  // (BLK_M,BLK_N)

#if 1
    if ((threadIdx.x) == 0 && (blockIdx.x == 0) && (blockIdx.y == 0)) {        
        print("gB.shape()=");print(gB.shape());print("\n");
        print("gG.shape()=");print(gG.shape());print("\n");
        print("size<0,0>(gG)=");print(size<0,0>(gG));print("\n");
        print("size<1,0>(gG)=");print(size<1,0>(gG));print("\n");
        print("size<2,0>(gG)=");print(size<2,0>(gG));print("\n");
        for (int n = 0; n < size<0,0>(gG); ++n)
            for (int z = 0; z < Z::value; ++z)
            for (int p = 0; p < P::value; ++p)
            for (int q = 0; q < Q::value; ++q)
                for (int ic = 0; ic < size<1,0>(gG); ++ic)
                for (int bc = 0; bc < size<2,0>(gG); ++bc)
                    for (int t = 0; t < T::value; ++t)
                    for (int r = 0; r < R::value; ++r)
                    for (int s = 0; s < S::value; ++s)
                    {
                        if (&gB(make_tuple(n,z,p,q),make_tuple(ic,1,1,1),make_tuple(bc,t,r,s)) !=
                            gB.data()+C::value*gG(make_tuple(n,z+t,p+r,q+s),make_tuple(ic,1,1,1),make_tuple(bc,0,0,0))+bc*size<1,0>(gG)+ic)
                        {
                            print("Pointer discrepancy ((n,z,p,q),(ic,1,1,1),(bc,t,r,s))=(");
                            print("(");print(n);print(",");print(z);print(",");print(p);print(",");print(q);print("),");
                            print("(");print(ic);print(",1,1,1),");
                            print("(");print(bc);print(",");print(t);print(",");print(r);print(",");print(s);print(")");
                            print(")\n");
                        }
                        if ((gG(make_tuple(n,z+t,p+r,q+s),make_tuple(ic,1,1,1),make_tuple(bc,0,0,0)) == 0) &&
                            (gB(make_tuple(n,z,p,q),make_tuple(ic,1,1,1),make_tuple(bc,t,r,s)) != 0.f))
                        {
                            print("Null value discrepancy ((n,z,p,q),(ic,1,1,1),(bc,t,r,s))=(");
                            print("(");print(n);print(",");print(z);print(",");print(p);print(",");print(q);print("),");
                            print("(");print(ic);print(",1,1,1),");
                            print("(");print(bc);print(",");print(t);print(",");print(r);print(",");print(s);print(")");
                            print(")\n");
                        }
                    }
    }
    __syncthreads();
#endif

    static_assert(size(TileP{}) % MaxThreadsPerBlock == 0);
    auto sG_ptr = &reinterpret_cast<SharedStorage*>(smem_buf)->mainloop.sGpMatrix[0];
    auto gG_ptr = gG.data();
    auto gather_predicate_layout = make_layout(
      shape(gB),
      make_stride(
        make_stride(D{}*H{}*W{}, H{}*W{},  W{}, _1{}),
        make_stride(       _0{},    _0{}, _0{}, _0{}),
        make_stride(       _0{}, H{}*W{},  W{}, _1{})));
    Tensor sP = make_tensor(make_smem_ptr(sG_ptr), gather_predicate_layout);    
    for (int i = 0; i < size(TileP{}); i += MaxThreadsPerBlock)
        sG_ptr[i+threadIdx.x] = gG_ptr[i+threadIdx.x];
    __syncthreads();

 #if 0
    if (thread0())
        for (int n = 0; n < size<0,0>(gG); ++n)
            for (int d = 0; d < D::value; ++d)
            for (int h = 0; h < H::value; ++h)
            for (int w = 0; w < W::value; ++w)
                sP(make_tuple(n,d,h,w),make_tuple(0,0,0,0),make_tuple(0,0,0,0)) =
                    gG(make_tuple(n,d,h,w),make_tuple(0,0,0,0),make_tuple(0,0,0,0));
    __syncthreads();
#endif

#if 0
    if ((threadIdx.x) == 0 && (blockIdx.x == 0) && (blockIdx.y == 0)) {        
        print("gG.layout()=");print(gG.layout());print("\n");
        print("gG.stride()=");print(gG.stride());print("\n");
        print("cosize(gG.layout())=");print(cosize(gG.layout()));print("\n");
        print("sP.layout()=");print(sP.layout());print("\n");
        print("sP.stride()=");print(sP.stride());print("\n");
        print("cosize(sP.layout())=");print(cosize(sP.layout()));print("\n");
        
#if 1
        for (int n = 0; n < size<0,0>(gG); ++n)
            for (int z = 0; z < Z::value; ++z)
            for (int p = 0; p < P::value; ++p)
            for (int q = 0; q < Q::value; ++q)
                for (int ic = 0; ic < size<1,0>(gG); ++ic)
                for (int bc = 0; bc < size<2,0>(gG); ++bc)
                    for (int t = 0; t < T::value; ++t)
                    for (int r = 0; r < R::value; ++r)
                    for (int s = 0; s < S::value; ++s)
                    {
                        if (!sP(make_tuple(n,z,p,q),make_tuple(ic,1,1,1),make_tuple(bc,t,r,s)) &&
                            (gB(make_tuple(n,z,p,q),make_tuple(ic,1,1,1),make_tuple(bc,t,r,s)) != 0.f))
                        {
                            print("Null value discrepancy ((n,z,p,q),(ic,1,1,1),(bc,t,r,s))=(");
                            print("(");print(n);print(",");print(z);print(",");print(p);print(",");print(q);print("),");
                            print("(");print(ic);print(",1,1,1),");
                            print("(");print(bc);print(",");print(t);print(",");print(r);print(",");print(s);print(")");
                            print(")\n");
                        }
                    }
#endif
    }
    __syncthreads();
#endif

#if 0
    if ((threadIdx.x) == 0 && (blockIdx.x == 0) && (blockIdx.y == 0)) {        
        print("gA.shape()=");print(gA.shape());print("\n");
        print("gB.shape()=");print(gB.shape());print("\n");
        print("sP.shape()=");print(sP.shape());print("\n");
        print("cosize(sP.layout())=");print(cosize(sP.layout()));print("\n");
        print("gG.shape()=");print(gG.shape());print("\n");
        print("cosize(gG.layout())=");print(cosize(gG.layout()));print("\n");
        for (int n = 0; n < Tiler_N::value; ++n)
            for (int z = 0; z < Z::value; ++z)
            for (int p = 0; p < P::value; ++p)
            for (int q = 0; q < Q::value; ++q)
                for (int c = 0; c < /*C::value*/1; ++c)
                    for (int k = 0; k < /*size<2,0>(sP)*/1; ++k)
                        for (int t = 0; t < T::value; ++t)
                        for (int r = 0; r < R::value; ++r)
                        for (int s = 0; s < S::value; ++s)
                            if (!sP(make_tuple(n,z,p,q),make_tuple(c,1,1,1),make_tuple(k,t,r,s)))
                                if (gG(make_tuple(n,z+t,p+r,q+s),make_tuple(c,1,1,1),make_tuple(k,1,1,1)))
                        {
#if 1
                            print("((n,z,p,q))=(");
                            print("(");print(n);print(",");print(z);print(",");print(p);print(",");print(q);print("),");
                            print("(");print(c);print(",1,1,1),");
                            print("(");print(k);print(",");print(t);print(",");print(r);print(",");print(s);print(")");
                            print(")\n");
#endif
       }
    }
    __syncthreads();
#endif

    
    auto k_tile_iter = cute::make_coord_iterator(size<2>(gA));
    int k_tile_count = size<2>(gA);

    CollectiveMainloop collective_mma;
    collective_mma(
      accum,
      gA,
      gB,
      sP,
      accum,
      k_tile_iter, k_tile_count,
      Underscore{}, // no residue since we do not support predication
      threadIdx.x,
      smem_buf);

    //
    // Epilogue
    //
    SharedStorage& storage = *reinterpret_cast<SharedStorage*>(smem_buf);
    Tensor sC = make_tensor(make_smem_ptr(&storage.epilogue.sCMatrix[0]), SmemLayoutOut{});

    auto smem_tiled_copy_C = make_tiled_copy_C(SmemCopyAtomOut{}, tiled_mma);
    auto smem_thr_copy_C = smem_tiled_copy_C.get_slice(threadIdx.x);
    auto tCrC = smem_thr_copy_C.retile_S(accum);
    auto tCsC = smem_thr_copy_C.partition_D(sC);
    copy(smem_tiled_copy_C, tCrC, tCsC);

    __syncthreads();

    GmemTiledCopyOut gmem_tiled_copy_C;
    auto gmem_thr_copy_C = gmem_tiled_copy_C.get_slice(threadIdx.x);
    auto tDsC = gmem_thr_copy_C.partition_S(sC);
    auto tDgC = gmem_thr_copy_C.partition_D(gC);
    copy(gmem_tiled_copy_C, tDsC, tDgC);

    #if 0
      if (thread0()) {
        print("mAct = "); print(mAct);          print('\n');
        print("mFlt = "); print(mFlt);          print('\n');
        print("mOut = "); print(mOut);          print('\n');
        print("gA   = "); print(gA);            print('\n');
        print("gB   = "); print(gB);            print('\n');
        print("gC   = "); print(gC);            print('\n');
        print("sA   = "); print(sA.layout());   print('\n');
        print("sB   = "); print(sB.layout());   print('\n');
        print("sC   = "); print(sC.layout());   print('\n');
        print("tAgA = "); print(tAgA.layout()); print('\n');
        print("tBgB = "); print(tBgB.layout()); print('\n');
        print("tAsA = "); print(tAsA.layout()); print('\n');
        print("tBsB = "); print(tBsB.layout()); print('\n');
        print("tCsA = "); print(tCsA.layout()); print('\n');
        print("tCsB = "); print(tCsB.layout()); print('\n');
        print("tCrC = "); print(tCrC.layout()); print('\n');
        print("tCsC = "); print(tCsC.layout()); print('\n');
        print("tDsC = "); print(tDsC.layout()); print('\n');
        print("tDgC = "); print(tDgC.layout()); print('\n');
        print("gmem tiled copy A = "); print(gmem_tiled_copy_A); print('\n');
        print("gmem tiled copy B = "); print(gmem_tiled_copy_B); print('\n');
        print("gmem tiled copy C = "); print(gmem_tiled_copy_C); print('\n');
        print("k_tile_count = "); print(size<2>(gA)); print('\n');
        print("k_tile_iter  = "); print(*k_tile_iter); print('\n');
        print("K_BLOCK_MAX  = "); print(K_BLOCK_MAX); print('\n');
    }
    #endif
  }
};
