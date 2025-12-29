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
    using Tiler_NN = Shape<_1,_1,_2,_2>;
    using TileM  = Tiler_K;
    using TileN  = Shape<Tiler_N , Z, P, Q>;
    using TileNN = Shape<Tiler_NN, Z, P, Q>;
    using TileK  = Shape<Tiler_C ,_1,_1,_1>;
    using TileP  = Shape<Tiler_N , D, H, W>; // Including halo in spatial dimensions
    using PIPE  = _3;
    using TilerFlt = Shape<TileM, TileK>;
    using TilerAct       = Shape<TileNN, TileK>;
    using TilerActLegacy = Shape<TileN , TileK>;

    using TilerGIx = Shape<TileP, TileK>;

    using TilerOut       = Shape<TileM, TileNN>;
    using TilerOutLegacy = Shape<TileM, TileN >;

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

    struct SharedStorage {
        union {
            struct {
                ElementFlt sAMatrix[size(TileM{}) * size(TileK{}) * size(PIPE{})];
                ElementAct sBMatrix[size(TileN{}) * size(TileK{}) * size(PIPE{})];
                bool       sGpMatrix[size(TileP{})]; // Gather predicate        
            } mainloop;

            struct {
                ElementOut sCMatrix[size(TileM{}) * size(TileN{})];
            } epilogue;
        };
        bool sSpMatrix[size(TileN{})]; // Scatter predicate
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
            Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS_ZFILL<uint128_t>, ElementAct>{},
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
    template <class EngineFlt,
        class TensorActivation,   class TensorActivationLegacy,
        class TensorGatherIndex,  class TensorGatherIndexLegacy,
        class TensorOutput,       class TensorOutputLegacy,
        class TensorScatterIndex, class TensorScatterIndexLegacy>
    void __device__
    operator()(cute::Tensor<EngineFlt, GmemLayoutFlt> mFlt,       // ( K,        (C,T,R,S))
        TensorActivation                              mAct,       // ((N,Z,P,Q), (C,T,R,S))
        TensorActivationLegacy                        mActLegacy, // ((N,Z,P,Q), (C,T,R,S))
        TensorGatherIndex                             mGIx_,      // ((N,D,H,W), (C,1,1,1))
        TensorGatherIndexLegacy                       mGIxLegacy, // ((N,D,H,W), (C,1,1,1))
        TensorOutput                                  mOut,       // ( K,        (N,Z,P,Q))
        TensorOutputLegacy                            mOutLegacy, // ( K,        (N,Z,P,Q))
        TensorScatterIndex                            mSIx,       // ( K,        (N,Z,P,Q))      
        TensorScatterIndexLegacy                      mSIxLegacy, // ( K,        (N,Z,P,Q))      
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
        Tensor accumLegacy = partition_fragment_C(tiled_mma, TilerOutLegacy{});
        clear(accumLegacy);
        Tensor accum = partition_fragment_C(tiled_mma, TilerOut{});
        clear(accum);

        // Set up tensors
        // NOTE: blockIdx.x projects onto act-NDHW mode, y along the flt-K mode for the sake of higher dynamic range in NDHW
        Tensor gA_mk = local_tile(mFlt, TilerFlt{}, make_coord(_,_));                            // (BLK_M,BLK_K,m',k')
        Tensor gBLegacy_nk = local_tile(mActLegacy, TilerActLegacy{}, make_coord(_,_));                      // (BLK_N,BLK_K,n',_1)
        Tensor gB_nk = local_tile(mAct, TilerAct{}, make_coord(_,_));                      // (BLK_N,BLK_K,n',_1)
        Tensor gG_nk = local_tile(mGIxLegacy, TilerGIx{}, make_coord(_,_));                // (BLK_N,BLK_K,n',_1)

        Tensor gCLegacy_mn = local_tile(mOutLegacy, TilerOutLegacy{}, make_coord(_,_));    // (BLK_M,BLK_N,m',n')
        Tensor gC_mn       = local_tile(      mOut,       TilerOut{}, make_coord(_,_));    // (BLK_M,BLK_N,m',n')
        Tensor gSLegacy_mn = local_tile(mSIxLegacy, TilerOutLegacy{}, make_coord(_,_));    // (BLK_M,BLK_N,m',n')
        Tensor gS_mn       = local_tile(      mSIx,       TilerOut{}, make_coord(_,_));    // (BLK_M,BLK_N,m',n')
#if 1
        if (thread0() && block0()) {
            // print("shape(mActLegacy)=");print(shape(mActLegacy));print("\n");
            // print("shape(TilerActLegacy{})=");print(shape(TilerActLegacy{}));print("\n");
            // print("shape(gBLegacy_nk)=");print(shape(gBLegacy_nk));print("\n");            
            // print("shape(mAct)=");print(shape(mAct));print("\n");
            // print("shape(TilerAct{})=");print(shape(TilerAct{}));print("\n");
            print("\nshape(gB_nk)=");print(shape(gB_nk));print("\n");            
            print("shape(gS_mn)=");print(shape(gS_mn));print("\n");            
            print("shape(gSLegacy_mn)=");print(shape(gSLegacy_mn));print("\n");
            print("layout(accum)=");print(layout(accum));print("\n");
            print("layout(accumLegacy)=");print(layout(accum));print("\n");
            
        }
#endif
        // __syncthreads();
        // Compute m_coord and n_coord with their post-tiled shapes
        auto m_coord = idx2crd(int(blockIdx.y), shape<2>(gA_mk));



        auto n_coord = idx2crd(int(blockIdx.x), shape<2>(gBLegacy_nk));
        // auto N_coord = idx2crd(int(blockIdx.x), make_layout(shape<2>(gB_nk), GenRowMajor{}));
        auto N_coord = idx2crd(int(blockIdx.x), shape<2>(gB_nk));
        auto test_stride = stride(make_layout(shape<2>(gB_nk), GenRowMajor{}));
        auto NN_coord = idx2crd(int(blockIdx.x), shape<2>(gB_nk), test_stride);

#if 0
        if ((threadIdx.x == 0) && (blockIdx.x == 1) && (blockIdx.y == 0))
        {
            print("blockIdx.x = ");print(blockIdx.x);
            print(", n_coord = ");print(n_coord);
            print(", N_coord = ");print(N_coord);
            print("\n");
        }
        __syncthreads();
#endif

        Tensor gA = gA_mk(_,_,m_coord,_);                                                        // (BLK_M,BLK_K,k')
        Tensor gBLegacy = gBLegacy_nk(_,_,n_coord,_);                                                        // (BLK_N,BLK_K,_1)
        Tensor gB = gB_nk(_,_,NN_coord,_);
        Tensor gG = gG_nk(_,_,n_coord,_);                                                        // (BLK_N,BLK_K,_1)

        Tensor gCLegacy = gCLegacy_mn(_,_,m_coord, n_coord);                                                  // (BLK_M,BLK_N)
        Tensor gC       = gC_mn      (_,_,m_coord,NN_coord);                                                  // (BLK_M,BLK_N)
        Tensor gSLegacy = gSLegacy_mn(_,_,m_coord, n_coord);                                                  // (BLK_M,BLK_N)
        Tensor gS       = gS_mn      (_,_,m_coord,NN_coord);                                                  // (BLK_M,BLK_N)

#if 1
        auto blockLayout = make_layout(shape<1,0>(gS), GenRowMajor{});
        if ((threadIdx.x == 0) && (threadIdx.y == 0) & (threadIdx.z == 0))
        {
            for (int kk = 0; kk < size<0>(gS); ++kk)
                for (int bii = 0; bii < size<1,0,1>(gS); ++bii)
                for (int bjj = 0; bjj < size<1,0,2>(gS); ++bjj)
                for (int bkk = 0; bkk < size<1,0,3>(gS); ++bkk)
                  for (int iii = 0; iii < size<1,1>(gS); ++iii)
                  for (int jjj = 0; jjj < size<1,2>(gS); ++jjj)
                  for (int kkk = 0; kkk < size<1,3>(gS); ++kkk)
                {

                    auto coord = 
                        make_tuple
                                             (kk,
                            make_tuple
                                                 (
                                make_tuple
                                                  (0,bii,bjj,bkk),iii,jjj,kkk));
                    int bIdx = blockLayout(make_tuple(0,bii,bjj,bkk));
                    auto coordLegacy = 
                        make_tuple
                                             (kk,
                            make_tuple
                                                 (
                                                             bIdx,iii,jjj,kkk));
                    if (gS(coord) != gSLegacy(coordLegacy))
                        printf("Mismatch (gS!=gSLegacy) at blockIdx = (%d,%d,%d)\n", blockIdx.x, blockIdx.y, blockIdx.z);
                    if (&gC(coord) != &gCLegacy(coordLegacy))
                        printf("Mismatch (&gC!=&gCLegacy) at blockIdx = (%d,%d,%d)\n", blockIdx.x, blockIdx.y, blockIdx.z);
                }
        }
        __syncthreads();
#endif
        
#if 0
        if ((threadIdx.x == 0) && (threadIdx.y == 0) & (threadIdx.z == 0))
        // if ((blockIdx.x  == 7) && (blockIdx.y  == 0) & (blockIdx.z  == 0))
        {
            //for (auto [it,count] = std::tuple{cute::make_coord_iterator(shape(gB)), (int)size(shape(gB))}; count; ++it, --count)
            //    if(gB(*it) != gB(*it))
            //        printf("Mismatch at blockIdx = (%d,%d,%d)\n", blockIdx.x, blockIdx.y, blockIdx.z);
            if(gB(0) != gBLegacy(0))
                printf("Mismatch at blockIdx = (%d,%d,%d)\n", blockIdx.x, blockIdx.y, blockIdx.z);

#if 0
            print("shape(mActLegacy)=");print(shape(mActLegacy));print("\n");
            print("shape(TilerActLegacy{})=");print(shape(TilerActLegacy{}));print("\n");
            print("shape(gBLegacy_nk)=");print(shape(gBLegacy_nk));print("\n");            
            print("shape(mAct)=");print(shape(mAct));print("\n");
            print("shape(TilerAct{})=");print(shape(TilerAct{}));print("\n");
            print("shape(gB_nk)=");print(shape(gB_nk));print("\n");            

            print("shape(gBLegacy)=");print(shape(gBLegacy));print("\n");
            print("shape(gB)=");print(shape(gB));print("\n");
            print("shape<2,0>(gB_nk)=");print(shape<2,0>(gB_nk));print("\n");
            print("reverse(shape<2,0>(gB_nk))=");print(reverse(shape<2,0>(gB_nk)));print("\n");

            print("n_coord = ");print(n_coord);print("\n");
            print("N_coord = ");print(N_coord);print("\n");
            print("NN_coord = ");print(NN_coord);print("\n");
#endif
        }
        __syncthreads();
#endif        

        // Build gather predicate tensor in SMEM
        static_assert(size(TileP{}) % MaxThreadsPerBlock == 0);
        auto sG_ptr = &reinterpret_cast<SharedStorage*>(smem_buf)->mainloop.sGpMatrix[0];
        auto gG_ptr = gG.data();
        auto gather_predicate_layout = make_layout(
            shape(gBLegacy),
            make_stride(
                make_stride(D{}*H{}*W{}, H{}*W{},  W{}, _1{}),
                make_stride(       _0{},    _0{}, _0{}, _0{}),
                make_stride(       _0{}, H{}*W{},  W{}, _1{})));
        for (int i = 0; i < size(TileP{}); i += MaxThreadsPerBlock)
            sG_ptr[i+threadIdx.x] = gG_ptr[i+threadIdx.x];
        Tensor sG = make_tensor(make_smem_ptr(sG_ptr), gather_predicate_layout);    

        // Build scatter predicate tensor in SMEM
        static_assert(size(TileN{}) <= MaxThreadsPerBlock);
        auto sS_ptr = &reinterpret_cast<SharedStorage*>(smem_buf)->sSpMatrix[0];
        auto gSLegacy_ptr = gSLegacy.data();
        auto scatter_predicate_layout = make_layout(
            shape(gCLegacy),
            make_stride(
                _0{},
                make_stride(Z{}*P{}*Q{}, P{}*Q{},  Q{}, _1{})));
        Tensor sS = make_tensor(make_smem_ptr(sS_ptr), scatter_predicate_layout);
        if (threadIdx.x < size(TileN{}))
            sS_ptr[threadIdx.x] = gSLegacy_ptr[threadIdx.x];

        __syncthreads();

        auto k_tile_iter = cute::make_coord_iterator(size<2>(gA));
        int k_tile_count = size<2>(gA);

        CollectiveMainloop collective_mma;
        collective_mma(
            accumLegacy,
            // accum,
            gA,
            gBLegacy,
            sG,
            accumLegacy,
            // accum,
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
        auto tCrC = smem_thr_copy_C.retile_S(accumLegacy);
        // auto tCrC = smem_thr_copy_C.retile_S(accum);
        auto tCsC = smem_thr_copy_C.partition_D(sC);
        copy(smem_tiled_copy_C, tCrC, tCsC);

        __syncthreads();

        GmemTiledCopyOut gmem_tiled_copy_C;
        auto gmem_thr_copy_C = gmem_tiled_copy_C.get_slice(threadIdx.x);
        auto tDsC = gmem_thr_copy_C.partition_S(sC);
        auto tDgCLegacy = gmem_thr_copy_C.partition_D(gCLegacy);
        auto tDgC = gmem_thr_copy_C.partition_D(gC);
        auto tDsS = gmem_thr_copy_C.partition_D(sS);

        // copy   (gmem_tiled_copy_C,       tDsC, tDgCLegacy);
        // copy   (gmem_tiled_copy_C,       tDsC, tDgC);
        copy_if(gmem_tiled_copy_C, tDsS, tDsC, tDgCLegacy);
        // copy_if(gmem_tiled_copy_C, tDsS, tDsC, tDgC);
    }
};
