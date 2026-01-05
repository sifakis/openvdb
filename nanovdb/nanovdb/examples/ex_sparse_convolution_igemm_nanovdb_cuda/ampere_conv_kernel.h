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

    using ZZ = Int<SettingsT::ZZ>;
    using PP = Int<SettingsT::PP>;
    using QQ = Int<SettingsT::QQ>;

    using Cx = Int<SettingsT::Cx>;
    using Cy = Int<SettingsT::Cy>;
    using Cz = Int<SettingsT::Cz>;

    using C = Int<SettingsT::C>;
    using K = Int<SettingsT::K>;

    // Tiler config
    using Tiler_K  = decltype(cute::min(K{}, _32{}));
    using Tiler_C  = decltype(cute::min(C{}, _32{}));
    using Tiler_N  = Shape<_1, ZZ, PP, QQ>;
    using TileM    = Tiler_K;
    using TileN    = Shape<Tiler_N, Z, P, Q>;
    using TileK    = Shape<Tiler_C,_1,_1,_1>;
    using PIPE     = _3;
    using TilerFlt = Shape<TileM, TileK>;
    using TilerAct = Shape<TileN, TileK>;
    using TilerOut = Shape<TileM, TileN>;

    using TileSizeM = Int<size(TileM{})>;
    using TileSizeN = Int<size(TileN{})>;
    using TileSizeK = Int<size(TileK{})>;
    static constexpr int Stages = PIPE::value;

    using ElementFlt = tfloat32_t;
    using ElementAct = tfloat32_t;
    using ElementOut = float;

    using ClusterShape = Shape<Cx,Cy,Cz>;

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
            } mainloop;

            struct {
                ElementOut sCMatrix[size(TileM{}) * size(TileN{})];
            } epilogue;
        };

        // uint64_t sBIdxMatrix[SettingsT::VoxelsPerLeafnodeWithHalo()];
        uint64_t sCIdxMatrix[SettingsT::VoxelsPerLeafnodeNoHalo()];
        bool sBPredMatrix[SettingsT::VoxelsPerLeafnodeWithHalo()];
        bool sCPredMatrix[SettingsT::VoxelsPerLeafnodeNoHalo()];
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
    using SmemLayoutOut       = Layout<Shape<TileSizeM, TileSizeN>>;

    //
    // Conv functor (predicated IGEMM)
    //
    template <class BuildT, class EngineFlt, class ActivationTensor, class ActivationIndexTensor, class OutputTensor, class OutputIndexTensor>
    void __device__
    operator()(cute::Tensor<EngineFlt, GmemLayoutFlt> mFlt,        // (                   K,           (C,T,R,S))
        ActivationTensor                              mAct,        // (((N,Bx,By,Bz),Z,P,Q),           (C,T,R,S))
        ActivationIndexTensor                         mActIdx,     // (((N,Bx,By,Bz),Z,P,Q),           (C,T,R,S))
        OutputTensor                                  mOut,        // ( K,                  ((N,Bx,By,Bz),Z,P,Q))
        OutputIndexTensor                             mOutIdx,     // ( K,                  ((N,Bx,By,Bz),Z,P,Q))
        const nanovdb::NanoGrid<BuildT>               *mActGrid,
        const nanovdb::NanoGrid<BuildT>               *mOutGrid,
        const float                                   *actData,
        float                                         *outData,
        char* smem_buf) const
    {
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

        int leafID = blockIdx.x;

        // Populate output (scatter) indices
        const auto& outLeaf = mOutGrid->tree().template getFirstNode<0>()[leafID];
        auto sCIdx_ptr = &reinterpret_cast<SharedStorage*>(smem_buf)->sCIdxMatrix[0];
        for (int i = 0; i < SettingsT::VoxelsPerLeafnodeNoHalo(); i += MaxThreadsPerBlock)
            sCIdx_ptr[i+threadIdx.x] = outLeaf.getValue(i+threadIdx.x);

        __syncthreads();

#if 0
        if (threadIdx.x == 0) {
            auto gCIdx_ptr = &mOutIdx(make_tuple(0,make_tuple(make_tuple(leafID,0,0,0),0,0,0)));
            for (int i = 0; i < SettingsT::VoxelsPerLeafnodeNoHalo(); ++i)
                if (sCIdx_ptr[i] != gCIdx_ptr[i])
                    printf("Inconsistency between on-the-fly and precomputed gather indices\n");
        }
        __syncthreads();
#endif

        TiledMma tiled_mma;
        Tensor accum = partition_fragment_C(tiled_mma, TilerOut{});

        // Set up tensors
        // NOTE: blockIdx.x projects onto act-NDHW mode, y along the flt-K mode for the sake of higher dynamic range in NDHW
        Tensor gA_mk    = local_tile(mFlt,    TilerFlt{}, make_coord(_,_));                // (BLK_M,BLK_K,m',k')
        Tensor gB_nk    = local_tile(mAct,    TilerAct{}, make_coord(_,_));                // (BLK_N,BLK_K,n',_1)
        Tensor gBIdx_nk = local_tile(mActIdx, TilerAct{}, make_coord(_,_));                // (BLK_N,BLK_K,n',_1)
        Tensor gC_mn    = local_tile(mOut,    TilerOut{}, make_coord(_,_));                // (BLK_M,BLK_N,m',n')
        Tensor gCIdx_mn = local_tile(mOutIdx, TilerOut{}, make_coord(_,_));                // (BLK_M,BLK_N,m',n')        
        
        for (int m_coord = 0; m_coord < size<2>(gA_mk); ++m_coord)
        for (int clusterID = 0; clusterID < size(ClusterShape{}); ++clusterID)
        {
            clear(accum);
        
#if 0
            auto n_layout = make_layout(shape<2>(gB_nk), GenRowMajor{});
            auto n_coord = idx2crd(int(8*leafID+clusterID), shape(n_layout), stride(n_layout));
#elif 0
            // This version produces the exactly same order as above
            auto clusterLayout = make_layout(ClusterShape{}, GenRowMajor{});
            auto clusterCoord = idx2crd(clusterID, shape(clusterLayout), stride(clusterLayout));
            auto n_coord = make_tuple(cute::prepend(clusterCoord, leafID),_0{},_0{},_0{});
#elif 1
            // Also correct, but clusters traversed in co-lex order
            auto clusterCoord = idx2crd(clusterID, ClusterShape{});
            auto n_coord = make_tuple(cute::prepend(clusterCoord, leafID),_0{},_0{},_0{});
#endif

            Tensor gA    = gA_mk   (_,_,m_coord,_);                                            // (BLK_M,BLK_K,k')
            Tensor gB    = gB_nk   (_,_,n_coord,_);                                            // (BLK_N,BLK_K,_1)
            Tensor gBIdx = gBIdx_nk(_,_,n_coord,_);                                            // (BLK_N,BLK_K,_1)
            Tensor gC    = gC_mn   (_,_,m_coord,n_coord);                                      // (BLK_M,BLK_N)
            Tensor gCIdx = gCIdx_mn(_,_,m_coord,n_coord);                                      // (BLK_M,BLK_N)
            
            // Build gather predicate tensors in SMEM
        
            auto sBPred_ptr = &reinterpret_cast<SharedStorage*>(smem_buf)->sBPredMatrix[0];
            auto gBIdx_ptr = gBIdx.data();
            Tensor sBPred = make_tensor(make_smem_ptr(sBPred_ptr), gBIdx.layout());
            auto sBPred_cosize = cosize(gBIdx.layout());
            for (int i = 0; i < sBPred_cosize; i += MaxThreadsPerBlock)
                if (i+threadIdx.x < sBPred_cosize)
                    sBPred_ptr[i+threadIdx.x] = gBIdx_ptr[i+threadIdx.x];
            
            auto sCPred_ptr = &reinterpret_cast<SharedStorage*>(smem_buf)->sCPredMatrix[0];
            auto gCIdx_ptr = gCIdx.data();
            auto sCPred_cosize = cosize(gCIdx.layout());
            Tensor sCPred = make_tensor(make_smem_ptr(sCPred_ptr), gCIdx.layout());
            for (int i = 0; i < sCPred_cosize; i += MaxThreadsPerBlock)
                if (i+threadIdx.x < sCPred_cosize)
                    sCPred_ptr[i+threadIdx.x] = gCIdx_ptr[i+threadIdx.x];
        
            __syncthreads();
        
            auto k_tile_iter = cute::make_coord_iterator(size<2>(gA));
            int k_tile_count = size<2>(gA);
        
            CollectiveMainloop collective_mma;
            collective_mma(
                accum,
                gA,
                gB,
                sBPred,
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
            auto tDsCPred = gmem_thr_copy_C.partition_D(sCPred);
        
            copy_if(gmem_tiled_copy_C, tDsCPred, tDsC, tDgC);

            __syncthreads(); // necessary while the predicate tensors are built once per iteration; TODO: revise
        }
    }
};
