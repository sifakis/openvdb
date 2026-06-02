// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/*!
    \file nanovdb/tools/cuda/ConnectedComponents.cuh

    \authors Efty Sifakis and JaeHyun Lee

    \brief Connected-components labeling of NanoVDB indexGrids on the device.

           Identifies connected components of active voxels in a ValueOnIndex grid: two
           active voxels share a component label iff they are connected through a path of
           adjacent active voxels.

    \warning The header file contains cuda device code so be sure
             to only include it in .cu files (or other .cuh files)
*/

#ifndef NVIDIA_TOOLS_CUDA_CONNECTEDCOMPONENTS_CUH_HAS_BEEN_INCLUDED
#define NVIDIA_TOOLS_CUDA_CONNECTEDCOMPONENTS_CUH_HAS_BEEN_INCLUDED

#include <nanovdb/NanoVDB.h>
#include <nanovdb/GridHandle.h>
#include <nanovdb/cuda/DeviceBuffer.h>
#include <nanovdb/util/cuda/DeviceGridTraits.cuh>
#include <nanovdb/util/cuda/Timer.h>

namespace nanovdb {

namespace tools::cuda {

template <typename BuildT>
class ConnectedComponents
{
    using GridT = NanoGrid<BuildT>;
    using TreeT = NanoTree<BuildT>;
    using RootT = NanoRoot<BuildT>;

public:

    /// @brief Constructor
    /// @param d_srcGrid source device indexGrid whose active voxels are to be labeled
    /// @param stream optional CUDA stream (defaults to CUDA stream 0)
    ConnectedComponents(const GridT* d_srcGrid, cudaStream_t stream = 0)
        : mStream(stream), mTimer(stream), mDeviceSrcGrid(d_srcGrid) {}

    /// @brief Toggle on and off verbose mode
    /// @param level Verbose level: 0=quiet, 1=timing, 2=benchmarking
    void setVerbose(int level = 1) { mVerbose = level; }

    /// @brief Compute per-leaf connected components.
    ///
    ///        Initially this only enumerates, for every leaf node, the number of distinct
    ///        connected components formed by that leaf's active voxels (treating each leaf
    ///        in isolation), storing the result in the per-leaf array mLeafComponentCounts.
    void processLeafConnectedComponents();

    /// @brief Device pointer to the per-leaf component-count array (one uint16_t per leaf),
    ///        valid after processLeafConnectedComponents(). A leaf has at most 256 components
    ///        (8^3 voxels, 6-connected worst case = 3D checkerboard), so uint16_t suffices.
    uint16_t* deviceLeafComponentCounts() { return static_cast<uint16_t*>(mLeafComponentCounts.deviceData()); }

private:

    cudaStream_t                 mStream{0};
    util::cuda::Timer            mTimer;
    int                          mVerbose{0};
    const GridT                 *mDeviceSrcGrid;

    nanovdb::cuda::DeviceBuffer  mLeafComponentCounts; // one uint16_t component count per leaf

}; // tools::cuda::ConnectedComponents<BuildT>

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

template <typename BuildT>
void ConnectedComponents<BuildT>::processLeafConnectedComponents()
{
    const uint32_t leafCount =
        util::cuda::DeviceGridTraits<BuildT>::getTreeData(mDeviceSrcGrid).mNodeCount[0];

    // Allocate one component-count per leaf (device-only). At most 256 components per 8^3
    // leaf (6-connected worst case), so uint16_t is sufficient.
    if (mVerbose==1) mTimer.start("Allocating per-leaf component counts");
    mLeafComponentCounts = nanovdb::cuda::DeviceBuffer::create(
        std::size_t(leafCount) * sizeof(uint16_t), nullptr, false);
    if (mVerbose==1) mTimer.stop();

    // TODO: launch a per-leaf kernel that counts the distinct connected components formed by
    //       each leaf's active voxels (in isolation) and writes them to mLeafComponentCounts.
}// ConnectedComponents<BuildT>::processLeafConnectedComponents

} // namespace tools::cuda

} // namespace nanovdb

#endif // NVIDIA_TOOLS_CUDA_CONNECTEDCOMPONENTS_CUH_HAS_BEEN_INCLUDED
