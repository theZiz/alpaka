/**
* \file
* Copyright 2014-2016 Benjamin Worpitz, Rene Widera
*
* This file is part of alpaka.
*
* alpaka is free software: you can redistribute it and/or modify
* it under the terms of the GNU Lesser General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* alpaka is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU Lesser General Public License for more details.
*
* You should have received a copy of the GNU Lesser General Public License
* along with alpaka.
* If not, see <http://www.gnu.org/licenses/>.
*/

#pragma once

#ifdef ALPAKA_ACC_GPU_HIP_ENABLED

#include <alpaka/core/Common.hpp>                   // ALPAKA_FN_*, BOOST_LANG_CUDA

// Base classes.

#include <alpaka/workdiv/WorkDivHipBuiltIn.hpp>    // as of now, just a renamed copy of it's CUDA counterpart
#include <alpaka/idx/gb/IdxGbHipBuiltIn.hpp>       // IdxGbHipBuiltIn (as of now, just a renamed copy of it's CUDA counterpart
#include <alpaka/idx/bt/IdxBtHipBuiltIn.hpp>	   // IdxBtHipBuiltIn (as of now, just a renamed copy of it's CUDA counterpart)
#include <alpaka/atomic/AtomicHipBuiltIn.hpp>		// as of now, just a renamed copy of it's CUDA counterpart
#include <alpaka/atomic/AtomicHierarchy.hpp>    // AtomicHierarchy
#include <alpaka/math/MathHipBuiltIn.hpp>	// as of now, just a renamed copy of it's CUDA counterpart
#include <alpaka/block/shared/dyn/BlockSharedMemDynHipBuiltIn.hpp>		// as of now, just a renamed copy of it's CUDA counterpart
#include <alpaka/block/shared/st/BlockSharedMemStHipBuiltIn.hpp>		// as of now, just a renamed copy of it's CUDA counterpart
#include <alpaka/block/sync/BlockSyncHipBuiltIn.hpp>		// as of now, just a renamed copy of it's CUDA counterpart
#include <alpaka/rand/RandHipRand.hpp>		// as of now, just a renamed copy of it's CUDA counterpart
#include <alpaka/time/TimeHipBuiltIn.hpp>          // as of now, just a renamed copy of it's CUDA counterpart

// Specialized traits.
#include <alpaka/acc/Traits.hpp>                    // acc::traits::AccType
#include <alpaka/dev/Traits.hpp>                    // dev::traits::DevType
#include <alpaka/exec/Traits.hpp>                   // exec::traits::ExecType
#include <alpaka/pltf/Traits.hpp>                   // pltf::traits::PltfType
#include <alpaka/size/Traits.hpp>                   // size::traits::SizeType

// Implementation details.
#include <alpaka/dev/DevHipRt.hpp>	// DevHipRt- as of now, this isn't implemented; DevCudaRt itself is used instead.
#include <alpaka/core/Hip.hpp>		    // as of now, just a renamed copy of it's CUDA coutnerpart

#include <boost/predef.h>                           // workarounds

#include <typeinfo>                                 // typeid

namespace alpaka
{
    namespace exec
    {
        template<
            typename TDim,
            typename TSize,
            typename TKernelFnObj,
            typename... TArgs>
        class ExecGpuCudaRt;
    }
    namespace acc
    {
        //#############################################################################
        //! The GPU CUDA accelerator.
        //!
        //! This accelerator allows parallel kernel execution on devices supporting CUDA.
        //#############################################################################
        template<
            typename TDim,
            typename TSize>
        class AccGpuCudaRt final :
            public workdiv::WorkDivCudaBuiltIn<TDim, TSize>,
            public idx::gb::IdxGbCudaBuiltIn<TDim, TSize>,
            public idx::bt::IdxBtCudaBuiltIn<TDim, TSize>,
            public atomic::AtomicHierarchy<
                atomic::AtomicCudaBuiltIn, // grid atomics
                atomic::AtomicCudaBuiltIn, // block atomics
                atomic::AtomicCudaBuiltIn  // thread atomics
            >,
            public math::MathCudaBuiltIn,
            public block::shared::dyn::BlockSharedMemDynCudaBuiltIn,
            public block::shared::st::BlockSharedMemStCudaBuiltIn,
            public block::sync::BlockSyncCudaBuiltIn,
// This is not currently supported by the clang native CUDA compiler.
#if !BOOST_COMP_CLANG_CUDA
            public rand::RandCuRand,
#endif
            public time::TimeCudaBuiltIn
        {
        public:
            //-----------------------------------------------------------------------------
            //! Constructor.
            //-----------------------------------------------------------------------------
            ALPAKA_FN_ACC_CUDA_ONLY AccGpuCudaRt(
                vec::Vec<TDim, TSize> const & threadElemExtent) :
                    workdiv::WorkDivCudaBuiltIn<TDim, TSize>(threadElemExtent),
                    idx::gb::IdxGbCudaBuiltIn<TDim, TSize>(),
                    idx::bt::IdxBtCudaBuiltIn<TDim, TSize>(),
                    atomic::AtomicHierarchy<
                        atomic::AtomicCudaBuiltIn, // atomics between grids
                        atomic::AtomicCudaBuiltIn, // atomics between blocks
                        atomic::AtomicCudaBuiltIn  // atomics between threads
                    >(),
                    math::MathCudaBuiltIn(),
                    block::shared::dyn::BlockSharedMemDynCudaBuiltIn(),
                    block::shared::st::BlockSharedMemStCudaBuiltIn(),
                    block::sync::BlockSyncCudaBuiltIn(),
// This is not currently supported by the clang native CUDA compiler.
#if !BOOST_COMP_CLANG_CUDA
                    rand::RandCuRand(),
#endif
                    time::TimeCudaBuiltIn()
            {}

        public:
            //-----------------------------------------------------------------------------
            //! Copy constructor.
            //-----------------------------------------------------------------------------
            ALPAKA_FN_ACC_CUDA_ONLY AccGpuCudaRt(AccGpuCudaRt const &) = delete;
            //-----------------------------------------------------------------------------
            //! Move constructor.
            //-----------------------------------------------------------------------------
            ALPAKA_FN_ACC_CUDA_ONLY AccGpuCudaRt(AccGpuCudaRt &&) = delete;
            //-----------------------------------------------------------------------------
            //! Copy assignment operator.
            //-----------------------------------------------------------------------------
            ALPAKA_FN_ACC_CUDA_ONLY auto operator=(AccGpuCudaRt const &) -> AccGpuCudaRt & = delete;
            //-----------------------------------------------------------------------------
            //! Move assignment operator.
            //-----------------------------------------------------------------------------
            ALPAKA_FN_ACC_CUDA_ONLY auto operator=(AccGpuCudaRt &&) -> AccGpuCudaRt & = delete;
            //-----------------------------------------------------------------------------
            //! Destructor.
            //-----------------------------------------------------------------------------
            ALPAKA_FN_ACC_CUDA_ONLY /*virtual*/ ~AccGpuCudaRt() = default;
        };
    }

    namespace acc
    {
        namespace traits
        {
            //#############################################################################
            //! The GPU CUDA accelerator accelerator type trait specialization.
            //#############################################################################
            template<
                typename TDim,
                typename TSize>
            struct AccType<
                acc::AccGpuCudaRt<TDim, TSize>>
            {
                using type = acc::AccGpuCudaRt<TDim, TSize>;
            };
            //#############################################################################
            //! The GPU CUDA accelerator device properties get trait specialization.
            //#############################################################################
            template<
                typename TDim,
                typename TSize>
            struct GetAccDevProps<
                acc::AccGpuCudaRt<TDim, TSize>>
            {
                //-----------------------------------------------------------------------------
                //
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto getAccDevProps(
                    dev::DevCudaRt const & dev)
                -> acc::AccDevProps<TDim, TSize>
                {
                    hipDeviceProp_t hipDevProp;
                    ALPAKA_HIP_RT_CHECK(hipGetDeviceProperties(
                        &hipDevProp,
                        dev.m_iDevice));

                    return {
                        // m_multiProcessorCount
                        static_cast<TSize>(hipDevProp.multiProcessorCount),
                        // m_gridBlockExtentMax
                        extent::getExtentVecEnd<TDim>(
                            vec::Vec<dim::DimInt<3u>, TSize>(
                                static_cast<TSize>(hipDevProp.maxGridSize[2]),
                                static_cast<TSize>(hipDevProp.maxGridSize[1]),
                                static_cast<TSize>(hipDevProp.maxGridSize[0]))),
                        // m_gridBlockCountMax
                        std::numeric_limits<TSize>::max(),
                        // m_blockThreadExtentMax
                        extent::getExtentVecEnd<TDim>(
                            vec::Vec<dim::DimInt<3u>, TSize>(
                                static_cast<TSize>(hipDevProp.maxThreadsDim[2]),
                                static_cast<TSize>(hipDevProp.maxThreadsDim[1]),
                                static_cast<TSize>(hipDevProp.maxThreadsDim[0]))),
                        // m_blockThreadCountMax
                        static_cast<TSize>(hipDevProp.maxThreadsPerBlock),
                        // m_threadElemExtentMax
                        vec::Vec<TDim, TSize>::all(std::numeric_limits<TSize>::max()),
                        // m_threadElemCountMax
                        std::numeric_limits<TSize>::max()};
                }
            };
            //#############################################################################
            //! The GPU CUDA accelerator name trait specialization.
            //#############################################################################
            template<
                typename TDim,
                typename TSize>
            struct GetAccName<
                acc::AccGpuCudaRt<TDim, TSize>>
            {
                //-----------------------------------------------------------------------------
                //
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto getAccName()
                -> std::string
                {
                    return "AccGpuCudaRt<" + std::to_string(TDim::value) + "," + typeid(TSize).name() + ">";
                }
            };
        }
    }
    namespace dev
    {
        namespace traits
        {
            //#############################################################################
            //! The GPU CUDA accelerator device type trait specialization.
            //#############################################################################
            template<
                typename TDim,
                typename TSize>
            struct DevType<
                acc::AccGpuCudaRt<TDim, TSize>>
            {
                using type = dev::DevCudaRt;
            };
        }
    }
    namespace dim
    {
        namespace traits
        {
            //#############################################################################
            //! The GPU CUDA accelerator dimension getter trait specialization.
            //#############################################################################
            template<
                typename TDim,
                typename TSize>
            struct DimType<
                acc::AccGpuCudaRt<TDim, TSize>>
            {
                using type = TDim;
            };
        }
    }
    namespace exec
    {
        namespace traits
        {
            //#############################################################################
            //! The GPU CUDA accelerator executor type trait specialization.
            //#############################################################################
            template<
                typename TDim,
                typename TSize,
                typename TKernelFnObj,
                typename... TArgs>
            struct ExecType<
                acc::AccGpuCudaRt<TDim, TSize>,
                TKernelFnObj,
                TArgs...>
            {
                using type = exec::ExecGpuCudaRt<TDim, TSize, TKernelFnObj, TArgs...>;
            };
        }
    }
    namespace pltf
    {
        namespace traits
        {
            //#############################################################################
            //! The CPU CUDA executor platform type trait specialization.
            //#############################################################################
            template<
                typename TDim,
                typename TSize>
            struct PltfType<
                acc::AccGpuCudaRt<TDim, TSize>>
            {
                using type = pltf::PltfCudaRt;
            };
        }
    }
    namespace size
    {
        namespace traits
        {
            //#############################################################################
            //! The GPU CUDA accelerator size type trait specialization.
            //#############################################################################
            template<
                typename TDim,
                typename TSize>
            struct SizeType<
                acc::AccGpuCudaRt<TDim, TSize>>
            {
                using type = TSize;
            };
        }
    }
}

#endif
