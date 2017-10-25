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

#include <alpaka/core/Common.hpp>                   // ALPAKA_FN_*, BOOST_LANG_HIP

// Base classes.

#include <alpaka/workdiv/WorkDivHipBuiltIn.hpp>
#include <alpaka/idx/gb/IdxGbHipBuiltIn.hpp>       // IdxGbHipBuiltIn
#include <alpaka/idx/bt/IdxBtHipBuiltIn.hpp>	   // IdxBtHipBuiltIn
#include <alpaka/atomic/AtomicHipBuiltIn.hpp>
#include <alpaka/atomic/AtomicHierarchy.hpp>    // AtomicHierarchy
#include <alpaka/math/MathHipBuiltIn.hpp>
#include <alpaka/block/shared/dyn/BlockSharedMemDynHipBuiltIn.hpp>
#include <alpaka/block/shared/st/BlockSharedMemStHipBuiltIn.hpp>
#include <alpaka/block/sync/BlockSyncHipBuiltIn.hpp>
#include <alpaka/rand/RandHipRand.hpp>
#include <alpaka/time/TimeHipBuiltIn.hpp>

// Specialized traits.
#include <alpaka/acc/Traits.hpp>                    // acc::traits::AccType
#include <alpaka/dev/Traits.hpp>                    // dev::traits::DevType
#include <alpaka/exec/Traits.hpp>                   // exec::traits::ExecType
#include <alpaka/pltf/Traits.hpp>                   // pltf::traits::PltfType
#include <alpaka/size/Traits.hpp>                   // size::traits::SizeType

// Implementation details.
#include <alpaka/dev/DevHipRt.hpp>	// DevHipRt
#include <alpaka/core/Hip.hpp>

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
        class ExecGpuHipRt;
    }
    namespace acc
    {
        //#############################################################################
        //! The GPU HIP accelerator.
        //!
        //! This accelerator allows parallel kernel execution on devices supporting HIP or HCC
        //#############################################################################
        template<
            typename TDim,
            typename TSize>
        class AccGpuHipRt final :
            public workdiv::WorkDivHipBuiltIn<TDim, TSize>,
            public idx::gb::IdxGbHipBuiltIn<TDim, TSize>,
            public idx::bt::IdxBtHipBuiltIn<TDim, TSize>,
            public atomic::AtomicHierarchy<
                atomic::AtomicHipBuiltIn, // grid atomics
                atomic::AtomicHipBuiltIn, // block atomics
                atomic::AtomicHipBuiltIn  // thread atomics
            >,
            public math::MathHipBuiltIn,
            public block::shared::dyn::BlockSharedMemDynHipBuiltIn,
            public block::shared::st::BlockSharedMemStHipBuiltIn,
            public block::sync::BlockSyncHipBuiltIn,
//            public rand::RandCuRand,
            public time::TimeHipBuiltIn
        {
        public:
            //-----------------------------------------------------------------------------
            //! Constructor.
            //-----------------------------------------------------------------------------
            ALPAKA_FN_ACC_HIP_ONLY AccGpuHipRt(
                vec::Vec<TDim, TSize> const & threadElemExtent) :
                    workdiv::WorkDivHipBuiltIn<TDim, TSize>(threadElemExtent),
                    idx::gb::IdxGbHipBuiltIn<TDim, TSize>(),
                    idx::bt::IdxBtHipBuiltIn<TDim, TSize>(),
                    atomic::AtomicHierarchy<
                        atomic::AtomicHipBuiltIn, // atomics between grids
                        atomic::AtomicHipBuiltIn, // atomics between blocks
                        atomic::AtomicHipBuiltIn  // atomics between threads
                    >(),
                    math::MathHipBuiltIn(),
                    block::shared::dyn::BlockSharedMemDynHipBuiltIn(),
                    block::shared::st::BlockSharedMemStHipBuiltIn(),
                    block::sync::BlockSyncHipBuiltIn(),
//                    rand::RandCuRand(),
                    time::TimeHipBuiltIn()
            {}

        public:
            //-----------------------------------------------------------------------------
            //! Copy constructor.
            //-----------------------------------------------------------------------------
            ALPAKA_FN_ACC_HIP_ONLY AccGpuHipRt(AccGpuHipRt const &) = delete;
            //-----------------------------------------------------------------------------
            //! Move constructor.
            //-----------------------------------------------------------------------------
            ALPAKA_FN_ACC_HIP_ONLY AccGpuHipRt(AccGpuHipRt &&) = delete;
            //-----------------------------------------------------------------------------
            //! Copy assignment operator.
            //-----------------------------------------------------------------------------
            ALPAKA_FN_ACC_HIP_ONLY auto operator=(AccGpuHipRt const &) -> AccGpuHipRt & = delete;
            //-----------------------------------------------------------------------------
            //! Move assignment operator.
            //-----------------------------------------------------------------------------
            ALPAKA_FN_ACC_HIP_ONLY auto operator=(AccGpuHipRt &&) -> AccGpuHipRt & = delete;
            //-----------------------------------------------------------------------------
            //! Destructor.
            //-----------------------------------------------------------------------------
            ALPAKA_FN_ACC_HIP_ONLY /*virtual*/ ~AccGpuHipRt() = default;
        };
    }

    namespace acc
    {
        namespace traits
        {
            //#############################################################################
            //! The GPU HIP accelerator accelerator type trait specialization.
            //#############################################################################
            template<
                typename TDim,
                typename TSize>
            struct AccType<
                acc::AccGpuHipRt<TDim, TSize>>
            {
                using type = acc::AccGpuHipRt<TDim, TSize>;
            };
            //#############################################################################
            //! The GPU HIP accelerator device properties get trait specialization.
            //#############################################################################
            template<
                typename TDim,
                typename TSize>
            struct GetAccDevProps<
                acc::AccGpuHipRt<TDim, TSize>>
            {
                //-----------------------------------------------------------------------------
                //
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto getAccDevProps(
                    dev::DevHipRt const & dev)
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
            //! The GPU Hip accelerator name trait specialization.
            //#############################################################################
            template<
                typename TDim,
                typename TSize>
            struct GetAccName<
                acc::AccGpuHipRt<TDim, TSize>>
            {
                //-----------------------------------------------------------------------------
                //
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto getAccName()
                -> std::string
                {
                    return "AccGpuHipRt<" + std::to_string(TDim::value) + "," + typeid(TSize).name() + ">";
                }
            };
        }
    }
    namespace dev
    {
        namespace traits
        {
            //#############################################################################
            //! The GPU HIP accelerator device type trait specialization.
            //#############################################################################
            template<
                typename TDim,
                typename TSize>
            struct DevType<
                acc::AccGpuHipRt<TDim, TSize>>
            {
                using type = dev::DevHipRt;
            };
        }
    }
    namespace dim
    {
        namespace traits
        {
            //#############################################################################
            //! The GPU HIP accelerator dimension getter trait specialization.
            //#############################################################################
            template<
                typename TDim,
                typename TSize>
            struct DimType<
                acc::AccGpuHipRt<TDim, TSize>>
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
            //! The GPU HIP accelerator executor type trait specialization.
            //#############################################################################
            template<
                typename TDim,
                typename TSize,
                typename TKernelFnObj,
                typename... TArgs>
            struct ExecType<
                acc::AccGpuHipRt<TDim, TSize>,
                TKernelFnObj,
                TArgs...>
            {
                using type = exec::ExecGpuHipRt<TDim, TSize, TKernelFnObj, TArgs...>;
            };
        }
    }
    namespace pltf
    {
        namespace traits
        {
            //#############################################################################
            //! The CPU HIP executor platform type trait specialization.
            //#############################################################################
            template<
                typename TDim,
                typename TSize>
            struct PltfType<
                acc::AccGpuHipRt<TDim, TSize>>
            {
                using type = pltf::PltfHipRt;
            };
        }
    }
    namespace size
    {
        namespace traits
        {
            //#############################################################################
            //! The GPU HIP accelerator size type trait specialization.
            //#############################################################################
            template<
                typename TDim,
                typename TSize>
            struct SizeType<
                acc::AccGpuHipRt<TDim, TSize>>
            {
                using type = TSize;
            };
        }
    }
}

#endif
