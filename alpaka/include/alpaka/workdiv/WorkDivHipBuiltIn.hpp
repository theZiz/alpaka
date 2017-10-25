/**
* \file
* Copyright 2014-2015 Benjamin Worpitz
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

#include <alpaka/core/Common.hpp>           // ALPAKA_FN_*, __HIPCC__

#include <alpaka/workdiv/Traits.hpp>        // workdiv::GetWorkDiv
#include <alpaka/size/Traits.hpp>           // size::Size

#include <alpaka/vec/Vec.hpp>               // Vec, getExtentVecEnd
#include <alpaka/core/Hip.hpp>

//#include <boost/core/ignore_unused.hpp>   // boost::ignore_unused

namespace alpaka
{
    namespace workdiv
    {
        //#############################################################################
        //! The GPU HIP accelerator work division.
        //#############################################################################
        template<
            typename TDim,
            typename TSize>
        class WorkDivHipBuiltIn
        {
        public:
            using WorkDivBase = WorkDivHipBuiltIn;

            //-----------------------------------------------------------------------------
            //! Default constructor.
            //-----------------------------------------------------------------------------
            ALPAKA_FN_ACC_HIP_ONLY WorkDivHipBuiltIn(
                vec::Vec<TDim, TSize> const & threadElemExtent) :
                    m_threadElemExtent(threadElemExtent)
            {}
            //-----------------------------------------------------------------------------
            //! Copy constructor.
            //-----------------------------------------------------------------------------
            ALPAKA_FN_ACC_HIP_ONLY WorkDivHipBuiltIn(WorkDivHipBuiltIn const &) = delete;
            //-----------------------------------------------------------------------------
            //! Move constructor.
            //-----------------------------------------------------------------------------
            ALPAKA_FN_ACC_HIP_ONLY WorkDivHipBuiltIn(WorkDivHipBuiltIn &&) = delete;
            //-----------------------------------------------------------------------------
            //! Copy assignment operator.
            //-----------------------------------------------------------------------------
            ALPAKA_FN_ACC_HIP_ONLY auto operator=(WorkDivHipBuiltIn const &) -> WorkDivHipBuiltIn & = delete;
            //-----------------------------------------------------------------------------
            //! Move assignment operator.
            //-----------------------------------------------------------------------------
            ALPAKA_FN_ACC_HIP_ONLY auto operator=(WorkDivHipBuiltIn &&) -> WorkDivHipBuiltIn & = delete;
            //-----------------------------------------------------------------------------
            //! Destructor.
            //-----------------------------------------------------------------------------
            ALPAKA_FN_ACC_HIP_ONLY /*virtual*/ ~WorkDivHipBuiltIn() = default;

        public:
            // \TODO: Optimize! Add WorkDivHipBuiltInNoElems that has no member m_threadElemExtent as well as AccGpuHipRtNoElems.
            // Use it instead of AccGpuHipRt if the thread element extent is one to reduce the register usage.
            vec::Vec<TDim, TSize> const & m_threadElemExtent;
        };
    }

    namespace dim
    {
        namespace traits
        {
            //#############################################################################
            //! The GPU HIP accelerator work division dimension get trait specialization.
            //#############################################################################
            template<
                typename TDim,
                typename TSize>
            struct DimType<
                workdiv::WorkDivHipBuiltIn<TDim, TSize>>
            {
                using type = TDim;
            };
        }
    }
    namespace size
    {
        namespace traits
        {
            //#############################################################################
            //! The GPU HIP accelerator work division size type trait specialization.
            //#############################################################################
            template<
                typename TDim,
                typename TSize>
            struct SizeType<
                workdiv::WorkDivHipBuiltIn<TDim, TSize>>
            {
                using type = TSize;
            };
        }
    }
    namespace workdiv
    {
        namespace traits
        {
            //#############################################################################
            //! The GPU HIP accelerator work division grid block extent trait specialization.
            //#############################################################################
            template<
                typename TDim,
                typename TSize>
            struct GetWorkDiv<
                WorkDivHipBuiltIn<TDim, TSize>,
                origin::Grid,
                unit::Blocks>
            {
                //-----------------------------------------------------------------------------
                //! \return The number of blocks in each dimension of the grid.
                //-----------------------------------------------------------------------------
                ALPAKA_FN_ACC_HIP_ONLY static auto getWorkDiv(
                    WorkDivHipBuiltIn<TDim, TSize> const & /*workDiv*/)
                -> vec::Vec<TDim, TSize>
                {
                    //boost::ignore_unused(workDiv);
                    return vec::cast<TSize>(extent::getExtentVecEnd<TDim>(dim3(hipGridDim_x ,hipGridDim_y ,hipGridDim_z)));
                }
            };

            //#############################################################################
            //! The GPU HIP accelerator work division block thread extent trait specialization.
            //#############################################################################
            template<
                typename TDim,
                typename TSize>
            struct GetWorkDiv<
                WorkDivHipBuiltIn<TDim, TSize>,
                origin::Block,
                unit::Threads>
            {
                //-----------------------------------------------------------------------------
                //! \return The number of threads in each dimension of a block.
                //-----------------------------------------------------------------------------
                ALPAKA_FN_ACC_HIP_ONLY static auto getWorkDiv(
                    WorkDivHipBuiltIn<TDim, TSize> const & /*workDiv*/)
                -> vec::Vec<TDim, TSize>
                {
                    //boost::ignore_unused(workDiv);
                    return vec::cast<TSize>(extent::getExtentVecEnd<TDim>(dim3(hipBlockDim_x,hipBlockDim_y,hipBlockDim_z)));
                }
            };

            //#############################################################################
            //! The GPU HIP accelerator work division thread element extent trait specialization.
            //#############################################################################
            template<
                typename TDim,
                typename TSize>
            struct GetWorkDiv<
                WorkDivHipBuiltIn<TDim, TSize>,
                origin::Thread,
                unit::Elems>
            {
                //-----------------------------------------------------------------------------
                //! \return The number of blocks in each dimension of the grid.
                //-----------------------------------------------------------------------------
                ALPAKA_FN_ACC_HIP_ONLY static auto getWorkDiv(
                    WorkDivHipBuiltIn<TDim, TSize> const & workDiv)
                -> vec::Vec<TDim, TSize>
                {
                    return workDiv.m_threadElemExtent;
                }
            };
        }
    }
}

#endif
