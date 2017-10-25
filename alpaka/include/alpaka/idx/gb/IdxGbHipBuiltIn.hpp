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

#include <alpaka/core/Common.hpp>       // ALPAKA_FN_*, __HIPCC__

#include <alpaka/idx/Traits.hpp>            // idx::getIdx
#include <alpaka/vec/Vec.hpp>               // Vec, offset::getOffsetVecEnd
#include <alpaka/core/Hip.hpp>

//#include <boost/core/ignore_unused.hpp>   // boost::ignore_unused

namespace alpaka
{
    namespace idx
    {
        namespace gb
        {
            //#############################################################################
            //! The HIP accelerator ND index provider.
            //#############################################################################
            template<
                typename TDim,
                typename TSize>
            class IdxGbHipBuiltIn
            {
            public:
                using IdxGbBase = IdxGbHipBuiltIn;

                //-----------------------------------------------------------------------------
                //! Default constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FN_ACC_HIP_ONLY IdxGbHipBuiltIn() = default;
                //-----------------------------------------------------------------------------
                //! Copy constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FN_ACC_HIP_ONLY IdxGbHipBuiltIn(IdxGbHipBuiltIn const &) = delete;
                //-----------------------------------------------------------------------------
                //! Move constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FN_ACC_HIP_ONLY IdxGbHipBuiltIn(IdxGbHipBuiltIn &&) = delete;
                //-----------------------------------------------------------------------------
                //! Copy assignment operator.
                //-----------------------------------------------------------------------------
                ALPAKA_FN_ACC_HIP_ONLY auto operator=(IdxGbHipBuiltIn const & ) -> IdxGbHipBuiltIn & = delete;
                //-----------------------------------------------------------------------------
                //! Move assignment operator.
                //-----------------------------------------------------------------------------
                ALPAKA_FN_ACC_HIP_ONLY auto operator=(IdxGbHipBuiltIn &&) -> IdxGbHipBuiltIn & = delete;
                //-----------------------------------------------------------------------------
                //! Destructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FN_ACC_HIP_ONLY /*virtual*/ ~IdxGbHipBuiltIn() = default;
            };
        }
    }

    namespace dim
    {
        namespace traits
        {
            //#############################################################################
            //! The GPU HIP accelerator index dimension get trait specialization.
            //#############################################################################
            template<
                typename TDim,
                typename TSize>
            struct DimType<
                idx::gb::IdxGbHipBuiltIn<TDim, TSize>>
            {
                using type = TDim;
            };
        }
    }
    namespace idx
    {
        namespace traits
        {
            //#############################################################################
            //! The GPU HIP accelerator grid block index get trait specialization.
            //#############################################################################
            template<
                typename TDim,
                typename TSize>
            struct GetIdx<
                idx::gb::IdxGbHipBuiltIn<TDim, TSize>,
                origin::Grid,
                unit::Blocks>
            {
                //-----------------------------------------------------------------------------
                //! \return The index of the current block in the grid.
                //-----------------------------------------------------------------------------
                template<
                    typename TWorkDiv>
                ALPAKA_FN_ACC_HIP_ONLY static auto getIdx(
                    idx::gb::IdxGbHipBuiltIn<TDim, TSize> const & /*idx*/,
                    TWorkDiv const &)
                -> vec::Vec<TDim, TSize>
                {
                    //boost::ignore_unused(idx);
                    return vec::cast<TSize>(offset::getOffsetVecEnd<TDim>(dim3(hipBlockIdx_x, hipBlockIdx_y, hipBlockIdx_z)));
                }
            };
        }
    }
    namespace size
    {
        namespace traits
        {
            //#############################################################################
            //! The GPU HIP accelerator grid block index size type trait specialization.
            //#############################################################################
            template<
                typename TDim,
                typename TSize>
            struct SizeType<
                idx::gb::IdxGbHipBuiltIn<TDim, TSize>>
            {
                using type = TSize;
            };
        }
    }
}

#endif
