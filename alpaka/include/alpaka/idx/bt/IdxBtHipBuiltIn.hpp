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

//#if !defined(__HIPCC__)
//    #error If ALPAKA_ACC_GPU_HIP_ENABLED is set, the compiler has to support HIP!
//#endif

#include <alpaka/idx/Traits.hpp>            // idx::getIdx

#include <alpaka/vec/Vec.hpp>               // Vec, offset::getOffsetVecEnd
#include <alpaka/core/Hip.hpp>		    // as of now, just a renamed copy of it's HIP coutnerpart

//#include <boost/core/ignore_unused.hpp>   // boost::ignore_unused

namespace alpaka
{
    namespace idx
    {
        namespace bt
        {
            //#############################################################################
            //! The HIP accelerator ND index provider.
            //#############################################################################
            template<
                typename TDim,
                typename TSize>
            class IdxBtHipBuiltIn
            {
            public:
                using IdxBtBase = IdxBtHipBuiltIn;

                //-----------------------------------------------------------------------------
                //! Default constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FN_ACC_HIP_ONLY IdxBtHipBuiltIn() = default;
                //-----------------------------------------------------------------------------
                //! Copy constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FN_ACC_HIP_ONLY IdxBtHipBuiltIn(IdxBtHipBuiltIn const &) = delete;
                //-----------------------------------------------------------------------------
                //! Move constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FN_ACC_HIP_ONLY IdxBtHipBuiltIn(IdxBtHipBuiltIn &&) = delete;
                //-----------------------------------------------------------------------------
                //! Copy assignment operator.
                //-----------------------------------------------------------------------------
                ALPAKA_FN_ACC_HIP_ONLY auto operator=(IdxBtHipBuiltIn const & ) -> IdxBtHipBuiltIn & = delete;
                //-----------------------------------------------------------------------------
                //! Move assignment operator.
                //-----------------------------------------------------------------------------
                ALPAKA_FN_ACC_HIP_ONLY auto operator=(IdxBtHipBuiltIn &&) -> IdxBtHipBuiltIn & = delete;
                //-----------------------------------------------------------------------------
                //! Destructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FN_ACC_HIP_ONLY /*virtual*/ ~IdxBtHipBuiltIn() = default;
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
                idx::bt::IdxBtHipBuiltIn<TDim, TSize>>
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
            //! The GPU HIP accelerator block thread index get trait specialization.
            //#############################################################################
            template<
                typename TDim,
                typename TSize>
            struct GetIdx<
                idx::bt::IdxBtHipBuiltIn<TDim, TSize>,
                origin::Block,
                unit::Threads>
            {
                //-----------------------------------------------------------------------------
                //! \return The index of the current thread in the block.
                //-----------------------------------------------------------------------------
                template<
                    typename TWorkDiv>
                ALPAKA_FN_ACC_HIP_ONLY static auto getIdx(
                    idx::bt::IdxBtHipBuiltIn<TDim, TSize> const & /*idx*/,
                    TWorkDiv const &)
                -> vec::Vec<TDim, TSize>
                {
                    //boost::ignore_unused(idx);
                    return vec::cast<TSize>(offset::getOffsetVecEnd<TDim>(dim3(hipThreadIdx_x,hipThreadIdx_y,hipThreadIdx_z)));
                }
            };
        }
    }
    namespace size
    {
        namespace traits
        {
            //#############################################################################
            //! The GPU HIP accelerator block thread index size type trait specialization.
            //#############################################################################
            template<
                typename TDim,
                typename TSize>
            struct SizeType<
                idx::bt::IdxBtHipBuiltIn<TDim, TSize>>
            {
                using type = TSize;
            };
        }
    }
}

#endif
