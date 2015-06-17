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

#include <alpaka/core/Common.hpp>   // ALPAKA_FCT_HOST_ACC

namespace alpaka
{
    namespace math
    {
        namespace traits
        {
            //#############################################################################
            //! The pow trait.
            //#############################################################################
            template<
                typename T,
                typename TBase,
                typename TExp,
                typename TSfinae = void>
            struct Pow;
        }

        //-----------------------------------------------------------------------------
        //! Computes the value of base raised to the power exp.
        //!
        //! \tparam T The type of the object specializing Pow.
        //! \tparam TBase The base type.
        //! \tparam TExp The exponent type.
        //! \param pow The object specializing Pow.
        //! \param base The base.
        //! \param exp The exponent.
        //-----------------------------------------------------------------------------
        template<
            typename T,
            typename TBase,
            typename TExp>
        ALPAKA_FCT_HOST_ACC auto pow(
            T const & pow,
            TBase const & base,
            TExp const & exp)
        -> decltype(
            traits::Pow<
                T,
                TBase,
                TExp>
            ::pow(
                pow,
                base,
                exp))
        {
            return traits::Pow<
                T,
                TBase,
                TExp>
            ::pow(
                pow,
                base,
                exp);
        }
    }
}
