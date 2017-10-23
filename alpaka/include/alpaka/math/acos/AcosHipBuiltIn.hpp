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

#if !defined(__HIPCC__)
    #error If ALPAKA_ACC_GPU_HIP_ENABLED is set, the compiler has to support HIP!
#endif

#include <alpaka/math/acos/Traits.hpp>  // Acos

#include <boost/core/ignore_unused.hpp> // boost::ignore_unused

#include <type_traits>                  // std::enable_if, std::is_floating_point
#include <math_functions.hpp>           // ::acos

namespace alpaka
{
    namespace math
    {
        //#############################################################################
        //! The standard library acos.
        //#############################################################################
        class AcosHipBuiltIn
        {
        public:
            using AcosBase = AcosHipBuiltIn;
        };

        namespace traits
        {
            //#############################################################################
            //! The standard library acos trait specialization.
            //#############################################################################
            template<
                typename TArg>
            struct Acos<
                AcosHipBuiltIn,
                TArg,
                typename std::enable_if<
                    std::is_floating_point<TArg>::value>::type>
            {
                ALPAKA_NO_HOST_ACC_WARNING
                ALPAKA_FN_ACC_HIP_ONLY static auto acos(
                    AcosHipBuiltIn const & /*acos*/,
                    TArg const & arg)
                -> decltype(::acos(arg))
                {
                    //boost::ignore_unused(acos);
                    return ::acos(arg);
                }
            };
        }
    }
}

#endif
