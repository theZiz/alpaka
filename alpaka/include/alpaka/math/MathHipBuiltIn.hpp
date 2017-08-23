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

#include <alpaka/core/Common.hpp>   // BOOST_LANG_CUDA

#include <alpaka/math/abs/AbsHipBuiltIn.hpp>	// as of now, just a renamed copy of it's CUDA counterpart
#include <alpaka/math/acos/AcosHipBuiltIn.hpp>	// as of now, just a renamed copy of it's CUDA counterpart
#include <alpaka/math/asin/AsinHipBuiltIn.hpp>	// as of now, just a renamed copy of it's CUDA counterpart
#include <alpaka/math/atan/AtanHipBuiltIn.hpp>	// as of now, just a renamed copy of it's CUDA counterpart
#include <alpaka/math/atan2/Atan2HipBuiltIn.hpp>// as of now, just a renamed copy of it's CUDA counterpart
#include <alpaka/math/cbrt/CbrtHipBuiltIn.hpp>	// as of now, just a renamed copy of it's CUDA counterpart
#include <alpaka/math/ceil/CeilHipBuiltIn.hpp>	// as of now, just a renamed copy of it's CUDA counterpart
#include <alpaka/math/cos/CosHipBuiltIn.hpp>	// as of now, just a renamed copy of it's CUDA counterpart
#include <alpaka/math/erf/ErfHipBuiltIn.hpp>	// as of now, just a renamed copy of it's CUDA counterpart
#include <alpaka/math/exp/ExpHipBuiltIn.hpp>	// as of now, just a renamed copy of it's CUDA counterpart
#include <alpaka/math/floor/FloorHipBuiltIn.hpp>// as of now, just a renamed copy of it's CUDA counterpart
#include <alpaka/math/fmod/FmodHipBuiltIn.hpp>	// as of now, just a renamed copy of it's CUDA counterpart
#include <alpaka/math/log/LogHipBuiltIn.hpp>	// as of now, just a renamed copy of it's CUDA counterpart
#include <alpaka/math/max/MaxHipBuiltIn.hpp>	// as of now, just a renamed copy of it's CUDA counterpart
#include <alpaka/math/min/MinHipBuiltIn.hpp>	// as of now, just a renamed copy of it's CUDA counterpart
#include <alpaka/math/pow/PowHipBuiltIn.hpp>	// as of now, just a renamed copy of it's CUDA counterpart
#include <alpaka/math/remainder/RemainderHipBuiltIn.hpp>// as of now, just a renamed copy of it's CUDA counterpart
#include <alpaka/math/round/RoundHipBuiltIn.hpp>// as of now, just a renamed copy of it's CUDA counterpart	
#include <alpaka/math/rsqrt/RsqrtHipBuiltIn.hpp>// as of now, just a renamed copy of it's CUDA counterpart
#include <alpaka/math/sin/SinHipBuiltIn.hpp>	// as of now, just a renamed copy of it's CUDA counterpart
#include <alpaka/math/sqrt/SqrtHipBuiltIn.hpp>	// as of now, just a renamed copy of it's CUDA counterpart
#include <alpaka/math/tan/TanHipBuiltIn.hpp>	// as of now, just a renamed copy of it's CUDA counterpart
#include <alpaka/math/trunc/TruncHipBuiltIn.hpp>// as of now, just a renamed copy of it's CUDA counterpart

namespace alpaka
{
    //-----------------------------------------------------------------------------
    //! The mathematical operation specifics.
    //-----------------------------------------------------------------------------
    namespace math
    {
        //#############################################################################
        //! The standard library math trait specializations.
        //#############################################################################
        class MathCudaBuiltIn :
            public AbsCudaBuiltIn,
            public AcosCudaBuiltIn,
            public AsinCudaBuiltIn,
            public AtanCudaBuiltIn,
            public Atan2CudaBuiltIn,
            public CbrtCudaBuiltIn,
            public CeilCudaBuiltIn,
            public CosCudaBuiltIn,
            public ErfCudaBuiltIn,
            public ExpCudaBuiltIn,
            public FloorCudaBuiltIn,
            public FmodCudaBuiltIn,
            public LogCudaBuiltIn,
            public MaxCudaBuiltIn,
            public MinCudaBuiltIn,
            public PowCudaBuiltIn,
            public RemainderCudaBuiltIn,
            public RoundCudaBuiltIn,
            public RsqrtCudaBuiltIn,
            public SinCudaBuiltIn,
            public SqrtCudaBuiltIn,
            public TanCudaBuiltIn,
            public TruncCudaBuiltIn
        {};
    }
}

#endif
