/**
* \file
* Copyright 2014-2015 Benjamin Worpitz, Rene Widera
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

#include <alpaka/dev/Traits.hpp>        // dev::traits::DevType
#include <alpaka/mem/buf/Traits.hpp>    // mem::buf::traits::BufType
#include <alpaka/pltf/Traits.hpp>       // pltf::traits::PltfType
#include <alpaka/wait/Traits.hpp>       // CurrentThreadWaitFor

#include <alpaka/core/Hip.hpp>

namespace alpaka
{
    namespace pltf
    {
        namespace traits
        {
            template<
                typename TPltf,
                typename TSfinae>
            struct GetDevByIdx;
        }
        class PltfHipRt;
    }

    namespace dev
    {
        //#############################################################################
        //! The HIP RT device handle.
        //#############################################################################
        class DevHipRt
        {
            friend struct pltf::traits::GetDevByIdx<pltf::PltfHipRt>;

        protected:
            //-----------------------------------------------------------------------------
            //! Constructor.
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST DevHipRt() = default;
        public:
            //-----------------------------------------------------------------------------
            //! Copy constructor.
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST DevHipRt(DevHipRt const &) = default;
            //-----------------------------------------------------------------------------
            //! Move constructor.
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST DevHipRt(DevHipRt &&) = default;
            //-----------------------------------------------------------------------------
            //! Copy assignment operator.
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST auto operator=(DevHipRt const &) -> DevHipRt & = default;
            //-----------------------------------------------------------------------------
            //! Move assignment operator.
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST auto operator=(DevHipRt &&) -> DevHipRt & = default;
            //-----------------------------------------------------------------------------
            //! Equality comparison operator.
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST auto operator==(DevHipRt const & rhs) const
            -> bool
            {
                return m_iDevice == rhs.m_iDevice;
            }
            //-----------------------------------------------------------------------------
            //! Inequality comparison operator.
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST auto operator!=(DevHipRt const & rhs) const
            -> bool
            {
                return !((*this) == rhs);
            }

        public:
            int m_iDevice;
        };
    }

    namespace dev
    {
        namespace traits
        {
            //#############################################################################
            //! The HIP RT device name get trait specialization.
            //#############################################################################
            template<>
            struct GetName<
                dev::DevHipRt>
            {
                //-----------------------------------------------------------------------------
                //
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto getName(
                    dev::DevHipRt const & dev)
                -> std::string
                {
                    hipDeviceProp_t hipDevProp;
                    ALPAKA_HIP_RT_CHECK(
                        hipGetDeviceProperties(
                            &hipDevProp,
                            dev.m_iDevice));

                    return std::string(hipDevProp.name);
                }
            };

            //#############################################################################
            //! The HIP RT device available memory get trait specialization.
            //#############################################################################
            template<>
            struct GetMemBytes<
                dev::DevHipRt>
            {
                //-----------------------------------------------------------------------------
                //
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto getMemBytes(
                    dev::DevHipRt const & dev)
                -> std::size_t
                {
                    // Set the current device to wait for.
                    ALPAKA_HIP_RT_CHECK(
                        hipSetDevice(
                            dev.m_iDevice));

                    std::size_t freeInternal(0u);
                    std::size_t totalInternal(0u);

                    // \TODO: Check which is faster: hipMemGetInfo().totalInternal vs hipGetDeviceProperties().totalGlobalMem
                    ALPAKA_HIP_RT_CHECK(
                        hipMemGetInfo(
                            &freeInternal,
                            &totalInternal));

                    return totalInternal;
                }
            };

            //#############################################################################
            //! The HIP RT device free memory get trait specialization.
            //#############################################################################
            template<>
            struct GetFreeMemBytes<
                dev::DevHipRt>
            {
                //-----------------------------------------------------------------------------
                //
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto getFreeMemBytes(
                    dev::DevHipRt const & dev)
                -> std::size_t
                {
                    // Set the current device to wait for.
                    ALPAKA_HIP_RT_CHECK(
                        hipSetDevice(
                            dev.m_iDevice));

                    std::size_t freeInternal(0u);
                    std::size_t totalInternal(0u);

                    ALPAKA_HIP_RT_CHECK(
                        hipMemGetInfo(
                            &freeInternal,
                            &totalInternal));

                    return freeInternal;
                }
            };

            //#############################################################################
            //! The HIP RT device reset trait specialization.
            //#############################################################################
            template<>
            struct Reset<
                dev::DevHipRt>
            {
                //-----------------------------------------------------------------------------
                //
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto reset(
                    dev::DevHipRt const & dev)
                -> void
                {
                    ALPAKA_DEBUG_FULL_LOG_SCOPE;

                    // Set the current device to wait for.
                    ALPAKA_HIP_RT_CHECK(
                        hipSetDevice(
                            dev.m_iDevice));
                    ALPAKA_HIP_RT_CHECK(
                        hipDeviceReset());
                }
            };
        }
    }
    namespace mem
    {
        namespace buf
        {
            template<
                typename TElem,
                typename TDim,
                typename TSize>
            class BufHipRt;

            namespace traits
            {
                //#############################################################################
                //! The HIP RT device memory buffer type trait specialization.
                //#############################################################################
                template<
                    typename TElem,
                    typename TDim,
                    typename TSize>
                struct BufType<
                    dev::DevHipRt,
                    TElem,
                    TDim,
                    TSize>
                {
                    using type = mem::buf::BufHipRt<TElem, TDim, TSize>;
                };
            }
        }
    }
    namespace pltf
    {
        namespace traits
        {
            //#############################################################################
            //! The HIP RT device platform type trait specialization.
            //#############################################################################
            template<>
            struct PltfType<
                dev::DevHipRt>
            {
                using type = pltf::PltfHipRt;
            };
        }
    }
    namespace wait
    {
        namespace traits
        {
            //#############################################################################
            //! The thread HIP device wait specialization.
            //!
            //! Blocks until the device has completed all preceding requested tasks.
            //! Tasks that are enqueued or streams that are created after this call is made are not waited for.
            //#############################################################################
            template<>
            struct CurrentThreadWaitFor<
                dev::DevHipRt>
            {
                //-----------------------------------------------------------------------------
                //
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto currentThreadWaitFor(
                    dev::DevHipRt const & dev)
                -> void
                {
                    ALPAKA_DEBUG_FULL_LOG_SCOPE;

                    // Set the current device to wait for.
                    ALPAKA_HIP_RT_CHECK(hipSetDevice(
                        dev.m_iDevice));
                    ALPAKA_HIP_RT_CHECK(hipDeviceSynchronize());
                }
            };
        }
    }
}

#endif
