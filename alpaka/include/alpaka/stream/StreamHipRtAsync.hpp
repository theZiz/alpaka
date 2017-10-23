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

#include <alpaka/dev/DevHipRt.hpp>	// DevHipRt- as of now, this isn't implemented; DevHipRt itself is used instead.

#include <alpaka/dev/Traits.hpp>        // dev::GetDev, dev::DevType
#include <alpaka/event/Traits.hpp>      // event::EventType
#include <alpaka/stream/Traits.hpp>     // stream::traits::Enqueue, ...
#include <alpaka/wait/Traits.hpp>       // CurrentThreadWaitFor, WaiterWaitFor

#include <alpaka/core/Hip.hpp>		    // as of now, just a renamed copy of it's HIP coutnerpart

#include <stdexcept>                    // std::runtime_error
#include <memory>                       // std::shared_ptr
#include <functional>                   // std::bind

namespace alpaka
{
    namespace event
    {
        class EventHipRt;
    }
}

namespace alpaka
{
    namespace stream
    {
        namespace hip
        {
            namespace detail
            {
                //#############################################################################
                //! The HIP RT stream implementation.
                //#############################################################################
                class StreamHipRtAsyncImpl final
                {
                public:
                    //-----------------------------------------------------------------------------
                    //! Constructor.
                    //-----------------------------------------------------------------------------
                    StreamHipRtAsyncImpl(
                        dev::DevHipRt const & dev) :
                            m_dev(dev),
                            m_HipStream()
                    {
                        ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                        // Set the current device.
                        ALPAKA_HIP_RT_CHECK(
                            hipSetDevice(
                                m_dev.m_iDevice));
                        // - hipStreamDefault: Default stream creation flag.
                        // - hipStreamNonBlocking: Specifies that work running in the created stream may run concurrently with work in stream 0 (the NULL stream),
                        //   and that the created stream should perform no implicit synchronization with stream 0.
                        // Create the stream on the current device.
                        // NOTE: hipStreamNonBlocking is required to match the semantic implemented in the alpaka CPU stream.
                        // It would be too much work to implement implicit default stream synchronization on CPU.
                        ALPAKA_HIP_RT_CHECK(
                            hipStreamCreateWithFlags(
                                &m_HipStream,
                                hipStreamNonBlocking));
                    }
                    //-----------------------------------------------------------------------------
                    //! Copy constructor.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST StreamHipRtAsyncImpl(StreamHipRtAsyncImpl const &) = delete;
                    //-----------------------------------------------------------------------------
                    //! Move constructor.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST StreamHipRtAsyncImpl(StreamHipRtAsyncImpl &&) = default;
                    //-----------------------------------------------------------------------------
                    //! Copy assignment operator.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST auto operator=(StreamHipRtAsyncImpl const &) -> StreamHipRtAsyncImpl & = delete;
                    //-----------------------------------------------------------------------------
                    //! Move assignment operator.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST auto operator=(StreamHipRtAsyncImpl &&) -> StreamHipRtAsyncImpl & = default;
                    //-----------------------------------------------------------------------------
                    //! Destructor.
                    //-----------------------------------------------------------------------------
                    ~StreamHipRtAsyncImpl()
                    {
                        ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                        // Set the current device. \TODO: Is setting the current device before hipStreamDestroy required?
                        ALPAKA_HIP_RT_CHECK(
                            hipSetDevice(
                                m_dev.m_iDevice));
                        // In case the device is still doing work in the stream when hipStreamDestroy() is called, the function will return immediately
                        // and the resources associated with stream will be released automatically once the device has completed all work in stream.
                        // -> No need to synchronize here.
                        ALPAKA_HIP_RT_CHECK(
                            hipStreamDestroy(
                                m_HipStream));
                    }

                public:
                    dev::DevHipRt const m_dev;   //!< The device this stream is bound to.
                    hipStream_t m_HipStream;
                };
            }
        }

        //#############################################################################
        //! The HIP RT stream.
        //#############################################################################
        class StreamHipRtAsync final
        {
        public:
            //-----------------------------------------------------------------------------
            //! Constructor.
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST StreamHipRtAsync(
                dev::DevHipRt const & dev) :
                m_spStreamHipRtAsyncImpl(std::make_shared<hip::detail::StreamHipRtAsyncImpl>(dev))
            {}
            //-----------------------------------------------------------------------------
            //! Copy constructor.
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST StreamHipRtAsync(StreamHipRtAsync const &) = default;
            //-----------------------------------------------------------------------------
            //! Move constructor.
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST StreamHipRtAsync(StreamHipRtAsync &&) = default;
            //-----------------------------------------------------------------------------
            //! Copy assignment operator.
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST auto operator=(StreamHipRtAsync const &) -> StreamHipRtAsync & = default;
            //-----------------------------------------------------------------------------
            //! Move assignment operator.
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST auto operator=(StreamHipRtAsync &&) -> StreamHipRtAsync & = default;
            //-----------------------------------------------------------------------------
            //! Equality comparison operator.
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST auto operator==(StreamHipRtAsync const & rhs) const
            -> bool
            {
                return (m_spStreamHipRtAsyncImpl->m_HipStream == rhs.m_spStreamHipRtAsyncImpl->m_HipStream);
            }
            //-----------------------------------------------------------------------------
            //! Equality comparison operator.
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST auto operator!=(StreamHipRtAsync const & rhs) const
            -> bool
            {
                return !((*this) == rhs);
            }
            //-----------------------------------------------------------------------------
            //! Destructor.
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST ~StreamHipRtAsync() = default;

        public:
            std::shared_ptr<hip::detail::StreamHipRtAsyncImpl> m_spStreamHipRtAsyncImpl;
        };
    }

    namespace dev
    {
        namespace traits
        {
            //#############################################################################
            //! The HIP RT stream device type trait specialization.
            //#############################################################################
            template<>
            struct DevType<
                stream::StreamHipRtAsync>
            {
                using type = dev::DevHipRt;
            };
            //#############################################################################
            //! The HIP RT stream device get trait specialization.
            //#############################################################################
            template<>
            struct GetDev<
                stream::StreamHipRtAsync>
            {
                //-----------------------------------------------------------------------------
                //
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto getDev(
                    stream::StreamHipRtAsync const & stream)
                -> dev::DevHipRt
                {
                    return stream.m_spStreamHipRtAsyncImpl->m_dev;
                }
            };
        }
    }
    namespace event
    {
        namespace traits
        {
            //#############################################################################
            //! The HIP RT stream event type trait specialization.
            //#############################################################################
            template<>
            struct EventType<
                stream::StreamHipRtAsync>
            {
                using type = event::EventHipRt;
            };
        }
    }
    namespace stream
    {
        namespace traits
        {
            //#############################################################################
            //! The HIP RT stream test trait specialization.
            //#############################################################################
            template<>
            struct Empty<
                stream::StreamHipRtAsync>
            {
                //-----------------------------------------------------------------------------
                //
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto empty(
                    stream::StreamHipRtAsync const & stream)
                -> bool
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                    // Query is allowed even for streams on non current device.
                    hipError_t ret = hipSuccess;
                    ALPAKA_HIP_RT_CHECK_IGNORE(
                        ret = hipStreamQuery(
                            stream.m_spStreamHipRtAsyncImpl->m_HipStream),
                        hipErrorNotReady);
                    return (ret == hipSuccess);
                }
            };
        }
    }
    namespace wait
    {
        namespace traits
        {
            //#############################################################################
            //! The HIP RT stream thread wait trait specialization.
            //!
            //! Blocks execution of the calling thread until the stream has finished processing all previously requested tasks (kernels, data copies, ...)
            //#############################################################################
            template<>
            struct CurrentThreadWaitFor<
                stream::StreamHipRtAsync>
            {
                //-----------------------------------------------------------------------------
                //
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto currentThreadWaitFor(
                    stream::StreamHipRtAsync const & stream)
                -> void
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                    // Sync is allowed even for streams on non current device.
                    ALPAKA_HIP_RT_CHECK(
                        hipStreamSynchronize(
                            stream.m_spStreamHipRtAsyncImpl->m_HipStream));
                }
            };
        }
    }
}

#endif
