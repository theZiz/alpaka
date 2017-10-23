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

#include <alpaka/core/Common.hpp>               // ALPAKA_FN_*, __HIPCC__

#include <alpaka/dev/DevHipRt.hpp>		// DevHipRt- as of now, this isn't implemented; DevHipRt itself is used instead.
#include <alpaka/dev/Traits.hpp>                // GetDev
#include <alpaka/event/Traits.hpp>              // event::traits::Test, ...
#include <alpaka/wait/Traits.hpp>               // CurrentThreadWaitFor


#include <alpaka/stream/StreamHipRtSync.hpp>   // stream::StreamHipRtSync (as of now, only a renamed copy of it's HIP counterpart)
#include <alpaka/stream/StreamHipRtAsync.hpp>  // stream::StreamHipRtAsync (as of now, only a renamed copy of it's HIP counterpart)

#include <alpaka/core/Hip.hpp>		    // as of now, just a renamed copy of it's HIP coutnerpart

#include <stdexcept>                            // std::runtime_error
#include <memory>                               // std::shared_ptr
#include <functional>                           // std::bind

namespace alpaka
{
    namespace event
    {
        namespace hip
        {
            namespace detail
            {
                //#############################################################################
                //! The HIP RT device event implementation.
                //#############################################################################
                class EventHipImpl final
                {
                public:
                    //-----------------------------------------------------------------------------
                    //! Constructor.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST EventHipImpl(
                        dev::DevHipRt const & dev,
                        bool bBusyWait) :
                            m_dev(dev),
                            m_HipEvent()
                    {
                        ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                        // Set the current device.
                        ALPAKA_HIP_RT_CHECK(
                            hipSetDevice(
                                m_dev.m_iDevice));
                        // Create the event on the current device with the specified flags. Valid flags include:
                        // - hipEventDefault: Default event creation flag.
                        // - hipEventBlockingSync : Specifies that event should use blocking synchronization.
                        //   A host thread that uses hipEventSynchronize() to wait on an event created with this flag will block until the event actually completes.
                        // - hipEventDisableTiming : Specifies that the created event does not need to record timing data.
                        //   Events created with this flag specified and the hipEventBlockingSync flag not specified will provide the best performance when used with hipStreamWaitEvent() and hipEventQuery().
                        ALPAKA_HIP_RT_CHECK(
                            hipEventCreateWithFlags(
                                &m_HipEvent,
                                (bBusyWait ? hipEventDefault : hipEventBlockingSync) | hipEventDisableTiming));
                    }
                    //-----------------------------------------------------------------------------
                    //! Copy constructor.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST EventHipImpl(EventHipImpl const &) = delete;
                    //-----------------------------------------------------------------------------
                    //! Move constructor.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST EventHipImpl(EventHipImpl &&) = default;
                    //-----------------------------------------------------------------------------
                    //! Copy assignment operator.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST auto operator=(EventHipImpl const &) -> EventHipImpl & = delete;
                    //-----------------------------------------------------------------------------
                    //! Move assignment operator.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST auto operator=(EventHipImpl &&) -> EventHipImpl & = default;
                    //-----------------------------------------------------------------------------
                    //! Destructor.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST ~EventHipImpl()
                    {
                        ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                        // Set the current device. \TODO: Is setting the current device before hipEventDestroy required?
                        ALPAKA_HIP_RT_CHECK(hipSetDevice(
                            m_dev.m_iDevice));
                        // In case event has been recorded but has not yet been completed when hipEventDestroy() is called, the function will return immediately
                        // and the resources associated with event will be released automatically once the device has completed event.
                        // -> No need to synchronize here.
                        ALPAKA_HIP_RT_CHECK(hipEventDestroy(
                            m_HipEvent));
                    }

                public:
                    dev::DevHipRt const m_dev;   //!< The device this event is bound to.
                    hipEvent_t m_HipEvent;
                };
            }
        }

        //#############################################################################
        //! The HIP RT device event.
        //#############################################################################
        class EventHipRt final
        {
        public:
            //-----------------------------------------------------------------------------
            //! Constructor.
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST EventHipRt(
                dev::DevHipRt const & dev,
                bool bBusyWait = true) :
                    m_spEventHipImpl(std::make_shared<hip::detail::EventHipImpl>(dev, bBusyWait))
            {
                ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;
            }
            //-----------------------------------------------------------------------------
            //! Copy constructor.
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST EventHipRt(EventHipRt const &) = default;
            //-----------------------------------------------------------------------------
            //! Move constructor.
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST EventHipRt(EventHipRt &&) = default;
            //-----------------------------------------------------------------------------
            //! Copy assignment operator.
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST auto operator=(EventHipRt const &) -> EventHipRt & = default;
            //-----------------------------------------------------------------------------
            //! Move assignment operator.
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST auto operator=(EventHipRt &&) -> EventHipRt & = default;
            //-----------------------------------------------------------------------------
            //! Equality comparison operator.
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST auto operator==(EventHipRt const & rhs) const
            -> bool
            {
                return (m_spEventHipImpl->m_HipEvent == rhs.m_spEventHipImpl->m_HipEvent);
            }
            //-----------------------------------------------------------------------------
            //! Equality comparison operator.
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST auto operator!=(EventHipRt const & rhs) const
            -> bool
            {
                return !((*this) == rhs);
            }
            //-----------------------------------------------------------------------------
            //! Destructor.
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST ~EventHipRt() = default;

        public:
            std::shared_ptr<hip::detail::EventHipImpl> m_spEventHipImpl;
        };
    }

    namespace dev
    {
        namespace traits
        {
            //#############################################################################
            //! The HIP RT device event device get trait specialization.
            //#############################################################################
            template<>
            struct GetDev<
                event::EventHipRt>
            {
                //-----------------------------------------------------------------------------
                //
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto getDev(
                    event::EventHipRt const & event)
                -> dev::DevHipRt
                {
                    return event.m_spEventHipImpl->m_dev;
                }
            };
        }
    }
    namespace event
    {
        namespace traits
        {
            //#############################################################################
            //! The HIP RT device event event type trait specialization.
            //#############################################################################
            template<>
            struct EventType<
                event::EventHipRt>
            {
                using type = event::EventHipRt;
            };
            //#############################################################################
            //! The CPU device event create trait specialization.
            //#############################################################################
            template<>
            struct Create<
                event::EventHipRt,
                dev::DevHipRt>
            {
                //-----------------------------------------------------------------------------
                //!
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto create(
                    dev::DevHipRt const & dev)
                -> event::EventHipRt
                {
                    return event::EventHipRt(dev);
                }
            };
            //#############################################################################
            //! The HIP RT device event test trait specialization.
            //#############################################################################
            template<>
            struct Test<
                event::EventHipRt>
            {
                //-----------------------------------------------------------------------------
                //
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto test(
                    event::EventHipRt const & event)
                -> bool
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                    // Query is allowed even for events on non current device.
                    hipError_t ret = hipSuccess;
                    ALPAKA_HIP_RT_CHECK_IGNORE(
                        ret = hipEventQuery(
                            event.m_spEventHipImpl->m_HipEvent),
                        hipErrorNotReady);
                    return (ret == hipSuccess);
                }
            };
        }
    }
    namespace stream
    {
        namespace traits
        {
            //#############################################################################
            //! The HIP RT stream enqueue trait specialization.
            //#############################################################################
            template<>
            struct Enqueue<
                stream::StreamHipRtAsync,
                event::EventHipRt>
            {
                //-----------------------------------------------------------------------------
                //
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto enqueue(
                    stream::StreamHipRtAsync & stream,
                    event::EventHipRt & event)
                -> void
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                    ALPAKA_HIP_RT_CHECK(hipEventRecord(
                        event.m_spEventHipImpl->m_HipEvent,
                        stream.m_spStreamHipRtAsyncImpl->m_HipStream));
                }
            };
            //#############################################################################
            //! The HIP RT stream enqueue trait specialization.
            //#############################################################################
            template<>
            struct Enqueue<
                stream::StreamHipRtSync,
                event::EventHipRt>
            {
                //-----------------------------------------------------------------------------
                //
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto enqueue(
                    stream::StreamHipRtSync & stream,
                    event::EventHipRt & event)
                -> void
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                    ALPAKA_HIP_RT_CHECK(hipEventRecord(
                        event.m_spEventHipImpl->m_HipEvent,
                        stream.m_spStreamHipRtSyncImpl->m_HipStream));
                }
            };
        }
    }
    namespace wait
    {
        namespace traits
        {
            //#############################################################################
            //! The HIP RT device event thread wait trait specialization.
            //!
            //! Waits until the event itself and therefore all tasks preceding it in the stream it is enqueued to have been completed.
            //! If the event is not enqueued to a stream the method returns immediately.
            //#############################################################################
            template<>
            struct CurrentThreadWaitFor<
                event::EventHipRt>
            {
                //-----------------------------------------------------------------------------
                //
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto currentThreadWaitFor(
                    event::EventHipRt const & event)
                -> void
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                    // Sync is allowed even for events on non current device.
                    ALPAKA_HIP_RT_CHECK(hipEventSynchronize(
                        event.m_spEventHipImpl->m_HipEvent));
                }
            };
            //#############################################################################
            //! The HIP RT stream event wait trait specialization.
            //#############################################################################
            template<>
            struct WaiterWaitFor<
                stream::StreamHipRtAsync,
                event::EventHipRt>
            {
                //-----------------------------------------------------------------------------
                //
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto waiterWaitFor(
                    stream::StreamHipRtAsync & stream,
                    event::EventHipRt const & event)
                -> void
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                    ALPAKA_HIP_RT_CHECK(hipStreamWaitEvent(
                        stream.m_spStreamHipRtAsyncImpl->m_HipStream,
                        event.m_spEventHipImpl->m_HipEvent,
                        0));
                }
            };
            //#############################################################################
            //! The HIP RT stream event wait trait specialization.
            //#############################################################################
            template<>
            struct WaiterWaitFor<
                stream::StreamHipRtSync,
                event::EventHipRt>
            {
                //-----------------------------------------------------------------------------
                //
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto waiterWaitFor(
                    stream::StreamHipRtSync & stream,
                    event::EventHipRt const & event)
                -> void
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                    ALPAKA_HIP_RT_CHECK(hipStreamWaitEvent(
                        stream.m_spStreamHipRtSyncImpl->m_HipStream,
                        event.m_spEventHipImpl->m_HipEvent,
                        0));
                }
            };
            //#############################################################################
            //! The HIP RT device event wait trait specialization.
            //!
            //! Any future work submitted in any stream of this device will wait for event to complete before beginning execution.
            //#############################################################################
            template<>
            struct WaiterWaitFor<
                dev::DevHipRt,
                event::EventHipRt>
            {
                //-----------------------------------------------------------------------------
                //
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto waiterWaitFor(
                    dev::DevHipRt & dev,
                    event::EventHipRt const & event)
                -> void
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                    // Set the current device.
                    ALPAKA_HIP_RT_CHECK(
                        hipSetDevice(
                            dev.m_iDevice));

                    ALPAKA_HIP_RT_CHECK(hipStreamWaitEvent(
                        0,
                        event.m_spEventHipImpl->m_HipEvent,
                        0));
                }
            };
        }
    }
}

#endif
