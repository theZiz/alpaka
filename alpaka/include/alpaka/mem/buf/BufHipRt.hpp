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

#include <alpaka/dev/DevHipRt.hpp>	    // dev::DevHipRt (as of now, the hip version itself is used)	
#include <alpaka/vec/Vec.hpp>               // Vec
#include <alpaka/core/Hip.hpp>		    // hipMalloc,...  		as of now, just a renamed copy of it's HIP coutnerpart	

#include <alpaka/dev/Traits.hpp>            // dev::traits::DevType
#include <alpaka/dim/DimIntegralConst.hpp>  // dim::DimInt<N>
#include <alpaka/mem/buf/Traits.hpp>        // mem::view::Copy, ...

#include <cassert>                          // assert
#include <memory>                           // std::shared_ptr

#include <hip/hip_runtime.h>		    // temporary fix



namespace alpaka
{
    namespace dev
    {
        class DevHipRt;
    }
    namespace mem
    {
        namespace buf
        {
            template<
                typename TElem,
                typename TDim,
                typename TSize>
            class BufCpu;
        }
    }
    namespace mem
    {
        namespace buf
        {
            //#############################################################################
            //! The HIP memory buffer.
            //#############################################################################
            template<
                typename TElem,
                typename TDim,
                typename TSize>
            class BufHipRt
            {
            private:
                using Elem = TElem;
                using Dim = TDim;

            public:
                //-----------------------------------------------------------------------------
                //! Constructor
                //-----------------------------------------------------------------------------
                template<
                    typename TExtent>
                ALPAKA_FN_HOST BufHipRt(
                    dev::DevHipRt const & dev,
                    TElem * const pMem,
                    TSize const & pitchBytes,
                    TExtent const & extent) :
                        m_dev(dev),
                        m_extentElements(extent::getExtentVecEnd<TDim>(extent)),
                        m_spMem(
                            pMem,
                            // NOTE: Because the BufHipRt object can be copied and the original object could have been destroyed,
                            // a std::ref(m_dev) or a this pointer can not be bound to the callback because they are not always valid at time of destruction.
                            std::bind(&BufHipRt::freeBuffer, std::placeholders::_1, m_dev)),
                        m_pitchBytes(pitchBytes)
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                    static_assert(
                        TDim::value == dim::Dim<TExtent>::value,
                        "The dimensionality of TExtent and the dimensionality of the TDim template parameter have to be identical!");
                    static_assert(
                        std::is_same<TSize, size::Size<TExtent>>::value,
                        "The size type of TExtent and the TSize template parameter have to be identical!");
                }

            private:
                //-----------------------------------------------------------------------------
                //! Frees the shared buffer.
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto freeBuffer(
                    TElem * memPtr,
                    dev::DevHipRt const & dev)
                -> void
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                    // Set the current device. \TODO: Is setting the current device before hipFree required?
                    ALPAKA_HIP_RT_CHECK(
                        hipSetDevice(
                            dev.m_iDevice));
                    // Free the buffer.
                    hipFree(reinterpret_cast<void *>(memPtr));
                }

            public:
                dev::DevHipRt m_dev;               // NOTE: The device has to be destructed after the memory pointer because it is required for destruction.
                vec::Vec<TDim, TSize> m_extentElements;
                std::shared_ptr<TElem> m_spMem;
                TSize m_pitchBytes;
            };
        }
    }

    //-----------------------------------------------------------------------------
    // Trait specializations for BufHipRt.
    //-----------------------------------------------------------------------------
    namespace dev
    {
        namespace traits
        {
            //#############################################################################
            //! The BufHipRt device type trait specialization.
            //#############################################################################
            template<
                typename TElem,
                typename TDim,
                typename TSize>
            struct DevType<
                mem::buf::BufHipRt<TElem, TDim, TSize>>
            {
                using type = dev::DevHipRt;
            };
            //#############################################################################
            //! The BufHipRt device get trait specialization.
            //#############################################################################
            template<
                typename TElem,
                typename TDim,
                typename TSize>
            struct GetDev<
                mem::buf::BufHipRt<TElem, TDim, TSize>>
            {
                ALPAKA_FN_HOST static auto getDev(
                    mem::buf::BufHipRt<TElem, TDim, TSize> const & buf)
                -> dev::DevHipRt
                {
                    return buf.m_dev;
                }
            };
        }
    }
    namespace dim
    {
        namespace traits
        {
            //#############################################################################
            //! The BufHipRt dimension getter trait specialization.
            //#############################################################################
            template<
                typename TElem,
                typename TDim,
                typename TSize>
            struct DimType<
                mem::buf::BufHipRt<TElem, TDim, TSize>>
            {
                using type = TDim;
            };
        }
    }
    namespace elem
    {
        namespace traits
        {
            //#############################################################################
            //! The BufHipRt memory element type get trait specialization.
            //#############################################################################
            template<
                typename TElem,
                typename TDim,
                typename TSize>
            struct ElemType<
                mem::buf::BufHipRt<TElem, TDim, TSize>>
            {
                using type = TElem;
            };
        }
    }
    namespace extent
    {
        namespace traits
        {
            //#############################################################################
            //! The BufHipRt extent get trait specialization.
            //#############################################################################
            template<
                typename TIdx,
                typename TElem,
                typename TDim,
                typename TSize>
            struct GetExtent<
                TIdx,
                mem::buf::BufHipRt<TElem, TDim, TSize>,
                typename std::enable_if<(TDim::value > TIdx::value)>::type>
            {
                //-----------------------------------------------------------------------------
                //!
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto getExtent(
                    mem::buf::BufHipRt<TElem, TDim, TSize> const & extent)
                -> TSize
                {
                    return extent.m_extentElements[TIdx::value];
                }
            };
        }
    }
    namespace mem
    {
        namespace view
        {
            namespace traits
            {
                //#############################################################################
                //! The BufHipRt native pointer get trait specialization.
                //#############################################################################
                template<
                    typename TElem,
                    typename TDim,
                    typename TSize>
                struct GetPtrNative<
                    mem::buf::BufHipRt<TElem, TDim, TSize>>
                {
                    //-----------------------------------------------------------------------------
                    //!
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST static auto getPtrNative(
                        mem::buf::BufHipRt<TElem, TDim, TSize> const & buf)
                    -> TElem const *
                    {
                        return buf.m_spMem.get();
                    }
                    //-----------------------------------------------------------------------------
                    //!
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST static auto getPtrNative(
                        mem::buf::BufHipRt<TElem, TDim, TSize> & buf)
                    -> TElem *
                    {
                        return buf.m_spMem.get();
                    }
                };
                //#############################################################################
                //! The BufHipRt pointer on device get trait specialization.
                //#############################################################################
                template<
                    typename TElem,
                    typename TDim,
                    typename TSize>
                struct GetPtrDev<
                    mem::buf::BufHipRt<TElem, TDim, TSize>,
                    dev::DevHipRt>
                {
                    //-----------------------------------------------------------------------------
                    //!
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST static auto getPtrDev(
                        mem::buf::BufHipRt<TElem, TDim, TSize> const & buf,
                        dev::DevHipRt const & dev)
                    -> TElem const *
                    {
                        if(dev == dev::getDev(buf))
                        {
                            return buf.m_spMem.get();
                        }
                        else
                        {
                            throw std::runtime_error("The buffer is not accessible from the given device!");
                        }
                    }
                    //-----------------------------------------------------------------------------
                    //!
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST static auto getPtrDev(
                        mem::buf::BufHipRt<TElem, TDim, TSize> & buf,
                        dev::DevHipRt const & dev)
                    -> TElem *
                    {
                        if(dev == dev::getDev(buf))
                        {
                            return buf.m_spMem.get();
                        }
                        else
                        {
                            throw std::runtime_error("The buffer is not accessible from the given device!");
                        }
                    }
                };
                //#############################################################################
                //! The BufHipRt pitch get trait specialization.
                //#############################################################################
                template<
                    typename TElem,
                    typename TDim,
                    typename TSize>
                struct GetPitchBytes<
                    dim::DimInt<TDim::value - 1u>,
                    mem::buf::BufHipRt<TElem, TDim, TSize>>
                {
                    //-----------------------------------------------------------------------------
                    //!
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST static auto getPitchBytes(
                        mem::buf::BufHipRt<TElem, TDim, TSize> const & buf)
                    -> TSize
                    {
                        return buf.m_pitchBytes;
                    }
                };
            }
        }
        namespace buf
        {
            namespace traits
            {
                //#############################################################################
                //! The HIP 1D memory allocation trait specialization.
                //#############################################################################
                template<
                    typename T,
                    typename TSize>
                struct Alloc<
                    T,
                    dim::DimInt<1u>,
                    TSize,
                    dev::DevHipRt>
                {
                    //-----------------------------------------------------------------------------
                    //!
                    //-----------------------------------------------------------------------------
                    template<
                        typename TExtent>
                    ALPAKA_FN_HOST static auto alloc(
                        dev::DevHipRt const & dev,
                        TExtent const & extent)
                    -> mem::buf::BufHipRt<T, dim::DimInt<1u>, TSize>
                    {
                        ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                        auto const width(extent::getWidth(extent));
                        auto const widthBytes(width * static_cast<TSize>(sizeof(T)));

                        // Set the current device.
                        ALPAKA_HIP_RT_CHECK(
                            hipSetDevice(
                                dev.m_iDevice));
                        // Allocate the buffer on this device.
                        void * memPtr;
                        ALPAKA_HIP_RT_CHECK(
                            hipMalloc(
                                &memPtr,
                                static_cast<std::size_t>(widthBytes)));

#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                        std::cout << BOOST_CURRENT_FUNCTION
                            << " ew: " << width
                            << " ewb: " << widthBytes
                            << " ptr: " << memPtr
                            << std::endl;
#endif
                        return
                            mem::buf::BufHipRt<T, dim::DimInt<1u>, TSize>(
                                dev,
                                reinterpret_cast<T *>(memPtr),
                                static_cast<TSize>(widthBytes),
                                extent);
                    }
                };
                //#############################################################################
                //! The HIP 2D memory allocation trait specialization.
                //#############################################################################
                template<
                    typename T,
                    typename TSize>
                struct Alloc<
                    T,
                    dim::DimInt<2u>,
                    TSize,
                    dev::DevHipRt>
                {
                    //-----------------------------------------------------------------------------
                    //!
                    //-----------------------------------------------------------------------------
                    template<
                        typename TExtent>
                    ALPAKA_FN_HOST static auto alloc(
                        dev::DevHipRt const & dev,
                        TExtent const & extent)
                    -> mem::buf::BufHipRt<T, dim::DimInt<2u>, TSize>
                    {
                        ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                        auto const width(extent::getWidth(extent));
                        auto const widthBytes(width * static_cast<TSize>(sizeof(T)));
                        auto const height(extent::getHeight(extent));

                        // Set the current device.
                        ALPAKA_HIP_RT_CHECK(
                            hipSetDevice(
                                dev.m_iDevice));
                        // Allocate the buffer on this device.
                        void * memPtr;
                        std::size_t pitchBytes;
                        ALPAKA_HIP_RT_CHECK(
                            hipMallocPitch(
                                &memPtr,
                                &pitchBytes,
                                static_cast<std::size_t>(widthBytes),
                                static_cast<std::size_t>(height)));
                        assert(pitchBytes>=widthBytes||(width*height)==0);

#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                        std::cout << BOOST_CURRENT_FUNCTION
                            << " ew: " << width
                            << " eh: " << height
                            << " ewb: " << widthBytes
                            << " ptr: " << memPtr
                            << " pitch: " << pitchBytes
                            << std::endl;
#endif
                        return
                            mem::buf::BufHipRt<T, dim::DimInt<2u>, TSize>(
                                dev,
                                reinterpret_cast<T *>(memPtr),
                                static_cast<TSize>(pitchBytes),
                                extent);
                    }
                };
                //#############################################################################
                //! The HIP 3D memory allocation trait specialization.
                //#############################################################################
                //~ template<
                    //~ typename T,
                    //~ typename TSize>
                //~ struct Alloc<
                    //~ T,
                    //~ dim::DimInt<3u>,
                    //~ TSize,
                    //~ dev::DevHipRt>
                //~ {
                    //~ //-----------------------------------------------------------------------------
                    //~ //!
                    //~ //-----------------------------------------------------------------------------
                    //~ template<
                        //~ typename TExtent>
                    //~ ALPAKA_FN_HOST static auto alloc(
                        //~ dev::DevHipRt const & dev,
                        //~ TExtent const & extent)
                    //~ -> mem::buf::BufHipRt<T, dim::DimInt<3u>, TSize>
                    //~ {
                        //~ ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                         //~ hipExtent const hipExtentVal(
                             //~ make_hipExtent(
                                //~ static_cast<std::size_t>(extent::getWidth(extent) * static_cast<TSize>(sizeof(T))),
                                //~ static_cast<std::size_t>(extent::getHeight(extent)),
                                //~ static_cast<std::size_t>(extent::getDepth(extent))));

                        //~ // Set the current device.
                        //~ ALPAKA_HIP_RT_CHECK(
                            //~ hipSetDevice(
                                //~ dev.m_iDevice));
                        //~ // Allocate the buffer on this device.
                        //~ hipPitchedPtr hipPitchedPtrVal;
                        //~ ALPAKA_HIP_RT_CHECK(
                            //~ hipMalloc3D(
                                //~ &hipPitchedPtrVal,
                                //~ hipExtentVal));


//~ #if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                        //~ std::cout << BOOST_CURRENT_FUNCTION
                            //~ << " ew: " << extent::getWidth(extent)
                            //~ << " eh: " << hipExtentVal.height
                            //~ << " ed: " << hipExtentVal.depth
                            //~ << " ewb: " << hipExtentVal.width
                            //~ << " ptr: " << hipPitchedPtrVal.ptr
                            //~ << " pitch: " << hipPitchedPtrVal.pitch
                            //~ << " wb: " << hipPitchedPtrVal.xsize
                            //~ << " h: " << hipPitchedPtrVal.ysize
                            //~ << std::endl;
//~ #endif
                        //~ return
                            //~ mem::buf::BufHipRt<T, dim::DimInt<3u>, TSize>(
                                //~ dev,
                                //~ reinterpret_cast<T *>(hipPitchedPtrVal.ptr),
                                //~ static_cast<TSize>(hipPitchedPtrVal.pitch),
                                //~ extent);
                    //~ }
                //~ };
                //#############################################################################
                //! The BufHipRt HIP device memory mapping trait specialization.
                //#############################################################################
                template<
                    typename TElem,
                    typename TDim,
                    typename TSize>
                struct Map<
                    mem::buf::BufHipRt<TElem, TDim, TSize>,
                    dev::DevHipRt>
                {
                    //-----------------------------------------------------------------------------
                    //!
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST static auto map(
                        mem::buf::BufHipRt<TElem, TDim, TSize> const & buf,
                        dev::DevHipRt const & dev)
                    -> void
                    {
                        ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                        if(dev::getDev(buf) != dev)
                        {
                            throw std::runtime_error("Mapping memory from one HIP device into an other HIP device not implemented!");
                        }
                        // If it is already the same device, nothing has to be mapped.
                    }
                };
                //#############################################################################
                //! The BufHipRt HIP device memory unmapping trait specialization.
                //#############################################################################
                template<
                    typename TElem,
                    typename TDim,
                    typename TSize>
                struct Unmap<
                    mem::buf::BufHipRt<TElem, TDim, TSize>,
                    dev::DevHipRt>
                {
                    //-----------------------------------------------------------------------------
                    //!
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST static auto unmap(
                        mem::buf::BufHipRt<TElem, TDim, TSize> const & buf,
                        dev::DevHipRt const & dev)
                    -> void
                    {
                        ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                        if(dev::getDev(buf) != dev)
                        {
                            throw std::runtime_error("Unmapping memory mapped from one HIP device into an other HIP device not implemented!");
                        }
                        // If it is already the same device, nothing has to be unmapped.
                    }
                };
                //#############################################################################
                //! The BufHipRt memory pinning trait specialization.
                //#############################################################################
                template<
                    typename TElem,
                    typename TDim,
                    typename TSize>
                struct Pin<
                    mem::buf::BufHipRt<TElem, TDim, TSize>>
                {
                    //-----------------------------------------------------------------------------
                    //!
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST static auto pin(
                        mem::buf::BufHipRt<TElem, TDim, TSize> &)
                    -> void
                    {
                        ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                        // HIP device memory is always pinned, it can not be swapped out.
                    }
                };
                //#############################################################################
                //! The BufHipRt memory unpinning trait specialization.
                //#############################################################################
                template<
                    typename TElem,
                    typename TDim,
                    typename TSize>
                struct Unpin<
                    mem::buf::BufHipRt<TElem, TDim, TSize>>
                {
                    //-----------------------------------------------------------------------------
                    //!
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST static auto unpin(
                        mem::buf::BufHipRt<TElem, TDim, TSize> &)
                    -> void
                    {
                        ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                        // HIP device memory is always pinned, it can not be swapped out.
                    }
                };
                //#############################################################################
                //! The BufHipRt memory pin state trait specialization.
                //#############################################################################
                template<
                    typename TElem,
                    typename TDim,
                    typename TSize>
                struct IsPinned<
                    mem::buf::BufHipRt<TElem, TDim, TSize>>
                {
                    //-----------------------------------------------------------------------------
                    //!
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST static auto isPinned(
                        mem::buf::BufHipRt<TElem, TDim, TSize> const &)
                    -> bool
                    {
                        ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                        // HIP device memory is always pinned, it can not be swapped out.
                        return true;
                    }
                };
            }
        }
    }
    namespace offset
    {
        namespace traits
        {
            //#############################################################################
            //! The BufHipRt offset get trait specialization.
            //#############################################################################
            template<
                typename TIdx,
                typename TElem,
                typename TDim,
                typename TSize>
            struct GetOffset<
                TIdx,
                mem::buf::BufHipRt<TElem, TDim, TSize>>
            {
                //-----------------------------------------------------------------------------
                //!
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto getOffset(
                   mem::buf::BufHipRt<TElem, TDim, TSize> const &)
                -> TSize
                {
                    return 0u;
                }
            };
        }
    }
    namespace size
    {
        namespace traits
        {
            //#############################################################################
            //! The BufHipRt size type trait specialization.
            //#############################################################################
            template<
                typename TElem,
                typename TDim,
                typename TSize>
            struct SizeType<
                mem::buf::BufHipRt<TElem, TDim, TSize>>
            {
                using type = TSize;
            };
        }
    }

    //-----------------------------------------------------------------------------
    // Trait specializations for BufCpu.
    //-----------------------------------------------------------------------------
    namespace mem
    {
        namespace buf
        {
            namespace traits
            {
                //#############################################################################
                //! The BufCpu HIP device memory mapping trait specialization.
                //#############################################################################
                template<
                    typename TElem,
                    typename TDim,
                    typename TSize>
                struct Map<
                    mem::buf::BufCpu<TElem, TDim, TSize>,
                    dev::DevHipRt>
                {
                    //-----------------------------------------------------------------------------
                    //!
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST static auto map(
                        mem::buf::BufCpu<TElem, TDim, TSize> & buf,
                        dev::DevHipRt const & dev)
                    -> void
                    {
                        ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                        if(dev::getDev(buf) != dev)
                        {
                            // hipHostRegisterMapped:
                            //   Maps the allocation into the HIP address space.The device pointer to the memory may be obtained by calling hipHostGetDevicePointer().
                            //   This feature is available only on GPUs with compute capability greater than or equal to 1.1.
                            ALPAKA_HIP_RT_CHECK(
                                hipHostRegister(
                                    const_cast<void *>(reinterpret_cast<void const *>(mem::view::getPtrNative(buf))),
                                    extent::getProductOfExtent(buf) * sizeof(elem::Elem<BufCpu<TElem, TDim, TSize>>),
                                    hipHostRegisterMapped));
                        }
                        // If it is already the same device, nothing has to be mapped.
                    }
                };
                //#############################################################################
                //! The BufCpu HIP device memory unmapping trait specialization.
                //#############################################################################
                template<
                    typename TElem,
                    typename TDim,
                    typename TSize>
                struct Unmap<
                    mem::buf::BufCpu<TElem, TDim, TSize>,
                    dev::DevHipRt>
                {
                    //-----------------------------------------------------------------------------
                    //!
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST static auto unmap(
                        mem::buf::BufCpu<TElem, TDim, TSize> & buf,
                        dev::DevHipRt const & dev)
                    -> void
                    {
                        ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                        if(dev::getDev(buf) != dev)
                        {
                            // Unmaps the memory range whose base address is specified by ptr, and makes it pageable again.
                            // \FIXME: If the memory has separately been pinned before we destroy the pinning state.
                            ALPAKA_HIP_RT_CHECK(
                                hipHostUnregister(
                                    const_cast<void *>(reinterpret_cast<void const *>(mem::view::getPtrNative(buf)))));
                        }
                        // If it is already the same device, nothing has to be unmapped.
                    }
                };
            }
        }
        namespace view
        {
            namespace traits
            {
                //#############################################################################
                //! The BufCpu pointer on HIP device get trait specialization.
                //#############################################################################
                template<
                    typename TElem,
                    typename TDim,
                    typename TSize>
                struct GetPtrDev<
                    mem::buf::BufCpu<TElem, TDim, TSize>,
                    dev::DevHipRt>
                {
                    //-----------------------------------------------------------------------------
                    //!
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST static auto getPtrDev(
                        mem::buf::BufCpu<TElem, TDim, TSize> const & buf,
                        dev::DevHipRt const &)
                    -> TElem const *
                    {
                        // TODO: Check if the memory is mapped at all!
                        TElem * pDev(nullptr);
                        ALPAKA_HIP_RT_CHECK(
                            hipHostGetDevicePointer(
                                &pDev,
                                const_cast<void *>(reinterpret_cast<void const *>(mem::view::getPtrNative(buf))),
                                0));
                        return pDev;
                    }
                    //-----------------------------------------------------------------------------
                    //!
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST static auto getPtrDev(
                        mem::buf::BufCpu<TElem, TDim, TSize> & buf,
                        dev::DevHipRt const &)
                    -> TElem *
                    {
                        // TODO: Check if the memory is mapped at all!
                        TElem * pDev(nullptr);
                        ALPAKA_HIP_RT_CHECK(
                            hipHostGetDevicePointer(
                                &pDev,
                                mem::view::getPtrNative(buf),
                                0));
                        return pDev;
                    }
                };
            }
        }
    }
}

#include <alpaka/mem/buf/hip/Copy.hpp>		//as of now, just a renamed copy of it's HIP counterpart

#include <alpaka/mem/buf/hip/Set.hpp>		//as of now, just a renamed copy of it's HIP counterpart

#endif
