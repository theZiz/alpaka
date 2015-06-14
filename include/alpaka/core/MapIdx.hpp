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

#include <alpaka/core/Vec.hpp>          // Vec
#include <alpaka/core/Common.hpp>       // ALPAKA_FCT_HOST_ACC

#include <boost/core/ignore_unused.hpp> // boost::ignore_unused

namespace alpaka
{
    namespace detail
    {
        //#############################################################################
        //! Maps a linear index to a N dimensional index.
        //#############################################################################
        template<
            UInt TuiIdxDimOut,
            UInt TuiIdxDimIn>
        struct MapIdx;
        //#############################################################################
        //! Maps a linear index to a linear index.
        //#############################################################################
        template<>
        struct MapIdx<
            1u,
            1u>
        {
            //-----------------------------------------------------------------------------
            // \tparam TElem Type of the index values.
            // \param Index Idx to be mapped.
            // \param Extents Spatial size to map the index to.
            // \return Vector of dimension TuiDimDst.
            //-----------------------------------------------------------------------------
            template<
                typename TElem>
            ALPAKA_FCT_HOST_ACC static auto mapIdx(
                Vec<dim::Dim1, TElem> const & idx,
                Vec<dim::Dim1, TElem> const & extents)
            -> Vec<dim::Dim1, TElem>
            {
                boost::ignore_unused(extents);
                return idx;
            }
        };
        //#############################################################################
        //! Maps a linear index to a 3 dimensional index.
        //#############################################################################
        template<>
        struct MapIdx<
            3u,
            1u>
        {
            //-----------------------------------------------------------------------------
            // \tparam TElem Type of the index values.
            // \param Index Idx to be mapped.
            // \param Extents Spatial size to map the index to.
            // \return Vector of dimension TuiDimDst.
            //-----------------------------------------------------------------------------
            template<
                typename TElem>
            ALPAKA_FCT_HOST_ACC static auto mapIdx(
                Vec<dim::Dim1, TElem> const & idx,
                Vec<dim::Dim3, TElem> const & extents)
            -> Vec<dim::Dim3, TElem>
            {
                auto const & uiIdx(idx[0]);
                auto const uiXyExtentsProd(extents.prod());
                auto const & uiExtentX(extents[2]);

                return {
                    uiIdx / uiXyExtentsProd,
                    (uiIdx % uiXyExtentsProd) / uiExtentX,
                    uiIdx % uiExtentX};
            }
        };
        //#############################################################################
        //! Maps a linear index to a 2 dimensional index.
        //#############################################################################
        template<>
        struct MapIdx<
            2u,
            1u>
        {
            //-----------------------------------------------------------------------------
            // \tparam TElem Type of the index values.
            // \param Index Idx to be mapped.
            // \param Extents Spatial size to map the index to.
            // \return Vector of dimension TuiDimDst.
            //-----------------------------------------------------------------------------
            template<
                typename TElem>
            ALPAKA_FCT_HOST_ACC static auto mapIdx(
                Vec<dim::Dim1, TElem> const & idx,
                Vec<dim::Dim2, TElem> const & extents)
            -> Vec<dim::Dim2, TElem>
            {
                auto const & uiIdx(idx[0]);
                auto const & uiExtentX(extents[1]);

                return {
                    uiIdx / uiExtentX,
                    uiIdx % uiExtentX};
            }
        };
        //#############################################################################
        //! Maps a 3 dimensional index to a linear index.
        //#############################################################################
        template<>
        struct MapIdx<
            1u,
            3u>
        {
            //-----------------------------------------------------------------------------
            // \tparam TElem Type of the index values.
            // \param Index Idx to be mapped.
            // \param Extents Spatial size to map the index to.
            // \return Vector of dimension TuiDimDst.
            //-----------------------------------------------------------------------------
            template<
                typename TElem>
            ALPAKA_FCT_HOST_ACC static auto mapIdx(
                Vec<dim::Dim3, TElem> const & idx,
                Vec<dim::Dim3, TElem> const & extents)
            -> Vec<dim::Dim1, TElem>
            {
                return (idx[0u] * extents[1u] + idx[1u]) * extents[2u] + idx[2u];
            }
        };
        //#############################################################################
        //! Maps a 2 dimensional index to a linear index.
        //#############################################################################
        template<>
        struct MapIdx<
            1u,
            2u>
        {
            //-----------------------------------------------------------------------------
            // \tparam TElem Type of the index values.
            // \param Index Idx to be mapped.
            // \param Extents Spatial size to map the index to.
            // \return Vector of dimension TuiDimDst.
            //-----------------------------------------------------------------------------
            template<
                typename TElem>
            ALPAKA_FCT_HOST_ACC static auto mapIdx(
                Vec<dim::Dim2, TElem> const & idx,
                Vec<dim::Dim2, TElem> const & extents)
            -> Vec<dim::Dim1, TElem>
            {
                return idx[0u] * extents[1u] + idx[1u];
            }
        };
    }

    //#############################################################################
    //! Maps a N dimensional index to a N dimensional position.
    //!
    //! \tparam TuiIdxDimOut Dimension of the index vector to map to.
    //! \tparam TuiIdxDimIn Dimension of the index vector to map from.
    //! \tparam TuiIdxDimExt Dimension of the extents vector to map use for mapping.
    //! \tparam TElem Type of the elements of the index vector to map from.
    //#############################################################################
    template<
        UInt TuiIdxDimOut,
        UInt TuiIdxDimIn,
        typename TElem>
    ALPAKA_FCT_HOST_ACC auto mapIdx(
        Vec<dim::Dim<TuiIdxDimIn>, TElem> const & idx,
        Vec<dim::Dim<(TuiIdxDimOut < TuiIdxDimIn) ? TuiIdxDimIn : TuiIdxDimOut>, TElem> const & extents)
    -> Vec<dim::Dim<TuiIdxDimOut>, TElem>
    {
        return detail::MapIdx<
            TuiIdxDimOut,
            TuiIdxDimIn>
        ::mapIdx(
            idx,
            extents);
    }
}