//=================================================================================================
/*!
//  \file blaze_tensor/math/IntegerSequence.h
//  \brief Header file for the integer_sequence and index_sequence aliases
//
//  Copyright (C) 2012-2019 Klaus Iglberger - All Rights Reserved
//  Copyright (C) 2018-2019 Hartmut Kaiser - All Rights Reserved
//
//  This file is part of the Blaze library. You can redistribute it and/or modify it under
//  the terms of the New (Revised) BSD License. Redistribution and use in source and binary
//  forms, with or without modification, are permitted provided that the following conditions
//  are met:
//
//  1. Redistributions of source code must retain the above copyright notice, this list of
//     conditions and the following disclaimer.
//  2. Redistributions in binary form must reproduce the above copyright notice, this list
//     of conditions and the following disclaimer in the documentation and/or other materials
//     provided with the distribution.
//  3. Neither the names of the Blaze development group nor the names of its contributors
//     may be used to endorse or promote products derived from this software without specific
//     prior written permission.
//
//  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY
//  EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
//  OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT
//  SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
//  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
//  TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
//  BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
//  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
//  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
//  DAMAGE.
*/
//=================================================================================================

#ifndef _BLAZE_TENSOR_MATH_INTEGERSEQUENCE_H_
#define _BLAZE_TENSOR_MATH_INTEGERSEQUENCE_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <tuple>
#include <utility>
#include <blaze/math/IntegerSequence.h>
#include <blaze/util/StaticAssert.h>


namespace blaze {

//=================================================================================================
//
//  UTILITY FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Dilates the given index sequence by a given step-size.
// \ingroup math
//
// \param sequence The given index sequence
// \return The dilated index sequence.
*/
template< size_t Dilation   // The step-size for the dilate operation
        , size_t... Is >  // The sequence of indices
constexpr decltype(auto) dilate( std::index_sequence<Is...> /*sequence*/ )
{
   return std::index_sequence< ( Is * Dilation )... >();
}
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  ALIAS DEFINITIONS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Auxiliary alias declaration for the setup of dilated index sequences.
// \ingroup math
//
// The make_dilated_index_sequence alias template provides a convenient way to create index
// sequences with specific initial index and a specific number of indices. The following code
// example demonstrates the use of make_shifted_index_sequence:

   \code
   // Creating the index sequence <2,4,6,8>
   using Type = make_dilated_index_sequence<2UL,4UL>;
   \endcode
*/
template< size_t Offset    // The offset for the shift operation
        , size_t N         // The total number of indices in the index sequence
        , size_t Dilation >// The step-sizeof the index sequence
using make_dilated_index_sequence = decltype( dilate<Dilation>( shift<Offset>( make_index_sequence<N>() ) ) );
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Auxiliary alias declaration for the setup of dilated index subsequences.
// \ingroup math
//
// The make_dilated_index_subsequence alias template provides a convenient way to create a
// subsequence of an index sequences with specific initial index and a specific number of indices.
// The following code example demonstrates the use of make_shifted_index_subsequence:

   \code
   // Creating the subsequence <6,12,16> from the index sequence <2,3,4,5,6,7,8>
   using Type = make_dilated_index_subsequence<2UL,6UL,1UL,4UL,6UL>;
   \endcode
*/
template< size_t Offset    // The offset for the shift operation
        , size_t N         // The total number of indices in the index sequence
        , size_t Dilation  // The step-size of the index sequence
        , size_t ... Is >  // The indices to be selected
using make_dilated_index_subsequence =
   decltype( subsequence<Is...>( dilate<Dilation>( shift<Offset>( make_index_sequence<N>() ) ) ) );
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Auxiliary function object allowing to split index sequences into two.
// \ingroup math
*/
template< size_t... Is1, size_t... Is2, typename... Ts, typename... Dims >
BLAZE_ALWAYS_INLINE decltype( auto ) fused_indices( index_sequence< Is1... >,
   size_t index, index_sequence< Is2... >, std::tuple< Ts... > indices, Dims... dims )
{
   constexpr size_t N = sizeof...(Is1) + sizeof...(Is2) + sizeof...(Dims) + 1;
   return std::array< size_t, N >{
         size_t( std::get< Is1 >( indices ) )...,
         index,
         size_t( std::get< Is2 >( indices ) )...,
         size_t( dims )...};
}

template< size_t M, typename... Dims >
BLAZE_ALWAYS_INLINE decltype(auto) fused_indices( size_t index, Dims... dims )
{
   BLAZE_STATIC_ASSERT( M < sizeof...( Dims ) );

   return fused_indices( make_index_sequence< M >{}, index,
      make_shifted_index_sequence< M, sizeof...( Dims ) >{},
      std::forward_as_tuple( dims... ) );
}

template< size_t N, size_t... Is >
BLAZE_ALWAYS_INLINE decltype(auto) array_to_tuple( std::array< size_t, N > const& indices, index_sequence< Is... > )
{
   return std::forward_as_tuple( indices[Is]... );
}

template< size_t N >
BLAZE_ALWAYS_INLINE decltype(auto) array_to_tuple(std::array< size_t, N > const& indices)
{
   return array_to_tuple( indices, make_index_sequence< N >{} );
}

template< size_t M, size_t N, typename... Dims >
BLAZE_ALWAYS_INLINE decltype(auto) fused_indices( size_t index, std::array< size_t, N > const& indices, Dims... dims )
{
   BLAZE_STATIC_ASSERT( M < N );

   return fused_indices(
      make_index_sequence< M >{}, index, make_shifted_index_sequence< M, N >{},
      array_to_tuple( indices ), dims... );
}
//*************************************************************************************************

} // namespace blaze

#endif
