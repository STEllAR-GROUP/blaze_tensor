//=================================================================================================
/*!
//  \file blaze/util/ArrayForEach.h
//  \brief Header file for the ArrayForEach function
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

#ifndef _BLAZE_TENSOR_UTIL_ARRAYFOREACH_H_
#define _BLAZE_TENSOR_UTIL_ARRAYFOREACH_H_

#include <array>
#include <utility>

#include <blaze/util/SmallArray.h>

namespace blaze {

template< size_t Shift = 1, size_t N >
std::array< size_t, N - Shift > ArrayShift( std::array< size_t, N > const& dims )
{
   BLAZE_STATIC_ASSERT(N >= 2 && N > Shift);     // cannot shift more than elements

   std::array<size_t, N - Shift> result;
   for( size_t i = 0; i != N - Shift; ++i ) {
      result[i] = dims[i];
   }
   return result;
}

//*************************************************************************************************
/*!\brief ArrayForEach function to iterate over arbitrary dimension data.
// \ingroup util
*/
//    N == 4
//
//    for( size_t c = 0UL; c < l_; ++c ) {
//       size_t x1 = ( c + 0 ) * o_;
//
//       N == 3
//       for( size_t k = 0UL; k < o_; ++k ) {
//          size_t x2 = ( x1 + k ) * m_;
//
//          N == 2
//          for( size_t i = 0UL; i < m_; ++i ) {
//             size_t x3 = ( x2 + i ) * nn_;
//
//             N == 1
//             for( size_t j = n_; j < n_; ++j ) {
//                size_t x4 = ( x3 + j );
//                v_[x4] = Type();
//             }
//          }
//       }
//    }
//
template< typename F >
void ArrayForEach(
   std::array< size_t, 1 > const& dims, F const& f, size_t base = 0 )
{
   for( size_t i = 0; i != dims[0]; ++i ) {
      f( i + base );
   }
}

template< size_t N, typename F >
void ArrayForEach(
   std::array< size_t, N > const& dims, F const& f, size_t base = 0 )
{
   BLAZE_STATIC_ASSERT( N >= 2 );
   std::array< size_t, N - 1 > shifted_dims = ArrayShift( dims );
   for( size_t i = base * dims[N - 2], j = 0; j != dims[N - 1]; i += dims[N - 2], ++j) {
      ArrayForEach( shifted_dims, f, i );
   }
}
//*************************************************************************************************

//*************************************************************************************************
/*!\brief ArrayForEachPadded function to iterate over arbitrary dimension data.
// \ingroup util
*/
//    N == 4
//
//    for( size_t c = 0UL; c < l_; ++c ) {
//       size_t x1 = ( c + 0 ) * o_;
//
//       N == 3
//       for( size_t k = 0UL; k < o_; ++k ) {
//          size_t x2 = ( x1 + k ) * m_;
//
//          N == 2
//          for( size_t i = 0UL; i < m_; ++i ) {
//             size_t x3 = ( x2 + i ) * nn_;
//
//             N == 1
//             for( size_t j = n_; j < nn_; ++j ) {
//                size_t x4 = ( x3 + j );
//                v_[x4] = Type();
//             }
//          }
//       }
//    }
//
template< typename F >
void ArrayForEachPadded(
   std::array< size_t, 1 > const& dims, size_t nn, F const& f, size_t base = 0 )
{
   for( size_t i = dims[0]; i != nn; ++i ) {
      f( i + base );
   }
}

template< size_t N, typename F >
void ArrayForEachPadded(
   std::array< size_t, 2 > const& dims, size_t nn, F const& f, size_t base = 0 )
{
   std::array< size_t, 1 > shifted_dims = ArrayShift( dims );
   for( size_t i = base * nn, j = 0; j != dims[1]; i += nn, ++j ) {
      ArrayForEachPadded( shifted_dims, nn, f, i );
   }
}

template< size_t N, typename F >
void ArrayForEachPadded(
   std::array< size_t, N > const& dims, size_t nn, F const& f, size_t base = 0 )
{
   BLAZE_STATIC_ASSERT( N >= 2 );
   std::array< size_t, N - 1 > shifted_dims = ArrayShift( dims );
   for( size_t i = base * dims[N - 2], j = 0; j != dims[N - 1]; i += dims[N - 2], ++j ) {
      ArrayForEachPadded( shifted_dims, nn, f, i );
   }
}
//*************************************************************************************************

//*************************************************************************************************
/*!\brief ArrayForEachGrouped function to iterate over arbitrary dimension data.
// \ingroup util
*/
template< typename F, size_t M >
void ArrayForEachGrouped(
   size_t dim0, F const& f, std::array< size_t, M >& currdims )
{
   for( currdims[0] = 0; currdims[0] != dim0; ++currdims[0] ) {
      f( currdims );
   }
}

template< typename F, size_t M >
void ArrayForEachGrouped(
   std::array< size_t, 2 > const& dims, F const& f, std::array< size_t, M >& currdims )
{
   for( currdims[1] = 0; currdims[1] != dims[1]; ++currdims[1] ) {
      ArrayForEachGrouped( dims[0], f, currdims );
   }
}

template< size_t N, typename F, size_t M >
void ArrayForEachGrouped(
   std::array< size_t, N > const& dims, F const& f, std::array< size_t, M >& currdims )
{
   BLAZE_STATIC_ASSERT( N > 2 );
   std::array< size_t, N - 1 > shifted_dims = ArrayShift( dims );
   for( currdims[N - 1] = 0; currdims[N - 1] != dims[N - 1]; ++currdims[N - 1] ) {
      ArrayForEachGrouped( shifted_dims, f, currdims );
   }
}

template< size_t N, typename F >
void ArrayForEachGrouped(
   std::array< size_t, N > const& dims, F const& f )
{
   std::array< size_t, N > currdims{};
   ArrayForEachGrouped( dims, f, currdims );
}
//*************************************************************************************************

//*************************************************************************************************
/*!\brief ArrayForEachGrouped function to iterate over arbitrary dimension data.
// \ingroup util
*/
template< typename F, size_t M >
void ArrayForEachGrouped(
   size_t dim0, F const& f, std::array< size_t, M >& currdims, size_t base )
{
   for( currdims[0] = 0; currdims[0] != dim0; ++currdims[0], ++base ) {
      f( base, currdims );
   }
}

template< typename F, size_t M >
void ArrayForEachGrouped(
   std::array< size_t, 2 > const& dims, size_t nn, F const& f, std::array< size_t, M >& currdims, size_t base = 0 )
{
   std::array< size_t, 1 > shifted_dims = ArrayShift( dims );
   currdims[1] = 0;
   for( size_t i = base * nn; currdims[1] != dims[1]; i += nn, ++currdims[1]) {
      ArrayForEachGrouped( dims[0], f, currdims, i );
   }
}

template< size_t N, typename F, size_t M >
void ArrayForEachGrouped(
   std::array< size_t, N > const& dims, size_t nn, F const& f, std::array< size_t, M >& currdims, size_t base = 0 )
{
   BLAZE_STATIC_ASSERT( N > 2 );
   std::array< size_t, N - 1 > shifted_dims = ArrayShift( dims );
   currdims[N - 1] = 0;
   for( size_t i = base * dims[N - 2]; currdims[N - 1] != dims[N - 1]; i += dims[N - 2], ++currdims[N - 1]) {
      ArrayForEachGrouped( shifted_dims, nn, f, currdims, i );
   }
}

template< size_t N, typename F >
void ArrayForEachGrouped(
   std::array< size_t, N > const& dims, size_t nn, F const& f )
{
   std::array< size_t, N > currdims{};
   ArrayForEachGrouped( dims, nn, f, currdims );
}
//*************************************************************************************************

//*************************************************************************************************
/*!\brief ArrayForEach2 function to iterate over arbitrary dimension data.
// \ingroup util
*/
//    N == 4
//
//    for( size_t c = 0UL; c < l_; ++c ) {
//       size_t x1 = ( c + 0 ) * o_;
//
//       N == 3
//       for( size_t k = 0UL; k < o_; ++k ) {
//          size_t x2 = ( x1 + k ) * m_;
//
//          N == 2
//          for( size_t i = 0UL; i < m_; ++i ) {
//             size_t x3 = ( x2 + i ) * nn_;
//             size_t x31 = ( x2 + i ) * n_;
//
//             N == 1
//             for( size_t j = n_; j < n_; ++j ) {
//                v_[x3 + j] = array[x31 + j];
//             }
//          }
//       }
//    }
//
template< typename F >
void ArrayForEach2(
   size_t dim0, F const& f, size_t base1, size_t base2 )
{
   for( size_t i = 0; i != dims[0]; ++i ) {
      f( i + base1, i + base2 );
   }
}

template< typename F >
void ArrayForEach2(
   std::array< size_t, 2 > const& dims, size_t nn, F const& f, size_t base )
{
   for( size_t i = 0; i != dims[1]; ++i ) {
      size_t index1 = (i + base) * dims[0];
      size_t index2 = (i + base) * nn;
      ArrayForEach2( dims[0], f, index1, index2 );
   }
}

template< size_t N, typename F >
void ArrayForEach2(
   std::array< size_t, N > const& dims, size_t nn, F const& f, size_t base = 0 )
{
   BLAZE_STATIC_ASSERT( N > 2 );
   std::array< size_t, N - 1 > shifted_dims = ArrayShift( dims );
   for( size_t i = base * dims[N - 2], j = 0; j != dims[N - 1]; i += dims[N - 2], ++j ) {
      ArrayForEach2( shifted_dims, nn, f, i );
   }
}
//*************************************************************************************************

//*************************************************************************************************
/*!\brief ArrayForEachGroupedAnyOf function to iterate over arbitrary dimension data.
// \ingroup util
*/
template< typename F, size_t M >
bool ArrayForEachGroupedAnyOf( std::array< size_t, 1 > const& dims, F const& f,
   std::array< size_t, M >& currdims )
{
   while( currdims[0] != dims[0] ) {
      if( f( currdims ) )
         return true;
      ++currdims[0];
   }
   return false;
}

template< size_t N, typename F, size_t M >
bool ArrayForEachGroupedAnyOf( std::array< size_t, N > const& dims, F const& f,
   std::array< size_t, M >& currdims )
{
   while( currdims[N - 1] != dims[N - 1] ) {
      currdims[N - 2] = 0;
      if( ArrayForEachGroupedAnyOf( ArrayShift( dims ), f, currdims ) ) {
         return true;
      }
      ++currdims[N - 1];
   }
   return false;
}

template< size_t N, typename F >
bool ArrayForEachGroupedAnyOf(
   std::array< size_t, N > const& dims, F const& f )
{
   BLAZE_STATIC_ASSERT( N >= 3 );
   std::array< size_t, N > currdims{};
   return ArrayForEachGroupedAnyOf( dims, f, currdims );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief ArrayForEachGroupedAnyOf function to iterate over arbitrary dimension data.
// \ingroup util
*/
template< typename F, size_t M >
bool ArrayForEachGroupedAllOf( std::array< size_t, 1 > const& dims, F const& f,
   std::array< size_t, M >& currdims )
{
   while( currdims[0] != dims[0] ) {
      if( !f( currdims ) )
         return false;
      ++currdims[0];
   }
   return true;
}

template< size_t N, typename F, size_t M >
bool ArrayForEachGroupedAllOf( std::array< size_t, N > const& dims, F const& f,
   std::array< size_t, M >& currdims )
{
   while( currdims[N - 1] != dims[N - 1] ) {
      currdims[N - 2] = 0;
      if( !ArrayForEachGroupedAllOf( ArrayShift( dims ), f, currdims ) ) {
         return false;
      }
      ++currdims[N - 1];
   }
   return true;
}

template< size_t N, typename F >
bool ArrayForEachGroupedAllOf(
   std::array< size_t, N > const& dims, F const& f )
{
   BLAZE_STATIC_ASSERT( N >= 3 );
   std::array< size_t, N > currdims{};
   return ArrayForEachGroupedAllOf( dims, f, currdims );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief ArrayDimForEach function to iterate over arbitrary dimension data.
// \ingroup util
*/
template< size_t N, typename F >
void ArrayDimForEach( std::array< size_t, N > const& dims, F const& f)
{
   for( size_t i = 0; i < N; ++i ) {
      f(i);
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief ArrayDimAnyOf function to iterate over arbitrary dimension data.
// \ingroup util
*/
template< size_t N, typename F >
bool ArrayDimAnyOf( std::array< size_t, N > const& dims, F const& f)
{
   for( size_t i = 0; i < N; ++i ) {
      if( f( dims[i] ) ) {
         return true;
      }
   }
   return false;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief ArrayDimAllOf function to iterate over arbitrary dimension data.
// \ingroup util
*/
template< size_t N, typename F >
bool ArrayDimAllOf( std::array< size_t, N > const& dims, F const& f)
{
   for( size_t i = 0; i < N; ++i ) {
      if( !f( dims[i] ) ) {
         return false;
      }
   }
   return true;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief ArrayDimNoneOf function to iterate over arbitrary dimension data.
// \ingroup util
*/
template< size_t N, typename F >
bool ArrayDimNoneOf( std::array< size_t, N > const& dims, F const& f)
{
   for( size_t i = 0; i < N; ++i ) {
      if( f( dims[i] ) ) {
         return false;
      }
   }
   return true;
}
//*************************************************************************************************

} // namespace blaze

#endif
