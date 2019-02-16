//=================================================================================================
/*!
//  \file blaze_tensor/math/dense/Transposition.h
//  \brief Header file for the implementation of in-place transposing a 3D tensor
//
//  Copyright (C) 2012-2018 Klaus Iglberger - All Rights Reserved
//  Copyright (C) 2018 Hartmut Kaiser - All Rights Reserved
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

#ifndef _BLAZE_TENSOR_MATH_DENSE_TRANSPOSITION_H_
#define _BLAZE_TENSOR_MATH_DENSE_TRANSPOSITION_H_

//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <blaze/system/Blocking.h>
#include <blaze/util/Assert.h>

#include <blaze_tensor/math/Forward.h>
#include <blaze_tensor/math/dense/DenseTensor.h>
#include <blaze_tensor/math/expressions/DenseTensor.h>

#include <algorithm>

namespace blaze {

//=================================================================================================
//
//  TRANSPOSITION FUNCTIONS FOR OxMxN TENSORS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename TT >
inline void transposeGeneral021( DenseTensor<TT>& dt );

template< typename TT >
inline void transposeGeneral102( DenseTensor<TT>& dt );

template< typename TT >
inline void transposeGeneral120( DenseTensor<TT>& dt );

template< typename TT >
inline void transposeGeneral201( DenseTensor<TT>& dt );

template< typename TT >
inline void transposeGeneral210( DenseTensor<TT>& dt );
/*! \endcond */
//*************************************************************************************************

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief In-place transposition of the given general dense tensor.
// \ingroup dense_tensor
//
// \param dt The general dense tensor to be transposed.
// \return void
//
// This function transposes the given general dense tensor.
*/
template< typename TT >     // Type of the dense tensor
inline void transposeGeneral( DenseTensor<TT>& dt )
{
   transposeGeneral210( dt );
}
/*! \endcond */
//*************************************************************************************************

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief In-place transposition of the given general dense tensor.
// \ingroup dense_tensor
//
// \param dt The general dense tensor to be transposed.
// \return void
//
// This function transposes the given general dense tensor.
*/
template< typename TT      // Type of the dense tensor
        , typename T >     // Type of the mapping indices
inline void transposeGeneral( DenseTensor<TT>& dt, const T* indices, size_t n )
{
   BLAZE_USER_ASSERT( n == 3, "Invalid number of transposition axes" );

   if ( indices[0] == 0 )
   {
      if ( indices[1] == 2 )
      {
         transposeGeneral021( dt );       // {0, 2, 1}
      }
   }
   else if ( indices[0] == 1 )
   {
      if ( indices[1] == 2 )
      {
         transposeGeneral120( dt );       // {1, 2, 0}
      }
      else
      {
         transposeGeneral102( dt );       // {1, 0, 2}
      }
   }
   else
   {
      // indices[0] == 2
      if ( indices[1] == 1 )
      {
         transposeGeneral210( dt );       // {2, 1, 0}
      }
      else
      {
         transposeGeneral201( dt );       // {2, 0, 1}
      }
   }
}
/*! \endcond */
//*************************************************************************************************

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief In-place transposition of the given general dense tensor.
// \ingroup dense_tensor
//
// \param dt The general dense tensor to be transposed.
// \return void
//
// This function transposes the given general dense tensor using {0, 2, 1} as the axis mapping.
*/
template< typename TT >     // Type of the dense tensor
inline void transposeGeneral021( DenseTensor<TT>& dt )
{
   using std::swap;

   constexpr size_t block( BLOCK_SIZE );

   TT& t( ~dt );

   for( size_t kk = 0UL; kk < t.pages(); kk += block )
   {
      const size_t kend( min( kk + block, t.pages() ) );
      for( size_t ii = 0UL; ii < t.rows(); ii += block )
      {
         for( size_t jj = 0UL; jj <= ii; jj += block )
         {
            const size_t iend( min( ii + block, t.rows() ) );
            for( size_t k = kk; k < kend; ++k )
            {
               for( size_t i = ii; i < iend; ++i )
               {
                  const size_t jend( min( min( jj + block, t.columns() ), i ) );
                  for( size_t j = jj; j < jend; ++j )
                  {
                     swap( t( k, i, j ), t( k, j, i ) );
                  }
               }
            }
         }
      }
   }
}
/*! \endcond */
//*************************************************************************************************

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief In-place transposition of the given general dense tensor.
// \ingroup dense_tensor
//
// \param dt The general dense tensor to be transposed.
// \return void
//
// This function transposes the given general dense tensor using {1, 0, 2} as the axis mapping.
*/
template< typename TT >     // Type of the dense tensor
inline void transposeGeneral102( DenseTensor<TT>& dt )
{
   using std::swap;

   constexpr size_t block( BLOCK_SIZE );

   TT& t( ~dt );

   for( size_t jj = 0UL; jj < t.columns(); jj += block )
   {
      const size_t jend( std::min( jj + block, t.columns() ) );
      for( size_t kk = 0UL; kk < t.pages(); kk += block )
      {
         for( size_t ii = 0UL; ii <= kk; ii += block )
         {
            const size_t kend( std::min( kk + block, t.pages() ) );
            for( size_t j = jj; j < jend; ++j )
            {
               for( size_t k = kk; k < kend; ++k )
               {
                  const size_t iend( std::min( std::min( ii + block, t.rows() ), k ) );
                  for( size_t i = ii; i < iend; ++i )
                  {
                     swap( t( k, i, j ), t( i, k, j ) );
                  }
               }
            }
         }
      }
   }
}
/*! \endcond */
//*************************************************************************************************

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief In-place transposition of the given general dense tensor.
// \ingroup dense_tensor
//
// \param dt The general dense tensor to be transposed.
// \return void
//
// This function transposes the given general dense tensor using {1, 2, 0} as the axis mapping.
*/
template< typename TT >     // Type of the dense tensor
inline void transposeGeneral120( DenseTensor<TT>& dt )
{
   using std::swap;

   constexpr size_t block( BLOCK_SIZE );

   TT& t( ~dt );

   for( size_t kk = 0UL; kk < t.pages(); kk += block )
   {
      const size_t kend( min( kk + block, t.pages() ) );
      for( size_t jj = 0UL; jj < t.columns(); jj += block )
      {
         const size_t jend( min( jj + block, t.columns() ) );
         for( size_t k = kk; k < kend; ++k )
         {
            for( size_t j = jj; j < jend; ++j )
            {
               if( k == j )
                  continue;

               swap( t( j, k, k ), t( k, j, k ) );
               swap( t( k, k, j ), t( k, j, k ) );
            }
         }
      }
      for( size_t ii = kk + 1; ii < t.rows(); ii += block )
      {
         const size_t iend( min( ii + block, t.rows() ) );
         for( size_t jj = ii + 1; jj < t.columns(); jj += block )
         {
            const size_t jend( min( jj + block, t.columns() ) );
            for( size_t k = kk; k < kend; ++k )
            {
               for( size_t i = k + 1; i < iend; ++i )
               {
                  for( size_t j = i + 1; j < jend; ++j )
                  {
                     swap( t( j, k, i ), t( i, j, k ) );
                     swap( t( k, i, j ), t( i, j, k ) );

                     swap( t( i, k, j ), t( j, i, k ) );
                     swap( t( k, j, i ), t( j, i, k ) );
                  }
               }
            }
         }
      }
   }
}
/*! \endcond */
//*************************************************************************************************

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief In-place transposition of the given general dense tensor.
// \ingroup dense_tensor
//
// \param dt The general dense tensor to be transposed.
// \return void
//
// This function transposes the given general dense tensor using {2, 0, 1} as the axis mapping.
*/
template< typename TT >     // Type of the dense tensor
inline void transposeGeneral201( DenseTensor<TT>& dt )
{
   using std::swap;

   constexpr size_t block( BLOCK_SIZE );

   TT& t( ~dt );

   for( size_t ii = 0UL; ii < t.rows(); ii += block )
   {
      const size_t iend( min( ii + block, t.rows() ) );
      for( size_t kk = 0UL; kk < t.pages(); kk += block )
      {
         const size_t kend( min( kk + block, t.pages() ) );
         for( size_t i = ii; i < iend; ++i )
         {
            for( size_t k = kk; k < kend; ++k )
            {
               if( i == k )
                  continue;

               swap( t( i, i, k ), t( i, k, i ) );
               swap( t( k, i, i ), t( i, k, i ) );
            }
         }
      }
      for( size_t jj = ii + 1; jj < t.columns(); jj += block )
      {
         const size_t jend( min( jj + block, t.columns() ) );
         for( size_t kk = jj + 1; kk < t.pages(); kk += block )
         {
            const size_t kend( min( kk + block, t.columns() ) );
            for( size_t i = ii; i < iend; ++i )
            {
               for( size_t j = i + 1; j < jend; ++j )
               {
                  for( size_t k = j + 1; k < kend; ++k )
                  {
                     swap( t( i, j, k ), t( j, k, i ) );
                     swap( t( k, i, j ), t( j, k, i ) );

                     swap( t( i, k, j ), t( k, j, i ) );
                     swap( t( j, i, k ), t( k, j, i ) );
                  }
               }
            }
         }
      }
   }
}
/*! \endcond */
//*************************************************************************************************

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief In-place transposition of the given general dense tensor.
// \ingroup dense_tensor
//
// \param dt The general dense tensor to be transposed.
// \return void
//
// This function transposes the given general dense tensor using {2, 1, 0} as the axis mapping.
*/
template< typename TT >     // Type of the dense tensor
inline void transposeGeneral210( DenseTensor<TT>& dt )
{
   using std::swap;

   constexpr size_t block( BLOCK_SIZE );

   TT& t( ~dt );

   for( size_t ii = 0UL; ii < t.rows(); ii += block )
   {
      const size_t iend( std::min( ii + block, t.rows() ) );
      for( size_t kk = 0UL; kk <t.pages(); kk += block )
      {
         for( size_t jj = 0UL; jj <= kk; jj += block )
         {
            const size_t kend( std::min( kk + block, t.pages() ) );
            for( size_t i = ii; i < iend; ++i )
            {
               for( size_t k = kk; k < kend; ++k )
               {
                  const size_t jend( std::min( std::min( jj + block, t.columns() ), k ) );
                  for( size_t j = jj; j < jend; ++j )
                  {
                     swap( t( k, i, j ), t( j, i, k ) );
                  }
               }
            }
         }
      }
   }
}
/*! \endcond */
//*************************************************************************************************

} // namespace blaze

#endif
