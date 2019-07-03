//=================================================================================================
/*!
//  \file blaze_array/math/DynamicArray.h
//  \brief Header file for the complete DynamicArray implementation
//
//  Copyright (C) 2012-2018 Klaus Iglberger - All Rights Reserved
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

#ifndef _BLAZE_TENSOR_MATH_DYNAMICARRAY_H_
#define _BLAZE_TENSOR_MATH_DYNAMICARRAY_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <blaze/math/DynamicMatrix.h>
#include <blaze/math/DynamicVector.h>
#include <blaze/math/Exception.h>
#include <blaze/math/IdentityMatrix.h>
#include <blaze/math/shims/Conjugate.h>
#include <blaze/math/shims/Real.h>
#include <blaze/math/typetraits/UnderlyingBuiltin.h>
#include <blaze/math/ZeroMatrix.h>
#include <blaze/util/Assert.h>
#include <blaze/util/constraints/Numeric.h>
#include <blaze/util/Random.h>

#include <blaze_tensor/math/DenseArray.h>
#include <blaze_tensor/math/dense/DynamicArray.h>
#include <blaze_tensor/util/ArrayForEach.h>

namespace blaze {

//=================================================================================================
//
//  RAND SPECIALIZATION
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Specialization of the Rand class template for DynamicArray.
// \ingroup random
//
// This specialization of the Rand class creates random instances of DynamicArray.
*/
template< size_t N, typename Type > // Data type of the array
class Rand< DynamicArray<N, Type> >
{
 public:
   //**Generate functions**************************************************************************
   /*!\name Generate functions */
   //@{
   template< typename... Dims >
   inline const DynamicArray<N, Type> generate( Dims... dims ) const;

   template< typename Arg, typename... Dims >
   inline const DynamicArray<N, Type> generate( const Arg& min, const Arg& max, Dims... dims ) const;
   //@}
   //**********************************************************************************************

   //**Randomize functions*************************************************************************
   /*!\name Randomize functions */
   //@{
   inline void randomize( DynamicArray<N, Type>& array ) const;

   template< typename Arg >
   inline void randomize( DynamicArray<N, Type>& array, const Arg& min, const Arg& max ) const;
   //@}
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Generation of a random DynamicArray.
//
// \param m The number of rows of the random array.
// \param n The number of columns of the random array.
// \return The generated random array.
*/
template< size_t N         // The dimensionality of the array
        , typename Type >  // Data type of the array
template< typename... Dims >
inline const DynamicArray<N, Type>
   Rand< DynamicArray<N, Type> >::generate( Dims... dims ) const
{
   DynamicArray<N, Type> array( dims... );
   randomize( array );
   return array;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Generation of a random DynamicArray.
//
// \param m The number of rows of the random array.
// \param n The number of columns of the random array.
// \param min The smallest possible value for a array element.
// \param max The largest possible value for a array element.
// \return The generated random array.
*/
template< size_t N         // The dimensionality of the array
        , typename Type >  // Data type of the array
template< typename Arg, typename... Dims >  // Min/max argument type
inline const DynamicArray<N, Type>
   Rand< DynamicArray<N, Type> >::generate( const Arg& min, const Arg& max, Dims... dims ) const
{
   DynamicArray<N, Type> array( dims... );
   randomize( array, min, max );
   return array;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Randomization of a DynamicArray.
//
// \param array The array to be randomized.
// \return void
*/
template< size_t N         // The dimensionality of the array
        , typename Type >  // Data type of the array
inline void Rand< DynamicArray<N, Type> >::randomize( DynamicArray<N, Type>& array ) const
{
   using blaze::randomize;

   ArrayForEachGrouped(
      array.dimensions(), [&]( std::array< size_t, N > const& dims ) {
         randomize( array( dims ) );
      } );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Randomization of a DynamicArray.
//
// \param array The array to be randomized.
// \param min The smallest possible value for a array element.
// \param max The largest possible value for a array element.
// \return void
*/
template< size_t N         // The dimensionality of the array
        , typename Type >  // Data type of the array
template< typename Arg >    // Min/max argument type
inline void Rand< DynamicArray< N, Type > >::randomize(
   DynamicArray< N, Type >& array, const Arg& min, const Arg& max ) const
{
   using blaze::randomize;

   ArrayForEachGrouped(
      array.dimensions(), [&]( std::array< size_t, N > const& dims ) {
         randomize( array( dims ), min, max );
      } );
}
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  MAKE FUNCTIONS
//
//=================================================================================================

} // namespace blaze

#endif
