//=================================================================================================
/*!
//  \file blaze_tensor/math/CustomArray.h
//  \brief Header file for the complete CustomArray implementation
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

#ifndef _BLAZE_TENSOR_MATH_CUSTOMARRAY_H_
#define _BLAZE_TENSOR_MATH_CUSTOMARRAY_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <array>

#include <blaze/math/CustomMatrix.h>

#include <blaze_tensor/math/CustomTensor.h>
#include <blaze_tensor/math/DenseArray.h>
#include <blaze_tensor/math/dense/CustomArray.h>
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
/*!\brief Specialization of the Rand class template for CustomArray.
// \ingroup random
//
// This specialization of the Rand class creates random instances of CustomArray.
*/
template< size_t N       // Dimensionality of the array
        , typename Type  // Data type of the tensor
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , typename RT >  // Result type
class Rand< CustomArray<N,Type,AF,PF,RT> >
{
 public:
   //**Randomize functions*************************************************************************
   /*!\name Randomize functions */
   //@{
   inline void randomize( CustomArray<N,Type,AF,PF,RT>& tensor ) const;

   template< typename Arg >
   inline void randomize( CustomArray<N,Type,AF,PF,RT>& tensor, const Arg& min, const Arg& max ) const;
   //@}
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Randomization of a CustomArray.
//
// \param tensor The tensor to be randomized.
// \return void
*/
template< size_t N       // Dimensionality of the array
        , typename Type  // Data type of the tensor
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , typename RT >  // Result type
inline void Rand< CustomArray<N,Type,AF,PF,RT> >::randomize( CustomArray<N,Type,AF,PF,RT>& tensor ) const
{
   using blaze::randomize;

   ArrayForEachGrouped(
      tensor.dimensions(), [&]( std::array< size_t, N > const& dims ) {
         randomize( tensor( dims ) );
      } );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Randomization of a CustomArray.
//
// \param tensor The tensor to be randomized.
// \param min The smallest possible value for a tensor element.
// \param max The largest possible value for a tensor element.
// \return void
*/
template< size_t N       // Dimensionality of the array
        , typename Type   // Data type of the tensor
        , bool AF         // Alignment flag
        , bool PF         // Padding flag
        , typename RT >   // Result type
template< typename Arg >  // Min/max argument type
inline void Rand< CustomArray<N,Type,AF,PF,RT> >::randomize( CustomArray<N,Type,AF,PF,RT>& tensor,
                                                             const Arg& min, const Arg& max ) const
{
   using blaze::randomize;

   ArrayForEachGrouped(
      tensor.dimensions(), [&]( std::array< size_t, N > const& dims ) {
         randomize( tensor( dims ), min, max );
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
