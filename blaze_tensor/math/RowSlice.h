//=================================================================================================
/*!
//  \file blaze_tensor/math/RowSlice.h
//  \brief Header file for the complete RowSlice implementation
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

#ifndef _BLAZE_TENSOR_MATH_ROWSLICE_H_
#define _BLAZE_TENSOR_MATH_ROWSLICE_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <blaze/math/Aliases.h>
#include <blaze/math/constraints/DenseMatrix.h>
#include <blaze/math/Exception.h>
#include <blaze/math/smp/DenseMatrix.h>
#include <blaze/util/Random.h>
#include <blaze/util/typetraits/RemoveReference.h>

#include <blaze_tensor/math/constraints/RowSlice.h>
#include <blaze_tensor/math/views/RowSlice.h>

namespace blaze {

//=================================================================================================
//
//  RAND SPECIALIZATION FOR DENSE ROWSLICES
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Specialization of the Rand class template for dense rowslices.
// \ingroup random
//
// This specialization of the Rand class randomizes dense rowslices.
*/
template< typename MT       // Type of the matrix
        , size_t... CRAs >  // Compile time rowslice arguments
class Rand< RowSlice<MT,CRAs...> >
{
 public:
   //**Randomize functions*************************************************************************
   /*!\name Randomize functions */
   //@{
   template< typename RT >
   inline void randomize( RT&& rowslice ) const;

   template< typename RT, typename Arg >
   inline void randomize( RT&& rowslice, const Arg& min, const Arg& max ) const;
   //@}
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Randomization of a dense rowslice.
//
// \param rowslice The rowslice to be randomized.
// \return void
*/
template< typename MT       // Type of the matrix
        , size_t... CRAs >  // Compile time rowslice arguments
template< typename RT >     // Type of the rowslice
inline void Rand< RowSlice<MT,CRAs...> >::randomize( RT&& rowslice ) const
{
   using blaze::randomize;

   using RowSliceType = RemoveReference_t<RT>;

   BLAZE_CONSTRAINT_MUST_BE_ROWSLICE_TYPE    ( RowSliceType );
   BLAZE_CONSTRAINT_MUST_BE_DENSE_MATRIX_TYPE( RowSliceType );

   for( size_t i=0UL; i<rowslice.size(); ++i ) {
      randomize( rowslice[i] );
   }
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Randomization of a dense rowslice.
//
// \param rowslice The rowslice to be randomized.
// \param min The smallest possible value for a rowslice element.
// \param max The largest possible value for a rowslice element.
// \return void
*/
template< typename MT       // Type of the matrix
        , size_t... CRAs >  // Compile time rowslice arguments
template< typename RT       // Type of the rowslice
        , typename Arg >    // Min/max argument type
inline void Rand< RowSlice<MT,CRAs...> >::randomize( RT&& rowslice, const Arg& min, const Arg& max ) const
{
   using blaze::randomize;

   using RowSliceType = RemoveReference_t<RT>;

   BLAZE_CONSTRAINT_MUST_BE_ROWSLICE_TYPE    ( RowSliceType );
   BLAZE_CONSTRAINT_MUST_BE_DENSE_MATRIX_TYPE( RowSliceType );

   for( size_t i=0UL; i<rowslice.size(); ++i ) {
      randomize( rowslice[i], min, max );
   }
}
/*! \endcond */
//*************************************************************************************************

} // namespace blaze

#endif
