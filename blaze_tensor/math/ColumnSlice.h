//=================================================================================================
/*!
//  \file blaze_tensor/math/ColumnSlice.h
//  \brief Header file for the complete ColumnSlice implementation
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

#ifndef _BLAZE_TENSOR_MATH_COLUMNSLICE_H_
#define _BLAZE_TENSOR_MATH_COLUMNSLICE_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <blaze/math/Aliases.h>
#include <blaze/math/constraints/DenseMatrix.h>
#include <blaze/math/Exception.h>
#include <blaze/math/smp/DenseMatrix.h>
#include <blaze/util/Random.h>
#include <blaze/util/typetraits/RemoveReference.h>

#include <blaze_tensor/math/constraints/ColumnSlice.h>
#include <blaze_tensor/math/views/ColumnSlice.h>

namespace blaze {

//=================================================================================================
//
//  RAND SPECIALIZATION FOR DENSE PAGESLICES
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Specialization of the Rand class template for dense columnslices.
// \ingroup random
//
// This specialization of the Rand class randomizes dense columnslices.
*/
template< typename MT       // Type of the matrix
        , size_t... CRAs >  // Compile time columnslice arguments
class Rand< ColumnSlice<MT,CRAs...> >
{
 public:
   //**Randomize functions*************************************************************************
   /*!\name Randomize functions */
   //@{
   template< typename RT >
   inline void randomize( RT&& columnslice ) const;

   template< typename RT, typename Arg >
   inline void randomize( RT&& columnslice, const Arg& min, const Arg& max ) const;
   //@}
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Randomization of a dense columnslice.
//
// \param columnslice The columnslice to be randomized.
// \return void
*/
template< typename MT       // Type of the matrix
        , size_t... CRAs >  // Compile time columnslice arguments
template< typename RT >     // Type of the columnslice
inline void Rand< ColumnSlice<MT,CRAs...> >::randomize( RT&& columnslice ) const
{
   using blaze::randomize;

   using ColumnSliceType = RemoveReference_t<RT>;

   BLAZE_CONSTRAINT_MUST_BE_COLUMNSLICE_TYPE ( ColumnSliceType );
   BLAZE_CONSTRAINT_MUST_BE_DENSE_MATRIX_TYPE( ColumnSliceType );

   for( size_t i=0UL; i<columnslice.rows(); ++i ) {
      for( size_t j=0UL; j<columnslice.columns(); ++j ) {
         randomize( columnslice(i, j) );
      }
   }
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Randomization of a dense columnslice.
//
// \param columnslice The columnslice to be randomized.
// \param min The smallest possible value for a columnslice element.
// \param max The largest possible value for a columnslice element.
// \return void
*/
template< typename MT       // Type of the matrix
        , size_t... CRAs >  // Compile time columnslice arguments
template< typename RT       // Type of the columnslice
        , typename Arg >    // Min/max argument type
inline void Rand< ColumnSlice<MT,CRAs...> >::randomize( RT&& columnslice, const Arg& min, const Arg& max ) const
{
   using blaze::randomize;

   using ColumnSliceType = RemoveReference_t<RT>;

   BLAZE_CONSTRAINT_MUST_BE_COLUMNSLICE_TYPE ( ColumnSliceType );
   BLAZE_CONSTRAINT_MUST_BE_DENSE_MATRIX_TYPE( ColumnSliceType );

   for( size_t i=0UL; i<columnslice.rows(); ++i ) {
      for( size_t j=0UL; j<columnslice.columns(); ++j ) {
         randomize( columnslice(i, j), min, max );
      }
   }
}
/*! \endcond */
//*************************************************************************************************

} // namespace blaze

#endif
