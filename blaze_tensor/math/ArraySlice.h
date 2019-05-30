//=================================================================================================
/*!
//  \file blaze_tensor/math/ArraySlice.h
//  \brief Header file for the complete ArraySlice implementation
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

#ifndef _BLAZE_TENSOR_MATH_ARRAYSLICE_H_
#define _BLAZE_TENSOR_MATH_ARRAYSLICE_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <blaze/math/Aliases.h>
#include <blaze/math/constraints/DenseMatrix.h>
#include <blaze/math/Exception.h>
#include <blaze/math/smp/DenseMatrix.h>
#include <blaze/util/Random.h>
#include <blaze/util/typetraits/RemoveReference.h>

#include <blaze_tensor/math/constraints/ArraySlice.h>
#include <blaze_tensor/math/views/ArraySlice.h>
#include <blaze_tensor/util/ArrayForEach.h>

namespace blaze {

//=================================================================================================
//
//  RAND SPECIALIZATION FOR DENSE ARRAYSLICES
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Specialization of the Rand class template for dense arrayslices.
// \ingroup random
//
// This specialization of the Rand class randomizes dense arrayslices.
*/
template< size_t M            // Dimension of the ArraysSlice
        , typename MT         // Type of the dense array
        , size_t... CRAs >    // Compile time arrayslice arguments
class Rand< ArraySlice<M,MT,CRAs...> >
{
 public:
   //**Randomize functions*************************************************************************
   /*!\name Randomize functions */
   //@{
   template< typename RT >
   inline void randomize( RT&& arrayslice ) const;

   template< typename RT, typename Arg >
   inline void randomize( RT&& arrayslice, const Arg& min, const Arg& max ) const;
   //@}
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Randomization of a dense arrayslice.
//
// \param arrayslice The arrayslice to be randomized.
// \return void
*/
template< size_t M            // Dimension of the ArraysSlice
        , typename MT         // Type of the dense array
        , size_t... CRAs >    // Compile time arrayslice arguments
template< typename RT >       // Type of the arrayslice
inline void Rand< ArraySlice<M,MT,CRAs...> >::randomize( RT&& arrayslice ) const
{
   using blaze::randomize;

   using ArraySliceType = RemoveReference_t<RT>;

   BLAZE_CONSTRAINT_MUST_BE_ARRAYSLICE_TYPE ( ArraySliceType );
   BLAZE_CONSTRAINT_MUST_BE_DENSE_ARRAY_TYPE( ArraySliceType );

   constexpr size_t N = ArraySliceType::num_dimensions();

   ArrayForEachGrouped(
      arrayslice.dimensions(), [&]( std::array< size_t, N > const& indices ) {
         randomize( arrayslice( indices ) );
      } );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Randomization of a dense arrayslice.
//
// \param arrayslice The arrayslice to be randomized.
// \param min The smallest possible value for a arrayslice element.
// \param max The largest possible value for a arrayslice element.
// \return void
*/
template< size_t M            // Dimension of the ArraysSlice
        , typename MT         // Type of the dense array
        , size_t... CRAs >    // Compile time arrayslice arguments
template< typename RT         // Type of the arrayslice
        , typename Arg >      // Min/max argument type
inline void Rand< ArraySlice<M,MT,CRAs...> >::randomize( RT&& arrayslice, const Arg& min, const Arg& max ) const
{
   using blaze::randomize;

   using ArraySliceType = RemoveReference_t<RT>;

   BLAZE_CONSTRAINT_MUST_BE_ARRAYSLICE_TYPE ( ArraySliceType );
   BLAZE_CONSTRAINT_MUST_BE_DENSE_ARRAY_TYPE( ArraySliceType );

   constexpr size_t N = ArraySliceType::num_dimensions();

   ArrayForEachGrouped(
      arrayslice.dimensions(), [&]( std::array< size_t, N > const& indices ) {
         randomize( arrayslice( indices ), min, max );
      } );
}
/*! \endcond */
//*************************************************************************************************

} // namespace blaze

#endif
