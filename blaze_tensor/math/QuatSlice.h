//=================================================================================================
/*!
//  \file blaze_tensor/math/QuatSlice.h
//  \brief Header file for the complete QuatSlice implementation
//
//  Copyright (C) 2012-2018 Klaus Iglberger - All Rights Reserved
//  Copyright (C) 2018-2019 Hartmut Kaiser - All Rights Reserved
//  Copyright (C) 2019 Bita Hasheminezhad - All Rights Reserved
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

#ifndef _BLAZE_TENSOR_MATH_QUATSLICE_H_
#define _BLAZE_TENSOR_MATH_QUATSLICE_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <blaze/math/Aliases.h>
#include <blaze/math/Exception.h>
#include <blaze/util/Random.h>
#include <blaze/util/typetraits/RemoveReference.h>

#include <blaze_tensor/math/constraints/DenseTensor.h>
#include <blaze_tensor/math/constraints/QuatSlice.h>
#include <blaze_tensor/math/smp/DenseTensor.h>
#include <blaze_tensor/math/views/QuatSlice.h>

namespace blaze {

//=================================================================================================
//
//  RAND SPECIALIZATION FOR DENSE QUATSLICES
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Specialization of the Rand class template for dense quatslices.
// \ingroup random
//
// This specialization of the Rand class randomizes dense quatslices.
*/
template< typename MT       // Type of the matrix
        , size_t... CRAs >  // Compile time quatslice arguments
class Rand< QuatSlice<MT,CRAs...> >
{
 public:
   //**Randomize functions*************************************************************************
   /*!\name Randomize functions */
   //@{
   template< typename RT >
   inline void randomize( RT&& quatslice ) const;

   template< typename RT, typename Arg >
   inline void randomize( RT&& quatslice, const Arg& min, const Arg& max ) const;
   //@}
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Randomization of a dense quatslice.
//
// \param quatslice The quatslice to be randomized.
// \return void
*/
template< typename MT       // Type of the matrix
        , size_t... CRAs >  // Compile time quatslice arguments
template< typename RT >     // Type of the quatslice
inline void Rand< QuatSlice<MT,CRAs...> >::randomize( RT&& quatslice ) const
{
   using blaze::randomize;

   using QuatSliceType = RemoveReference_t<RT>;

   BLAZE_CONSTRAINT_MUST_BE_QUATSLICE_TYPE   ( QuatSliceType );
   BLAZE_CONSTRAINT_MUST_BE_DENSE_TENSOR_TYPE( QuatSliceType );

   for (size_t k = 0UL; k < quatslice.pages(); ++k) {
      for (size_t i = 0UL; i < quatslice.rows(); ++i) {
         for (size_t j = 0UL; j < quatslice.columns(); ++j) {
            randomize(quatslice(k, i, j));
         }
      }
   }
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Randomization of a dense quatslice.
//
// \param quatslice The quatslice to be randomized.
// \param min The smallest possible value for a quatslice element.
// \param max The largest possible value for a quatslice element.
// \return void
*/
template< typename MT       // Type of the matrix
        , size_t... CRAs >  // Compile time quatslice arguments
template< typename RT       // Type of the quatslice
        , typename Arg >    // Min/max argument type
inline void Rand< QuatSlice<MT,CRAs...> >::randomize( RT&& quatslice, const Arg& min, const Arg& max ) const
{
   using blaze::randomize;

   using QuatSliceType = RemoveReference_t<RT>;

   BLAZE_CONSTRAINT_MUST_BE_QUATSLICE_TYPE   ( QuatSliceType );
   BLAZE_CONSTRAINT_MUST_BE_DENSE_TENSOR_TYPE( QuatSliceType );

   for (size_t k = 0UL; k < quatslice.pages(); ++k) {
      for (size_t i = 0UL; i < quatslice.rows(); ++i) {
         for (size_t j = 0UL; j < quatslice.columns(); ++j) {
            randomize(quatslice(k, i, j), min, max);
         }
      }
   }
}
/*! \endcond */
//*************************************************************************************************

} // namespace blaze

#endif
