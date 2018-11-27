//=================================================================================================
/*!
//  \file blaze_tensor/math/DynamicTensor.h
//  \brief Header file for the complete DynamicTensor implementation
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

#ifndef _BLAZE_TENSOR_MATH_DYNAMICTENSOR_H_
#define _BLAZE_TENSOR_MATH_DYNAMICTENSOR_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <blaze/math/DynamicMatrix.h>

#include <blaze_tensor/math/DenseTensor.h>
#include <blaze_tensor/math/dense/DynamicTensor.h>

namespace blaze {

//=================================================================================================
//
//  RAND SPECIALIZATION
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Specialization of the Rand class template for DynamicTensor.
// \ingroup random
//
// This specialization of the Rand class creates random instances of DynamicTensor.
*/
template< typename Type > // Data type of the tensor
class Rand< DynamicTensor<Type> >
{
 public:
   //**Generate functions**************************************************************************
   /*!\name Generate functions */
   //@{
   inline const DynamicTensor<Type> generate( size_t o, size_t m, size_t n ) const;

   template< typename Arg >
   inline const DynamicTensor<Type> generate( size_t o, size_t m, size_t n, const Arg& min, const Arg& max ) const;
   //@}
   //**********************************************************************************************

   //**Randomize functions*************************************************************************
   /*!\name Randomize functions */
   //@{
   inline void randomize( DynamicTensor<Type>& tensor ) const;

   template< typename Arg >
   inline void randomize( DynamicTensor<Type>& tensor, const Arg& min, const Arg& max ) const;
   //@}
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Generation of a random DynamicTensor.
//
// \param m The number of rows of the random tensor.
// \param n The number of columns of the random tensor.
// \return The generated random tensor.
*/
template< typename Type > // Data type of the tensor
inline const DynamicTensor<Type>
   Rand< DynamicTensor<Type> >::generate( size_t o, size_t m, size_t n ) const
{
   DynamicTensor<Type> tensor( o, m, n );
   randomize( tensor );
   return tensor;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Generation of a random DynamicTensor.
//
// \param m The number of rows of the random tensor.
// \param n The number of columns of the random tensor.
// \param min The smallest possible value for a tensor element.
// \param max The largest possible value for a tensor element.
// \return The generated random tensor.
*/
template< typename Type > // Data type of the tensor
template< typename Arg >  // Min/max argument type
inline const DynamicTensor<Type>
   Rand< DynamicTensor<Type> >::generate( size_t o, size_t m, size_t n, const Arg& min, const Arg& max ) const
{
   DynamicTensor<Type> tensor( o, m, n );
   randomize( tensor, min, max );
   return tensor;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Randomization of a DynamicTensor.
//
// \param tensor The tensor to be randomized.
// \return void
*/
template< typename Type > // Data type of the tensor
inline void Rand< DynamicTensor<Type> >::randomize( DynamicTensor<Type>& tensor ) const
{
   using blaze::randomize;

   const size_t m( tensor.rows()    );
   const size_t n( tensor.columns() );
   const size_t o( tensor.pages()   );

   for (size_t k=0UL; k<o; ++k) {
      for (size_t i=0UL; i<m; ++i) {
         for (size_t j=0UL; j<n; ++j) {
            randomize(tensor(k, i, j));
         }
      }
   }
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Randomization of a DynamicTensor.
//
// \param tensor The tensor to be randomized.
// \param min The smallest possible value for a tensor element.
// \param max The largest possible value for a tensor element.
// \return void
*/
template< typename Type > // Data type of the tensor
template< typename Arg >  // Min/max argument type
inline void Rand< DynamicTensor<Type> >::randomize( DynamicTensor<Type>& tensor,
                                                       const Arg& min, const Arg& max ) const
{
   using blaze::randomize;

   const size_t m( tensor.rows()    );
   const size_t n( tensor.columns() );
   const size_t o( tensor.pages()   );

   for (size_t k=0UL; k<o; ++k) {
      for (size_t i=0UL; i<m; ++i) {
         for (size_t j=0UL; j<n; ++j) {
            randomize(tensor(k, i, j), min, max);
         }
      }
   }
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
