//=================================================================================================
/*!
//  \file blaze_tensor/math/StaticTensor.h
//  \brief Header file for the complete StaticTensor implementation
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

#ifndef _BLAZE_TENSOR_MATH_STATICTENSOR_H_
#define _BLAZE_TENSOR_MATH_STATICTENSOR_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <blaze/math/StaticMatrix.h>
#include <blaze/math/shims/Conjugate.h>
#include <blaze/math/shims/Real.h>
#include <blaze/math/typetraits/UnderlyingBuiltin.h>
#include <blaze/util/Assert.h>
#include <blaze/util/Random.h>
#include <blaze/util/StaticAssert.h>
#include <blaze/util/constraints/Numeric.h>

// #include <blaze_tensor/math/HybridTensor.h>
// #include <blaze_tensor/math/IdentityTensor.h>
#include <blaze_tensor/math/dense/StaticTensor.h>
#include <blaze_tensor/math/DenseTensor.h>

namespace blaze {

//=================================================================================================
//
//  RAND SPECIALIZATION
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Specialization of the Rand class template for StaticTensor.
// \ingroup random
//
// This specialization of the Rand class creates random instances of StaticTensor.
*/
template< typename Type  // Data type of the tensor
        , size_t O       // Number of pages
        , size_t M       // Number of rows
        , size_t N >     // Number of columns
class Rand< StaticTensor<Type,O,M,N> >
{
 public:
   //**Generate functions**************************************************************************
   /*!\name Generate functions */
   //@{
   inline const StaticTensor<Type,O,M,N> generate() const;

   template< typename Arg >
   inline const StaticTensor<Type,O,M,N> generate( const Arg& min, const Arg& max ) const;
   //@}
   //**********************************************************************************************

   //**Randomize functions*************************************************************************
   /*!\name Randomize functions */
   //@{
   inline void randomize( StaticTensor<Type,O,M,N>& tensor ) const;

   template< typename Arg >
   inline void randomize( StaticTensor<Type,O,M,N>& tensor, const Arg& min, const Arg& max ) const;
   //@}
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Generation of a random StaticTensor.
//
// \return The generated random tensor.
*/
template< typename Type  // Data type of the tensor
        , size_t O       // Number of pages
        , size_t M       // Number of rows
        , size_t N >     // Number of columns
inline const StaticTensor<Type,O,M,N> Rand< StaticTensor<Type,O,M,N> >::generate() const
{
   StaticTensor<Type,O,M,N> tensor;
   randomize( tensor );
   return tensor;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Generation of a random StaticTensor.
//
// \param min The smallest possible value for a tensor element.
// \param max The largest possible value for a tensor element.
// \return The generated random tensor.
*/
template< typename Type  // Data type of the tensor
        , size_t O       // Number of pages
        , size_t M       // Number of rows
        , size_t N >     // Number of columns
template< typename Arg >  // Min/max argument type
inline const StaticTensor<Type,O,M,N>
   Rand< StaticTensor<Type,O,M,N> >::generate( const Arg& min, const Arg& max ) const
{
   StaticTensor<Type,O,M,N> tensor;
   randomize( tensor, min, max );
   return tensor;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Randomization of a StaticTensor.
//
// \param tensor The tensor to be randomized.
// \return void
*/
template< typename Type  // Data type of the tensor
        , size_t O       // Number of pages
        , size_t M       // Number of rows
        , size_t N >     // Number of columns
inline void Rand< StaticTensor<Type,O,M,N> >::randomize( StaticTensor<Type,O,M,N>& tensor ) const
{
   using blaze::randomize;

   for( size_t k=0UL; k<O; ++k ) {
      for( size_t i=0UL; i<M; ++i ) {
         for( size_t j=0UL; j<N; ++j ) {
            randomize( tensor(k,i,j) );
         }
      }
   }
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Randomization of a StaticTensor.
//
// \param tensor The tensor to be randomized.
// \param min The smallest possible value for a tensor element.
// \param max The largest possible value for a tensor element.
// \return void
*/
template< typename Type  // Data type of the tensor
        , size_t O       // Number of pages
        , size_t M       // Number of rows
        , size_t N >     // Number of columns
template< typename Arg >  // Min/max argument type
inline void Rand< StaticTensor<Type,O,M,N> >::randomize( StaticTensor<Type,O,M,N>& tensor,
                                                          const Arg& min, const Arg& max ) const
{
   using blaze::randomize;

   for( size_t k=0UL; k<O; ++k ) {
      for( size_t i=0UL; i<M; ++i ) {
         for( size_t j=0UL; j<N; ++j ) {
            randomize( tensor(k,i,j), min, max );
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

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Setup of a random symmetric StaticTensor.
//
// \param tensor The tensor to be randomized.
// \return void
// \exception std::invalid_argument Invalid non-square tensor provided.
*/
// template< typename Type  // Data type of the tensor
//         , size_t O       // Number of pages
//         , size_t M       // Number of rows
//         , size_t N >     // Number of columns
// void makeSymmetric( StaticTensor<Type,O,M,N>& tensor )
// {
//    using blaze::randomize;
//
//    BLAZE_STATIC_ASSERT( M == N );
//
//    for( size_t i=0UL; i<N; ++i ) {
//       for( size_t j=0UL; j<i; ++j ) {
//          randomize( tensor(i,j) );
//          tensor(j,i) = tensor(i,j);
//       }
//       randomize( tensor(i,i) );
//    }
//
//    BLAZE_INTERNAL_ASSERT( isSymmetric( tensor ), "Non-symmetric tensor detected" );
// }
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Setup of a random symmetric StaticTensor.
//
// \param tensor The tensor to be randomized.
// \param min The smallest possible value for a tensor element.
// \param max The largest possible value for a tensor element.
// \return void
// \exception std::invalid_argument Invalid non-square tensor provided.
*/
// template< typename Type   // Data type of the tensor
//         , size_t O        // Number of pages
//         , size_t M        // Number of rows
//         , size_t N        // Number of columns
//         , typename Arg >  // Min/max argument type
// void makeSymmetric( StaticTensor<Type,O,M,N>& tensor, const Arg& min, const Arg& max )
// {
//    using blaze::randomize;
//
//    BLAZE_STATIC_ASSERT( M == N );
//
//    for( size_t i=0UL; i<N; ++i ) {
//       for( size_t j=0UL; j<i; ++j ) {
//          randomize( tensor(i,j), min, max );
//          tensor(j,i) = tensor(i,j);
//       }
//       randomize( tensor(i,i), min, max );
//    }
//
//    BLAZE_INTERNAL_ASSERT( isSymmetric( tensor ), "Non-symmetric tensor detected" );
// }
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Setup of a random Hermitian StaticTensor.
//
// \param tensor The tensor to be randomized.
// \return void
// \exception std::invalid_argument Invalid non-square tensor provided.
*/
// template< typename Type  // Data type of the tensor
//         , size_t O       // Number of pages
//         , size_t M       // Number of rows
//         , size_t N >     // Number of columns
// void makeHermitian( StaticTensor<Type,O,M,N>& tensor )
// {
//    using blaze::randomize;
//
//    BLAZE_STATIC_ASSERT( M == N );
//    BLAZE_CONSTRAINT_MUST_BE_NUMERIC_TYPE( Type );
//
//    using BT = UnderlyingBuiltin_t<Type>;
//
//    for( size_t i=0UL; i<N; ++i ) {
//       for( size_t j=0UL; j<i; ++j ) {
//          randomize( tensor(i,j) );
//          tensor(j,i) = conj( tensor(i,j) );
//       }
//       tensor(i,i) = rand<BT>();
//    }
//
//    BLAZE_INTERNAL_ASSERT( isHermitian( tensor ), "Non-Hermitian tensor detected" );
// }
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Setup of a random Hermitian StaticTensor.
//
// \param tensor The tensor to be randomized.
// \param min The smallest possible value for a tensor element.
// \param max The largest possible value for a tensor element.
// \return void
// \exception std::invalid_argument Invalid non-square tensor provided.
*/
// template< typename Type   // Data type of the tensor
//         , size_t O        // Number of pages
//         , size_t M        // Number of rows
//         , size_t N        // Number of columns
//         , typename Arg >  // Min/max argument type
// void makeHermitian( StaticTensor<Type,O,M,N>& tensor, const Arg& min, const Arg& max )
// {
//    using blaze::randomize;
//
//    BLAZE_STATIC_ASSERT( M == N );
//    BLAZE_CONSTRAINT_MUST_BE_NUMERIC_TYPE( Type );
//
//    using BT = UnderlyingBuiltin_t<Type>;
//
//    for( size_t i=0UL; i<N; ++i ) {
//       for( size_t j=0UL; j<i; ++j ) {
//          randomize( tensor(i,j), min, max );
//          tensor(j,i) = conj( tensor(i,j) );
//       }
//       tensor(i,i) = rand<BT>( real( min ), real( max ) );
//    }
//
//    BLAZE_INTERNAL_ASSERT( isHermitian( tensor ), "Non-Hermitian tensor detected" );
// }
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Setup of a random (Hermitian) positive definite StaticTensor.
//
// \param tensor The tensor to be randomized.
// \return void
// \exception std::invalid_argument Invalid non-square tensor provided.
*/
// template< typename Type  // Data type of the tensor
//         , size_t O       // Number of pages
//         , size_t M       // Number of rows
//         , size_t N >     // Number of columns
// void makePositiveDefinite( StaticTensor<Type,O,M,N>& tensor )
// {
//    using blaze::randomize;
//
//    BLAZE_STATIC_ASSERT( M == N );
//    BLAZE_CONSTRAINT_MUST_BE_NUMERIC_TYPE( Type );
//
//    randomize( tensor );
//    tensor *= ctrans( tensor );
//
//    for( size_t i=0UL; i<N; ++i ) {
//       tensor(i,i) += Type(N);
//    }
//
//    BLAZE_INTERNAL_ASSERT( isHermitian( tensor ), "Non-symmetric tensor detected" );
// }
/*! \endcond */
//*************************************************************************************************

} // namespace blaze

#endif
