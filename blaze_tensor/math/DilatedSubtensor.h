//=================================================================================================
/*!
//  \file blaze_tensor/math/DilatedSubtensor.h
//  \brief Header file for the complete DilatedSubtensor implementation
//
//  Copyright (C) 2012-2019 Klaus Iglberger - All Rights Reserved
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

#ifndef _BLAZE_TENSOR_MATH_DILATEDSUBTENSOR_H_
#define _BLAZE_TENSOR_MATH_DILATEDSUBTENSOR_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <blaze/math/Aliases.h>
#include <blaze/math/Exception.h>

#include <blaze/util/Random.h>
#include <blaze/util/typetraits/RemoveReference.h>

#include <blaze_tensor/math/constraints/DenseTensor.h>
#include <blaze_tensor/math/constraints/DilatedSubtensor.h>
#include <blaze_tensor/math/dense/DynamicTensor.h>
#include <blaze_tensor/math/dense/StaticTensor.h>
#include <blaze_tensor/math/dense/UniformTensor.h>
#include <blaze_tensor/math/smp/DenseTensor.h>
#include <blaze_tensor/math/views/DilatedSubtensor.h>
#include <blaze_tensor/math/views/Subtensor.h>


namespace blaze {

//=================================================================================================
//
//  RAND SPECIALIZATION FOR DENSE DILATEDSUBTENSORS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Specialization of the Rand class template for dense dilatedsubtensors.
// \ingroup random
//
// This specialization of the Rand class randomizes dense dilatedsubtensors.
*/
template< typename TT       // Type of the tensor
        , size_t... CSAs >  // Compile time dilatedsubtensors arguments
class Rand< DilatedSubtensor<TT,true,CSAs...> >
{
 public:
   //**Randomize functions*************************************************************************
   /*!\name Randomize functions */
   //@{
   template< typename DSTT >
   inline void randomize( DSTT&& dilatedsubtensor ) const;

   template< typename DSTT, typename Arg >
   inline void randomize( DSTT&& dilatedsubtensor, const Arg& min, const Arg& max ) const;
   //@}
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Randomization of a dense dilatedsubtensor.
//
// \param dilatedsubtensor The dilatedsubtensor to be randomized.
// \return void
*/
template< typename TT       // Type of the dense tensor
        , size_t... CSAs >  // Compile time dilatedsubtensor arguments
template< typename DSTT >    // Type of the dilatedsubtensor
inline void Rand< DilatedSubtensor<TT,true,CSAs...> >::randomize( DSTT&& dilatedsubtensor ) const
{
   using blaze::randomize;

   using DilatedSubtensorType = RemoveReference_t<DSTT>;

   BLAZE_CONSTRAINT_MUST_BE_DILATEDSUBTENSOR_TYPE( DilatedSubtensorType );
   BLAZE_CONSTRAINT_MUST_BE_DENSE_TENSOR_TYPE( DilatedSubtensorType );

   for( size_t k=0UL; k<dilatedsubtensor.pages(); ++k ) {
      for( size_t i=0UL; i<dilatedsubtensor.rows(); ++i ) {
         for( size_t j=0UL; j<dilatedsubtensor.columns(); ++j ) {
            randomize( dilatedsubtensor(k,i,j) );
         }
      }
   }

}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Randomization of a dense dilatedsubtensor.
//
// \param dilatedsubtensor The dilatedsubtensor to be randomized.
// \param min The smallest possible value for a dilatedsubtensor element.
// \param max The largest possible value for a dilatedsubtensor element.
// \return void
*/
template< typename TT       // Type of the dense tensor
        , size_t... CSAs >  // Compile time dilatedsubtensor arguments
template< typename DSTT      // Type of the dilatedsubtensor
        , typename Arg >    // Min/max argument type
inline void Rand< DilatedSubtensor<TT,true,CSAs...> >::randomize( DSTT&& dilatedsubtensor,
                                                             const Arg& min, const Arg& max ) const
{
   using blaze::randomize;

   using DilatedSubtensorType = RemoveReference_t<DSTT>;

   BLAZE_CONSTRAINT_MUST_BE_DILATEDSUBTENSOR_TYPE( DilatedSubtensorType );
   BLAZE_CONSTRAINT_MUST_BE_DENSE_TENSOR_TYPE( DilatedSubtensorType );

   for( size_t k=0UL; k<dilatedsubtensor.pages(); ++k ) {
      for( size_t i=0UL; i<dilatedsubtensor.rows(); ++i ) {
         for( size_t j=0UL; j<dilatedsubtensor.columns(); ++j ) {
            randomize( dilatedsubtensor(k,i,j), min, max );
         }
      }
   }
}
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  RAND SPECIALIZATION FOR SPARSE DILATEDSUBTENSORS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Specialization of the Rand class template for sparse dilatedsubtensors.
// \ingroup random
//
// This specialization of the Rand class randomizes sparse dilatedsubtensors.
*/
//template< typename TT       // Type of the dense tensor
//        , size_t... CSAs >  // Compile time dilatedsubtensor arguments
//class Rand< DilatedSubtensor<TT,CSAs...> >
//{
// public:
//   //**Randomize functions*************************************************************************
//   /*!\name Randomize functions */
//   //@{
//   template< typename DSTT >
//   inline void randomize( DSTT&& dilatedsubtensor ) const;
//
//   template< typename DSTT >
//   inline void randomize( DSTT&& dilatedsubtensor, size_t nonzeros ) const;
//
//   template< typename DSTT, typename Arg >
//   inline void randomize( DSTT&& dilatedsubtensor, const Arg& min, const Arg& max ) const;
//
//   template< typename DSTT, typename Arg >
//   inline void randomize( DSTT&& dilatedsubtensor, size_t nonzeros, const Arg& min, const Arg& max ) const;
//   //@}
//   //**********************************************************************************************
//};
///*! \endcond */
////*************************************************************************************************
//
//
////*************************************************************************************************
///*! \cond BLAZE_INTERNAL */
///*!\brief Randomization of a sparse dilatedsubtensor.
////
//// \param dilatedsubtensor The dilatedsubtensor to be randomized.
//// \return void
//*/
//template< typename TT       // Type of the dense tensor
//        , bool SO           // Storage order
//        , size_t... CSAs >  // Compile time dilatedsubtensor arguments
//template< typename DSTT >    // Type of the dilatedsubtensor
//inline void Rand< DilatedSubtensor<TT,false,CSAs...> >::randomize( DSTT&& dilatedsubtensor ) const
//{
//   using DilatedSubtensorType = RemoveReference_t<DSTT>;
//   using ElementType   = ElementType_t<DilatedSubtensorType>;
//
//   BLAZE_CONSTRAINT_MUST_BE_DILATEDSUBTENSOR_TYPE( DilatedSubtensorType );
//   BLAZE_CONSTRAINT_MUST_BE_SPARSE_TENSOR_TYPE( DilatedSubtensorType );
//
//   const size_t m( dilatedsubtensor.rows()    );
//   const size_t n( dilatedsubtensor.columns() );
//
//   if( m == 0UL || n == 0UL ) return;
//
//   const size_t nonzeros( rand<size_t>( 1UL, std::ceil( 0.5*m*n ) ) );
//
//   dilatedsubtensor.reset();
//   dilatedsubtensor.reserve( nonzeros );
//
//   while( dilatedsubtensor.nonZeros() < nonzeros ) {
//      dilatedsubtensor( rand<size_t>( 0UL, m-1UL ), rand<size_t>( 0UL, n-1UL ) ) = rand<ElementType>();
//   }
//}
///*! \endcond */
////*************************************************************************************************
//
//
////*************************************************************************************************
///*! \cond BLAZE_INTERNAL */
///*!\brief Randomization of a sparse dilatedsubtensor.
////
//// \param dilatedsubtensor The dilatedsubtensor to be randomized.
//// \param nonzeros The number of non-zero elements of the random dilatedsubtensor.
//// \return void
//// \exception std::invalid_argument Invalid number of non-zero elements.
//*/
//template< typename TT       // Type of the dense tensor
//        , bool SO           // Storage order
//        , size_t... CSAs >  // Compile time dilatedsubtensor arguments
//template< typename DSTT >    // Type of the dilatedsubtensor
//inline void Rand< DilatedSubtensor<TT,false,CSAs...> >::randomize( DSTT&& dilatedsubtensor, size_t nonzeros ) const
//{
//   using DilatedSubtensorType = RemoveReference_t<DSTT>;
//   using ElementType   = ElementType_t<DilatedSubtensorType>;
//
//   BLAZE_CONSTRAINT_MUST_BE_DILATEDSUBTENSOR_TYPE( DilatedSubtensorType );
//   BLAZE_CONSTRAINT_MUST_BE_SPARSE_TENSOR_TYPE( DilatedSubtensorType );
//
//   const size_t m( dilatedsubtensor.rows()    );
//   const size_t n( dilatedsubtensor.columns() );
//
//   if( nonzeros > m*n ) {
//      BLAZE_THROW_INVALID_ARGUMENT( "Invalid number of non-zero elements" );
//   }
//
//   if( m == 0UL || n == 0UL ) return;
//
//   dilatedsubtensor.reset();
//   dilatedsubtensor.reserve( nonzeros );
//
//   while( dilatedsubtensor.nonZeros() < nonzeros ) {
//      dilatedsubtensor( rand<size_t>( 0UL, m-1UL ), rand<size_t>( 0UL, n-1UL ) ) = rand<ElementType>();
//   }
//}
///*! \endcond */
////*************************************************************************************************
//
//
////*************************************************************************************************
///*! \cond BLAZE_INTERNAL */
///*!\brief Randomization of a sparse dilatedsubtensor.
////
//// \param dilatedsubtensor The dilatedsubtensor to be randomized.
//// \param min The smallest possible value for a dilatedsubtensor element.
//// \param max The largest possible value for a dilatedsubtensor element.
//// \return void
//*/
//template< typename TT       // Type of the dense tensor
//        , bool SO           // Storage order
//        , size_t... CSAs >  // Compile time dilatedsubtensor arguments
//template< typename DSTT      // Type of the dilatedsubtensor
//        , typename Arg >    // Min/max argument type
//inline void Rand< DilatedSubtensor<TT,false,CSAs...> >::randomize( DSTT&& dilatedsubtensor,
//                                                                  const Arg& min, const Arg& max ) const
//{
//   using DilatedSubtensorType = RemoveReference_t<DSTT>;
//   using ElementType   = ElementType_t<DilatedSubtensorType>;
//
//   BLAZE_CONSTRAINT_MUST_BE_DILATEDSUBTENSOR_TYPE( DilatedSubtensorType );
//   BLAZE_CONSTRAINT_MUST_BE_SPARSE_TENSOR_TYPE( DilatedSubtensorType );
//
//   const size_t m( dilatedsubtensor.rows()    );
//   const size_t n( dilatedsubtensor.columns() );
//
//   if( m == 0UL || n == 0UL ) return;
//
//   const size_t nonzeros( rand<size_t>( 1UL, std::ceil( 0.5*m*n ) ) );
//
//   dilatedsubtensor.reset();
//   dilatedsubtensor.reserve( nonzeros );
//
//   while( dilatedsubtensor.nonZeros() < nonzeros ) {
//      dilatedsubtensor( rand<size_t>( 0UL, m-1UL ), rand<size_t>( 0UL, n-1UL ) ) = rand<ElementType>( min, max );
//   }
//}
///*! \endcond */
////*************************************************************************************************
//
//
////*************************************************************************************************
///*! \cond BLAZE_INTERNAL */
///*!\brief Randomization of a sparse dilatedsubtensor.
////
//// \param dilatedsubtensor The dilatedsubtensor to be randomized.
//// \param nonzeros The number of non-zero elements of the random dilatedsubtensor.
//// \param min The smallest possible value for a dilatedsubtensor element.
//// \param max The largest possible value for a dilatedsubtensor element.
//// \return void
//// \exception std::invalid_argument Invalid number of non-zero elements.
//*/
//template< typename TT       // Type of the dense tensor
//        , bool SO           // Storage order
//        , size_t... CSAs >  // Compile time dilatedsubtensor arguments
//template< typename DSTT      // Type of the dilatedsubtensor
//        , typename Arg >    // Min/max argument type
//inline void Rand< DilatedSubtensor<TT,false,CSAs...> >::randomize( DSTT&& dilatedsubtensor, size_t nonzeros,
//                                                                  const Arg& min, const Arg& max ) const
//{
//   using DilatedSubtensorType = RemoveReference_t<DSTT>;
//   using ElementType   = ElementType_t<DilatedSubtensorType>;
//
//   BLAZE_CONSTRAINT_MUST_BE_DILATEDSUBTENSOR_TYPE( DilatedSubtensorType );
//   BLAZE_CONSTRAINT_MUST_BE_SPARSE_TENSOR_TYPE( DilatedSubtensorType );
//
//   const size_t m( dilatedsubtensor.rows()    );
//   const size_t n( dilatedsubtensor.columns() );
//
//   if( nonzeros > m*n ) {
//      BLAZE_THROW_INVALID_ARGUMENT( "Invalid number of non-zero elements" );
//   }
//
//   if( m == 0UL || n == 0UL ) return;
//
//   dilatedsubtensor.reset();
//   dilatedsubtensor.reserve( nonzeros );
//
//   while( dilatedsubtensor.nonZeros() < nonzeros ) {
//      dilatedsubtensor( rand<size_t>( 0UL, m-1UL ), rand<size_t>( 0UL, n-1UL ) ) = rand<ElementType>( min, max );
//   }
//}
/*! \endcond */
//*************************************************************************************************

} // namespace blaze

#endif
