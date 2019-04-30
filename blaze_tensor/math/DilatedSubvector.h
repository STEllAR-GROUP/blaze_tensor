//=================================================================================================
/*!
//  \file blaze_tensor/math/DilatedSubvector.h
//  \brief Header file for the complete DilatedSubvector implementation
//
//  Copyright (C) 2012-2019 Klaus Iglberger - All Rights Reserved
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

#ifndef _BLAZE_TENSOR_MATH_DILATEDSUBVECTOR_H_
#define _BLAZE_TENSOR_MATH_DILATEDSUBVECTOR_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <blaze/math/Aliases.h>
#include <blaze/math/constraints/DenseVector.h>
#include <blaze/math/constraints/SparseVector.h>
#include <blaze/math/Exception.h>
#include <blaze/math/smp/DenseVector.h>
#include <blaze/math/smp/SparseVector.h>
#include <blaze/math/views/Submatrix.h>
#include <blaze/util/Random.h>
#include <blaze/util/typetraits/RemoveReference.h>

#include <blaze_tensor/math/constraints/DilatedSubvector.h>
#include <blaze_tensor/math/dense/DynamicVector.h>
#include <blaze_tensor/math/dense/HybridVector.h>
#include <blaze_tensor/math/dense/StaticVector.h>
#include <blaze_tensor/math/dense/UniformVector.h>
#include <blaze_tensor/math/views/DilatedSubvector.h>

namespace blaze {

//=================================================================================================
//
//  RAND SPECIALIZATION FOR DENSE DILATEDSUBVECTORS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Specialization of the Rand class template for dense dilatedsubvectors.
// \ingroup random
//
// This specialization of the Rand class randomizes dense dilatedsubvectors.
*/
template< typename VT       // Type of the vector
        , bool TF           // Transpose flag
        , size_t... CSAs >  // Compile time dilatedsubvector arguments
class Rand< DilatedSubvector<VT,TF,true,CSAs...> >
{
 public:
   //**Randomize functions*************************************************************************
   /*!\name Randomize functions */
   //@{
   template< typename DSVT >
   inline void randomize( DSVT&& dilatedsubvector ) const;

   template< typename DSVT, typename Arg >
   inline void randomize( DSVT&& dilatedsubvector, const Arg& min, const Arg& max ) const;
   //@}
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Randomization of a dense dilatedsubvector.
//
// \param dilatedsubvector The dilatedsubvector to be randomized.
// \return void
*/
template< typename VT       // Type of the vector
        , bool TF           // Transpose flag
        , size_t... CSAs >  // Compile time dilatedsubvector arguments
template< typename DSVT >    // Type of the dilatedsubvector
inline void Rand< DilatedSubvector<VT,TF,true,CSAs...> >::randomize( DSVT&& dilatedsubvector ) const
{
   using blaze::randomize;

   using DilatedSubvectorType = RemoveReference_t<DSVT>;

   BLAZE_CONSTRAINT_MUST_BE_DILATEDSUBVECTOR_TYPE( DilatedSubvectorType );
   BLAZE_CONSTRAINT_MUST_BE_DENSE_VECTOR_TYPE( DilatedSubvectorType );

   for( size_t i=0UL; i<dilatedsubvector.size(); ++i ) {
      randomize( dilatedsubvector[i] );
   }
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Randomization of a dense dilatedsubvector.
//
// \param dilatedsubvector The dilatedsubvector to be randomized.
// \param min The smallest possible value for a dilatedsubvector element.
// \param max The largest possible value for a dilatedsubvector element.
// \return void
*/
template< typename VT       // Type of the vector
        , bool TF           // Transpose flag
        , size_t... CSAs >  // Compile time dilatedsubvector arguments
template< typename DSVT      // Type of the dilatedsubvector
        , typename Arg >    // Min/max argument type
inline void Rand< DilatedSubvector<VT,TF,true,CSAs...> >::randomize( DSVT&& dilatedsubvector, const Arg& min, const Arg& max ) const
{
   using blaze::randomize;

   using DilatedSubvectorType = RemoveReference_t<DSVT>;

   BLAZE_CONSTRAINT_MUST_BE_DILATEDSUBVECTOR_TYPE( DilatedSubvectorType );
   BLAZE_CONSTRAINT_MUST_BE_DENSE_VECTOR_TYPE( DilatedSubvectorType );

   for( size_t i=0UL; i<dilatedsubvector.size(); ++i ) {
      randomize( dilatedsubvector[i], min, max );
   }
}
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  RAND SPECIALIZATION FOR SPARSE DILATEDSUBVECTORS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Specialization of the Rand class template for sparse dilatedsubvectors.
// \ingroup random
//
// This specialization of the Rand class randomizes sparse dilatedsubvectors.
*/
template< typename VT       // Type of the vector
        , bool TF           // Transpose flag
        , size_t... CSAs >  // Compile time dilatedsubvector arguments
class Rand< DilatedSubvector<VT,TF,false,CSAs...> >
{
 public:
   //**Randomize functions*************************************************************************
   /*!\name Randomize functions */
   //@{
   template< typename DSVT >
   inline void randomize( DSVT&& dilatedsubvector ) const;

   template< typename DSVT >
   inline void randomize( DSVT&& dilatedsubvector, size_t nonzeros ) const;

   template< typename DSVT, typename Arg >
   inline void randomize( DSVT&& dilatedsubvector, const Arg& min, const Arg& max ) const;

   template< typename DSVT, typename Arg >
   inline void randomize( DSVT&& dilatedsubvector, size_t nonzeros, const Arg& min, const Arg& max ) const;
   //@}
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Randomization of a sparse dilatedsubvector.
//
// \param dilatedsubvector The dilatedsubvector to be randomized.
// \return void
*/
template< typename VT       // Type of the vector
        , bool TF           // Transpose flag
        , size_t... CSAs >  // Compile time dilatedsubvector arguments
template< typename DSVT >    // Type of the dilatedsubvector
inline void Rand< DilatedSubvector<VT,TF,false,CSAs...> >::randomize( DSVT&& dilatedsubvector ) const
{
   using DilatedSubvectorType = RemoveReference_t<DSVT>;
   using ElementType   = ElementType_t<DilatedSubvectorType>;

   BLAZE_CONSTRAINT_MUST_BE_DILATEDSUBVECTOR_TYPE( DilatedSubvectorType );
   BLAZE_CONSTRAINT_MUST_BE_SPARSE_VECTOR_TYPE( DilatedSubvectorType );

   const size_t size( dilatedsubvector.size() );

   if( size == 0UL ) return;

   const size_t nonzeros( rand<size_t>( 1UL, std::ceil( 0.5*size ) ) );

   dilatedsubvector.reset();
   dilatedsubvector.reserve( nonzeros );

   while( dilatedsubvector.nonZeros() < nonzeros ) {
      dilatedsubvector[ rand<size_t>( 0UL, size-1UL ) ] = rand<ElementType>();
   }
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Randomization of a sparse dilatedsubvector.
//
// \param dilatedsubvector The dilatedsubvector to be randomized.
// \param nonzeros The number of non-zero elements of the random dilatedsubvector.
// \return void
// \exception std::invalid_argument Invalid number of non-zero elements.
*/
template< typename VT       // Type of the vector
        , bool TF           // Transpose flag
        , size_t... CSAs >  // Compile time dilatedsubvector arguments
template< typename DSVT >    // Type of the dilatedsubvector
inline void Rand< DilatedSubvector<VT,TF,false,CSAs...> >::randomize( DSVT&& dilatedsubvector, size_t nonzeros ) const
{
   using DilatedSubvectorType = RemoveReference_t<DSVT>;
   using ElementType   = ElementType_t<DilatedSubvectorType>;

   BLAZE_CONSTRAINT_MUST_BE_DILATEDSUBVECTOR_TYPE( DilatedSubvectorType );
   BLAZE_CONSTRAINT_MUST_BE_SPARSE_VECTOR_TYPE( DilatedSubvectorType );

   const size_t size( dilatedsubvector.size() );

   if( nonzeros > size ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid number of non-zero elements" );
   }

   if( size == 0UL ) return;

   dilatedsubvector.reset();
   dilatedsubvector.reserve( nonzeros );

   while( dilatedsubvector.nonZeros() < nonzeros ) {
      dilatedsubvector[ rand<size_t>( 0UL, size-1UL ) ] = rand<ElementType>();
   }
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Randomization of a sparse dilatedsubvector.
//
// \param dilatedsubvector The dilatedsubvector to be randomized.
// \param min The smallest possible value for a dilatedsubvector element.
// \param max The largest possible value for a dilatedsubvector element.
// \return void
*/
template< typename VT       // Type of the vector
        , bool TF           // Transpose flag
        , size_t... CSAs >  // Compile time dilatedsubvector arguments
template< typename DSVT      // Type of the dilatedsubvector
        , typename Arg >    // Min/max argument type
inline void Rand< DilatedSubvector<VT,TF,false,CSAs...> >::randomize( DSVT&& dilatedsubvector,
                                                                  const Arg& min, const Arg& max ) const
{
   using DilatedSubvectorType = RemoveReference_t<DSVT>;
   using ElementType   = ElementType_t<DilatedSubvectorType>;

   BLAZE_CONSTRAINT_MUST_BE_DILATEDSUBVECTOR_TYPE( DilatedSubvectorType );
   BLAZE_CONSTRAINT_MUST_BE_SPARSE_VECTOR_TYPE( DilatedSubvectorType );

   const size_t size( dilatedsubvector.size() );

   if( size == 0UL ) return;

   const size_t nonzeros( rand<size_t>( 1UL, std::ceil( 0.5*size ) ) );

   dilatedsubvector.reset();
   dilatedsubvector.reserve( nonzeros );

   while( dilatedsubvector.nonZeros() < nonzeros ) {
      dilatedsubvector[ rand<size_t>( 0UL, size-1UL ) ] = rand<ElementType>( min, max );
   }
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Randomization of a sparse dilatedsubvector.
//
// \param dilatedsubvector The dilatedsubvector to be randomized.
// \param nonzeros The number of non-zero elements of the random dilatedsubvector.
// \param min The smallest possible value for a dilatedsubvector element.
// \param max The largest possible value for a dilatedsubvector element.
// \return void
// \exception std::invalid_argument Invalid number of non-zero elements.
*/
template< typename VT       // Type of the vector
        , bool TF           // Transpose flag
        , size_t... CSAs >  // Compile time dilatedsubvector arguments
template< typename DSVT      // Type of the dilatedsubvector
        , typename Arg >    // Min/max argument type
inline void Rand< DilatedSubvector<VT,TF,false,CSAs...> >::randomize( DSVT&& dilatedsubvector, size_t nonzeros,
                                                                  const Arg& min, const Arg& max ) const
{
   using DilatedSubvectorType = RemoveReference_t<DSVT>;
   using ElementType   = ElementType_t<DilatedSubvectorType>;

   BLAZE_CONSTRAINT_MUST_BE_DILATEDSUBVECTOR_TYPE( DilatedSubvectorType );
   BLAZE_CONSTRAINT_MUST_BE_SPARSE_VECTOR_TYPE( DilatedSubvectorType );

   const size_t size( dilatedsubvector.size() );

   if( nonzeros > size ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid number of non-zero elements" );
   }

   if( size == 0UL ) return;

   dilatedsubvector.reset();
   dilatedsubvector.reserve( nonzeros );

   while( dilatedsubvector.nonZeros() < nonzeros ) {
      dilatedsubvector[ rand<size_t>( 0UL, size-1UL ) ] = rand<ElementType>( min, max );
   }
}
/*! \endcond */
//*************************************************************************************************

} // namespace blaze

#endif
