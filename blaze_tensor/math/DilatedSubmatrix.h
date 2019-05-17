//=================================================================================================
/*!
//  \file blaze_tensor/math/DilatedSubmatrix.h
//  \brief Header file for the complete DilatedSubmatrix implementation
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

#ifndef _BLAZE_TENSOR_MATH_DILATEDSUBMATRIX_H_
#define _BLAZE_TENSOR_MATH_DILATEDSUBMATRIX_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <blaze/math/Aliases.h>
#include <blaze/math/constraints/DenseMatrix.h>
#include <blaze/math/constraints/SparseMatrix.h>
#include <blaze/math/Exception.h>
#include <blaze/math/smp/DenseMatrix.h>
#include <blaze/math/smp/SparseMatrix.h>
#include <blaze/math/views/Subvector.h>
#include <blaze/util/Random.h>
#include <blaze/util/typetraits/RemoveReference.h>

#include <blaze_tensor/math/constraints/DilatedSubmatrix.h>
#include <blaze_tensor/math/dense/DynamicMatrix.h>
#include <blaze_tensor/math/dense/HybridMatrix.h>
#include <blaze_tensor/math/dense/StaticMatrix.h>
#include <blaze_tensor/math/dense/UniformMatrix.h>
#include <blaze_tensor/math/views/DilatedSubmatrix.h>


namespace blaze {

//=================================================================================================
//
//  RAND SPECIALIZATION FOR DENSE DILATEDSUBMATRICES
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Specialization of the Rand class template for dense dilatedsubmatrices.
// \ingroup random
//
// This specialization of the Rand class randomizes dense dilatedsubmatrices.
*/
template< typename MT       // Type of the matrix
        , bool SO           // Storage order
        , size_t... CSAs >  // Compile time dilatedsubmatrices arguments
class Rand< DilatedSubmatrix<MT,SO,true,CSAs...> >
{
 public:
   //**Randomize functions*************************************************************************
   /*!\name Randomize functions */
   //@{
   template< typename DSMT >
   inline void randomize( DSMT&& dilatedsubmatrix ) const;

   template< typename DSMT, typename Arg >
   inline void randomize( DSMT&& dilatedsubmatrix, const Arg& min, const Arg& max ) const;
   //@}
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Randomization of a dense dilatedsubmatrix.
//
// \param dilatedsubmatrix The dilatedsubmatrix to be randomized.
// \return void
*/
template< typename MT       // Type of the dense matrix
        , bool SO           // Storage order
        , size_t... CSAs >  // Compile time dilatedsubmatrix arguments
template< typename DSMT >    // Type of the dilatedsubmatrix
inline void Rand< DilatedSubmatrix<MT,SO,true,CSAs...> >::randomize( DSMT&& dilatedsubmatrix ) const
{
   using blaze::randomize;

   using DilatedSubmatrixType = RemoveReference_t<DSMT>;

   BLAZE_CONSTRAINT_MUST_BE_DILATEDSUBMATRIX_TYPE( DilatedSubmatrixType );
   BLAZE_CONSTRAINT_MUST_BE_DENSE_MATRIX_TYPE( DilatedSubmatrixType );

   if( SO == rowMajor ) {
      for( size_t i=0UL; i<dilatedsubmatrix.rows(); ++i ) {
         for( size_t j=0UL; j<dilatedsubmatrix.columns(); ++j ) {
            randomize( dilatedsubmatrix(i,j) );
         }
      }
   }
   else {
      for( size_t j=0UL; j<dilatedsubmatrix.columns(); ++j ) {
         for( size_t i=0UL; i<dilatedsubmatrix.rows(); ++i ) {
            randomize( dilatedsubmatrix(i,j) );
         }
      }
   }
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Randomization of a dense dilatedsubmatrix.
//
// \param dilatedsubmatrix The dilatedsubmatrix to be randomized.
// \param min The smallest possible value for a dilatedsubmatrix element.
// \param max The largest possible value for a dilatedsubmatrix element.
// \return void
*/
template< typename MT       // Type of the dense matrix
        , bool SO           // Storage order
        , size_t... CSAs >  // Compile time dilatedsubmatrix arguments
template< typename DSMT      // Type of the dilatedsubmatrix
        , typename Arg >    // Min/max argument type
inline void Rand< DilatedSubmatrix<MT,SO,true,CSAs...> >::randomize( DSMT&& dilatedsubmatrix,
                                                                 const Arg& min, const Arg& max ) const
{
   using blaze::randomize;

   using DilatedSubmatrixType = RemoveReference_t<DSMT>;

   BLAZE_CONSTRAINT_MUST_BE_DILATEDSUBMATRIX_TYPE( DilatedSubmatrixType );
   BLAZE_CONSTRAINT_MUST_BE_DENSE_MATRIX_TYPE( DilatedSubmatrixType );

   if( SO == rowMajor ) {
      for( size_t i=0UL; i<dilatedsubmatrix.rows(); ++i ) {
         for( size_t j=0UL; j<dilatedsubmatrix.columns(); ++j ) {
            randomize( dilatedsubmatrix(i,j), min, max );
         }
      }
   }
   else {
      for( size_t j=0UL; j<dilatedsubmatrix.columns(); ++j ) {
         for( size_t i=0UL; i<dilatedsubmatrix.rows(); ++i ) {
            randomize( dilatedsubmatrix(i,j), min, max );
         }
      }
   }
}
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  RAND SPECIALIZATION FOR SPARSE DILATEDSUBMATRICES
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Specialization of the Rand class template for sparse dilatedsubmatrices.
// \ingroup random
//
// This specialization of the Rand class randomizes sparse dilatedsubmatrices.
*/
template< typename MT       // Type of the dense matrix
        , bool SO           // Storage order
        , size_t... CSAs >  // Compile time dilatedsubmatrix arguments
class Rand< DilatedSubmatrix<MT,SO,false,CSAs...> >
{
 public:
   //**Randomize functions*************************************************************************
   /*!\name Randomize functions */
   //@{
   template< typename DSMT >
   inline void randomize( DSMT&& dilatedsubmatrix ) const;

   template< typename DSMT >
   inline void randomize( DSMT&& dilatedsubmatrix, size_t nonzeros ) const;

   template< typename DSMT, typename Arg >
   inline void randomize( DSMT&& dilatedsubmatrix, const Arg& min, const Arg& max ) const;

   template< typename DSMT, typename Arg >
   inline void randomize( DSMT&& dilatedsubmatrix, size_t nonzeros, const Arg& min, const Arg& max ) const;
   //@}
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Randomization of a sparse dilatedsubmatrix.
//
// \param dilatedsubmatrix The dilatedsubmatrix to be randomized.
// \return void
*/
template< typename MT       // Type of the dense matrix
        , bool SO           // Storage order
        , size_t... CSAs >  // Compile time dilatedsubmatrix arguments
template< typename DSMT >    // Type of the dilatedsubmatrix
inline void Rand< DilatedSubmatrix<MT,SO,false,CSAs...> >::randomize( DSMT&& dilatedsubmatrix ) const
{
   using DilatedSubmatrixType = RemoveReference_t<DSMT>;
   using ElementType   = ElementType_t<DilatedSubmatrixType>;

   BLAZE_CONSTRAINT_MUST_BE_DILATEDSUBMATRIX_TYPE( DilatedSubmatrixType );
   BLAZE_CONSTRAINT_MUST_BE_SPARSE_MATRIX_TYPE( DilatedSubmatrixType );

   const size_t m( dilatedsubmatrix.rows()    );
   const size_t n( dilatedsubmatrix.columns() );

   if( m == 0UL || n == 0UL ) return;

   const size_t nonzeros( rand<size_t>( 1UL, std::ceil( 0.5*m*n ) ) );

   dilatedsubmatrix.reset();
   dilatedsubmatrix.reserve( nonzeros );

   while( dilatedsubmatrix.nonZeros() < nonzeros ) {
      dilatedsubmatrix( rand<size_t>( 0UL, m-1UL ), rand<size_t>( 0UL, n-1UL ) ) = rand<ElementType>();
   }
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Randomization of a sparse dilatedsubmatrix.
//
// \param dilatedsubmatrix The dilatedsubmatrix to be randomized.
// \param nonzeros The number of non-zero elements of the random dilatedsubmatrix.
// \return void
// \exception std::invalid_argument Invalid number of non-zero elements.
*/
template< typename MT       // Type of the dense matrix
        , bool SO           // Storage order
        , size_t... CSAs >  // Compile time dilatedsubmatrix arguments
template< typename DSMT >    // Type of the dilatedsubmatrix
inline void Rand< DilatedSubmatrix<MT,SO,false,CSAs...> >::randomize( DSMT&& dilatedsubmatrix, size_t nonzeros ) const
{
   using DilatedSubmatrixType = RemoveReference_t<DSMT>;
   using ElementType   = ElementType_t<DilatedSubmatrixType>;

   BLAZE_CONSTRAINT_MUST_BE_DILATEDSUBMATRIX_TYPE( DilatedSubmatrixType );
   BLAZE_CONSTRAINT_MUST_BE_SPARSE_MATRIX_TYPE( DilatedSubmatrixType );

   const size_t m( dilatedsubmatrix.rows()    );
   const size_t n( dilatedsubmatrix.columns() );

   if( nonzeros > m*n ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid number of non-zero elements" );
   }

   if( m == 0UL || n == 0UL ) return;

   dilatedsubmatrix.reset();
   dilatedsubmatrix.reserve( nonzeros );

   while( dilatedsubmatrix.nonZeros() < nonzeros ) {
      dilatedsubmatrix( rand<size_t>( 0UL, m-1UL ), rand<size_t>( 0UL, n-1UL ) ) = rand<ElementType>();
   }
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Randomization of a sparse dilatedsubmatrix.
//
// \param dilatedsubmatrix The dilatedsubmatrix to be randomized.
// \param min The smallest possible value for a dilatedsubmatrix element.
// \param max The largest possible value for a dilatedsubmatrix element.
// \return void
*/
template< typename MT       // Type of the dense matrix
        , bool SO           // Storage order
        , size_t... CSAs >  // Compile time dilatedsubmatrix arguments
template< typename DSMT      // Type of the dilatedsubmatrix
        , typename Arg >    // Min/max argument type
inline void Rand< DilatedSubmatrix<MT,SO,false,CSAs...> >::randomize( DSMT&& dilatedsubmatrix,
                                                                  const Arg& min, const Arg& max ) const
{
   using DilatedSubmatrixType = RemoveReference_t<DSMT>;
   using ElementType   = ElementType_t<DilatedSubmatrixType>;

   BLAZE_CONSTRAINT_MUST_BE_DILATEDSUBMATRIX_TYPE( DilatedSubmatrixType );
   BLAZE_CONSTRAINT_MUST_BE_SPARSE_MATRIX_TYPE( DilatedSubmatrixType );

   const size_t m( dilatedsubmatrix.rows()    );
   const size_t n( dilatedsubmatrix.columns() );

   if( m == 0UL || n == 0UL ) return;

   const size_t nonzeros( rand<size_t>( 1UL, std::ceil( 0.5*m*n ) ) );

   dilatedsubmatrix.reset();
   dilatedsubmatrix.reserve( nonzeros );

   while( dilatedsubmatrix.nonZeros() < nonzeros ) {
      dilatedsubmatrix( rand<size_t>( 0UL, m-1UL ), rand<size_t>( 0UL, n-1UL ) ) = rand<ElementType>( min, max );
   }
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Randomization of a sparse dilatedsubmatrix.
//
// \param dilatedsubmatrix The dilatedsubmatrix to be randomized.
// \param nonzeros The number of non-zero elements of the random dilatedsubmatrix.
// \param min The smallest possible value for a dilatedsubmatrix element.
// \param max The largest possible value for a dilatedsubmatrix element.
// \return void
// \exception std::invalid_argument Invalid number of non-zero elements.
*/
template< typename MT       // Type of the dense matrix
        , bool SO           // Storage order
        , size_t... CSAs >  // Compile time dilatedsubmatrix arguments
template< typename DSMT      // Type of the dilatedsubmatrix
        , typename Arg >    // Min/max argument type
inline void Rand< DilatedSubmatrix<MT,SO,false,CSAs...> >::randomize( DSMT&& dilatedsubmatrix, size_t nonzeros,
                                                                  const Arg& min, const Arg& max ) const
{
   using DilatedSubmatrixType = RemoveReference_t<DSMT>;
   using ElementType   = ElementType_t<DilatedSubmatrixType>;

   BLAZE_CONSTRAINT_MUST_BE_DILATEDSUBMATRIX_TYPE( DilatedSubmatrixType );
   BLAZE_CONSTRAINT_MUST_BE_SPARSE_MATRIX_TYPE( DilatedSubmatrixType );

   const size_t m( dilatedsubmatrix.rows()    );
   const size_t n( dilatedsubmatrix.columns() );

   if( nonzeros > m*n ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid number of non-zero elements" );
   }

   if( m == 0UL || n == 0UL ) return;

   dilatedsubmatrix.reset();
   dilatedsubmatrix.reserve( nonzeros );

   while( dilatedsubmatrix.nonZeros() < nonzeros ) {
      dilatedsubmatrix( rand<size_t>( 0UL, m-1UL ), rand<size_t>( 0UL, n-1UL ) ) = rand<ElementType>( min, max );
   }
}
/*! \endcond */
//*************************************************************************************************

} // namespace blaze

#endif
