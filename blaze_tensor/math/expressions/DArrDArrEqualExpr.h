//=================================================================================================
/*!
//  \file blaze_array/math/expressions/DArrDArrEqualExpr.h
//  \brief Header file for the dense array/dense array equality comparison expression
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

#ifndef _BLAZE_TENSOR_MATH_EXPRESSIONS_DARRDARREQUALEXPR_H_
#define _BLAZE_TENSOR_MATH_EXPRESSIONS_DARRDARREQUALEXPR_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <blaze/math/expressions/DMatDMatEqualExpr.h>

#include <blaze_tensor/math/expressions/DenseArray.h>
#include <blaze_tensor/util/ArrayForEach.h>

namespace blaze {

//=================================================================================================
//
//  CLASS DEFINITION
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Auxiliary helper struct for the dense array/dense array equality comparison.
// \ingroup dense_array
*/
template< typename MT1    // Type of the left-hand side dense array
        , typename MT2 >  // Type of the right-hand side dense array
struct DArrDArrEqualExprHelper
{
   //**Type definitions****************************************************************************
   //! Composite type of the left-hand side dense array expression.
   using CT1 = RemoveReference_t< CompositeType_t<MT1> >;

   //! Composite type of the right-hand side dense array expression.
   using CT2 = RemoveReference_t< CompositeType_t<MT2> >;
   //**********************************************************************************************

   //**********************************************************************************************
   static constexpr bool value =
      ( useOptimizedKernels &&
        CT1::simdEnabled &&
        CT2::simdEnabled &&
        HasSIMDEqual_v< ElementType_t<CT1>, ElementType_t<CT2> > );
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  GLOBAL BINARY RELATIONAL OPERATORS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Default equality check of two row-major dense arrays.
// \ingroup dense_array
//
// \param lhs The left-hand side dense array for the comparison.
// \param rhs The right-hand side dense array for the comparison.
// \return \a true if the two arrays are equal, \a false if not.
//
// Equal function for the comparison of two dense arrays. Due to the limited machine accuracy,
// a direct comparison of two floating point numbers should be avoided. This function offers the
// possibility to compare two floating-point arrays with a certain accuracy margin.
*/
template< bool RF         // Relaxation flag
        , typename MT1    // Type of the left-hand side dense array
        , typename MT2 >  // Type of the right-hand side dense array
inline bool //DisableIf_t< DArrDArrEqualExprHelper<MT1,MT2>::value, bool >
   equal( const DenseArray<MT1>& lhs, const DenseArray<MT2>& rhs )
{
   using CT1 = CompositeType_t<MT1>;
   using CT2 = CompositeType_t<MT2>;

   // Early exit in case the array sizes don't match
   if( ( ~lhs ).dimensions() != ( ~rhs ).dimensions() ) {
      return false;
   }

   constexpr size_t N = MT1::num_dimensions();

   // Evaluation of the two dense array operands
   CT1 A( ~lhs );
   CT2 B( ~rhs );

   // In order to compare the two arrays, the data values of the lower-order data
   // type are converted to the higher-order data type within the equal function.
   return ArrayForEachGroupedAllOf(
      ( ~lhs ).dimensions(), [&]( std::array< size_t, N > const& dims ) {
         return equal( A( dims ), B( dims ) );
      } );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief SIMD optimized equality check of two row-major dense arrays.
// \ingroup dense_array
//
// \param lhs The left-hand side dense array for the comparison.
// \param rhs The right-hand side dense array for the comparison.
// \return \a true if the two arrays are equal, \a false if not.
//
// Equal function for the comparison of two dense arrays. Due to the limited machine accuracy,
// a direct comparison of two floating point numbers should be avoided. This function offers the
// possibility to compare two floating-point arrays with a certain accuracy margin.
*/
// template< bool RF         // Relaxation flag
//         , typename MT1    // Type of the left-hand side dense array
//         , typename MT2 >  // Type of the right-hand side dense array
// inline EnableIf_t< DArrDArrEqualExprHelper<MT1,MT2>::value, bool >
//    equal( const DenseArray<MT1>& lhs, const DenseArray<MT2>& rhs )
// {
//    using CT1 = CompositeType_t<MT1>;
//    using CT2 = CompositeType_t<MT2>;
//    using XT1 = RemoveReference_t<CT1>;
//    using XT2 = RemoveReference_t<CT2>;
//
//    // Early exit in case the array sizes don't match
//    auto const& rhsdims = ( ~rhs ).dimensions();
//    if( ArrayDimAnyOf( ( ~lhs ).dimensions(),
//           [&]( size_t i, size_t dim ) { return dim != rhsdims[i]; } ) ) {
//       return false;
//    }
//
//    // Evaluation of the two dense array operands
//    CT1 A( ~lhs );
//    CT2 B( ~rhs );
//
//    constexpr size_t SIMDSIZE = SIMDTrait< ElementType_t<MT1> >::size;
//    constexpr bool remainder( !usePadding || !IsPadded_v<XT1> || !IsPadded_v<XT2> );
//
//    const size_t M( A.rows()    );
//    const size_t N( A.columns() );
//    const size_t O( A.pages()   );
//
//    const size_t jpos( ( remainder )?( N & size_t(-SIMDSIZE) ):( N ) );
//    BLAZE_INTERNAL_ASSERT( !remainder || ( N - ( N % SIMDSIZE ) ) == jpos, "Invalid end calculation" );
//
//    for (size_t k=0UL; k<O; ++k)
//    {
//       for (size_t i=0UL; i<M; ++i)
//       {
//          size_t j(0UL);
//
//          for (; (j+SIMDSIZE*3UL) < jpos; j+=SIMDSIZE*4UL) {
//             if (!equal(A.load(k, i, j), B.load(k, i, j))) return false;
//             if (!equal(A.load(k, i, j+SIMDSIZE), B.load(k, i, j+SIMDSIZE))) return false;
//             if (!equal(A.load(k, i, j+SIMDSIZE*2UL), B.load(k, i, j+SIMDSIZE*2UL))) return false;
//             if (!equal(A.load(k, i, j+SIMDSIZE*3UL), B.load(k, i, j+SIMDSIZE*3UL))) return false;
//          }
//          for (; (j+SIMDSIZE) < jpos; j+=SIMDSIZE*2UL) {
//             if (!equal(A.load(k, i, j), B.load(k, i, j))) return false;
//             if (!equal(A.load(k, i, j+SIMDSIZE), B.load(k, i, j+SIMDSIZE))) return false;
//          }
//          for (; j<jpos; j+=SIMDSIZE) {
//             if (!equal(A.load(k, i, j), B.load(k, i, j))) return false;
//          }
//          for (; remainder && j<A.columns(); ++j) {
//             if (!equal(A(k, i, j), B(k, i, j))) return false;
//          }
//       }
//    }
//    return true;
// }
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Equality operator for the comparison of two dense arrays.
// \ingroup dense_array
//
// \param lhs The left-hand side array for the comparison.
// \param rhs The right-hand side array for the comparison.
// \return \a true if the two arrays are equal, \a false if not.
*/
template< typename MT1  // Type of the left-hand side dense array
        , typename MT2 > // Type of the right-hand side dense array
inline bool operator==( const DenseArray<MT1>& lhs, const DenseArray<MT2>& rhs )
{
   return equal<relaxed>( lhs, rhs );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Inequality operator for the comparison of two dense arrays.
// \ingroup dense_array
//
// \param lhs The left-hand side dense array for the comparison.
// \param rhs The right-hand side dense array for the comparison.
// \return \a true if the two arrays are not equal, \a false if they are equal.
*/
template< typename MT1  // Type of the left-hand side dense array
        , typename MT2 > // Type of the right-hand side dense array
inline bool operator!=( const DenseArray<MT1>& lhs, const DenseArray<MT2>& rhs )
{
   return !equal<relaxed>( lhs, rhs );
}
//*************************************************************************************************

} // namespace blaze

#endif
