//=================================================================================================
/*!
//  \file blaze_tensor/math/expressions/DTensDTensEqualExpr.h
//  \brief Header file for the dense tensor/dense tensor equality comparison expression
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

#ifndef _BLAZE_TENSOR_MATH_EXPRESSIONS_DTENSDTENSEQUALEXPR_H_
#define _BLAZE_TENSOR_MATH_EXPRESSIONS_DTENSDTENSEQUALEXPR_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <blaze/math/expressions/DMatDMatEqualExpr.h>

#include <blaze_tensor/math/expressions/DenseTensor.h>

namespace blaze {

//=================================================================================================
//
//  CLASS DEFINITION
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Auxiliary helper struct for the dense tensor/dense tensor equality comparison.
// \ingroup dense_tensor
*/
template< typename MT1    // Type of the left-hand side dense tensor
        , typename MT2 >  // Type of the right-hand side dense tensor
struct DTensDTensEqualExprHelper
{
   //**Type definitions****************************************************************************
   //! Composite type of the left-hand side dense tensor expression.
   using CT1 = RemoveReference_t< CompositeType_t<MT1> >;

   //! Composite type of the right-hand side dense tensor expression.
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
/*!\brief Default equality check of two row-major dense matrices.
// \ingroup dense_tensor
//
// \param lhs The left-hand side dense tensor for the comparison.
// \param rhs The right-hand side dense tensor for the comparison.
// \return \a true if the two matrices are equal, \a false if not.
//
// Equal function for the comparison of two dense matrices. Due to the limited machine accuracy,
// a direct comparison of two floating point numbers should be avoided. This function offers the
// possibility to compare two floating-point matrices with a certain accuracy margin.
*/
template< bool RF         // Relaxation flag
        , typename MT1    // Type of the left-hand side dense tensor
        , typename MT2 >  // Type of the right-hand side dense tensor
inline DisableIf_t< DTensDTensEqualExprHelper<MT1,MT2>::value, bool >
   equal( const DenseTensor<MT1>& lhs, const DenseTensor<MT2>& rhs )
{
   using CT1 = CompositeType_t<MT1>;
   using CT2 = CompositeType_t<MT2>;

   // Early exit in case the tensor sizes don't match
   if( (~lhs).rows() != (~rhs).rows() || (~lhs).columns() != (~rhs).columns() || (~lhs).pages() != (~rhs).pages() )
      return false;

   // Evaluation of the two dense tensor operands
   CT1 A( ~lhs );
   CT2 B( ~rhs );

   // In order to compare the two matrices, the data values of the lower-order data
   // type are converted to the higher-order data type within the equal function.
   for (size_t k=0UL; k<A.pages(); ++k) {
      for (size_t i=0UL; i<A.rows(); ++i) {
         for (size_t j=0UL; j<A.columns(); ++j) {
            if (!equal(A(k, i, j), B(k, i, j)))
               return false;
         }
      }
   }
   return true;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief SIMD optimized equality check of two row-major dense matrices.
// \ingroup dense_tensor
//
// \param lhs The left-hand side dense tensor for the comparison.
// \param rhs The right-hand side dense tensor for the comparison.
// \return \a true if the two matrices are equal, \a false if not.
//
// Equal function for the comparison of two dense matrices. Due to the limited machine accuracy,
// a direct comparison of two floating point numbers should be avoided. This function offers the
// possibility to compare two floating-point matrices with a certain accuracy margin.
*/
template< bool RF         // Relaxation flag
        , typename MT1    // Type of the left-hand side dense tensor
        , typename MT2 >  // Type of the right-hand side dense tensor
inline EnableIf_t< DTensDTensEqualExprHelper<MT1,MT2>::value, bool >
   equal( const DenseTensor<MT1>& lhs, const DenseTensor<MT2>& rhs )
{
   using CT1 = CompositeType_t<MT1>;
   using CT2 = CompositeType_t<MT2>;
   using XT1 = RemoveReference_t<CT1>;
   using XT2 = RemoveReference_t<CT2>;

   // Early exit in case the tensor sizes don't match
   if( (~lhs).rows() != (~rhs).rows() || (~lhs).columns() != (~rhs).columns() || (~lhs).pages() != (~rhs).pages() )
      return false;

   // Evaluation of the two dense tensor operands
   CT1 A( ~lhs );
   CT2 B( ~rhs );

   constexpr size_t SIMDSIZE = SIMDTrait< ElementType_t<MT1> >::size;
   constexpr bool remainder( !usePadding || !IsPadded_v<XT1> || !IsPadded_v<XT2> );

   const size_t M( A.rows()    );
   const size_t N( A.columns() );
   const size_t O( A.pages()   );

   const size_t jpos( ( remainder )?( N & size_t(-SIMDSIZE) ):( N ) );
   BLAZE_INTERNAL_ASSERT( !remainder || ( N - ( N % SIMDSIZE ) ) == jpos, "Invalid end calculation" );

   for (size_t k=0UL; k<O; ++k)
   {
      for (size_t i=0UL; i<M; ++i)
      {
         size_t j(0UL);

         for (; (j+SIMDSIZE*3UL) < jpos; j+=SIMDSIZE*4UL) {
            if (!equal(A.load(k, i, j), B.load(k, i, j))) return false;
            if (!equal(A.load(i, j+SIMDSIZE, k), B.load(i, j+SIMDSIZE, k))) return false;
            if (!equal(A.load(i, j+SIMDSIZE*2UL, k), B.load(i, j+SIMDSIZE*2UL, k))) return false;
            if (!equal(A.load(i, j+SIMDSIZE*3UL, k), B.load(i, j+SIMDSIZE*3UL, k))) return false;
         }
         for (; (j+SIMDSIZE) < jpos; j+=SIMDSIZE*2UL) {
            if (!equal(A.load(k, i, j), B.load(k, i, j))) return false;
            if (!equal(A.load(i, j+SIMDSIZE, k), B.load(i, j+SIMDSIZE, k))) return false;
         }
         for (; j<jpos; j+=SIMDSIZE) {
            if (!equal(A.load(k, i, j), B.load(k, i, j))) return false;
         }
         for (; remainder && j<A.columns(); ++j) {
            if (!equal(A(k, i, j), B(k, i, j))) return false;
         }
      }
   }
   return true;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Equality operator for the comparison of two dense matrices.
// \ingroup dense_tensor
//
// \param lhs The left-hand side tensor for the comparison.
// \param rhs The right-hand side tensor for the comparison.
// \return \a true if the two matrices are equal, \a false if not.
*/
template< typename MT1  // Type of the left-hand side dense tensor
        , typename MT2 > // Type of the right-hand side dense tensor
inline bool operator==( const DenseTensor<MT1>& lhs, const DenseTensor<MT2>& rhs )
{
   return equal<relaxed>( lhs, rhs );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Inequality operator for the comparison of two dense matrices.
// \ingroup dense_tensor
//
// \param lhs The left-hand side dense tensor for the comparison.
// \param rhs The right-hand side dense tensor for the comparison.
// \return \a true if the two matrices are not equal, \a false if they are equal.
*/
template< typename MT1  // Type of the left-hand side dense tensor
        , typename MT2 > // Type of the right-hand side dense tensor
inline bool operator!=( const DenseTensor<MT1>& lhs, const DenseTensor<MT2>& rhs )
{
   return !equal<relaxed>( lhs, rhs );
}
//*************************************************************************************************

} // namespace blaze

#endif
