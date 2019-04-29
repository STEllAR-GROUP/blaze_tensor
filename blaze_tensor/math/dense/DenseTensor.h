//=================================================================================================
/*!
//  \file blaze_tensor/math/dense/DenseTensor.h
//  \brief Header file for utility functions for dense matrices
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

#ifndef _BLAZE_TENSOR_MATH_DENSE_DENSETENSOR_H_
#define _BLAZE_TENSOR_MATH_DENSE_DENSETENSOR_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <blaze/math/Aliases.h>
#include <blaze/math/constraints/RequiresEvaluation.h>
#include <blaze/math/shims/Equal.h>
#include <blaze/math/shims/IsDefault.h>
#include <blaze/math/shims/IsNaN.h>
#include <blaze/math/shims/IsOne.h>
#include <blaze/math/shims/IsReal.h>
#include <blaze/math/shims/IsZero.h>
#include <blaze/math/typetraits/IsExpression.h>
#include <blaze/math/typetraits/IsRestricted.h>
#include <blaze/util/Assert.h>
#include <blaze/util/DecltypeAuto.h>
#include <blaze/util/EnableIf.h>
#include <blaze/util/FalseType.h>
#include <blaze/util/mpl/If.h>
#include <blaze/util/TrueType.h>
#include <blaze/util/Types.h>
#include <blaze/util/typetraits/IsBuiltin.h>
#include <blaze/util/typetraits/IsNumeric.h>

#include <blaze_tensor/math/expressions/DTensDTensAddExpr.h>
#include <blaze_tensor/math/expressions/DTensDTensEqualExpr.h>
#include <blaze_tensor/math/expressions/DTensDTensMapExpr.h>
#include <blaze_tensor/math/expressions/DTensDTensMultExpr.h>
#include <blaze_tensor/math/expressions/DTensDTensSchurExpr.h>
#include <blaze_tensor/math/expressions/DTensDTensSubExpr.h>
#include <blaze_tensor/math/expressions/DTensEvalExpr.h>
#include <blaze_tensor/math/expressions/DTensMapExpr.h>
#include <blaze_tensor/math/expressions/DTensNormExpr.h>
#include <blaze_tensor/math/expressions/DTensReduceExpr.h>
#include <blaze_tensor/math/expressions/DTensScalarDivExpr.h>
#include <blaze_tensor/math/expressions/DTensScalarMultExpr.h>
#include <blaze_tensor/math/expressions/DTensSerialExpr.h>
#include <blaze_tensor/math/expressions/DenseTensor.h>

namespace blaze {

//=================================================================================================
//
//  GLOBAL OPERATORS
//
//=================================================================================================

//*************************************************************************************************
/*!\name DenseTensor operators */
//@{
template< typename T1, typename T2 >
inline auto operator==( const DenseTensor<T1>& tens, T2 scalar )
   -> EnableIf_t< IsNumeric_v<T2>, bool >;

template< typename T1, typename T2 >
inline auto operator==( T1 scalar, const DenseTensor<T2>& tens )
   -> EnableIf_t< IsNumeric_v<T2>, bool >;

template< typename T1, typename T2 >
inline auto operator!=( const DenseTensor<T1>& tens, T2 scalar )
   -> EnableIf_t< IsNumeric_v<T2>, bool >;

template< typename T1, typename T2 >
inline auto operator!=( T1 scalar, const DenseTensor<T2>& tens )
   -> EnableIf_t< IsNumeric_v<T2>, bool >;

template< typename TT, typename ST >
inline auto operator*=( DenseTensor<TT>& tens, ST scalar )
   -> EnableIf_t< IsNumeric_v<ST>, TT& >;

template< typename TT, typename ST >
inline auto operator/=( DenseTensor<TT>& tens, ST scalar )
   -> EnableIf_t< IsNumeric_v<ST>, TT& >;
//@}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Equality operator for the comparison of a row-major dense tensor and a scalar value.
// \ingroup dense_tensor
//
// \param tens The left-hand side row-major dense tensor for the comparison.
// \param scalar The right-hand side scalar value for the comparison.
// \return \a true if all elements of the tensor are equal to the scalar, \a false if not.
//
// If all values of the tensor are equal to the scalar value, the equality test returns \a true,
// otherwise \a false. Note that this function can only be used with built-in, numerical data
// types!
*/
template< typename T1    // Type of the left-hand side dense tensor
        , typename T2 >  // Type of the right-hand side scalar
inline auto operator==( const DenseTensor<T1>& tens, T2 scalar )
   -> EnableIf_t< IsNumeric_v<T2>, bool >
{
   using CT1 = CompositeType_t<T1>;

   // Evaluation of the dense tensor operand
   CT1 A( ~tens );

   // In order to compare the tensor and the scalar value, the data values of the lower-order
   // data type are converted to the higher-order data type within the equal function.
   for (size_t k=0; k<A.pages(); ++k) {
      for (size_t i=0; i<A.rows(); ++i) {
         for (size_t j=0; j<A.columns(); ++j) {
            if (!equal(A(k, i, j), scalar)) return false;
         }
      }
   }

   return true;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Equality operator for the comparison of a scalar value and a dense tensor.
// \ingroup dense_tensor
//
// \param scalar The left-hand side scalar value for the comparison.
// \param tens The right-hand side dense tensor for the comparison.
// \return \a true if all elements of the tensor are equal to the scalar, \a false if not.
//
// If all values of the tensor are equal to the scalar value, the equality test returns \a true,
// otherwise \a false. Note that this function can only be used with built-in, numerical data
// types!
*/
template< typename T1  // Type of the left-hand side scalar
        , typename T2 >  // Type of the right-hand side dense tensor
inline auto operator==( T1 scalar, const DenseTensor<T2>& tens )
   -> EnableIf_t< IsNumeric_v<T1>, bool >
{
   return ( tens == scalar );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Inequality operator for the comparison of a dense tensor and a scalar value.
// \ingroup dense_tensor
//
// \param tens The left-hand side dense tensor for the comparison.
// \param scalar The right-hand side scalar value for the comparison.
// \return \a true if at least one element of the tensor is different from the scalar, \a false if not.
//
// If one value of the tensor is inequal to the scalar value, the inequality test returns \a true,
// otherwise \a false. Note that this function can only be used with built-in, numerical data
// types!
*/
template< typename T1  // Type of the left-hand side scalar
        , typename T2 >  // Type of the right-hand side dense tensor
inline auto operator!=( const DenseTensor<T1>& tens, T2 scalar )
   -> EnableIf_t< IsNumeric_v<T2>, bool >
{
   return !( tens == scalar );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Inequality operator for the comparison of a scalar value and a dense tensor.
// \ingroup dense_tensor
//
// \param scalar The left-hand side scalar value for the comparison.
// \param tens The right-hand side dense tensor for the comparison.
// \return \a true if at least one element of the tensor is different from the scalar, \a false if not.
//
// If one value of the tensor is inequal to the scalar value, the inequality test returns \a true,
// otherwise \a false. Note that this function can only be used with built-in, numerical data
// types!
*/
template< typename T1  // Type of the left-hand side scalar
        , typename T2 >  // Type of the right-hand side dense tensor
inline auto operator!=( T1 scalar, const DenseTensor<T2>& tens )
   -> EnableIf_t< IsNumeric_v<T1>, bool >
{
   return !( tens == scalar );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Multiplication assignment operator for the multiplication of a dense tensor and
//        a scalar value (\f$ A*=s \f$).
// \ingroup dense_tensor
//
// \param tens The left-hand side dense tensor for the multiplication.
// \param scalar The right-hand side scalar value for the multiplication.
// \return Reference to the left-hand side dense tensor.
// \exception std::invalid_argument Invalid scaling of restricted tensor.
//
// In case the tensor \a TT is restricted and the assignment would violate an invariant of the
// tensor, a \a std::invalid_argument exception is thrown.
*/
template< typename TT    // Type of the left-hand side dense tensor
        , typename ST >  // Data type of the right-hand side scalar
inline auto operator*=( DenseTensor<TT>& tens, ST scalar )
   -> EnableIf_t< IsNumeric_v<ST>, TT& >
{
   if( IsRestricted_v<TT> ) {
      if( !tryMult( ~tens, 0UL, 0UL, 0UL, (~tens).pages(), (~tens).rows(), (~tens).columns(), scalar ) ) {
         BLAZE_THROW_INVALID_ARGUMENT( "Invalid scaling of restricted tensor" );
      }
   }

   BLAZE_DECLTYPE_AUTO( left, derestrict( ~tens ) );

   smpAssign( left, left * scalar );

   BLAZE_INTERNAL_ASSERT( isIntact( ~tens ), "Invariant violation detected" );

   return ~tens;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Multiplication assignment operator for the multiplication of a temporary dense tensor
//        and a scalar value (\f$ A*=s \f$).
// \ingroup dense_tensor
//
// \param tens The left-hand side temporary dense tensor for the multiplication.
// \param scalar The right-hand side scalar value for the multiplication.
// \return Reference to the left-hand side dense tensor.
// \exception std::invalid_argument Invalid scaling of restricted tensor.
//
// In case the tensor \a TT is restricted and the assignment would violate an invariant of the
// tensor, a \a std::invalid_argument exception is thrown.
*/
template< typename TT    // Type of the left-hand side dense tensor
        , typename ST >  // Data type of the right-hand side scalar
inline auto operator*=( DenseTensor<TT>&& tens, ST scalar )
   -> EnableIf_t< IsNumeric_v<ST>, TT& >
{
   return operator*=( ~tens, scalar );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Division assignment operator for the division of a dense tensor by a scalar value
//        (\f$ A/=s \f$).
// \ingroup dense_tensor
//
// \param tens The left-hand side dense tensor for the division.
// \param scalar The right-hand side scalar value for the division.
// \return Reference to the left-hand side dense tensor.
// \exception std::invalid_argument Invalid scaling of restricted tensor.
//
// In case the tensor \a TT is restricted and the assignment would violate an invariant of the
// tensor, a \a std::invalid_argument exception is thrown.
//
// \note A division by zero is only checked by an user assert.
*/
template< typename TT    // Type of the left-hand side dense tensor
        , typename ST >  // Data type of the right-hand side scalar
inline auto operator/=( DenseTensor<TT>& tens, ST scalar )
   -> EnableIf_t< IsNumeric_v<ST>, TT& >
{
//    BLAZE_CONSTRAINT_MUST_NOT_BE_UNITRIANGULAR_TENSOR_TYPE( TT );

   BLAZE_USER_ASSERT( !isZero( scalar ), "Division by zero detected" );

   if( IsRestricted_v<TT> ) {
      if( !tryDiv( ~tens, 0UL, 0UL, 0UL, (~tens).pages(), (~tens).rows(), (~tens).columns(), scalar ) ) {
         BLAZE_THROW_INVALID_ARGUMENT( "Invalid scaling of restricted tensor" );
      }
   }

   BLAZE_DECLTYPE_AUTO( left, derestrict( ~tens ) );

   smpAssign( left, left / scalar );

   BLAZE_INTERNAL_ASSERT( isIntact( ~tens ), "Invariant violation detected" );

   return ~tens;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Division assignment operator for the division of a temporary dense tensor by a scalar
//        value (\f$ A/=s \f$).
// \ingroup dense_tensor
//
// \param tens The left-hand side temporary dense tensor for the division.
// \param scalar The right-hand side scalar value for the division.
// \return Reference to the left-hand side dense tensor.
// \exception std::invalid_argument Invalid scaling of restricted tensor.
//
// In case the tensor \a TT is restricted and the assignment would violate an invariant of the
// tensor, a \a std::invalid_argument exception is thrown.
//
// \note A division by zero is only checked by an user assert.
*/
template< typename TT    // Type of the left-hand side dense tensor
        , typename ST >  // Data type of the right-hand side scalar
inline auto operator/=( DenseTensor<TT>&& tens, ST scalar )
   -> EnableIf_t< IsNumeric_v<ST>, TT& >
{
   return operator/=( ~tens, scalar );
}
//*************************************************************************************************




//=================================================================================================
//
//  GLOBAL FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*!\name DenseTensor functions */
//@{
template< typename TT >
bool isnan( const DenseTensor<TT>& dm );

// template< bool RF, typename MT >
// bool isSymmetric( const DenseTensor<MT>& dm );
//
// template< bool RF, typename MT >
// bool isHermitian( const DenseTensor<MT>& dm );

template< bool RF, typename MT >
bool isUniform( const DenseTensor<MT>& dm );

// template< bool RF, typename MT >
// bool isLower( const DenseTensor<MT>& dm );
//
// template< bool RF, typename MT >
// bool isUniLower( const DenseTensor<MT>& dm );
//
// template< bool RF, typename MT >
// bool isStrictlyLower( const DenseTensor<MT>& dm );
//
// template< bool RF, typename MT >
// bool isUpper( const DenseTensor<MT>& dm );
//
// template< bool RF, typename MT >
// bool isUniUpper( const DenseTensor<MT>& dm );
//
// template< bool RF, typename MT >
// bool isStrictlyUpper( const DenseTensor<MT>& dm );
//
// template< bool RF, typename MT >
// bool isDiagonal( const DenseTensor<MT>& dm );
//
// template< bool RF, typename MT >
// bool isIdentity( const DenseTensor<MT>& dm );

template< typename MT >
auto softmax( const DenseTensor<MT>& dm );
//@}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Checks the given dense tensor for not-a-number elements.
// \ingroup dense_tensor
//
// \param dm The tensor to be checked for not-a-number elements.
// \return \a true if at least one element of the tensor is not-a-number, \a false otherwise.
//
// This function checks the dense tensor for not-a-number (NaN) elements. If at least one
// element of the tensor is not-a-number, the function returns \a true, otherwise it returns
// \a false.

   \code
   blaze::DynamicTensor<double> A( 5UL, 3UL, 4UL );
   // ... Initialization
   if( isnan( A ) ) { ... }
   \endcode

// Note that this function only works for matrices with floating point elements. The attempt to
// use it for a tensor with a non-floating point element type results in a compile time error.
*/
template< typename TT > // Type of the dense tensor
bool isnan( const DenseTensor<TT>& dm )
{
   using CT = CompositeType_t<TT>;

   CT A( ~dm );  // Evaluation of the dense tensor operand

   for (size_t k=0UL; k<A.pages(); ++k) {
      for (size_t i=0UL; i<A.rows(); ++i) {
         for (size_t j=0UL; j<A.columns(); ++j)
            if (isnan(A(k, i, j))) return true;
      }
   }

   return false;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Computes the softmax function for the given dense tensor.
// \ingroup dense_tensor
//
// \param dm The given dense tensor for the softmax computation.
// \return The resulting tensor.
//
// This function computes the softmax function (i.e. the normalized exponential function) for
// the given dense tensor \a dm (see also https://en.wikipedia.org/wiki/Softmax_function). The
// resulting dense tensor consists of real values in the range (0..1], which add up to 1.
*/
template< typename MT > // Type of the dense tensor
auto softmax( const DenseTensor<MT>& dm )
{
   auto tmp( evaluate( exp( ~dm ) ) );
   const auto scalar( sum( ~tmp ) );
   tmp /= scalar;
   return tmp;
}
//*************************************************************************************************

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Checks if the given row-major general dense tensor is a uniform tensor.
// \ingroup dense_tensor
//
// \param dm The dense tensor to be checked.
// \return \a true if the tensor is a uniform tensor, \a false if not.
*/
template< bool RF        // Relaxation flag
        , typename MT >  // Type of the dense tensor
bool isUniform_backend( const DenseTensor<MT>& dm )
{
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( MT );

   BLAZE_INTERNAL_ASSERT( (~dm).pages()   != 0UL, "Invalid number of pages detected"    );
   BLAZE_INTERNAL_ASSERT( (~dm).rows()    != 0UL, "Invalid number of rows detected"    );
   BLAZE_INTERNAL_ASSERT( (~dm).columns() != 0UL, "Invalid number of columns detected" );

   const auto& cmp( (~dm)(0UL,0UL,0UL) );

   for( size_t k=0UL; k<(~dm).pages(); ++k ) {
      for( size_t i=0UL; i<(~dm).rows(); ++i ) {
         for( size_t j=0UL; j<(~dm).columns(); ++j ) {
            if( !equal<RF>( (~dm)(k,i,j), cmp ) )
               return false;
         }
      }
   }

   return true;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Checks if the given dense tensor is a uniform tensor.
// \ingroup dense_tensor
//
// \param dm The dense tensor to be checked.
// \return \a true if the tensor is a uniform tensor, \a false if not.
//
// This function checks if the given dense tensor is a uniform tensor. The tensor is considered
// to be uniform if all its elements are identical. The following code example demonstrates the
// use of the function:

   \code
   blaze::DynamicTensor<int,blaze::rowMajor> A, B;
   // ... Initialization
   if( isUniform( A ) ) { ... }
   \endcode

// Optionally, it is possible to switch between strict semantics (blaze::strict) and relaxed
// semantics (blaze::relaxed):

   \code
   if( isUniform<relaxed>( A ) ) { ... }
   \endcode

// It is also possible to check if a tensor expression results is a uniform tensor:

   \code
   if( isUniform( A * B ) ) { ... }
   \endcode

// However, note that this might require the complete evaluation of the expression, including
// the generation of a temporary tensor.
*/
template< bool RF      // Relaxation flag
        , typename MT > // Type of the dense tensor
bool isUniform( const DenseTensor<MT>& dm )
{
   if( IsUniform_v<MT> ||
       (~dm).pages() == 0UL || (~dm).rows() == 0UL || (~dm).columns() == 0UL ||
       ( (~dm).pages() == 1UL && (~dm).rows() == 1UL && (~dm).columns() == 1UL ) )
      return true;

   if( IsUniTriangular_v<MT> )
      return false;

   CompositeType_t<MT> A( ~dm );  // Evaluation of the dense tensor operand

   return isUniform_backend<RF>( A );
}
//*************************************************************************************************


} // namespace blaze

#endif
