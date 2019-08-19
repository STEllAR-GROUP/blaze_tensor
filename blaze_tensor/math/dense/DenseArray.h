//=================================================================================================
/*!
//  \file blaze_tensor/math/dense/DenseArray.h
//  \brief Header file for utility functions for dense arrays
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

#ifndef _BLAZE_TENSOR_MATH_DENSE_DENSEARRAY_H_
#define _BLAZE_TENSOR_MATH_DENSE_DENSEARRAY_H_


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
#include <blaze/util/EnableIf.h>
#include <blaze/util/IntegralConstant.h>
#include <blaze/util/mpl/If.h>
#include <blaze/util/Types.h>
#include <blaze/util/typetraits/IsBuiltin.h>
#include <blaze/util/typetraits/IsNumeric.h>
#include <blaze/util/typetraits/RemoveCV.h>
#include <blaze/util/typetraits/RemoveReference.h>

// #include <blaze_tensor/math/expressions/DTensDTensAddExpr.h>
#include <blaze_tensor/math/expressions/DArrDArrEqualExpr.h>
#include <blaze_tensor/math/expressions/DArrDArrMapExpr.h>
// #include <blaze_tensor/math/expressions/DTensDTensMultExpr.h>
// #include <blaze_tensor/math/expressions/DTensDTensSchurExpr.h>
// #include <blaze_tensor/math/expressions/DTensDTensSubExpr.h>
// #include <blaze_tensor/math/expressions/DTensEvalExpr.h>
#include <blaze_tensor/math/expressions/DArrMapExpr.h>
#include <blaze_tensor/math/expressions/DTensNormExpr.h>
#include <blaze_tensor/math/expressions/DArrReduceExpr.h>
#include <blaze_tensor/math/expressions/DArrScalarDivExpr.h>
#include <blaze_tensor/math/expressions/DArrScalarMultExpr.h>
// #include <blaze_tensor/math/expressions/DTensSerialExpr.h>
#include <blaze_tensor/math/expressions/DenseArray.h>
#include <blaze_tensor/util/ArrayForEach.h>

namespace blaze {

//=================================================================================================
//
//  GLOBAL OPERATORS
//
//=================================================================================================

//*************************************************************************************************
/*!\name DenseArray operators */
//@{
template< typename T1, typename T2 >
inline auto operator==( const DenseArray<T1>& arr, T2 scalar )
   -> EnableIf_t< IsNumeric_v<T2>, bool >;

template< typename T1, typename T2 >
inline auto operator==( T1 scalar, const DenseArray<T2>& arr )
   -> EnableIf_t< IsNumeric_v<T2>, bool >;

template< typename T1, typename T2 >
inline auto operator!=( const DenseArray<T1>& arr, T2 scalar )
   -> EnableIf_t< IsNumeric_v<T2>, bool >;

template< typename T1, typename T2 >
inline auto operator!=( T1 scalar, const DenseArray<T2>& arr )
   -> EnableIf_t< IsNumeric_v<T2>, bool >;

template< typename TT, typename ST >
inline auto operator*=( DenseArray<TT>& arr, ST scalar )
   -> EnableIf_t< IsNumeric_v<ST>, TT& >;

template< typename TT, typename ST >
inline auto operator/=( DenseArray<TT>& arr, ST scalar )
   -> EnableIf_t< IsNumeric_v<ST>, TT& >;
//@}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Equality operator for the comparison of a row-major dense array and a scalar value.
// \ingroup dense_tensor
//
// \param arr The left-hand side row-major dense array for the comparison.
// \param scalar The right-hand side scalar value for the comparison.
// \return \a true if all elements of the array are equal to the scalar, \a false if not.
//
// If all values of the array are equal to the scalar value, the equality test returns \a true,
// otherwise \a false. Note that this function can only be used with built-in, numerical data
// types!
*/
template< typename T1    // Type of the left-hand side dense array
        , typename T2 >  // Type of the right-hand side scalar
inline auto operator==( const DenseArray<T1>& arr, T2 scalar )
   -> EnableIf_t< IsNumeric_v<T2>, bool >
{
   using CT1 = CompositeType_t<T1>;

   constexpr size_t N =
      RemoveCV_t< RemoveReference_t< decltype( ~arr ) > >::num_dimensions;

   // Evaluation of the dense array operand
   CT1 A( ~arr );

   // In order to compare the array and the scalar value, the data values of the lower-order
   // data type are converted to the higher-order data type within the equal function.
   return ArrayForEachGroupedAllOf(
      ( ~arr ).dimensions(), [&]( std::array< size_t, N > const& dims ) {
         return equal( A( dims ), scalar );
      } );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Equality operator for the comparison of a scalar value and a dense array.
// \ingroup dense_tensor
//
// \param scalar The left-hand side scalar value for the comparison.
// \param arr The right-hand side dense array for the comparison.
// \return \a true if all elements of the array are equal to the scalar, \a false if not.
//
// If all values of the array are equal to the scalar value, the equality test returns \a true,
// otherwise \a false. Note that this function can only be used with built-in, numerical data
// types!
*/
template< typename T1  // Type of the left-hand side scalar
        , typename T2 >  // Type of the right-hand side dense array
inline auto operator==( T1 scalar, const DenseArray<T2>& arr )
   -> EnableIf_t< IsNumeric_v<T1>, bool >
{
   return ( arr == scalar );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Inequality operator for the comparison of a dense array and a scalar value.
// \ingroup dense_tensor
//
// \param arr The left-hand side dense array for the comparison.
// \param scalar The right-hand side scalar value for the comparison.
// \return \a true if at least one element of the array is different from the scalar, \a false if not.
//
// If one value of the array is inequal to the scalar value, the inequality test returns \a true,
// otherwise \a false. Note that this function can only be used with built-in, numerical data
// types!
*/
template< typename T1  // Type of the left-hand side scalar
        , typename T2 >  // Type of the right-hand side dense array
inline auto operator!=( const DenseArray<T1>& arr, T2 scalar )
   -> EnableIf_t< IsNumeric_v<T2>, bool >
{
   return !( arr == scalar );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Inequality operator for the comparison of a scalar value and a dense array.
// \ingroup dense_tensor
//
// \param scalar The left-hand side scalar value for the comparison.
// \param arr The right-hand side dense array for the comparison.
// \return \a true if at least one element of the array is different from the scalar, \a false if not.
//
// If one value of the array is inequal to the scalar value, the inequality test returns \a true,
// otherwise \a false. Note that this function can only be used with built-in, numerical data
// types!
*/
template< typename T1  // Type of the left-hand side scalar
        , typename T2 >  // Type of the right-hand side dense array
inline auto operator!=( T1 scalar, const DenseArray<T2>& arr )
   -> EnableIf_t< IsNumeric_v<T1>, bool >
{
   return !( arr == scalar );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Multiplication assignment operator for the multiplication of a dense array and
//        a scalar value (\f$ A*=s \f$).
// \ingroup dense_tensor
//
// \param arr The left-hand side dense array for the multiplication.
// \param scalar The right-hand side scalar value for the multiplication.
// \return Reference to the left-hand side dense array.
// \exception std::invalid_argument Invalid scaling of restricted array.
//
// In case the array \a TT is restricted and the assignment would violate an invariant of the
// array, a \a std::invalid_argument exception is thrown.
*/
template< typename TT    // Type of the left-hand side dense array
        , typename ST >  // Data type of the right-hand side scalar
inline auto operator*=( DenseArray<TT>& arr, ST scalar )
   -> EnableIf_t< IsNumeric_v<ST>, TT& >
{
   if( IsRestricted_v<TT> ) {
      constexpr size_t N =
         RemoveCV_t< RemoveReference_t< decltype( ~arr ) > >::num_dimensions;

      std::array< size_t, N > dims{};
      if( !tryMult( ~arr, dims, (~arr).dimensions(), scalar ) ) {
         BLAZE_THROW_INVALID_ARGUMENT( "Invalid scaling of restricted array" );
      }
   }

   decltype(auto) left( derestrict( ~arr ) );

   smpAssign( left, left * scalar );

   BLAZE_INTERNAL_ASSERT( isIntact( ~arr ), "Invariant violation detected" );

   return ~arr;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Multiplication assignment operator for the multiplication of a temporary dense array
//        and a scalar value (\f$ A*=s \f$).
// \ingroup dense_tensor
//
// \param arr The left-hand side temporary dense array for the multiplication.
// \param scalar The right-hand side scalar value for the multiplication.
// \return Reference to the left-hand side dense array.
// \exception std::invalid_argument Invalid scaling of restricted array.
//
// In case the array \a TT is restricted and the assignment would violate an invariant of the
// array, a \a std::invalid_argument exception is thrown.
*/
template< typename TT    // Type of the left-hand side dense array
        , typename ST >  // Data type of the right-hand side scalar
inline auto operator*=( DenseArray<TT>&& arr, ST scalar )
   -> EnableIf_t< IsNumeric_v<ST>, TT& >
{
   return operator*=( ~arr, scalar );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Division assignment operator for the division of a dense array by a scalar value
//        (\f$ A/=s \f$).
// \ingroup dense_tensor
//
// \param arr The left-hand side dense array for the division.
// \param scalar The right-hand side scalar value for the division.
// \return Reference to the left-hand side dense array.
// \exception std::invalid_argument Invalid scaling of restricted array.
//
// In case the array \a TT is restricted and the assignment would violate an invariant of the
// array, a \a std::invalid_argument exception is thrown.
//
// \note A division by zero is only checked by an user assert.
*/
template< typename TT    // Type of the left-hand side dense array
        , typename ST >  // Data type of the right-hand side scalar
inline auto operator/=( DenseArray<TT>& arr, ST scalar )
   -> EnableIf_t< IsNumeric_v<ST>, TT& >
{
//    BLAZE_CONSTRAINT_MUST_NOT_BE_UNITRIANGULAR_TENSOR_TYPE( TT );

   BLAZE_USER_ASSERT( !isZero( scalar ), "Division by zero detected" );

   if( IsRestricted_v<TT> ) {
      constexpr size_t N =
         RemoveCV_t< RemoveReference_t< decltype( ~arr ) > >::num_dimensions;

      std::array< size_t, N > dims{};
      if( !tryDiv( ~arr, dims, (~arr).dimensions(), scalar ) ) {
         BLAZE_THROW_INVALID_ARGUMENT( "Invalid scaling of restricted array" );
      }
   }

   decltype(auto) left( derestrict( ~arr ) );

   smpAssign( left, left / scalar );

   BLAZE_INTERNAL_ASSERT( isIntact( ~arr ), "Invariant violation detected" );

   return ~arr;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Division assignment operator for the division of a temporary dense array by a scalar
//        value (\f$ A/=s \f$).
// \ingroup dense_tensor
//
// \param arr The left-hand side temporary dense array for the division.
// \param scalar The right-hand side scalar value for the division.
// \return Reference to the left-hand side dense array.
// \exception std::invalid_argument Invalid scaling of restricted array.
//
// In case the array \a TT is restricted and the assignment would violate an invariant of the
// array, a \a std::invalid_argument exception is thrown.
//
// \note A division by zero is only checked by an user assert.
*/
template< typename TT    // Type of the left-hand side dense array
        , typename ST >  // Data type of the right-hand side scalar
inline auto operator/=( DenseArray<TT>&& arr, ST scalar )
   -> EnableIf_t< IsNumeric_v<ST>, TT& >
{
   return operator/=( ~arr, scalar );
}
//*************************************************************************************************




//=================================================================================================
//
//  GLOBAL FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*!\name DenseArray functions */
//@{
template< typename TT >
bool isnan( const DenseArray<TT>& dm );

// template< bool RF, typename MT >
// bool isSymmetric( const DenseArray<MT>& dm );
//
// template< bool RF, typename MT >
// bool isHermitian( const DenseArray<MT>& dm );

template< bool RF, typename MT >
bool isUniform( const DenseArray<MT>& dm );

// template< bool RF, typename MT >
// bool isLower( const DenseArray<MT>& dm );
//
// template< bool RF, typename MT >
// bool isUniLower( const DenseArray<MT>& dm );
//
// template< bool RF, typename MT >
// bool isStrictlyLower( const DenseArray<MT>& dm );
//
// template< bool RF, typename MT >
// bool isUpper( const DenseArray<MT>& dm );
//
// template< bool RF, typename MT >
// bool isUniUpper( const DenseArray<MT>& dm );
//
// template< bool RF, typename MT >
// bool isStrictlyUpper( const DenseArray<MT>& dm );
//
// template< bool RF, typename MT >
// bool isDiagonal( const DenseArray<MT>& dm );
//
// template< bool RF, typename MT >
// bool isIdentity( const DenseArray<MT>& dm );

template< typename MT >
auto softmax( const DenseArray<MT>& dm );
//@}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Checks the given dense array for not-a-number elements.
// \ingroup dense_tensor
//
// \param dm The array to be checked for not-a-number elements.
// \return \a true if at least one element of the array is not-a-number, \a false otherwise.
//
// This function checks the dense array for not-a-number (NaN) elements. If at least one
// element of the array is not-a-number, the function returns \a true, otherwise it returns
// \a false.

   \code
   blaze::DynamicArray<double> A( 5UL, 3UL, 4UL );
   // ... Initialization
   if( isnan( A ) ) { ... }
   \endcode

// Note that this function only works for arrays with floating point elements. The attempt to
// use it for a array with a non-floating point element type results in a compile time error.
*/
template< typename TT > // Type of the dense array
bool isnan( const DenseArray<TT>& dm )
{
   using CT = CompositeType_t<TT>;

   constexpr size_t N =
      RemoveCV_t< RemoveReference_t< decltype( ~dm ) > >::num_dimensions;

   CT A( ~dm );  // Evaluation of the dense array operand

   return ArrayForEachGroupedAnyOf(
      ( ~dm ).dimensions(), [&]( std::array< size_t, N > const& dims ) {
         return isnan( A( dims ) );
      } );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Computes the softmax function for the given dense array.
// \ingroup dense_tensor
//
// \param dm The given dense array for the softmax computation.
// \return The resulting array.
//
// This function computes the softmax function (i.e. the normalized exponential function) for
// the given dense array \a dm (see also https://en.wikipedia.org/wiki/Softmax_function). The
// resulting dense array consists of real values in the range (0..1], which add up to 1.
*/
template< typename MT > // Type of the dense array
auto softmax( const DenseArray<MT>& dm )
{
   auto tmp( evaluate( exp( ~dm ) ) );
   const auto scalar( sum( ~tmp ) );
   tmp /= scalar;
   return tmp;
}
//*************************************************************************************************

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Checks if the given row-major general dense array is a uniform array.
// \ingroup dense_tensor
//
// \param dm The dense array to be checked.
// \return \a true if the array is a uniform array, \a false if not.
*/
template< bool RF        // Relaxation flag
        , typename MT >  // Type of the dense array
bool isUniform_backend( const DenseArray<MT>& dm )
{
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( MT );

#if defined(BLAZE_INTERNAL_ASSERTION)
   ArrayDimForEach( ( ~dm ).dimensions(), [&]( size_t, size_t dim ) {
      BLAZE_INTERNAL_ASSERT( dim != 0, "Invalid array dimension detected" );
   } );
#endif

   constexpr size_t N =
      RemoveCV_t< RemoveReference_t< decltype( ~dm ) > >::num_dimensions;

   std::array< size_t, N > dims{};
   const auto& cmp( (~dm)( dims ) );

   return ArrayForEachGroupedAllOf( ( ~dm ).dimensions(),
      [&]( std::array< size_t, N > const& dims ) {
         return equal< RF >( ( ~dm )( dims ), cmp );
      } );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Checks if the given dense array is a uniform array.
// \ingroup dense_tensor
//
// \param dm The dense array to be checked.
// \return \a true if the array is a uniform array, \a false if not.
//
// This function checks if the given dense array is a uniform array. The array is considered
// to be uniform if all its elements are identical. The following code example demonstrates the
// use of the function:

   \code
   blaze::DynamicArray<int,blaze::rowMajor> A, B;
   // ... Initialization
   if( isUniform( A ) ) { ... }
   \endcode

// Optionally, it is possible to switch between strict semantics (blaze::strict) and relaxed
// semantics (blaze::relaxed):

   \code
   if( isUniform<relaxed>( A ) ) { ... }
   \endcode

// It is also possible to check if a array expression results is a uniform array:

   \code
   if( isUniform( A * B ) ) { ... }
   \endcode

// However, note that this might require the complete evaluation of the expression, including
// the generation of a temporary array.
*/
template< bool RF      // Relaxation flag
        , typename MT > // Type of the dense array
bool isUniform( const DenseArray<MT>& dm )
{
   if( IsUniform_v< MT > ||
      ArrayDimAnyOf( ( ~dm ).dimensions(),
         []( size_t, size_t dim ) { return dim == 0; } ) ||
      ArrayDimAllOf( ( ~dm ).dimensions(),
         []( size_t, size_t dim ) { return dim == 1; } ) ) {
      return true;
   }

   if( IsUniTriangular_v<MT> )
      return false;

   CompositeType_t<MT> A( ~dm );  // Evaluation of the dense array operand

   return isUniform_backend<RF>( A );
}
//*************************************************************************************************


} // namespace blaze

#endif
