//=================================================================================================
/*!
//  \file blaze_array/math/expressions/Array.h
//  \brief Header file for the Array base class
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

#ifndef _BLAZE_TENSOR_MATH_EXPRESSIONS_ARRAY_H_
#define _BLAZE_TENSOR_MATH_EXPRESSIONS_ARRAY_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <blaze/math/Aliases.h>
#include <blaze/math/constraints/Symmetric.h>
#include <blaze/math/Exception.h>
#include <blaze/math/expressions/Forward.h>
#include <blaze/math/typetraits/IsResizable.h>
#include <blaze/math/typetraits/IsShrinkable.h>
#include <blaze/math/typetraits/IsSquare.h>
#include <blaze/math/typetraits/IsSymmetric.h>
#include <blaze/system/Inline.h>
#include <blaze/util/Assert.h>
#include <blaze/util/EnableIf.h>
#include <blaze/util/FunctionTrace.h>
#include <blaze/util/MaybeUnused.h>
#include <blaze/util/Types.h>
#include <blaze/util/typetraits/IsSame.h>

#include <blaze_tensor/math/expressions/Forward.h>
#include <blaze_tensor/util/ArrayForEach.h>

#include <initializer_list>
#include <type_traits>

namespace blaze {

//=================================================================================================
//
//  CLASS DEFINITION
//
//=================================================================================================

//*************************************************************************************************
/*!\defgroup array Arrays
// \ingroup math
*/
/*!\brief Base class for arrays.
// \ingroup array
//
// The Array class is a base class for all dense and sparse array classes within the Blaze
// library. It provides an abstraction from the actual type of the array, but enables a
// conversion back to this type via the 'Curiously Recurring Template Pattern' (CRTP).
*/
template< typename TT > // Type of the array
struct Array
{
   //**Type definitions****************************************************************************
   using ArrayType = TT;  //!< Type of the array.
   //**********************************************************************************************

   //**Non-const conversion operator***************************************************************
   /*!\brief Conversion operator for non-constant arrays.
   //
   // \return Reference of the actual type of the array.
   */
   BLAZE_ALWAYS_INLINE constexpr ArrayType& operator~() noexcept {
      return *static_cast<ArrayType*>( this );
   }
   //**********************************************************************************************

   //**Const conversion operator*******************************************************************
   /*!\brief Conversion operator for constant arrays.
   //
   // \return Constant reference of the actual type of the array.
   */
   BLAZE_ALWAYS_INLINE constexpr const ArrayType& operator~() const noexcept {
      return *static_cast<const ArrayType*>( this );
   }
   //**********************************************************************************************
};
//*************************************************************************************************

//=================================================================================================
//
//  GLOBAL OPERATORS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Multiplication assignment operator for the multiplication of two arrays (\f$ A*=B \f$).
// \ingroup array
//
// \param lhs The left-hand side array for the multiplication.
// \param rhs The right-hand side array for the multiplication.
// \return Reference to the left-hand side array.
// \exception std::invalid_argument Array sizes do not match.
//
// In case the current number of columns of \a lhs and the current number of rows of \a rhs
// don't match, a \a std::invalid_argument is thrown.
*/
template< typename TT1  // Type of the left-hand side array
        , typename TT2 > // Type of the right-hand side array
inline TT1& operator*=( Array<TT1>& lhs, const Array<TT2>& rhs )
{
   ResultType_t<TT1> tmp( (~lhs) * (~rhs) );
   (~lhs) = std::move( tmp );
   return (~lhs);
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Multiplication assignment operator for the multiplication of a temporary array with
//        another array (\f$ A*=B \f$).
// \ingroup array
//
// \param lhs The left-hand side temporary array for the multiplication.
// \param rhs The right-hand side array for the multiplication.
// \return Reference to the left-hand side array.
// \exception std::invalid_argument Array sizes do not match.
//
// In case the current number of columns of \a lhs and the current number of rows of \a rhs
// don't match, a \a std::invalid_argument is thrown.
*/
template< typename TT1  // Type of the left-hand side array
        , typename TT2 > // Type of the right-hand side array
inline TT1& operator*=( Array<TT1>&& lhs, const Array<TT2>& rhs )
{
   return (~lhs) *= (~rhs);
}
/*! \endcond */
//*************************************************************************************************



//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Predict invariant violations by setting a single element of an array.
// \ingroup array
//
// \param mat The target array.
// \param i The row index of the element to be set.
// \param j The column index of the element to be set.
// \param k The page index of the element to be set.
// \param value The value to be set to the element.
// \return \a true in case the operation would be successful, \a false if not.
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename MT    // Type of the array
        , size_t N       // Number of dimensions
        , typename ET >  // Type of the element
BLAZE_ALWAYS_INLINE bool trySet( const Array<MT>& arr, std::array< size_t, N > const& dims, const ET& value )
{
#if defined(BLAZE_INTERNAL_ASSERTION)
   auto const& arrdims = ( ~arr ).dimensions();
   ArrayDimForEach( arrdims, [&]( size_t i, size_t dim ) {
      BLAZE_INTERNAL_ASSERT( dims[i] < dim, "Invalid array access index" );
   } );
#endif

   MAYBE_UNUSED( arr, dims, value );

   return true;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Predict invariant violations by adding to a single element of an array.
// \ingroup array
//
// \param mat The target array.
// \param i The row index of the element to be modified.
// \param j The column index of the element to be modified.
// \param k The page index of the element to be modified.
// \param value The value to be added to the element.
// \return \a true in case the operation would be successful, \a false if not.
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename MT    // Type of the array
        , size_t N       // Number of dimensions
        , typename ET >  // Type of the element
BLAZE_ALWAYS_INLINE bool tryAdd( const Array<MT>& arr, std::array< size_t, N > const& dims, const ET& value )
{
#if defined(BLAZE_INTERNAL_ASSERTION)
   auto const& arrdims = ( ~arr ).dimensions();
   ArrayDimForEach( arrdims, [&]( size_t i, size_t dim ) {
      BLAZE_INTERNAL_ASSERT( dims[i] < dim, "Invalid array access index" );
   } );
#endif

   MAYBE_UNUSED( arr, dims, value );

   return true;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Predict invariant violations by subtracting from a single element of an array.
// \ingroup array
//
// \param mat The target array.
// \param i The row index of the element to be modified.
// \param j The column index of the element to be modified.
// \param k The page index of the element to be modified.
// \param value The value to be subtracted from the element.
// \return \a true in case the operation would be successful, \a false if not.
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename MT    // Type of the array
        , size_t N       // Number of dimensions
        , typename ET >  // Type of the element
BLAZE_ALWAYS_INLINE bool trySub( const Array<MT>& arr, std::array< size_t, N > const& dims, const ET& value )
{
#if defined(BLAZE_INTERNAL_ASSERTION)
   auto const& arrdims = ( ~arr ).dimensions();
   ArrayDimForEach( arrdims, [&]( size_t i, size_t dim ) {
      BLAZE_INTERNAL_ASSERT( dims[i] < dim, "Invalid array access index" );
   } );
#endif

   MAYBE_UNUSED( arr, dims, value );

   return true;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Predict invariant violations by scaling a single element of an array.
// \ingroup array
//
// \param tens The target array.
// \param i The row index of the element to be modified.
// \param j The column index of the element to be modified.
// \param k The page index of the element to be modified.
// \param value The factor for the element.
// \return \a true in case the operation would be successful, \a false if not.
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename MT    // Type of the array
        , size_t N       // Number of dimensions
        , typename ET >  // Type of the element
BLAZE_ALWAYS_INLINE bool tryMult( const Array<MT>& arr, std::array< size_t, N > const& dims, const ET& value )
{
#if defined(BLAZE_INTERNAL_ASSERTION)
   auto const& arrdims = ( ~arr ).dimensions();
   ArrayDimForEach( arrdims, [&]( size_t i, size_t dim ) {
      BLAZE_INTERNAL_ASSERT( dims[i] < dim, "Invalid array access index" );
   } );
#endif

   MAYBE_UNUSED( arr, dims, value );

   return true;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Predict invariant violations by scaling a range of elements of an array.
// \ingroup array
//
// \param tens The target array.
// \param row The index of the first row of the range to be modified.
// \param column The index of the first column of the range to be modified.
// \param page The index of the first page of the range to be modified.
// \param m The number of rows of the range to be modified.
// \param n The number of columns of the range to be modified.
// \param o The number of pages of the range to be modified.
// \param value The factor for the elements.
// \return \a true in case the operation would be successful, \a false if not.
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename MT    // Type of the array
        , size_t N       // Number of dimensions
        , typename ET >  // Type of the element
BLAZE_ALWAYS_INLINE bool
   tryMult( const Array<MT>& arr, std::array< size_t, N > const& sizes, std::array< size_t, N > const& indices, const ET& value )
{
//    BLAZE_INTERNAL_ASSERT( row <= (~tens).rows(), "Invalid row access index" );
//    BLAZE_INTERNAL_ASSERT( column <= (~tens).columns(), "Invalid column access index" );
//    BLAZE_INTERNAL_ASSERT( page <= (~tens).pages(), "Invalid page access index" );
//    BLAZE_INTERNAL_ASSERT( row + m <= (~tens).rows(), "Invalid number of rows" );
//    BLAZE_INTERNAL_ASSERT( column + n <= (~tens).columns(), "Invalid number of columns" );
//    BLAZE_INTERNAL_ASSERT( page + o <= (~tens).pages(), "Invalid number of pages" );
//
//    MAYBE_UNUSED( tens, page, row, column, o, m, n, value );

   return true;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Predict invariant violations by scaling a single element of an array.
// \ingroup array
//
// \param mat The target array.
// \param i The row index of the element to be modified.
// \param j The column index of the element to be modified.
// \param k The page index of the element to be modified.
// \param value The divisor for the element.
// \return \a true in case the operation would be successful, \a false if not.
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename MT    // Type of the array
        , size_t N       // Number of dimensions
        , typename ET >  // Type of the element
BLAZE_ALWAYS_INLINE bool tryDiv( const Array<MT>& arr, std::array< size_t, N > const& dims, const ET& value )
{
#if defined(BLAZE_INTERNAL_ASSERTION)
   auto const& arrdims = ( ~arr ).dimensions();
   ArrayDimForEach( arrdims, [&]( size_t i, size_t dim ) {
      BLAZE_INTERNAL_ASSERT( dims[i] < dim, "Invalid array access index" );
   } );
#endif

   MAYBE_UNUSED( arr, dims, value );

   return true;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Predict invariant violations by scaling a range of elements of an array.
// \ingroup array
//
// \param mat The target array.
// \param row The index of the first row of the range to be modified.
// \param column The index of the first column of the range to be modified.
// \param page The index of the first page of the range to be modified.
// \param m The number of rows of the range to be modified.
// \param n The number of columns of the range to be modified.
// \param o The number of pages of the range to be modified.
// \param value The divisor for the elements.
// \return \a true in case the operation would be successful, \a false if not.
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename MT    // Type of the array
        , size_t M       // number of current dimensions
        , size_t N       // number of dimensions
        , typename ET >  // Type of the element
BLAZE_ALWAYS_INLINE bool
   tryDiv( const Array<MT>& arr, std::array< size_t, M > const& currdims,
      std::array< size_t, N > const& dims, const ET& value )
{
   BLAZE_STATIC_ASSERT( M == N );

#if defined(BLAZE_INTERNAL_ASSERTION)
   ArrayDimForEach( dims, [&]( size_t i, size_t dim ) {
      BLAZE_INTERNAL_ASSERT( currdims[i] < dim, "Invalid array access index" );
   } );
#endif

   MAYBE_UNUSED( arr, currdims, dims, value );

   return true;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Predict invariant violations by the assignment of an array to an array.
// \ingroup array
//
// \param lhs The target left-hand side array.
// \param rhs The right-hand side array to be assigned.
// \param row The row index of the first element to be modified.
// \param column The column index of the first element to be modified.
// \return \a true in case the assignment would be successful, \a false if not.
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename MT  // Type of the left-hand side array
        , typename VT  // Type of the right-hand side array
        , size_t N >   // Number of dimensions to check
BLAZE_ALWAYS_INLINE bool tryAssign( const Array<MT>& lhs, const Array<VT>& rhs,
                                    std::array< size_t, N > const& dims )
{
#if defined(BLAZE_INTERNAL_ASSERTION)
   auto const& rhsdims = ( ~rhs ).dimensions();
   ArrayDimForEach( ( ~lhs ).dimensions(), [&]( size_t i, size_t dim ) {
      BLAZE_INTERNAL_ASSERT( rhsdims[i] != dim, "Invalid array access index" );
      BLAZE_INTERNAL_ASSERT( dims[i] < dim, "Invalid array access index" );
   } );
#endif

   MAYBE_UNUSED( lhs, rhs, dims );

   return true;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Predict invariant violations by the addition assignment of an array to an array.
// \ingroup array
//
// \param lhs The target left-hand side array.
// \param rhs The right-hand side array to be added.
// \param row The row index of the first element to be modified.
// \param column The column index of the first element to be modified.
// \return \a true in case the assignment would be successful, \a false if not.
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename TT1  // Type of the left-hand side array
        , typename TT2  // Type of the right-hand side array
        , size_t N  >   // Number of dimensions to check
BLAZE_ALWAYS_INLINE bool tryAddAssign( const Array<TT1>& lhs, const Array<TT2>& rhs,
                                       std::array< size_t, N > const& dims )
{
#if defined(BLAZE_INTERNAL_ASSERTION)
   auto const& rhsdims = ( ~rhs ).dimensions();
   ArrayDimForEach( ( ~lhs ).dimensions(), [&]( size_t i, size_t dim ) {
      BLAZE_INTERNAL_ASSERT( rhsdims[i] != dim, "Invalid array access index" );
      BLAZE_INTERNAL_ASSERT( dims[i] < dim, "Invalid array access index" );
   } );
#endif

   MAYBE_UNUSED( lhs, rhs, dims );

   return true;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Predict invariant violations by the subtraction assignment of an array to an array.
// \ingroup array
//
// \param lhs The target left-hand side array.
// \param rhs The right-hand side array to be subtracted.
// \param row The row index of the first element to be modified.
// \param column The column index of the first element to be modified.
// \return \a true in case the assignment would be successful, \a false if not.
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename TT1  // Type of the left-hand side array
        , typename TT2  // Type of the right-hand side array
        , size_t N  >   // Number of dimensions to check
BLAZE_ALWAYS_INLINE bool trySubAssign( const Array<TT1>& lhs, const Array<TT2>& rhs,
                                       std::array< size_t, N > const& dims )
{
#if defined(BLAZE_INTERNAL_ASSERTION)
   auto const& rhsdims = ( ~rhs ).dimensions();
   ArrayDimForEach( ( ~lhs ).dimensions(), [&]( size_t i, size_t dim ) {
      BLAZE_INTERNAL_ASSERT( rhsdims[i] != dim, "Invalid array access index" );
      BLAZE_INTERNAL_ASSERT( dims[i] < dim, "Invalid array access index" );
   } );
#endif

   MAYBE_UNUSED( lhs, rhs, dims );

   return true;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Predict invariant violations by the multiplication assignment of an array to an array.
// \ingroup array
//
// \param lhs The target left-hand side array.
// \param rhs The right-hand side array to be multiplied.
// \param band The index of the band the right-hand side array is assigned to.
// \param row The row index of the first element to be modified.
// \param column The column index of the first element to be modified.
// \return \a true in case the assignment would be successful, \a false if not.
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename TT1  // Type of the left-hand side array
        , typename TT2  // Type of the right-hand side array
        , size_t N  >   // Number of dimensions to check
BLAZE_ALWAYS_INLINE bool tryMultAssign( const Array<TT1>& lhs, const Array<TT2>& rhs,
                                         std::array< size_t, N > const& dims )
{
#if defined(BLAZE_INTERNAL_ASSERTION)
   auto const& rhsdims = ( ~rhs ).dimensions();
   ArrayDimForEach( ( ~lhs ).dimensions(), [&]( size_t i, size_t dim ) {
      BLAZE_INTERNAL_ASSERT( rhsdims[i] != dim, "Invalid array access index" );
      BLAZE_INTERNAL_ASSERT( dims[i] < dim, "Invalid array access index" );
   } );
#endif

   MAYBE_UNUSED( lhs, rhs, dims );

   return true;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Predict invariant violations by the Schur product assignment of an array to an array.
// \ingroup array
//
// \param lhs The target left-hand side array.
// \param rhs The right-hand side array for the Schur product.
// \param row The row index of the first element to be modified.
// \param column The column index of the first element to be modified.
// \return \a true in case the assignment would be successful, \a false if not.
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename TT1  // Type of the left-hand side array
        , typename TT2  // Type of the right-hand side array
        , size_t N  >   // Number of dimensions to check
BLAZE_ALWAYS_INLINE bool trySchurAssign( const Array<TT1>& lhs, const Array<TT2>& rhs,
                                         std::array< size_t, N > const& dims )
{
#if defined(BLAZE_INTERNAL_ASSERTION)
   auto const& rhsdims = ( ~rhs ).dimensions();
   ArrayDimForEach( ( ~lhs ).dimensions(), [&]( size_t i, size_t dim ) {
      BLAZE_INTERNAL_ASSERT( rhsdims[i] != dim, "Invalid array access index" );
      BLAZE_INTERNAL_ASSERT( dims[i] < dim, "Invalid array access index" );
   } );
#endif

   MAYBE_UNUSED( lhs, rhs, dims );

   return true;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Predict invariant violations by the division assignment of an array to an array.
// \ingroup array
//
// \param lhs The target left-hand side array.
// \param rhs The right-hand side array divisor.
// \param row The row index of the first element to be modified.
// \param column The column index of the first element to be modified.
// \return \a true in case the assignment would be successful, \a false if not.
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename TT1  // Type of the left-hand side array
        , typename TT2  // Type of the right-hand side array
        , size_t N  >   // Number of dimensions to check
BLAZE_ALWAYS_INLINE bool tryDivAssign( const Array<TT1>& lhs, const Array<TT2>& rhs,
                                       std::array< size_t, N > const& dims )
{
#if defined(BLAZE_INTERNAL_ASSERTION)
   auto const& rhsdims = ( ~rhs ).dimensions();
   ArrayDimForEach( ( ~lhs ).dimensions(), [&]( size_t i, size_t dim ) {
      BLAZE_INTERNAL_ASSERT( rhsdims[i] != dim, "Invalid array access index" );
      BLAZE_INTERNAL_ASSERT( dims[i] < dim, "Invalid array access index" );
   } );
#endif

   MAYBE_UNUSED( lhs, rhs, dims );

   return true;
}
/*! \endcond */
//*************************************************************************************************


//=================================================================================================
//
//  GLOBAL FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*!\name Array global functions */
//@{
template< typename MT >
BLAZE_ALWAYS_INLINE typename MT::Iterator begin( Array<MT>& array, size_t i, size_t k );

template< typename MT >
BLAZE_ALWAYS_INLINE typename MT::ConstIterator begin( const Array<MT>& array, size_t i, size_t k );

template< typename MT >
BLAZE_ALWAYS_INLINE typename MT::ConstIterator cbegin( const Array<MT>& array, size_t i, size_t k );

template< typename MT >
BLAZE_ALWAYS_INLINE typename MT::Iterator end( Array<MT>& array, size_t i, size_t k );

template< typename MT >
BLAZE_ALWAYS_INLINE typename MT::ConstIterator end( const Array<MT>& array, size_t i, size_t k );

template< typename MT >
BLAZE_ALWAYS_INLINE typename MT::ConstIterator cend( const Array<MT>& array, size_t i, size_t k );

template< typename MT >
BLAZE_ALWAYS_INLINE constexpr size_t rows( const Array<MT>& array ) noexcept;

template< typename MT >
BLAZE_ALWAYS_INLINE constexpr size_t columns( const Array<MT>& array ) noexcept;

template< typename MT >
BLAZE_ALWAYS_INLINE constexpr size_t pages( const Array<MT>& array ) noexcept;

template< typename MT >
BLAZE_ALWAYS_INLINE constexpr size_t quats( const Array<MT>& array ) noexcept;

template< typename MT >
BLAZE_ALWAYS_INLINE constexpr size_t size( const Array<MT>& array ) noexcept;

template< typename MT >
BLAZE_ALWAYS_INLINE size_t capacity( const Array<MT>& array ) noexcept;

template< typename MT >
BLAZE_ALWAYS_INLINE size_t capacity( const Array<MT>& array, size_t i, size_t k ) noexcept;

template< typename MT >
BLAZE_ALWAYS_INLINE size_t nonZeros( const Array<MT>& array );

template< typename MT >
BLAZE_ALWAYS_INLINE size_t nonZeros( const Array<MT>& array, size_t i, size_t k );

template< typename MT >
BLAZE_ALWAYS_INLINE void resize( Array<MT>& array, size_t rows, size_t columns, size_t pages, bool preserve=true );

template< typename MT >
BLAZE_ALWAYS_INLINE void shrinkToFit( Array<MT>& array );

template< typename MT >
BLAZE_ALWAYS_INLINE void transpose( Array<MT>& array );

template< typename MT, typename T >
BLAZE_ALWAYS_INLINE void transpose( Array<MT>& array, const T*, size_t );

template< typename MT, typename T >
BLAZE_ALWAYS_INLINE void transpose( Array<MT>& array, std::initializer_list<T> );

template< typename MT >
BLAZE_ALWAYS_INLINE void ctranspose( Array<MT>& array );

template< typename MT, typename T >
BLAZE_ALWAYS_INLINE void ctranspose( Array<MT>& array, const T*, size_t );

template< typename MT, typename T >
BLAZE_ALWAYS_INLINE void ctranspose( Array<MT>& array, std::initializer_list<T> );


template< typename MT >
inline const typename MT::ResultType evaluate( const Array<MT>& array );

template< typename MT >
BLAZE_ALWAYS_INLINE constexpr bool isEmpty( const Array<MT>& array ) noexcept;

template< typename MT >
BLAZE_ALWAYS_INLINE bool isSquare( const Array<MT>& array ) noexcept;

template< typename TT1, typename TT2 >
BLAZE_ALWAYS_INLINE bool isSame( const Array<TT1>& a, const Array<TT2>& b ) noexcept;
//@}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns an iterator to the first element of row/column \a i.
// \ingroup array
//
// \param array The given dense or sparse array.
// \param i The row/column index.
// \return Iterator to the first element of row/column \a i.
//
// This function returns a row/column iterator to the first element of row/column \a i. In case
// the given array is a row-major array the function returns an iterator to the first element
// of row \a i, in case it is a column-major array the function returns an iterator to the first
// element of column \a i.
*/
template< typename MT > // Type of the array
BLAZE_ALWAYS_INLINE typename MT::Iterator begin( Array<MT>& array, size_t i, size_t k )
{
   return (~array).begin(i, k);
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns an iterator to the first element of row/column \a i.
// \ingroup array
//
// \param array The given dense or sparse array.
// \param i The row/column index.
// \return Iterator to the first element of row/column \a i.
//
// This function returns a row/column iterator to the first element of row/column \a i. In case
// the given array is a row-major array the function returns an iterator to the first element
// of row \a i, in case it is a column-major array the function returns an iterator to the first
// element of column \a i.
*/
template< typename MT > // Type of the array
BLAZE_ALWAYS_INLINE typename MT::ConstIterator begin( const Array<MT>& array, size_t i, size_t k )
{
   return (~array).begin(i, k);
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns an iterator to the first element of row/column \a i.
// \ingroup array
//
// \param array The given dense or sparse array.
// \param i The row/column index.
// \return Iterator to the first element of row/column \a i.
//
// This function returns a row/column iterator to the first element of row/column \a i. In case
// the given array is a row-major array the function returns an iterator to the first element
// of row \a i, in case it is a column-major array the function returns an iterator to the first
// element of column \a i.
*/
template< typename MT > // Type of the array
BLAZE_ALWAYS_INLINE typename MT::ConstIterator cbegin( const Array<MT>& array, size_t i, size_t k )
{
   return (~array).cbegin(i, k);
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns an iterator just past the last element of row/column \a i.
// \ingroup array
//
// \param array The given dense or sparse array.
// \param i The row/column index.
// \return Iterator just past the last element of row/column \a i.
//
// This function returns an row/column iterator just past the last element of row/column \a i.
// In case the given array is a row-major array the function returns an iterator just past
// the last element of row \a i, in case it is a column-major array the function returns an
// iterator just past the last element of column \a i.
*/
template< typename MT > // Type of the array
BLAZE_ALWAYS_INLINE typename MT::Iterator end( Array<MT>& array, size_t i, size_t k )
{
   return (~array).end(i, k);
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns an iterator just past the last element of row/column \a i.
// \ingroup array
//
// \param array The given dense or sparse array.
// \param i The row/column index.
// \return Iterator just past the last element of row/column \a i.
//
// This function returns an row/column iterator just past the last element of row/column \a i.
// In case the given array is a row-major array the function returns an iterator just past
// the last element of row \a i, in case it is a column-major array the function returns an
// iterator just past the last element of column \a i.
*/
template< typename MT > // Type of the array
BLAZE_ALWAYS_INLINE typename MT::ConstIterator end( const Array<MT>& array, size_t i, size_t k )
{
   return (~array).end(i, k);
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns an iterator just past the last element of row/column \a i.
// \ingroup array
//
// \param array The given dense or sparse array.
// \param i The row/column index.
// \return Iterator just past the last element of row/column \a i.
//
// This function returns an row/column iterator just past the last element of row/column \a i.
// In case the given array is a row-major array the function returns an iterator just past
// the last element of row \a i, in case it is a column-major array the function returns an
// iterator just past the last element of column \a i.
*/
template< typename MT > // Type of the array
BLAZE_ALWAYS_INLINE typename MT::ConstIterator cend( const Array<MT>& array, size_t i, size_t k )
{
   return (~array).cend(i, k);
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns the current number of rows of the array.
// \ingroup array
//
// \param array The given array.
// \return The number of rows of the array.
*/
template< typename MT > // Type of the array
BLAZE_ALWAYS_INLINE constexpr size_t rows( const Array<MT>& array ) noexcept
{
   return (~array).rows();
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns the current number of columns of the array.
// \ingroup array
//
// \param array The given array.
// \return The number of columns of the array.
*/
template< typename MT > // Type of the array
BLAZE_ALWAYS_INLINE constexpr size_t columns( const Array<MT>& array ) noexcept
{
   return (~array).columns();
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns the current number of pages of the array.
// \ingroup array
//
// \param array The given array.
// \return The number of pages of the array.
*/
template< typename MT > // Type of the array
BLAZE_ALWAYS_INLINE constexpr size_t pages( const Array<MT>& array ) noexcept
{
   return (~array).pages();
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns the current number of quats of the array.
// \ingroup array
//
// \param array The given array.
// \return The number of quats of the array.
*/
template< typename MT > // Type of the array
BLAZE_ALWAYS_INLINE constexpr size_t quats( const Array<MT>& array ) noexcept
{
   return (~array).quats();
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns the total number of elements of the array.
// \ingroup array
//
// \param array The given array.
// \return The total number of elements of the array.
*/
template< typename MT > // Type of the array
BLAZE_ALWAYS_INLINE constexpr size_t size( const Array<MT>& array ) noexcept
{
   return (~array).rows() * (~array).columns() * (~array).pages();
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns the maximum capacity of the array.
// \ingroup array
//
// \param array The given array.
// \return The capacity of the array.
*/
template< typename MT > // Type of the array
BLAZE_ALWAYS_INLINE size_t capacity( const Array<MT>& array ) noexcept
{
   return (~array).capacity();
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns the current capacity of the specified row/column.
// \ingroup array
//
// \param array The given array.
// \param i The index of the row/column.
// \return The current capacity of row/column \a i.
//
// This function returns the current capacity of the specified row/column. In case the
// storage order is set to \a rowMajor the function returns the capacity of row \a i,
// in case the storage flag is set to \a columnMajor the function returns the capacity
// of column \a i.
*/
template< typename MT > // Type of the array
BLAZE_ALWAYS_INLINE size_t capacity( const Array<MT>& array, size_t i, size_t k ) noexcept
{
   return (~array).capacity( i, k );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns the total number of non-zero elements in the array
// \ingroup array
//
// \param array The given array.
// \return The number of non-zero elements in the dense array.
*/
template< typename MT > // Type of the array
BLAZE_ALWAYS_INLINE size_t nonZeros( const Array<MT>& array )
{
   return (~array).nonZeros();
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns the number of non-zero elements in the specified row/column.
// \ingroup array
//
// \param array The given array.
// \param i The index of the row/column.
// \return The number of non-zero elements of row/column \a i.
//
// This function returns the current number of non-zero elements in the specified row/column.
// In case the storage order is set to \a rowMajor the function returns the number of non-zero
// elements in row \a i, in case the storage flag is set to \a columnMajor the function returns
// the number of non-zero elements in column \a i.
*/
template< typename MT > // Type of the array
BLAZE_ALWAYS_INLINE size_t nonZeros( const Array<MT>& array, size_t i, size_t k )
{
   return (~array).nonZeros( i, k );
}
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Backend implementation of the \c resize() function for non-resizable arrays.
// \ingroup array
//
// \param array The given array to be resized.
// \param m The new number of rows of the array.
// \param n The new number of columns of the array.
// \param preserve \a true if the old values of the array should be preserved, \a false if not.
// \return void
// \exception std::invalid_argument Array cannot be resized.
//
// This function tries to change the number of rows and columns of a non-resizable array. Since
// the array cannot be resized, in case the specified number of rows and columns is not identical
// to the current number of rows and columns of the array, a \a std::invalid_argument exception
// is thrown.
*/
template< typename MT > // Type of the array
BLAZE_ALWAYS_INLINE EnableIf_t< !IsResizable_v<MT> >
   resize_backend( Array<MT>& array, size_t o, size_t m, size_t n, bool preserve )
{
   MAYBE_UNUSED( preserve );

   if( (~array).rows() != m || (~array).columns() != n || (~array).pages() != o) {
      BLAZE_THROW_INVALID_ARGUMENT( "Array cannot be resized" );
   }
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Backend implementation of the \c resize() function for resizable, non-square arrays.
// \ingroup array
//
// \param array The given array to be resized.
// \param m The new number of rows of the array.
// \param n The new number of columns of the array.
// \param preserve \a true if the old values of the array should be preserved, \a false if not.
// \return void
//
// This function changes the number of rows and columns of the given resizable, non-square array.
*/
template< typename MT > // Type of the array
BLAZE_ALWAYS_INLINE EnableIf_t< IsResizable_v<MT> && !IsSquare_v<MT> >
   resize_backend( Array<MT>& array, size_t o, size_t m, size_t n, bool preserve )
{
   (~array).resize( o, m, n, preserve );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Backend implementation of the \c resize() function for resizable, square arrays.
// \ingroup array
//
// \param array The given array to be resized.
// \param m The new number of rows of the array.
// \param n The new number of columns of the array.
// \param preserve \a true if the old values of the array should be preserved, \a false if not.
// \return void
// \exception std::invalid_argument Invalid resize arguments for square array.
//
// This function changes the number of rows and columns of the given resizable, square array.
*/
// template< typename MT > // Type of the array
// BLAZE_ALWAYS_INLINE EnableIf_t< IsResizable_v<MT> && IsSquare_v<MT> >
//    resize_backend( Array<MT>& array, size_t o, size_t m, size_t n, bool preserve )
// {
//    if( m != n || m != o ) {
//       BLAZE_THROW_INVALID_ARGUMENT( "Invalid resize arguments for square array" );
//    }
//
//    (~array).resize( m, preserve );
// }
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Changing the size of the array.
// \ingroup array
//
// \param array The given array to be resized.
// \param m The new number of rows of the array.
// \param n The new number of columns of the array.
// \param preserve \a true if the old values of the array should be preserved, \a false if not.
// \return void
// \exception std::invalid_argument Invalid resize arguments for square array.
// \exception std::invalid_argument Array cannot be resized.
//
// This function provides a unified interface to resize dense and sparse arrays. In contrast
// to the \c resize() member function, which is only available on resizable array types, this
// function can be used on both resizable and non-resizable arrays. In case the given array
// of type \a MT is resizable (i.e. provides a \c resize function) the type-specific \c resize()
// member function is called. Depending on the type \a MT, this may result in the allocation of
// new dynamic memory and the invalidation of existing views (subarrays, rows, columns, ...).
// Note that in case the array is a compile time square array (as for instance the
// blaze::SymmetricArray adaptor, ...) the specified number of rows must be identical to the
// number of columns. Otherwise a \a std::invalid_argument exception is thrown. If the array
// type \a MT is non-resizable (i.e. does not provide a \c resize() function) and if the specified
// number of rows and columns is not identical to the current number of rows and columns of the
// array, a \a std::invalid_argument exception is thrown.

   \code
   blaze::DynamicArray<int> A( 3UL, 3UL );
   resize( A, 5UL, 2UL );  // OK: regular resize operation

   blaze::SymmetricArray< DynamicArray<int> > B( 3UL );
   resize( B, 4UL, 4UL );  // OK: Number of rows and columns is identical
   resize( B, 3UL, 5UL );  // Error: Invalid arguments for square array!

   blaze::StaticArray<int,3UL,3UL> C;
   resize( C, 3UL, 3UL );  // OK: No resize necessary
   resize( C, 5UL, 2UL );  // Error: Array cannot be resized!
   \endcode
*/
template< typename MT > // Type of the array
BLAZE_ALWAYS_INLINE void resize( Array<MT>& array, size_t o, size_t m, size_t n, bool preserve )
{
   resize_backend( array, o, m, n, preserve );
}
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Backend implementation of the \c shrinkToFit() function for non-shrinkable arrays.
// \ingroup array
//
// \param array The given array to be shrunk.
// \return void
*/
template< typename MT > // Type of the array
BLAZE_ALWAYS_INLINE EnableIf_t< !IsShrinkable_v<MT> >
   shrinkToFit_backend( Array<MT>& array )
{
   MAYBE_UNUSED( array );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Backend implementation of the \c shrinkToFit() function for shrinkable arrays.
// \ingroup array
//
// \param array The given array to be shrunk.
// \return void
*/
template< typename MT > // Type of the array
BLAZE_ALWAYS_INLINE EnableIf_t< IsShrinkable_v<MT> >
   shrinkToFit_backend( Array<MT>& array )
{
   (~array).shrinkToFit();
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Requesting the removal of unused capacity.
// \ingroup array
//
// \param array The given array to be shrunk.
// \return void
//
// This function tries to minimize the capacity of the array by removing unused capacity.
// Please note that in case of a shrinkable array, due to padding the capacity might not be
// reduced exactly to the number of rows times the number of columns. Please also note that
// in case a reallocation occurs, all iterators (including end() iterators), all pointers and
// references to elements of this array are invalidated. In case of an unshrinkable array
// the function has no effect.
*/
template< typename MT > // Type of the array
BLAZE_ALWAYS_INLINE void shrinkToFit( Array<MT>& array )
{
   shrinkToFit_backend( array );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief In-place transpose of the given array.
// \ingroup array
//
// \param array The given array to be transposed.
// \return void
// \exception std::logic_error Array cannot be transposed.
//
// This function transposes the given array in-place. The function fails if ...
//
//  - ... the given array has a fixed size and is non-square;
//  - ... the given array is a triangular array;
//  - ... the given subarray affects the restricted parts of a triangular array;
//  - ... the given subarray would cause non-deterministic results in a symmetric/Hermitian array.
//
// In all failure cases a \a std::logic_error exception is thrown.
*/
template< typename MT > // Type of the array
BLAZE_ALWAYS_INLINE void transpose( Array<MT>& array )
{
   (~array).transpose( );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief In-place transpose of the given array.
// \ingroup array
//
// \param array The given array to be transposed.
// \return void
// \exception std::logic_error Array cannot be transposed.
//
// This function transposes the given array in-place. The function fails if ...
//
//  - ... the given array has a fixed size and is non-square;
//  - ... the given array is a triangular array;
//  - ... the given subarray affects the restricted parts of a triangular array;
//  - ... the given subarray would cause non-deterministic results in a symmetric/Hermitian array.
//
// In all failure cases a \a std::logic_error exception is thrown.
*/
template< typename MT   // Type of the array
        , typename T >  // Type of the index initializer
BLAZE_ALWAYS_INLINE void transpose( Array<MT>& array, const T* indices, size_t n )
{
   (~array).transpose( indices, n );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief In-place transpose of the given array.
// \ingroup array
//
// \param array The given array to be transposed.
// \return void
// \exception std::logic_error Array cannot be transposed.
//
// This function transposes the given array in-place. The function fails if ...
//
//  - ... the given array has a fixed size and is non-square;
//  - ... the given array is a triangular array;
//  - ... the given subarray affects the restricted parts of a triangular array;
//  - ... the given subarray would cause non-deterministic results in a symmetric/Hermitian array.
//
// In all failure cases a \a std::logic_error exception is thrown.
*/
template< typename MT   // Type of the array
        , typename T >  // Type of the index initializer
BLAZE_ALWAYS_INLINE void transpose( Array<MT>& array, std::initializer_list<T> indices )
{
   (~array).transpose( indices.begin(), indices.size() );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief In-place conjugate transpose of the given array.
// \ingroup array
//
// \param array The given array to be transposed.
// \return void
// \exception std::logic_error Array cannot be transposed.
//
// This function transposes the given array in-place. The function fails if ...
//
//  - ... the given array has a fixed size and is non-square;
//  - ... the given array is a triangular array;
//  - ... the given subarray affects the restricted parts of a triangular array;
//  - ... the given subarray would cause non-deterministic results in a symmetric/Hermitian array.
//
// In all failure cases a \a std::logic_error exception is thrown.
*/
template< typename MT > // Type of the array
BLAZE_ALWAYS_INLINE void ctranspose( Array<MT>& array )
{
   (~array).ctranspose();
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief In-place conjugate transpose of the given array.
// \ingroup array
//
// \param array The given array to be transposed.
// \return void
// \exception std::logic_error Array cannot be transposed.
//
// This function transposes the given array in-place. The function fails if ...
//
//  - ... the given array has a fixed size and is non-square;
//  - ... the given array is a triangular array;
//  - ... the given subarray affects the restricted parts of a triangular array;
//  - ... the given subarray would cause non-deterministic results in a symmetric/Hermitian array.
//
// In all failure cases a \a std::logic_error exception is thrown.
*/
template< typename MT   // Type of the array
        , typename T >  // Type of the index initializer
BLAZE_ALWAYS_INLINE void ctranspose( Array<MT>& array, const T* indices, size_t n )
{
   (~array).ctranspose( indices, n );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief In-place conjugate transpose of the given array.
// \ingroup array
//
// \param array The given array to be transposed.
// \return void
// \exception std::logic_error Array cannot be transposed.
//
// This function transposes the given array in-place. The function fails if ...
//
//  - ... the given array has a fixed size and is non-square;
//  - ... the given array is a triangular array;
//  - ... the given subarray affects the restricted parts of a triangular array;
//  - ... the given subarray would cause non-deterministic results in a symmetric/Hermitian array.
//
// In all failure cases a \a std::logic_error exception is thrown.
*/
template< typename MT   // Type of the array
        , typename T >  // Type of the index initializer
BLAZE_ALWAYS_INLINE void ctranspose( Array<MT>& array, std::initializer_list<T> indices )
{
   (~array).ctranspose( indices.begin(), indices.size() );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Evaluates the given array expression.
// \ingroup array
//
// \param array The array to be evaluated.
// \return The result of the evaluated array expression.
//
// This function forces an evaluation of the given array expression and enables an automatic
// deduction of the correct result type of an operation. The following code example demonstrates
// its intended use for the multiplication of a lower and a strictly lower dense array:

   \code
   using blaze::DynamicArray;
   using blaze::LowerArray;
   using blaze::StrictlyLowerArray;

   LowerArray< DynamicArray<double> > A;
   StrictlyLowerArray< DynamicArray<double> > B;
   // ... Resizing and initialization

   auto C = evaluate( A * B );
   \endcode

// In this scenario, the \a evaluate() function assists in deducing the exact result type of
// the operation via the 'auto' keyword. Please note that if \a evaluate() is used in this
// way, no temporary array is created and no copy operation is performed. Instead, the result
// is directly written to the target array due to the return value optimization (RVO). However,
// if \a evaluate() is used in combination with an explicit target type, a temporary will be
// created and a copy operation will be performed if the used type differs from the type
// returned from the function:

   \code
   StrictlyLowerArray< DynamicArray<double> > D( A * B );  // No temporary & no copy operation
   LowerArray< DynamicArray<double> > E( A * B );          // Temporary & copy operation
   DynamicArray<double> F( A * B );                         // Temporary & copy operation
   D = evaluate( A * B );                                    // Temporary & copy operation
   \endcode

// Sometimes it might be desirable to explicitly evaluate a sub-expression within a larger
// expression. However, please note that \a evaluate() is not intended to be used for this
// purpose. This task is more elegantly and efficiently handled by the \a eval() function:

   \code
   blaze::DynamicArray<double> A, B, C, D;

   D = A + evaluate( B * C );  // Unnecessary creation of a temporary array
   D = A + eval( B * C );      // No creation of a temporary array
   \endcode

// In contrast to the \a evaluate() function, \a eval() can take the complete expression into
// account and therefore can guarantee the most efficient way to evaluate it.
*/
template< typename MT > // Type of the array
inline const typename MT::ResultType evaluate( const Array<MT>& array )
{
   const typename MT::ResultType tmp( ~array );
   return tmp;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Checks if the given array is empty.
// \ingroup array
//
// \param array The array to be checked.
// \return \a true if the array is empty, \a false if not.
//
// This function checks if the total number of elements of the given array is zero. If the
// total number of elements is zero the function returns \a true, otherwise it returns \a false.
*/
template< typename MT > // Type of the array
BLAZE_ALWAYS_INLINE constexpr bool isEmpty( const Array<MT>& array ) noexcept
{
   return size( ~array ) == 0UL;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Checks if the given array is a square array.
// \ingroup array
//
// \param array The array to be checked.
// \return \a true if the array is a square array, \a false if not.
//
// This function checks if the number of rows and columns of the given array are equal. If
// they are, the function returns \a true, otherwise it returns \a false.
*/
template< typename MT > // Type of the array
BLAZE_ALWAYS_INLINE bool isSquare( const Array<MT>& array ) noexcept
{
   return ( IsSquare_v<MT> || ( (~array).rows() == (~array).columns() && (~array).rows() == (~array).pages() ) );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns whether the two given arrays represent the same observable state.
// \ingroup array
//
// \param a The first array to be tested for its state.
// \param b The second array to be tested for its state.
// \return \a true in case the two arrays share a state, \a false otherwise.
//
// The isSame function provides an abstract interface for testing if the two given arrays
// represent the same observable state. This happens for instance in case \c a and \c b refer
// to the same array or in case \c a and \c b are aliases for the same array. In case both
// arrays represent the same observable state, the function returns \a true, other it returns
// \a false.

   \code
   blaze::DynamicArray<int> mat1( 4UL, 5UL );  // Setup of a 4x5 dynamic array
   blaze::DynamicArray<int> mat2( 4UL, 5UL );  // Setup of a second 4x5 dynamic array

   auto sub1 = subarray( mat1, 4UL, 0UL, 0UL, 5UL );  // Subarray fully covering mat1
   auto sub2 = subarray( mat1, 2UL, 1UL, 1UL, 3UL );  // Subarray partially covering mat1
   auto sub3 = subarray( mat1, 2UL, 1UL, 1UL, 3UL );  // Subarray partially covering mat1

   isSame( mat1, mat1 );  // returns true since both objects refer to the same array
   isSame( mat1, mat2 );  // returns false since mat1 and mat2 are two different arrays
   isSame( mat1, sub1 );  // returns true since sub1 represents the same observable state as mat1
   isSame( mat1, sub3 );  // returns false since sub3 only covers part of mat1
   isSame( sub2, sub3 );  // returns true since sub1 and sub2 refer to exactly the same part of mat1
   isSame( sub1, sub3 );  // returns false since sub1 and sub3 refer to different parts of mat1
   \endcode
*/
template< typename TT1  // Type of the left-hand side array
        , typename TT2 > // Type of the right-hand side array
BLAZE_ALWAYS_INLINE bool isSame( const Array<TT1>& a, const Array<TT2>& b ) noexcept
{
   return ( IsSame_v<TT1,TT2> &&
            reinterpret_cast<const void*>( &a ) == reinterpret_cast<const void*>( &b ) );
}
//*************************************************************************************************



//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Backend implementation of the assignment of two arrays with the same storage order.
// \ingroup array
//
// \param lhs The target left-hand side array.
// \param rhs The right-hand side array to be assigned.
// \return void
*/
template< typename TT1  // Type of the left-hand side array
        , typename TT2 > // Type of the right-hand side array
BLAZE_ALWAYS_INLINE void assign_backend( Array<TT1>& lhs, const Array<TT2>& rhs )
{
   BLAZE_FUNCTION_TRACE;

   (~lhs).assign( ~rhs );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Default implementation of the assignment of an array to an array.
// \ingroup array
//
// \param lhs The target left-hand side array.
// \param rhs The right-hand side array to be assigned.
// \return void
//
// This function implements the default assignment of an array to an array.\n
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename TT1  // Type of the left-hand side array
        , typename TT2 > // Type of the right-hand side array
BLAZE_ALWAYS_INLINE void assign( Array<TT1>& lhs, const Array<TT2>& rhs )
{
   BLAZE_FUNCTION_TRACE;

   BLAZE_INTERNAL_ASSERT( (~lhs).rows()    == (~rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( (~lhs).columns() == (~rhs).columns(), "Invalid number of columns" );
   BLAZE_INTERNAL_ASSERT( (~lhs).pages()   == (~rhs).pages(),   "Invalid number of pages"   );

   assign_backend( ~lhs, ~rhs );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Backend implementation of the addition assignment of two arrays with the same
//        storage order.
// \ingroup array
//
// \param lhs The target left-hand side array.
// \param rhs The right-hand side array to be added.
// \return void
*/
template< typename TT1  // Type of the left-hand side array
        , typename TT2 > // Type of the right-hand side array
BLAZE_ALWAYS_INLINE void addAssign_backend( Array<TT1>& lhs, const Array<TT2>& rhs )
{
   BLAZE_FUNCTION_TRACE;

   (~lhs).addAssign( ~rhs );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Default implementation of the addition assignment of an array to an array.
// \ingroup array
//
// \param lhs The target left-hand side array.
// \param rhs The right-hand side array to be added.
// \return void
//
// This function implements the default addition assignment of an array to an array.\n
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename TT1  // Type of the left-hand side array
        , typename TT2 > // Type of the right-hand side array
BLAZE_ALWAYS_INLINE void addAssign( Array<TT1>& lhs, const Array<TT2>& rhs )
{
   BLAZE_FUNCTION_TRACE;

   BLAZE_INTERNAL_ASSERT( (~lhs).rows()    == (~rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( (~lhs).columns() == (~rhs).columns(), "Invalid number of columns" );
   BLAZE_INTERNAL_ASSERT( (~lhs).pages()   == (~rhs).pages(),   "Invalid number of pages"   );

   addAssign_backend( ~lhs, ~rhs );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Backend implementation of the subtraction assignment of two arrays with the same
//        storage order.
// \ingroup array
//
// \param lhs The target left-hand side array.
// \param rhs The right-hand side array to be subtracted.
// \return void
*/
template< typename TT1  // Type of the left-hand side array
        , typename TT2 > // Type of the right-hand side array
BLAZE_ALWAYS_INLINE void subAssign_backend( Array<TT1>& lhs, const Array<TT2>& rhs )
{
   BLAZE_FUNCTION_TRACE;

   (~lhs).subAssign( ~rhs );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Default implementation of the subtraction assignment of an array to array.
// \ingroup array
//
// \param lhs The target left-hand side array.
// \param rhs The right-hand side array to be subtracted.
// \return void
//
// This function implements the default subtraction assignment of an array to an array.\n
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename TT1  // Type of the left-hand side array
        , typename TT2 > // Type of the right-hand side array
BLAZE_ALWAYS_INLINE void subAssign( Array<TT1>& lhs, const Array<TT2>& rhs )
{
   BLAZE_FUNCTION_TRACE;

   BLAZE_INTERNAL_ASSERT( (~lhs).rows()    == (~rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( (~lhs).columns() == (~rhs).columns(), "Invalid number of columns" );
   BLAZE_INTERNAL_ASSERT( (~lhs).pages()   == (~rhs).pages(),   "Invalid number of pages"   );

   subAssign_backend( ~lhs, ~rhs );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Backend implementation of the Schur product assignment of two arrays with the same
//        storage order.
// \ingroup array
//
// \param lhs The target left-hand side array.
// \param rhs The right-hand side array for the Schur product.
// \return void
*/
template< typename TT1  // Type of the left-hand side array
        , typename TT2 > // Type of the right-hand side array
BLAZE_ALWAYS_INLINE void schurAssign_backend( Array<TT1>& lhs, const Array<TT2>& rhs )
{
   BLAZE_FUNCTION_TRACE;

   (~lhs).schurAssign( ~rhs );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Default implementation of the Schur product assignment of an array to array.
// \ingroup array
//
// \param lhs The target left-hand side array.
// \param rhs The right-hand side array for the Schur product.
// \return void
//
// This function implements the default Schur product assignment of an array to an array.\n
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename TT1  // Type of the left-hand side array
        , typename TT2 > // Type of the right-hand side array
BLAZE_ALWAYS_INLINE void schurAssign( Array<TT1>& lhs, const Array<TT2>& rhs )
{
   BLAZE_FUNCTION_TRACE;

   BLAZE_INTERNAL_ASSERT( (~lhs).rows()    == (~rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( (~lhs).columns() == (~rhs).columns(), "Invalid number of columns" );
   BLAZE_INTERNAL_ASSERT( (~lhs).pages()   == (~rhs).pages(),   "Invalid number of pages"   );

   schurAssign_backend( ~lhs, ~rhs );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Default implementation of the multiplication assignment of an array to an array.
// \ingroup array
//
// \param lhs The target left-hand side array.
// \param rhs The right-hand side array to be multiplied.
// \return void
//
// This function implements the default multiplication assignment of an array to an array.\n
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename TT1  // Type of the left-hand side array
        , typename TT2 > // Type of the right-hand side array
BLAZE_ALWAYS_INLINE void multAssign( Array<TT1>& lhs, const Array<TT2>& rhs )
{
   BLAZE_FUNCTION_TRACE;

   BLAZE_INTERNAL_ASSERT( (~lhs).columns() == (~rhs).rows(), "Invalid array sizes" );

   (~lhs).multAssign( ~rhs );
}
/*! \endcond */
//*************************************************************************************************



//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Removal of all restrictions on the data access to the given array.
// \ingroup array
//
// \param array The array to be derestricted.
// \return Reference to the array without access restrictions.
//
// This function removes all restrictions on the data access to the given array. It returns a
// reference to the array that does provide the same interface but does not have any restrictions
// on the data access.\n
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in the violation of invariants, erroneous results and/or in compilation errors.
*/
template< typename TT > // Type of the array
BLAZE_ALWAYS_INLINE TT& derestrict( Array<TT>& array )
{
   return ~array;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Removal of the top-level view on the given array.
// \ingroup column
//
// \param c The given array.
// \return Reference to the array without view.
//
// This function removes the top-level view on the given array and returns a reference to the
// unviewed array.\n
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in the violation of invariants, erroneous results and/or in compilation errors.
*/
template< typename TT > // Type of the array
inline decltype(auto) unview( Array<TT>& array )
{
   return ~array;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Removal of the top-level view on the given array.
// \ingroup column
//
// \param c The given array.
// \return Reference to the array without view.
//
// This function removes the top-level view on the given array and returns a reference to the
// unviewed array.\n
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in the violation of invariants, erroneous results and/or in compilation errors.
*/
template< typename TT > // Type of the array
inline decltype(auto) unview( const Array<TT>& array )
{
   return ~array;
}
/*! \endcond */
//*************************************************************************************************


} // namespace blaze

#endif
