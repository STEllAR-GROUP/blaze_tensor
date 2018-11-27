//=================================================================================================
/*!
//  \file blaze_tensor/math/expressions/Tensor.h
//  \brief Header file for the Tensor base class
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

#ifndef _BLAZE_TENSOR_MATH_EXPRESSIONS_TENSOR_H_
#define _BLAZE_TENSOR_MATH_EXPRESSIONS_TENSOR_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <blaze/math/expressions/Matrix.h>

namespace blaze {

//=================================================================================================
//
//  CLASS DEFINITION
//
//=================================================================================================

//*************************************************************************************************
/*!\defgroup tensor Tensors
// \ingroup math
*/
/*!\brief Base class for tensors.
// \ingroup tensor
//
// The Tensor class is a base class for all dense and sparse tensor classes within the Blaze
// library. It provides an abstraction from the actual type of the tensor, but enables a
// conversion back to this type via the 'Curiously Recurring Template Pattern' (CRTP).
*/
template< typename TT > // Type of the tensor
struct Tensor
{
   //**Type definitions****************************************************************************
   using TensorType = TT;  //!< Type of the tensor.
   //**********************************************************************************************

   //**Non-const conversion operator***************************************************************
   /*!\brief Conversion operator for non-constant tensors.
   //
   // \return Reference of the actual type of the tensor.
   */
   BLAZE_ALWAYS_INLINE constexpr TensorType& operator~() noexcept {
      return *static_cast<TensorType*>( this );
   }
   //**********************************************************************************************

   //**Const conversion operator*******************************************************************
   /*!\brief Conversion operator for constant tensors.
   //
   // \return Constant reference of the actual type of the tensor.
   */
   BLAZE_ALWAYS_INLINE constexpr const TensorType& operator~() const noexcept {
      return *static_cast<const TensorType*>( this );
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
/*!\brief Multiplication assignment operator for the multiplication of two tensors (\f$ A*=B \f$).
// \ingroup tensor
//
// \param lhs The left-hand side tensor for the multiplication.
// \param rhs The right-hand side tensor for the multiplication.
// \return Reference to the left-hand side tensor.
// \exception std::invalid_argument Tensor sizes do not match.
//
// In case the current number of columns of \a lhs and the current number of rows of \a rhs
// don't match, a \a std::invalid_argument is thrown.
*/
template< typename TT1  // Type of the left-hand side tensor
        , typename TT2 > // Type of the right-hand side tensor
inline TT1& operator*=( Tensor<TT1>& lhs, const Tensor<TT2>& rhs )
{
   ResultType_t<TT1> tmp( (~lhs) * (~rhs) );
   (~lhs) = std::move( tmp );
   return (~lhs);
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Multiplication assignment operator for the multiplication of a temporary tensor with
//        another tensor (\f$ A*=B \f$).
// \ingroup tensor
//
// \param lhs The left-hand side temporary tensor for the multiplication.
// \param rhs The right-hand side tensor for the multiplication.
// \return Reference to the left-hand side tensor.
// \exception std::invalid_argument Tensor sizes do not match.
//
// In case the current number of columns of \a lhs and the current number of rows of \a rhs
// don't match, a \a std::invalid_argument is thrown.
*/
template< typename TT1  // Type of the left-hand side tensor
        , typename TT2 > // Type of the right-hand side tensor
inline TT1& operator*=( Tensor<TT1>&& lhs, const Tensor<TT2>& rhs )
{
   return (~lhs) *= (~rhs);
}
/*! \endcond */
//*************************************************************************************************



//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Predict invariant violations by setting a single element of a tensor.
// \ingroup tensor
//
// \param mat The target tensor.
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
template< typename MT    // Type of the tensor
        , typename ET >  // Type of the element
BLAZE_ALWAYS_INLINE bool trySet( const Tensor<MT>& mat, size_t k, size_t i, size_t j, const ET& value )
{
   BLAZE_INTERNAL_ASSERT( i < (~mat).rows(), "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( j < (~mat).columns(), "Invalid column access index" );
   BLAZE_INTERNAL_ASSERT( k < (~mat).pages(), "Invalid page access index" );

   UNUSED_PARAMETER( mat, k, i, j, value );

   return true;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Predict invariant violations by adding to a single element of a tensor.
// \ingroup tensor
//
// \param mat The target tensor.
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
template< typename MT    // Type of the tensor
        , typename ET >  // Type of the element
BLAZE_ALWAYS_INLINE bool tryAdd( const Tensor<MT>& mat, size_t k, size_t i, size_t j, const ET& value )
{
   BLAZE_INTERNAL_ASSERT( i < (~mat).rows(), "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( j < (~mat).columns(), "Invalid column access index" );
   BLAZE_INTERNAL_ASSERT( k < (~mat).pages(), "Invalid page access index" );

   UNUSED_PARAMETER( mat, k, i, j, value );

   return true;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Predict invariant violations by subtracting from a single element of a tensor.
// \ingroup tensor
//
// \param mat The target tensor.
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
template< typename MT    // Type of the tensor
        , typename ET >  // Type of the element
BLAZE_ALWAYS_INLINE bool trySub( const Tensor<MT>& mat, size_t k, size_t i, size_t j, const ET& value )
{
   BLAZE_INTERNAL_ASSERT( i < (~mat).rows(), "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( j < (~mat).columns(), "Invalid column access index" );
   BLAZE_INTERNAL_ASSERT( k < (~mat).pages(), "Invalid page access index" );

   UNUSED_PARAMETER( mat, k, i, j, value );

   return true;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Predict invariant violations by scaling a single element of a tensor.
// \ingroup tensor
//
// \param tens The target tensor.
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
template< typename MT    // Type of the tensor
        , typename ET >  // Type of the element
BLAZE_ALWAYS_INLINE bool tryMult( const Tensor<MT>& tens, size_t k, size_t i, size_t j, const ET& value )
{
   BLAZE_INTERNAL_ASSERT( i < (~tens).rows(), "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( j < (~tens).columns(), "Invalid column access index" );
   BLAZE_INTERNAL_ASSERT( k < (~tens).pages(), "Invalid page access index" );

   UNUSED_PARAMETER( tens, k, i, j, value );

   return true;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Predict invariant violations by scaling a range of elements of a tensor.
// \ingroup tensor
//
// \param tens The target tensor.
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
template< typename MT    // Type of the tensor
        , typename ET >  // Type of the element
BLAZE_ALWAYS_INLINE bool
   tryMult( const Tensor<MT>& tens, size_t row, size_t column, size_t page, size_t o, size_t m, size_t n, const ET& value )
{
   BLAZE_INTERNAL_ASSERT( row <= (~tens).rows(), "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( column <= (~tens).columns(), "Invalid column access index" );
   BLAZE_INTERNAL_ASSERT( page <= (~tens).pages(), "Invalid page access index" );
   BLAZE_INTERNAL_ASSERT( row + m <= (~tens).rows(), "Invalid number of rows" );
   BLAZE_INTERNAL_ASSERT( column + n <= (~tens).columns(), "Invalid number of columns" );
   BLAZE_INTERNAL_ASSERT( page + o <= (~tens).pages(), "Invalid number of pages" );

   UNUSED_PARAMETER( tens, page, row, column, o, m, n, value );

   return true;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Predict invariant violations by scaling a single element of a tensor.
// \ingroup tensor
//
// \param mat The target tensor.
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
template< typename MT    // Type of the tensor
        , typename ET >  // Type of the element
BLAZE_ALWAYS_INLINE bool tryDiv( const Tensor<MT>& mat, size_t k, size_t i, size_t j, const ET& value )
{
   BLAZE_INTERNAL_ASSERT( i < (~mat).rows(), "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( j < (~mat).columns(), "Invalid column access index" );
   BLAZE_INTERNAL_ASSERT( k < (~tens).pages(), "Invalid page access index" );

   UNUSED_PARAMETER( mat, k, i, j, value );

   return true;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Predict invariant violations by scaling a range of elements of a tensor.
// \ingroup tensor
//
// \param mat The target tensor.
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
template< typename MT    // Type of the tensor
        , typename ET >  // Type of the element
BLAZE_ALWAYS_INLINE bool
   tryDiv( const Tensor<MT>& tens, size_t row, size_t column, size_t page, size_t o, size_t m, size_t n, const ET& value )
{
   BLAZE_INTERNAL_ASSERT( row <= (~tens).rows(), "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( column <= (~tens).columns(), "Invalid column access index" );
   BLAZE_INTERNAL_ASSERT( page <= (~tens).pages(), "Invalid page access index" );
   BLAZE_INTERNAL_ASSERT( row + m <= (~tens).rows(), "Invalid number of rows" );
   BLAZE_INTERNAL_ASSERT( column + n <= (~tens).columns(), "Invalid number of columns" );
   BLAZE_INTERNAL_ASSERT( page + o <= (~tens).pages(), "Invalid number of pages" );

   UNUSED_PARAMETER( tens, page, row, column, o, m, n, value );

   return true;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Predict invariant violations by the assignment of a tensor to a tensor.
// \ingroup tensor
//
// \param lhs The target left-hand side tensor.
// \param rhs The right-hand side tensor to be assigned.
// \param row The row index of the first element to be modified.
// \param column The column index of the first element to be modified.
// \return \a true in case the assignment would be successful, \a false if not.
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename MT  // Type of the left-hand side tensor
        , typename VT > // Type of the right-hand side matrix
BLAZE_ALWAYS_INLINE bool tryAssign( const Tensor<MT>& lhs, const Matrix<VT,false>& rhs,
                                    size_t row, size_t column, size_t page )
{
   BLAZE_INTERNAL_ASSERT( row <= (~lhs).rows(), "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( column <= (~lhs).columns(), "Invalid column access index" );
   BLAZE_INTERNAL_ASSERT( page <= (~lhs).pages(), "Invalid page access index" );
   BLAZE_INTERNAL_ASSERT( row + (~rhs).size() <= (~lhs).rows(), "Invalid number of rows" );
   BLAZE_INTERNAL_ASSERT( column + (~rhs).size() <= (~lhs).columns(), "Invalid number of columns" );
   BLAZE_INTERNAL_ASSERT( page + 1 <= (~lhs).pages(), "Invalid number of pages" );

   UNUSED_PARAMETER( lhs, rhs, page, row, column );

   return true;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Predict invariant violations by the assignment of a tensor to a tensor.
// \ingroup tensor
//
// \param lhs The target left-hand side tensor.
// \param rhs The right-hand side tensor to be assigned.
// \param row The row index of the first element to be modified.
// \param column The column index of the first element to be modified.
// \param page The page index of the first element to be modified.
// \return \a true in case the assignment would be successful, \a false if not.
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename TT1  // Type of the left-hand side tensor
        , typename TT2 > // Type of the right-hand side tensor
BLAZE_ALWAYS_INLINE bool tryAssign( const Tensor<TT1>& lhs, const Tensor<TT2>& rhs,
                                    size_t row, size_t column, size_t page )
{
   BLAZE_INTERNAL_ASSERT( row <= (~lhs).rows(), "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( column <= (~lhs).columns(), "Invalid column access index" );
   BLAZE_INTERNAL_ASSERT( page <= (~lhs).pages(), "Invalid page access index" );
   BLAZE_INTERNAL_ASSERT( row + (~rhs).rows() <= (~lhs).rows(), "Invalid number of rows" );
   BLAZE_INTERNAL_ASSERT( column + (~rhs).columns() <= (~lhs).columns(), "Invalid number of columns" );
   BLAZE_INTERNAL_ASSERT( page + (~rhs).pages() <= (~lhs).pages(), "Invalid number of pages" );

   UNUSED_PARAMETER( lhs, rhs, page, row, column );

   return true;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Predict invariant violations by the addition assignment of a matrix to a tensor.
// \ingroup tensor
//
// \param lhs The target left-hand side tensor.
// \param rhs The right-hand side matrix to be added.
// \param row The row index of the first element to be modified.
// \param column The column index of the first element to be modified.
// \return \a true in case the assignment would be successful, \a false if not.
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename MT  // Type of the left-hand side tensor
        , typename VT > // Type of the right-hand side matrix
BLAZE_ALWAYS_INLINE bool tryAddAssign( const Tensor<MT>& lhs, const Matrix<VT,false>& rhs,
                                       size_t row, size_t column, size_t page )
{
   BLAZE_INTERNAL_ASSERT( row <= (~lhs).rows(), "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( column <= (~lhs).columns(), "Invalid column access index" );
   BLAZE_INTERNAL_ASSERT( page <= (~lhs).pages(), "Invalid page access index" );
   BLAZE_INTERNAL_ASSERT( row + (~rhs).size() <= (~lhs).rows(), "Invalid number of rows" );
   BLAZE_INTERNAL_ASSERT( column + (~rhs).size() <= (~lhs).columns(), "Invalid number of columns" );
   BLAZE_INTERNAL_ASSERT( page + 1 <= (~lhs).pages(), "Invalid number of pages" );

   UNUSED_PARAMETER( lhs, rhs, page, row, column );

   return true;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Predict invariant violations by the addition assignment of a matrix to the band of a tensor.
// \ingroup tensor
//
// \param lhs The target left-hand side tensor.
// \param rhs The right-hand side matrix to be added.
// \param band The index of the band the right-hand side matrix is assigned to.
// \param row The row index of the first element to be modified.
// \param column The column index of the first element to be modified.
// \return \a true in case the assignment would be successful, \a false if not.
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename MT  // Type of the left-hand side tensor
        , typename VT > // Type of the right-hand side matrix
BLAZE_ALWAYS_INLINE bool tryAddAssign( const Tensor<MT>& lhs, const Matrix<VT,false>& rhs,
                                       ptrdiff_t band, size_t row, size_t column, size_t page )
{
   BLAZE_INTERNAL_ASSERT( row <= (~lhs).rows(), "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( column <= (~lhs).columns(), "Invalid column access index" );
   BLAZE_INTERNAL_ASSERT( page <= (~lhs).pages(), "Invalid page access index" );
   BLAZE_INTERNAL_ASSERT( row + (~rhs).size() <= (~lhs).rows(), "Invalid number of rows" );
   BLAZE_INTERNAL_ASSERT( column + (~rhs).size() <= (~lhs).columns(), "Invalid number of columns" );
   BLAZE_INTERNAL_ASSERT( page + 1 <= (~lhs).pages(), "Invalid number of pages" );

   UNUSED_PARAMETER( lhs, rhs, band, page, row, column );

   return true;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Predict invariant violations by the addition assignment of a tensor to a tensor.
// \ingroup tensor
//
// \param lhs The target left-hand side tensor.
// \param rhs The right-hand side tensor to be added.
// \param row The row index of the first element to be modified.
// \param column The column index of the first element to be modified.
// \return \a true in case the assignment would be successful, \a false if not.
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename TT1  // Type of the left-hand side tensor
        , typename TT2 > // Type of the right-hand side tensor
BLAZE_ALWAYS_INLINE bool tryAddAssign( const Tensor<TT1>& lhs, const Tensor<TT2>& rhs,
                                       size_t row, size_t column, size_t page )
{
   BLAZE_INTERNAL_ASSERT( row <= (~lhs).rows(), "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( column <= (~lhs).columns(), "Invalid column access index" );
   BLAZE_INTERNAL_ASSERT( page <= (~lhs).pages(), "Invalid page access index" );
   BLAZE_INTERNAL_ASSERT( row + (~rhs).rows() <= (~lhs).rows(), "Invalid number of rows" );
   BLAZE_INTERNAL_ASSERT( column + (~rhs).columns() <= (~lhs).columns(), "Invalid number of columns" );
   BLAZE_INTERNAL_ASSERT( page + (~rhs).pages() <= (~lhs).pages(), "Invalid number of pages" );

   UNUSED_PARAMETER( lhs, rhs, page, row, column );

   return true;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Predict invariant violations by the subtraction assignment of a matrix to a tensor.
// \ingroup tensor
//
// \param lhs The target left-hand side tensor.
// \param rhs The right-hand side matrix to be subtracted.
// \param row The row index of the first element to be modified.
// \param column The column index of the first element to be modified.
// \return \a true in case the assignment would be successful, \a false if not.
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename MT  // Type of the left-hand side tensor
        , typename VT > // Type of the right-hand side matrix
BLAZE_ALWAYS_INLINE bool trySubAssign( const Tensor<MT>& lhs, const Matrix<VT,false>& rhs,
                                       size_t row, size_t column, size_t page )
{
   BLAZE_INTERNAL_ASSERT( row <= (~lhs).rows(), "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( column <= (~lhs).columns(), "Invalid column access index" );
   BLAZE_INTERNAL_ASSERT( page <= (~lhs).pages(), "Invalid page access index" );
   BLAZE_INTERNAL_ASSERT( row + (~rhs).size() <= (~lhs).rows(), "Invalid number of rows" );
   BLAZE_INTERNAL_ASSERT( column + (~rhs).size() <= (~lhs).columns(), "Invalid number of columns" );
   BLAZE_INTERNAL_ASSERT( page + 1 <= (~lhs).pages(), "Invalid number of pages" );

   UNUSED_PARAMETER( lhs, rhs, page, row, column );

   return true;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Predict invariant violations by the subtraction assignment of a matrix to the band of
//        a tensor.
// \ingroup tensor
//
// \param lhs The target left-hand side tensor.
// \param rhs The right-hand side matrix to be subtracted.
// \param band The index of the band the right-hand side matrix is assigned to.
// \param row The row index of the first element to be modified.
// \param column The column index of the first element to be modified.
// \return \a true in case the assignment would be successful, \a false if not.
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename MT  // Type of the left-hand side tensor
        , typename VT > // Type of the right-hand side matrix
BLAZE_ALWAYS_INLINE bool trySubAssign( const Tensor<MT>& lhs, const Matrix<VT,false>& rhs,
                                       ptrdiff_t band, size_t row, size_t column, size_t page )
{
   BLAZE_INTERNAL_ASSERT( row <= (~lhs).rows(), "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( column <= (~lhs).columns(), "Invalid column access index" );
   BLAZE_INTERNAL_ASSERT( page <= (~lhs).pages(), "Invalid page access index" );
   BLAZE_INTERNAL_ASSERT( row + (~rhs).size() <= (~lhs).rows(), "Invalid number of rows" );
   BLAZE_INTERNAL_ASSERT( column + (~rhs).size() <= (~lhs).columns(), "Invalid number of columns" );
   BLAZE_INTERNAL_ASSERT( page + 1 <= (~lhs).pages(), "Invalid number of pages" );

   UNUSED_PARAMETER( lhs, rhs, band, page, row, column );

   return true;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Predict invariant violations by the subtraction assignment of a tensor to a tensor.
// \ingroup tensor
//
// \param lhs The target left-hand side tensor.
// \param rhs The right-hand side tensor to be subtracted.
// \param row The row index of the first element to be modified.
// \param column The column index of the first element to be modified.
// \return \a true in case the assignment would be successful, \a false if not.
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename TT1  // Type of the left-hand side tensor
        , typename TT2 > // Type of the right-hand side tensor
BLAZE_ALWAYS_INLINE bool trySubAssign( const Tensor<TT1>& lhs, const Tensor<TT2>& rhs,
                                       size_t row, size_t column, size_t page )
{
   BLAZE_INTERNAL_ASSERT( row <= (~lhs).rows(), "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( column <= (~lhs).columns(), "Invalid column access index" );
   BLAZE_INTERNAL_ASSERT( page <= (~lhs).pages(), "Invalid page access index" );
   BLAZE_INTERNAL_ASSERT( row + (~rhs).size() <= (~lhs).rows(), "Invalid number of rows" );
   BLAZE_INTERNAL_ASSERT( column + (~rhs).size() <= (~lhs).columns(), "Invalid number of columns" );
   BLAZE_INTERNAL_ASSERT( page + (~rhs).pages() <= (~lhs).pages(), "Invalid number of pages" );

   UNUSED_PARAMETER( lhs, rhs, page, row, column );

   return true;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Predict invariant violations by the multiplication assignment of a matrix to a tensor.
// \ingroup tensor
//
// \param lhs The target left-hand side tensor.
// \param rhs The right-hand side matrix to be multiplied.
// \param row The row index of the first element to be modified.
// \param column The column index of the first element to be modified.
// \return \a true in case the assignment would be successful, \a false if not.
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename MT  // Type of the left-hand side tensor
        , typename VT > // Type of the right-hand side matrix
BLAZE_ALWAYS_INLINE bool tryMultAssign( const Tensor<MT>& lhs, const Matrix<VT,false>& rhs,
                                        size_t row, size_t column, size_t page )
{
   BLAZE_INTERNAL_ASSERT( row <= (~lhs).rows(), "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( column <= (~lhs).columns(), "Invalid column access index" );
   BLAZE_INTERNAL_ASSERT( page <= (~lhs).pages(), "Invalid page access index" );
   BLAZE_INTERNAL_ASSERT( row + (~rhs).size() <= (~lhs).rows(), "Invalid number of rows" );
   BLAZE_INTERNAL_ASSERT( column + (~rhs).size() <= (~lhs).columns(), "Invalid number of columns" );
   BLAZE_INTERNAL_ASSERT( page + 1 <= (~lhs).pages(), "Invalid number of pages" );

   UNUSED_PARAMETER( lhs, rhs, page, row, column );

   return true;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Predict invariant violations by the multiplication assignment of a matrix to the band
//        of a tensor.
// \ingroup tensor
//
// \param lhs The target left-hand side tensor.
// \param rhs The right-hand side matrix to be multiplied.
// \param band The index of the band the right-hand side matrix is assigned to.
// \param row The row index of the first element to be modified.
// \param column The column index of the first element to be modified.
// \return \a true in case the assignment would be successful, \a false if not.
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename MT  // Type of the left-hand side tensor
        , typename VT > // Type of the right-hand side matrix
BLAZE_ALWAYS_INLINE bool tryMultAssign( const Tensor<MT>& lhs, const Matrix<VT,false>& rhs,
                                        ptrdiff_t band, size_t row, size_t column, size_t page )
{
   BLAZE_INTERNAL_ASSERT( row <= (~lhs).rows(), "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( column <= (~lhs).columns(), "Invalid column access index" );
   BLAZE_INTERNAL_ASSERT( page <= (~lhs).pages(), "Invalid page access index" );
   BLAZE_INTERNAL_ASSERT( row + (~rhs).size() <= (~lhs).rows(), "Invalid number of rows" );
   BLAZE_INTERNAL_ASSERT( column + (~rhs).size() <= (~lhs).columns(), "Invalid number of columns" );
   BLAZE_INTERNAL_ASSERT( page + 1 <= (~lhs).pages(), "Invalid number of pages" );

   UNUSED_PARAMETER( lhs, rhs, band, page, row, column );

   return true;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Predict invariant violations by the Schur product assignment of a tensor to a tensor.
// \ingroup tensor
//
// \param lhs The target left-hand side tensor.
// \param rhs The right-hand side tensor for the Schur product.
// \param row The row index of the first element to be modified.
// \param column The column index of the first element to be modified.
// \return \a true in case the assignment would be successful, \a false if not.
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename TT1  // Type of the left-hand side tensor
        , typename TT2 > // Type of the right-hand side tensor
BLAZE_ALWAYS_INLINE bool trySchurAssign( const Tensor<TT1>& lhs, const Tensor<TT2>& rhs,
                                         size_t row, size_t column, size_t page )
{
   BLAZE_INTERNAL_ASSERT( row <= (~lhs).rows(), "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( column <= (~lhs).columns(), "Invalid column access index" );
   BLAZE_INTERNAL_ASSERT( page <= (~lhs).pages(), "Invalid page access index" );
   BLAZE_INTERNAL_ASSERT( row + (~rhs).size() <= (~lhs).rows(), "Invalid number of rows" );
   BLAZE_INTERNAL_ASSERT( column + (~rhs).size() <= (~lhs).columns(), "Invalid number of columns" );
   BLAZE_INTERNAL_ASSERT( page + (~rhs).pages() <= (~lhs).pages(), "Invalid number of pages" );

   UNUSED_PARAMETER( lhs, rhs, page, row, column );

   return true;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Predict invariant violations by the Schur multiplication assignment of a matrix to a tensor.
// \ingroup tensor
//
// \param lhs The target left-hand side tensor.
// \param rhs The right-hand side matrix to be Schur-multiplied.
// \param band The index of the band the right-hand side matrix is assigned to.
// \param row The row index of the first element to be modified.
// \param column The column index of the first element to be modified.
// \return \a true in case the assignment would be successful, \a false if not.
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename MT  // Type of the left-hand side tensor
        , typename VT > // Type of the right-hand side matrix
BLAZE_ALWAYS_INLINE bool trySchurAssign( const Tensor<MT>& lhs, const Matrix<VT,false>& rhs,
                                        size_t row, size_t column, size_t page )
{
   BLAZE_INTERNAL_ASSERT( row <= (~lhs).rows(), "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( column <= (~lhs).columns(), "Invalid column access index" );
   BLAZE_INTERNAL_ASSERT( page <= (~lhs).pages(), "Invalid page access index" );
   BLAZE_INTERNAL_ASSERT( row + (~rhs).size() <= (~lhs).rows(), "Invalid number of rows" );
   BLAZE_INTERNAL_ASSERT( column + (~rhs).size() <= (~lhs).columns(), "Invalid number of columns" );
   BLAZE_INTERNAL_ASSERT( page + 1 <= (~lhs).pages(), "Invalid number of pages" );

   UNUSED_PARAMETER( lhs, rhs, page, row, column );

   return true;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Predict invariant violations by the division assignment of a matrix to a tensor.
// \ingroup tensor
//
// \param lhs The target left-hand side tensor.
// \param rhs The right-hand side matrix divisor.
// \param row The row index of the first element to be modified.
// \param column The column index of the first element to be modified.
// \return \a true in case the assignment would be successful, \a false if not.
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename MT  // Type of the left-hand side tensor
        , typename VT > // Type of the right-hand side matrix
BLAZE_ALWAYS_INLINE bool tryDivAssign( const Tensor<MT>& lhs, const Matrix<VT,false>& rhs,
                                       size_t row, size_t column, size_t page )
{
   BLAZE_INTERNAL_ASSERT( row <= (~lhs).rows(), "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( column <= (~lhs).columns(), "Invalid column access index" );
   BLAZE_INTERNAL_ASSERT( page <= (~lhs).pages(), "Invalid page access index" );
   BLAZE_INTERNAL_ASSERT( row + (~rhs).size() <= (~lhs).rows(), "Invalid number of rows" );
   BLAZE_INTERNAL_ASSERT( column + (~rhs).size() <= (~lhs).columns(), "Invalid number of columns" );
   BLAZE_INTERNAL_ASSERT( page + 1 <= (~lhs).pages(), "Invalid number of pages" );

   UNUSED_PARAMETER( lhs, rhs, page, row, column );

   return true;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Predict invariant violations by the division assignment of a matrix to the band of
//        a tensor.
// \ingroup tensor
//
// \param lhs The target left-hand side tensor.
// \param rhs The right-hand side matrix divisor.
// \param band The index of the band the right-hand side matrix is assigned to.
// \param row The row index of the first element to be modified.
// \param column The column index of the first element to be modified.
// \return \a true in case the assignment would be successful, \a false if not.
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename MT  // Type of the left-hand side tensor
        , typename VT > // Type of the right-hand side matrix
BLAZE_ALWAYS_INLINE bool tryDivAssign( const Tensor<MT>& lhs, const Matrix<VT,false>& rhs,
                                       ptrdiff_t band, size_t row, size_t column, size_t page )
{
   BLAZE_INTERNAL_ASSERT( row <= (~lhs).rows(), "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( column <= (~lhs).columns(), "Invalid column access index" );
   BLAZE_INTERNAL_ASSERT( page <= (~lhs).pages(), "Invalid page access index" );
   BLAZE_INTERNAL_ASSERT( row + (~rhs).size() <= (~lhs).rows(), "Invalid number of rows" );
   BLAZE_INTERNAL_ASSERT( column + (~rhs).size() <= (~lhs).columns(), "Invalid number of columns" );
   BLAZE_INTERNAL_ASSERT( page + 1 <= (~lhs).pages(), "Invalid number of pages" );

   UNUSED_PARAMETER( lhs, rhs, band, page, row, column );

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
/*!\name Tensor global functions */
//@{
template< typename MT >
BLAZE_ALWAYS_INLINE typename MT::Iterator begin( Tensor<MT>& tensor, size_t i, size_t k );

template< typename MT >
BLAZE_ALWAYS_INLINE typename MT::ConstIterator begin( const Tensor<MT>& tensor, size_t i, size_t k );

template< typename MT >
BLAZE_ALWAYS_INLINE typename MT::ConstIterator cbegin( const Tensor<MT>& tensor, size_t i, size_t k );

template< typename MT >
BLAZE_ALWAYS_INLINE typename MT::Iterator end( Tensor<MT>& tensor, size_t i, size_t k );

template< typename MT >
BLAZE_ALWAYS_INLINE typename MT::ConstIterator end( const Tensor<MT>& tensor, size_t i, size_t k );

template< typename MT >
BLAZE_ALWAYS_INLINE typename MT::ConstIterator cend( const Tensor<MT>& tensor, size_t i, size_t k );

template< typename MT >
BLAZE_ALWAYS_INLINE constexpr size_t rows( const Tensor<MT>& tensor ) noexcept;

template< typename MT >
BLAZE_ALWAYS_INLINE constexpr size_t columns( const Tensor<MT>& tensor ) noexcept;

template< typename MT >
BLAZE_ALWAYS_INLINE constexpr size_t pages( const Tensor<MT>& tensor ) noexcept;

template< typename MT >
BLAZE_ALWAYS_INLINE constexpr size_t size( const Tensor<MT>& tensor ) noexcept;

template< typename MT >
BLAZE_ALWAYS_INLINE size_t capacity( const Tensor<MT>& tensor ) noexcept;

template< typename MT >
BLAZE_ALWAYS_INLINE size_t capacity( const Tensor<MT>& tensor, size_t i, size_t k ) noexcept;

template< typename MT >
BLAZE_ALWAYS_INLINE size_t nonZeros( const Tensor<MT>& tensor );

template< typename MT >
BLAZE_ALWAYS_INLINE size_t nonZeros( const Tensor<MT>& tensor, size_t i, size_t k );

template< typename MT >
BLAZE_ALWAYS_INLINE void resize( Tensor<MT>& tensor, size_t rows, size_t columns, size_t pages, bool preserve=true );

template< typename MT >
BLAZE_ALWAYS_INLINE void shrinkToFit( Tensor<MT>& tensor );

// template< typename MT >
// BLAZE_ALWAYS_INLINE void transpose( Tensor<MT>& tensor );
//
// template< typename MT >
// BLAZE_ALWAYS_INLINE void ctranspose( Tensor<MT>& tensor );

template< typename MT >
inline const typename MT::ResultType evaluate( const Tensor<MT>& tensor );

template< typename MT >
BLAZE_ALWAYS_INLINE constexpr bool isEmpty( const Tensor<MT>& tensor ) noexcept;

template< typename MT >
BLAZE_ALWAYS_INLINE bool isSquare( const Tensor<MT>& tensor ) noexcept;

template< typename TT1, typename TT2 >
BLAZE_ALWAYS_INLINE bool isSame( const Tensor<TT1>& a, const Tensor<TT2>& b ) noexcept;
//@}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns an iterator to the first element of row/column \a i.
// \ingroup tensor
//
// \param tensor The given dense or sparse tensor.
// \param i The row/column index.
// \return Iterator to the first element of row/column \a i.
//
// This function returns a row/column iterator to the first element of row/column \a i. In case
// the given tensor is a row-major tensor the function returns an iterator to the first element
// of row \a i, in case it is a column-major tensor the function returns an iterator to the first
// element of column \a i.
*/
template< typename MT > // Type of the tensor
BLAZE_ALWAYS_INLINE typename MT::Iterator begin( Tensor<MT>& tensor, size_t i, size_t k )
{
   return (~tensor).begin(i, k);
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns an iterator to the first element of row/column \a i.
// \ingroup tensor
//
// \param tensor The given dense or sparse tensor.
// \param i The row/column index.
// \return Iterator to the first element of row/column \a i.
//
// This function returns a row/column iterator to the first element of row/column \a i. In case
// the given tensor is a row-major tensor the function returns an iterator to the first element
// of row \a i, in case it is a column-major tensor the function returns an iterator to the first
// element of column \a i.
*/
template< typename MT > // Type of the tensor
BLAZE_ALWAYS_INLINE typename MT::ConstIterator begin( const Tensor<MT>& tensor, size_t i, size_t k )
{
   return (~tensor).begin(i, k);
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns an iterator to the first element of row/column \a i.
// \ingroup tensor
//
// \param tensor The given dense or sparse tensor.
// \param i The row/column index.
// \return Iterator to the first element of row/column \a i.
//
// This function returns a row/column iterator to the first element of row/column \a i. In case
// the given tensor is a row-major tensor the function returns an iterator to the first element
// of row \a i, in case it is a column-major tensor the function returns an iterator to the first
// element of column \a i.
*/
template< typename MT > // Type of the tensor
BLAZE_ALWAYS_INLINE typename MT::ConstIterator cbegin( const Tensor<MT>& tensor, size_t i, size_t k )
{
   return (~tensor).cbegin(i, k);
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns an iterator just past the last element of row/column \a i.
// \ingroup tensor
//
// \param tensor The given dense or sparse tensor.
// \param i The row/column index.
// \return Iterator just past the last element of row/column \a i.
//
// This function returns an row/column iterator just past the last element of row/column \a i.
// In case the given tensor is a row-major tensor the function returns an iterator just past
// the last element of row \a i, in case it is a column-major tensor the function returns an
// iterator just past the last element of column \a i.
*/
template< typename MT > // Type of the tensor
BLAZE_ALWAYS_INLINE typename MT::Iterator end( Tensor<MT>& tensor, size_t i, size_t k )
{
   return (~tensor).end(i, k);
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns an iterator just past the last element of row/column \a i.
// \ingroup tensor
//
// \param tensor The given dense or sparse tensor.
// \param i The row/column index.
// \return Iterator just past the last element of row/column \a i.
//
// This function returns an row/column iterator just past the last element of row/column \a i.
// In case the given tensor is a row-major tensor the function returns an iterator just past
// the last element of row \a i, in case it is a column-major tensor the function returns an
// iterator just past the last element of column \a i.
*/
template< typename MT > // Type of the tensor
BLAZE_ALWAYS_INLINE typename MT::ConstIterator end( const Tensor<MT>& tensor, size_t i, size_t k )
{
   return (~tensor).end(i, k);
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns an iterator just past the last element of row/column \a i.
// \ingroup tensor
//
// \param tensor The given dense or sparse tensor.
// \param i The row/column index.
// \return Iterator just past the last element of row/column \a i.
//
// This function returns an row/column iterator just past the last element of row/column \a i.
// In case the given tensor is a row-major tensor the function returns an iterator just past
// the last element of row \a i, in case it is a column-major tensor the function returns an
// iterator just past the last element of column \a i.
*/
template< typename MT > // Type of the tensor
BLAZE_ALWAYS_INLINE typename MT::ConstIterator cend( const Tensor<MT>& tensor, size_t i, size_t k )
{
   return (~tensor).cend(i, k);
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns the current number of rows of the tensor.
// \ingroup tensor
//
// \param tensor The given tensor.
// \return The number of rows of the tensor.
*/
template< typename MT > // Type of the tensor
BLAZE_ALWAYS_INLINE constexpr size_t rows( const Tensor<MT>& tensor ) noexcept
{
   return (~tensor).rows();
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns the current number of columns of the tensor.
// \ingroup tensor
//
// \param tensor The given tensor.
// \return The number of columns of the tensor.
*/
template< typename MT > // Type of the tensor
BLAZE_ALWAYS_INLINE constexpr size_t columns( const Tensor<MT>& tensor ) noexcept
{
   return (~tensor).columns();
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns the current number of pages of the tensor.
// \ingroup tensor
//
// \param tensor The given tensor.
// \return The number of pages of the tensor.
*/
template< typename MT > // Type of the tensor
BLAZE_ALWAYS_INLINE constexpr size_t pages( const Tensor<MT>& tensor ) noexcept
{
   return (~tensor).pages();
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns the total number of elements of the tensor.
// \ingroup tensor
//
// \param tensor The given tensor.
// \return The total number of elements of the tensor.
*/
template< typename MT > // Type of the tensor
BLAZE_ALWAYS_INLINE constexpr size_t size( const Tensor<MT>& tensor ) noexcept
{
   return (~tensor).rows() * (~tensor).columns() * (~tensor).pages();
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns the maximum capacity of the tensor.
// \ingroup tensor
//
// \param tensor The given tensor.
// \return The capacity of the tensor.
*/
template< typename MT > // Type of the tensor
BLAZE_ALWAYS_INLINE size_t capacity( const Tensor<MT>& tensor ) noexcept
{
   return (~tensor).capacity();
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns the current capacity of the specified row/column.
// \ingroup tensor
//
// \param tensor The given tensor.
// \param i The index of the row/column.
// \return The current capacity of row/column \a i.
//
// This function returns the current capacity of the specified row/column. In case the
// storage order is set to \a rowMajor the function returns the capacity of row \a i,
// in case the storage flag is set to \a columnMajor the function returns the capacity
// of column \a i.
*/
template< typename MT > // Type of the tensor
BLAZE_ALWAYS_INLINE size_t capacity( const Tensor<MT>& tensor, size_t i, size_t k ) noexcept
{
   return (~tensor).capacity( i, k );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns the total number of non-zero elements in the tensor
// \ingroup tensor
//
// \param tensor The given tensor.
// \return The number of non-zero elements in the dense tensor.
*/
template< typename MT > // Type of the tensor
BLAZE_ALWAYS_INLINE size_t nonZeros( const Tensor<MT>& tensor )
{
   return (~tensor).nonZeros();
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns the number of non-zero elements in the specified row/column.
// \ingroup tensor
//
// \param tensor The given tensor.
// \param i The index of the row/column.
// \return The number of non-zero elements of row/column \a i.
//
// This function returns the current number of non-zero elements in the specified row/column.
// In case the storage order is set to \a rowMajor the function returns the number of non-zero
// elements in row \a i, in case the storage flag is set to \a columnMajor the function returns
// the number of non-zero elements in column \a i.
*/
template< typename MT > // Type of the tensor
BLAZE_ALWAYS_INLINE size_t nonZeros( const Tensor<MT>& tensor, size_t i, size_t k )
{
   return (~tensor).nonZeros( i, k );
}
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Backend implementation of the \c resize() function for non-resizable tensors.
// \ingroup tensor
//
// \param tensor The given tensor to be resized.
// \param m The new number of rows of the tensor.
// \param n The new number of columns of the tensor.
// \param preserve \a true if the old values of the tensor should be preserved, \a false if not.
// \return void
// \exception std::invalid_argument Tensor cannot be resized.
//
// This function tries to change the number of rows and columns of a non-resizable tensor. Since
// the tensor cannot be resized, in case the specified number of rows and columns is not identical
// to the current number of rows and columns of the tensor, a \a std::invalid_argument exception
// is thrown.
*/
template< typename MT > // Type of the tensor
BLAZE_ALWAYS_INLINE DisableIf_t< IsResizable_v<MT> >
   resize_backend( Tensor<MT>& tensor, size_t o, size_t m, size_t n, bool preserve )
{
   UNUSED_PARAMETER( preserve );

   if( (~tensor).rows() != m || (~tensor).columns() != n || (~tensor).pages() != o) {
      BLAZE_THROW_INVALID_ARGUMENT( "Tensor cannot be resized" );
   }
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Backend implementation of the \c resize() function for resizable, non-square tensors.
// \ingroup tensor
//
// \param tensor The given tensor to be resized.
// \param m The new number of rows of the tensor.
// \param n The new number of columns of the tensor.
// \param preserve \a true if the old values of the tensor should be preserved, \a false if not.
// \return void
//
// This function changes the number of rows and columns of the given resizable, non-square tensor.
*/
template< typename MT > // Type of the tensor
BLAZE_ALWAYS_INLINE EnableIf_t< IsResizable_v<MT> && !IsSquare_v<MT> >
   resize_backend( Tensor<MT>& tensor, size_t o, size_t m, size_t n, bool preserve )
{
   (~tensor).resize( o, m, n, preserve );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Backend implementation of the \c resize() function for resizable, square tensors.
// \ingroup tensor
//
// \param tensor The given tensor to be resized.
// \param m The new number of rows of the tensor.
// \param n The new number of columns of the tensor.
// \param preserve \a true if the old values of the tensor should be preserved, \a false if not.
// \return void
// \exception std::invalid_argument Invalid resize arguments for square tensor.
//
// This function changes the number of rows and columns of the given resizable, square tensor.
*/
// template< typename MT > // Type of the tensor
// BLAZE_ALWAYS_INLINE EnableIf_t< IsResizable_v<MT> && IsSquare_v<MT> >
//    resize_backend( Tensor<MT>& tensor, size_t o, size_t m, size_t n, bool preserve )
// {
//    if( m != n || m != o ) {
//       BLAZE_THROW_INVALID_ARGUMENT( "Invalid resize arguments for square tensor" );
//    }
//
//    (~tensor).resize( m, preserve );
// }
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Changing the size of the tensor.
// \ingroup tensor
//
// \param tensor The given tensor to be resized.
// \param m The new number of rows of the tensor.
// \param n The new number of columns of the tensor.
// \param preserve \a true if the old values of the tensor should be preserved, \a false if not.
// \return void
// \exception std::invalid_argument Invalid resize arguments for square tensor.
// \exception std::invalid_argument Tensor cannot be resized.
//
// This function provides a unified interface to resize dense and sparse tensors. In contrast
// to the \c resize() member function, which is only available on resizable tensor types, this
// function can be used on both resizable and non-resizable tensors. In case the given tensor
// of type \a MT is resizable (i.e. provides a \c resize function) the type-specific \c resize()
// member function is called. Depending on the type \a MT, this may result in the allocation of
// new dynamic memory and the invalidation of existing views (subtensors, rows, columns, ...).
// Note that in case the tensor is a compile time square tensor (as for instance the
// blaze::SymmetricTensor adaptor, ...) the specified number of rows must be identical to the
// number of columns. Otherwise a \a std::invalid_argument exception is thrown. If the tensor
// type \a MT is non-resizable (i.e. does not provide a \c resize() function) and if the specified
// number of rows and columns is not identical to the current number of rows and columns of the
// tensor, a \a std::invalid_argument exception is thrown.

   \code
   blaze::DynamicTensor<int> A( 3UL, 3UL );
   resize( A, 5UL, 2UL );  // OK: regular resize operation

   blaze::SymmetricTensor< DynamicTensor<int> > B( 3UL );
   resize( B, 4UL, 4UL );  // OK: Number of rows and columns is identical
   resize( B, 3UL, 5UL );  // Error: Invalid arguments for square tensor!

   blaze::StaticTensor<int,3UL,3UL> C;
   resize( C, 3UL, 3UL );  // OK: No resize necessary
   resize( C, 5UL, 2UL );  // Error: Tensor cannot be resized!
   \endcode
*/
template< typename MT > // Type of the tensor
BLAZE_ALWAYS_INLINE void resize( Tensor<MT>& tensor, size_t o, size_t m, size_t n, bool preserve )
{
   resize_backend( tensor, o, m, n, preserve );
}
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Backend implementation of the \c shrinkToFit() function for non-shrinkable tensors.
// \ingroup tensor
//
// \param tensor The given tensor to be shrunk.
// \return void
*/
template< typename MT > // Type of the tensor
BLAZE_ALWAYS_INLINE DisableIf_t< IsShrinkable_v<MT> >
   shrinkToFit_backend( Tensor<MT>& tensor )
{
   UNUSED_PARAMETER( tensor );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Backend implementation of the \c shrinkToFit() function for shrinkable tensors.
// \ingroup tensor
//
// \param tensor The given tensor to be shrunk.
// \return void
*/
template< typename MT > // Type of the tensor
BLAZE_ALWAYS_INLINE EnableIf_t< IsShrinkable_v<MT> >
   shrinkToFit_backend( Tensor<MT>& tensor )
{
   (~tensor).shrinkToFit();
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Requesting the removal of unused capacity.
// \ingroup tensor
//
// \param tensor The given tensor to be shrunk.
// \return void
//
// This function tries to minimize the capacity of the tensor by removing unused capacity.
// Please note that in case of a shrinkable tensor, due to padding the capacity might not be
// reduced exactly to the number of rows times the number of columns. Please also note that
// in case a reallocation occurs, all iterators (including end() iterators), all pointers and
// references to elements of this tensor are invalidated. In case of an unshrinkable tensor
// the function has no effect.
*/
template< typename MT > // Type of the tensor
BLAZE_ALWAYS_INLINE void shrinkToFit( Tensor<MT>& tensor )
{
   shrinkToFit_backend( tensor );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief In-place transpose of the given tensor.
// \ingroup tensor
//
// \param tensor The given tensor to be transposed.
// \return void
// \exception std::logic_error Tensor cannot be transposed.
//
// This function transposes the given tensor in-place. The function fails if ...
//
//  - ... the given tensor has a fixed size and is non-square;
//  - ... the given tensor is a triangular tensor;
//  - ... the given subtensor affects the restricted parts of a triangular tensor;
//  - ... the given subtensor would cause non-deterministic results in a symmetric/Hermitian tensor.
//
// In all failure cases a \a std::logic_error exception is thrown.
*/
// template< typename MT > // Type of the tensor
// BLAZE_ALWAYS_INLINE void transpose( Tensor<MT>& tensor )
// {
//    (~tensor).transpose();
// }
//*************************************************************************************************


//*************************************************************************************************
/*!\brief In-place conjugate transpose of the given tensor.
// \ingroup tensor
//
// \param tensor The given tensor to be transposed.
// \return void
// \exception std::logic_error Tensor cannot be transposed.
//
// This function transposes the given tensor in-place. The function fails if ...
//
//  - ... the given tensor has a fixed size and is non-square;
//  - ... the given tensor is a triangular tensor;
//  - ... the given subtensor affects the restricted parts of a triangular tensor;
//  - ... the given subtensor would cause non-deterministic results in a symmetric/Hermitian tensor.
//
// In all failure cases a \a std::logic_error exception is thrown.
*/
// template< typename MT > // Type of the tensor
// BLAZE_ALWAYS_INLINE void ctranspose( Tensor<MT>& tensor )
// {
//    (~tensor).ctranspose();
// }
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Evaluates the given tensor expression.
// \ingroup tensor
//
// \param tensor The tensor to be evaluated.
// \return The result of the evaluated tensor expression.
//
// This function forces an evaluation of the given tensor expression and enables an automatic
// deduction of the correct result type of an operation. The following code example demonstrates
// its intended use for the multiplication of a lower and a strictly lower dense tensor:

   \code
   using blaze::DynamicTensor;
   using blaze::LowerTensor;
   using blaze::StrictlyLowerTensor;

   LowerTensor< DynamicTensor<double> > A;
   StrictlyLowerTensor< DynamicTensor<double> > B;
   // ... Resizing and initialization

   auto C = evaluate( A * B );
   \endcode

// In this scenario, the \a evaluate() function assists in deducing the exact result type of
// the operation via the 'auto' keyword. Please note that if \a evaluate() is used in this
// way, no temporary tensor is created and no copy operation is performed. Instead, the result
// is directly written to the target tensor due to the return value optimization (RVO). However,
// if \a evaluate() is used in combination with an explicit target type, a temporary will be
// created and a copy operation will be performed if the used type differs from the type
// returned from the function:

   \code
   StrictlyLowerTensor< DynamicTensor<double> > D( A * B );  // No temporary & no copy operation
   LowerTensor< DynamicTensor<double> > E( A * B );          // Temporary & copy operation
   DynamicTensor<double> F( A * B );                         // Temporary & copy operation
   D = evaluate( A * B );                                    // Temporary & copy operation
   \endcode

// Sometimes it might be desirable to explicitly evaluate a sub-expression within a larger
// expression. However, please note that \a evaluate() is not intended to be used for this
// purpose. This task is more elegantly and efficiently handled by the \a eval() function:

   \code
   blaze::DynamicTensor<double> A, B, C, D;

   D = A + evaluate( B * C );  // Unnecessary creation of a temporary tensor
   D = A + eval( B * C );      // No creation of a temporary tensor
   \endcode

// In contrast to the \a evaluate() function, \a eval() can take the complete expression into
// account and therefore can guarantee the most efficient way to evaluate it.
*/
template< typename MT > // Type of the tensor
inline const typename MT::ResultType evaluate( const Tensor<MT>& tensor )
{
   const typename MT::ResultType tmp( ~tensor );
   return tmp;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Checks if the given tensor is empty.
// \ingroup tensor
//
// \param tensor The tensor to be checked.
// \return \a true if the tensor is empty, \a false if not.
//
// This function checks if the total number of elements of the given tensor is zero. If the
// total number of elements is zero the function returns \a true, otherwise it returns \a false.
*/
template< typename MT > // Type of the tensor
BLAZE_ALWAYS_INLINE constexpr bool isEmpty( const Tensor<MT>& tensor ) noexcept
{
   return size( ~tensor ) == 0UL;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Checks if the given tensor is a square tensor.
// \ingroup tensor
//
// \param tensor The tensor to be checked.
// \return \a true if the tensor is a square tensor, \a false if not.
//
// This function checks if the number of rows and columns of the given tensor are equal. If
// they are, the function returns \a true, otherwise it returns \a false.
*/
template< typename MT > // Type of the tensor
BLAZE_ALWAYS_INLINE bool isSquare( const Tensor<MT>& tensor ) noexcept
{
   return ( IsSquare_v<MT> || ( (~tensor).rows() == (~tensor).columns() && (~tensor).rows() == (~tensor).pages() ) );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns whether the two given tensors represent the same observable state.
// \ingroup tensor
//
// \param a The first tensor to be tested for its state.
// \param b The second tensor to be tested for its state.
// \return \a true in case the two tensors share a state, \a false otherwise.
//
// The isSame function provides an abstract interface for testing if the two given tensors
// represent the same observable state. This happens for instance in case \c a and \c b refer
// to the same tensor or in case \c a and \c b are aliases for the same tensor. In case both
// tensors represent the same observable state, the function returns \a true, other it returns
// \a false.

   \code
   blaze::DynamicTensor<int> mat1( 4UL, 5UL );  // Setup of a 4x5 dynamic tensor
   blaze::DynamicTensor<int> mat2( 4UL, 5UL );  // Setup of a second 4x5 dynamic tensor

   auto sub1 = subtensor( mat1, 4UL, 0UL, 0UL, 5UL );  // Subtensor fully covering mat1
   auto sub2 = subtensor( mat1, 2UL, 1UL, 1UL, 3UL );  // Subtensor partially covering mat1
   auto sub3 = subtensor( mat1, 2UL, 1UL, 1UL, 3UL );  // Subtensor partially covering mat1

   isSame( mat1, mat1 );  // returns true since both objects refer to the same tensor
   isSame( mat1, mat2 );  // returns false since mat1 and mat2 are two different tensors
   isSame( mat1, sub1 );  // returns true since sub1 represents the same observable state as mat1
   isSame( mat1, sub3 );  // returns false since sub3 only covers part of mat1
   isSame( sub2, sub3 );  // returns true since sub1 and sub2 refer to exactly the same part of mat1
   isSame( sub1, sub3 );  // returns false since sub1 and sub3 refer to different parts of mat1
   \endcode
*/
template< typename TT1  // Type of the left-hand side tensor
        , typename TT2 > // Type of the right-hand side tensor
BLAZE_ALWAYS_INLINE bool isSame( const Tensor<TT1>& a, const Tensor<TT2>& b ) noexcept
{
   return ( IsSame_v<TT1,TT2> &&
            reinterpret_cast<const void*>( &a ) == reinterpret_cast<const void*>( &b ) );
}
//*************************************************************************************************



//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Backend implementation of the assignment of two tensors with the same storage order.
// \ingroup tensor
//
// \param lhs The target left-hand side tensor.
// \param rhs The right-hand side tensor to be assigned.
// \return void
*/
template< typename TT1  // Type of the left-hand side tensor
        , typename TT2 > // Type of the right-hand side tensor
BLAZE_ALWAYS_INLINE void assign_backend( Tensor<TT1>& lhs, const Tensor<TT2>& rhs )
{
   BLAZE_FUNCTION_TRACE;

   (~lhs).assign( ~rhs );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Default implementation of the assignment of a tensor to a tensor.
// \ingroup tensor
//
// \param lhs The target left-hand side tensor.
// \param rhs The right-hand side tensor to be assigned.
// \return void
//
// This function implements the default assignment of a tensor to a tensor.\n
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename TT1  // Type of the left-hand side tensor
        , typename TT2 > // Type of the right-hand side tensor
BLAZE_ALWAYS_INLINE void assign( Tensor<TT1>& lhs, const Tensor<TT2>& rhs )
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
/*!\brief Backend implementation of the addition assignment of two tensors with the same
//        storage order.
// \ingroup tensor
//
// \param lhs The target left-hand side tensor.
// \param rhs The right-hand side tensor to be added.
// \return void
*/
template< typename TT1  // Type of the left-hand side tensor
        , typename TT2 > // Type of the right-hand side tensor
BLAZE_ALWAYS_INLINE void addAssign_backend( Tensor<TT1>& lhs, const Tensor<TT2>& rhs )
{
   BLAZE_FUNCTION_TRACE;

   (~lhs).addAssign( ~rhs );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Default implementation of the addition assignment of a tensor to a tensor.
// \ingroup tensor
//
// \param lhs The target left-hand side tensor.
// \param rhs The right-hand side tensor to be added.
// \return void
//
// This function implements the default addition assignment of a tensor to a tensor.\n
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename TT1  // Type of the left-hand side tensor
        , typename TT2 > // Type of the right-hand side tensor
BLAZE_ALWAYS_INLINE void addAssign( Tensor<TT1>& lhs, const Tensor<TT2>& rhs )
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
/*!\brief Backend implementation of the subtraction assignment of two tensors with the same
//        storage order.
// \ingroup tensor
//
// \param lhs The target left-hand side tensor.
// \param rhs The right-hand side tensor to be subtracted.
// \return void
*/
template< typename TT1  // Type of the left-hand side tensor
        , typename TT2 > // Type of the right-hand side tensor
BLAZE_ALWAYS_INLINE void subAssign_backend( Tensor<TT1>& lhs, const Tensor<TT2>& rhs )
{
   BLAZE_FUNCTION_TRACE;

   (~lhs).subAssign( ~rhs );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Default implementation of the subtraction assignment of a tensor to tensor.
// \ingroup tensor
//
// \param lhs The target left-hand side tensor.
// \param rhs The right-hand side tensor to be subtracted.
// \return void
//
// This function implements the default subtraction assignment of a tensor to a tensor.\n
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename TT1  // Type of the left-hand side tensor
        , typename TT2 > // Type of the right-hand side tensor
BLAZE_ALWAYS_INLINE void subAssign( Tensor<TT1>& lhs, const Tensor<TT2>& rhs )
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
/*!\brief Backend implementation of the Schur product assignment of two tensors with the same
//        storage order.
// \ingroup tensor
//
// \param lhs The target left-hand side tensor.
// \param rhs The right-hand side tensor for the Schur product.
// \return void
*/
template< typename TT1  // Type of the left-hand side tensor
        , typename TT2 > // Type of the right-hand side tensor
BLAZE_ALWAYS_INLINE void schurAssign_backend( Tensor<TT1>& lhs, const Tensor<TT2>& rhs )
{
   BLAZE_FUNCTION_TRACE;

   (~lhs).schurAssign( ~rhs );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Default implementation of the Schur product assignment of a tensor to tensor.
// \ingroup tensor
//
// \param lhs The target left-hand side tensor.
// \param rhs The right-hand side tensor for the Schur product.
// \return void
//
// This function implements the default Schur product assignment of a tensor to a tensor.\n
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename TT1  // Type of the left-hand side tensor
        , typename TT2 > // Type of the right-hand side tensor
BLAZE_ALWAYS_INLINE void schurAssign( Tensor<TT1>& lhs, const Tensor<TT2>& rhs )
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
/*!\brief Default implementation of the multiplication assignment of a tensor to a tensor.
// \ingroup tensor
//
// \param lhs The target left-hand side tensor.
// \param rhs The right-hand side tensor to be multiplied.
// \return void
//
// This function implements the default multiplication assignment of a tensor to a tensor.\n
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename TT1  // Type of the left-hand side tensor
        , typename TT2 > // Type of the right-hand side tensor
BLAZE_ALWAYS_INLINE void multAssign( Tensor<TT1>& lhs, const Tensor<TT2>& rhs )
{
   BLAZE_FUNCTION_TRACE;

   BLAZE_INTERNAL_ASSERT( (~lhs).columns() == (~rhs).rows(), "Invalid tensor sizes" );

   (~lhs).multAssign( ~rhs );
}
/*! \endcond */
//*************************************************************************************************



//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Removal of all restrictions on the data access to the given tensor.
// \ingroup tensor
//
// \param tensor The tensor to be derestricted.
// \return Reference to the tensor without access restrictions.
//
// This function removes all restrictions on the data access to the given tensor. It returns a
// reference to the tensor that does provide the same interface but does not have any restrictions
// on the data access.\n
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in the violation of invariants, erroneous results and/or in compilation errors.
*/
template< typename TT > // Type of the tensor
BLAZE_ALWAYS_INLINE TT& derestrict( Tensor<TT>& tensor )
{
   return ~tensor;
}
/*! \endcond */
//*************************************************************************************************


} // namespace blaze

#endif
