//=================================================================================================
/*!
//  \file blaze_tensor/math/views/Subtensor.h
//  \brief Header file for the implementation of the Subtensor view
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

#ifndef _BLAZE_TENSOR_MATH_VIEWS_SUBTENSOR_H_
#define _BLAZE_TENSOR_MATH_VIEWS_SUBTENSOR_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <blaze/math/AlignmentFlag.h>
#include <blaze/math/views/Submatrix.h>

#include <blaze_tensor/math/Aliases.h>
#include <blaze_tensor/math/expressions/Forward.h>
#include <blaze_tensor/math/expressions/TensEvalExpr.h>
#include <blaze_tensor/math/expressions/TensMapExpr.h>
#include <blaze_tensor/math/expressions/TensReduceExpr.h>
#include <blaze_tensor/math/expressions/TensScalarDivExpr.h>
#include <blaze_tensor/math/expressions/TensScalarMultExpr.h>
#include <blaze_tensor/math/expressions/TensTensAddExpr.h>
#include <blaze_tensor/math/expressions/TensTensMapExpr.h>
#include <blaze_tensor/math/expressions/TensTensMultExpr.h>
#include <blaze_tensor/math/expressions/TensTensSubExpr.h>
#include <blaze_tensor/math/views/subtensor/DenseAligned.h>
#include <blaze_tensor/math/views/subtensor/DenseUnaligned.h>


namespace blaze {

//=================================================================================================
//
//  GLOBAL FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Creating a view on a specific subtensor of the given tensor.
// \ingroup subtensor
//
// \param tensor The tensor containing the subtensor.
// \param args Optional subtensor arguments.
// \return View on the specific subtensor of the tensor.
// \exception std::invalid_argument Invalid subtensor specification.
//
// This function returns an expression representing the specified subtensor of the given tensor.
// The following example demonstrates the creation of a dense and sparse subtensor:

   \code
   blaze::DynamicTensor<double,blaze::rowMajor> D;
   blaze::CompressedTensor<int,blaze::columnMajor> S;
   // ... Resizing and initialization

   // Creating a dense subtensor of size 8x4, starting in row 0 and column 16
   auto dsm = subtensor<0UL,16UL,8UL,4UL>( D );

   // Creating a sparse subtensor of size 7x3, starting in row 2 and column 4
   auto ssm = subtensor<2UL,4UL,7UL,3UL>( S );
   \endcode

// By default, the provided subtensor arguments are checked at runtime. In case the subtensor
// is not properly specified (i.e. if the specified row or column is larger than the total number
// of rows or columns of the given tensor or the subtensor is specified beyond the number of rows
// or columns of the tensor) a \a std::invalid_argument exception is thrown. The checks can be
// skipped by providing the optional \a blaze::unchecked argument.

   \code
   auto dsm = subtensor<0UL,16UL,8UL,4UL>( D, unchecked );
   auto ssm = subtensor<2UL,4UL,7UL,3UL>( S, unchecked );
   \endcode

// Please note that this function creates an unaligned dense or sparse subtensor. For instance,
// the creation of the dense subtensor is equivalent to the following function call:

   \code
   auto dsm = subtensor<unaligned,0UL,16UL,8UL,4UL>( D );
   \endcode

// In contrast to unaligned subtensors, which provide full flexibility, aligned subtensors pose
// additional alignment restrictions. However, especially in case of dense subtensors this may
// result in considerable performance improvements. In order to create an aligned subtensor the
// following function call has to be used:

   \code
   auto dsm = subtensor<aligned,0UL,16UL,8UL,4UL>( D );
   \endcode

// Note however that in this case the given compile time arguments \a I, \a J, \a M, and \a N are
// subject to additional checks to guarantee proper alignment.
*/
template< size_t I            // Index of the first row
        , size_t J            // Index of the first column
        , size_t K            // Index of the first page
        , size_t M            // Number of rows
        , size_t N            // Number of columns
        , size_t O            // Number of pages
        , typename TT         // Type of the dense tensor
        , typename... RSAs >  // Optional subtensor arguments
inline decltype(auto) subtensor( Tensor<TT>& tensor, RSAs... args )
{
   BLAZE_FUNCTION_TRACE;

   return subtensor<unaligned,K,I,J,O,M,N>( ~tensor, args... );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Creating a view on a specific subtensor of the given constant tensor.
// \ingroup subtensor
//
// \param tensor The constant tensor containing the subtensor.
// \param args Optional subtensor arguments.
// \return View on the specific subtensor of the tensor.
// \exception std::invalid_argument Invalid subtensor specification.
//
// This function returns an expression representing the specified subtensor of the given constant
// tensor. The following example demonstrates the creation of a dense and sparse subtensor:

   \code
   const blaze::DynamicTensor<double,blaze::rowMajor> D( ... );
   const blaze::CompressedTensor<int,blaze::columnMajor> S( ... );

   // Creating a dense subtensor of size 8x4, starting in row 0 and column 16
   auto dsm = subtensor<0UL,16UL,8UL,4UL>( D );

   // Creating a sparse subtensor of size 7x3, starting in row 2 and column 4
   auto ssm = subtensor<2UL,4UL,7UL,3UL>( S );
   \endcode

// By default, the provided subtensor arguments are checked at runtime. In case the subtensor
// is not properly specified (i.e. if the specified row or column is larger than the total number
// of rows or columns of the given tensor or the subtensor is specified beyond the number of rows
// or columns of the tensor) a \a std::invalid_argument exception is thrown. The checks can be
// skipped by providing the optional \a blaze::unchecked argument.

   \code
   auto dsm = subtensor<0UL,16UL,8UL,4UL>( D, unchecked );
   auto ssm = subtensor<2UL,4UL,7UL,3UL>( S, unchecked );
   \endcode

// Please note that this function creates an unaligned dense or sparse subtensor. For instance,
// the creation of the dense subtensor is equivalent to the following three function calls:

   \code
   auto dsm = subtensor<unaligned,0UL,16UL,8UL,4UL>( D );
   \endcode

// In contrast to unaligned subtensors, which provide full flexibility, aligned subtensors pose
// additional alignment restrictions. However, especially in case of dense subtensors this may
// result in considerable performance improvements. In order to create an aligned subtensor the
// following function call has to be used:

   \code
   auto dsm = subtensor<aligned,0UL,16UL,8UL,4UL>( D );
   \endcode

// Note however that in this case the given compile time arguments \a I, \a J, \a M, and \a N are
// subject to additional checks to guarantee proper alignment.
*/
template< size_t I            // Index of the first row
        , size_t J            // Index of the first column
        , size_t K            // Index of the first page
        , size_t M            // Number of rows
        , size_t N            // Number of columns
        , size_t O            // Number of pages
        , typename TT         // Type of the dense tensor
        , typename... RSAs >  // Option subtensor arguments
inline decltype(auto) subtensor( const Tensor<TT>& tensor, RSAs... args )
{
   BLAZE_FUNCTION_TRACE;

   return subtensor<unaligned,K,I,J,O,M,N>( ~tensor, args... );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Creating a view on a specific subtensor of the given temporary tensor.
// \ingroup subtensor
//
// \param tensor The temporary tensor containing the subtensor.
// \param args Optional subtensor arguments.
// \return View on the specific subtensor of the tensor.
// \exception std::invalid_argument Invalid subtensor specification.
//
// This function returns an expression representing the specified subtensor of the given
// temporary tensor. In case the subtensor is not properly specified (i.e. if the specified
// row or column is greater than the total number of rows or columns of the given tensor or
// the subtensor is specified beyond the number of rows or columns of the tensor) a
// \a std::invalid_argument exception is thrown.
*/
template< size_t I            // Index of the first row
        , size_t J            // Index of the first column
        , size_t K            // Index of the first page
        , size_t M            // Number of rows
        , size_t N            // Number of columns
        , size_t O            // Number of pages
        , typename TT         // Type of the dense tensor
        , typename... RSAs >  // Option subtensor arguments
inline decltype(auto) subtensor( Tensor<TT>&& tensor, RSAs... args )
{
   BLAZE_FUNCTION_TRACE;

   return subtensor<unaligned,K,I,J,O,M,N>( ~tensor, args... );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Creating a view on a specific subtensor of the given tensor.
// \ingroup subtensor
//
// \param tensor The tensor containing the subtensor.
// \param args Optional subtensor arguments.
// \return View on the specific subtensor of the tensor.
// \exception std::invalid_argument Invalid subtensor specification.
//
// This function returns an expression representing an aligned or unaligned subtensor of the
// given dense or sparse tensor, based on the specified alignment flag \a AF. The following
// example demonstrates the creation of both an aligned and unaligned subtensor:

   \code
   blaze::DynamicTensor<double,blaze::rowMajor> D;
   blaze::CompressedTensor<int,blaze::columnMajor> S;
   // ... Resizing and initialization

   // Creating an aligned dense subtensor of size 8x4, starting in row 0 and column 16
   auto dsm = subtensor<aligned,0UL,16UL,8UL,4UL>( D );

   // Creating an unaligned sparse subtensor of size 7x3, starting in row 2 and column 4
   auto ssm = subtensor<unaligned,2UL,4UL,7UL,3UL>( S );
   \endcode

// By default, the provided subtensor arguments are checked at runtime. In case the subtensor
// is not properly specified (i.e. if the specified row or column is larger than the total number
// of rows or columns of the given tensor or the subtensor is specified beyond the number of rows
// or columns of the tensor) a \a std::invalid_argument exception is thrown. The checks can be
// skipped by providing the optional \a blaze::unchecked argument.

   \code
   auto dsm = subtensor<aligned,0UL,16UL,8UL,4UL>( D, unchecked );
   auto ssm = subtensor<unaligned,2UL,4UL,7UL,3UL>( S, unchecked );
   \endcode

// In contrast to unaligned subtensors, which provide full flexibility, aligned subtensors
// pose additional alignment restrictions and the given \a I, and \a J arguments are subject
// to additional checks to guarantee proper alignment. However, especially in case of dense
// subtensors this may result in considerable performance improvements.
//
// The alignment restrictions refer to system dependent address restrictions for the used element
// type and the available vectorization mode (SSE, AVX, ...). In order to be properly aligned the
// first element of each row/column of the subtensor must be aligned. The following source code
// gives some examples for a double precision row-major dynamic tensor, assuming that padding is
// enabled and that AVX is available, which packs 4 \c double values into a SIMD vector:

   \code
   blaze::DynamicTensor<double,blaze::rowMajor> D( 13UL, 17UL );
   // ... Resizing and initialization

   // OK: Starts at position (0,0), i.e. the first element of each row is aligned (due to padding)
   auto dsm1 = subtensor<aligned,0UL,0UL,7UL,11UL>( D );

   // OK: First column is a multiple of 4, i.e. the first element of each row is aligned (due to padding)
   auto dsm2 = subtensor<aligned,3UL,12UL,8UL,16UL>( D );

   // OK: First column is a multiple of 4 and the subtensor includes the last row and column
   auto dsm3 = subtensor<aligned,4UL,0UL,9UL,17UL>( D );

   // Error: First column is not a multiple of 4, i.e. the first element is not aligned
   auto dsm4 = subtensor<aligned,2UL,3UL,12UL,12UL>( D );
   \endcode

// In case any alignment restrictions are violated, a \a std::invalid_argument exception is thrown.
*/
template< AlignmentFlag AF    // Alignment flag
        , size_t K            // Index of the first page
        , size_t I            // Index of the first row
        , size_t J            // Index of the first column
        , size_t O            // Number of pages
        , size_t M            // Number of rows
        , size_t N            // Number of columns
        , typename TT         // Type of the dense tensor
        , typename... RSAs >  // Option subtensor arguments
inline decltype(auto) subtensor( Tensor<TT>& tensor, RSAs... args )
{
   BLAZE_FUNCTION_TRACE;

   using ReturnType = Subtensor_<TT,AF,K,I,J,O,M,N>;
   return ReturnType( ~tensor, args... );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Creating a view on a specific subtensor of the given constant tensor.
// \ingroup subtensor
//
// \param tensor The constant tensor containing the subtensor.
// \param args Optional subtensor arguments.
// \return View on the specific subtensor of the tensor.
// \exception std::invalid_argument Invalid subtensor specification.
//
// This function returns an expression representing an aligned or unaligned subtensor of the
// given constant dense or sparse tensor, based on the specified alignment flag \a AF. The
// following example demonstrates the creation of both an aligned and unaligned subtensor:

   \code
   const blaze::DynamicTensor<double,blaze::rowMajor> D( ... );
   const blaze::CompressedTensor<int,blaze::columnMajor> S( ... );

   // Creating an aligned dense subtensor of size 8x4, starting in row 0 and column 16
   auto dsm = subtensor<aligned,0UL,16UL,8UL,4UL>( D );

   // Creating an unaligned sparse subtensor of size 7x3, starting in row 2 and column 4
   auto ssm = subtensor<unaligned,2UL,4UL,7UL,3UL>( S );
   \endcode

// By default, the provided subtensor arguments are checked at runtime. In case the subtensor
// is not properly specified (i.e. if the specified row or column is larger than the total number
// of rows or columns of the given tensor or the subtensor is specified beyond the number of rows
// or columns of the tensor) a \a std::invalid_argument exception is thrown. The checks can be
// skipped by providing the optional \a blaze::unchecked argument.

   \code
   auto dsm = subtensor<aligned,0UL,16UL,8UL,4UL>( D, unchecked );
   auto ssm = subtensor<unaligned,2UL,4UL,7UL,3UL>( S, unchecked );
   \endcode

// In contrast to unaligned subtensors, which provide full flexibility, aligned subtensors
// pose additional alignment restrictions and the given \a I, and \a J arguments are subject
// to additional checks to guarantee proper alignment. However, especially in case of dense
// subtensors this may result in considerable performance improvements.
//
// The alignment restrictions refer to system dependent address restrictions for the used element
// type and the available vectorization mode (SSE, AVX, ...). In order to be properly aligned the
// first element of each row/column of the subtensor must be aligned. The following source code
// gives some examples for a double precision row-major dynamic tensor, assuming that padding is
// enabled and that AVX is available, which packs 4 \c double values into a SIMD vector:

   \code
   const blaze::DynamicTensor<double,blaze::rowMajor> D( ... );

   // OK: Starts at position (0,0), i.e. the first element of each row is aligned (due to padding)
   auto dsm1 = subtensor<aligned,0UL,0UL,7UL,11UL>( D );

   // OK: First column is a multiple of 4, i.e. the first element of each row is aligned (due to padding)
   auto dsm2 = subtensor<aligned,3UL,12UL,8UL,16UL>( D );

   // OK: First column is a multiple of 4 and the subtensor includes the last row and column
   auto dsm3 = subtensor<aligned,4UL,0UL,9UL,17UL>( D );

   // Error: First column is not a multiple of 4, i.e. the first element is not aligned
   auto dsm4 = subtensor<aligned,2UL,3UL,12UL,12UL>( D );
   \endcode

// In case any alignment restrictions are violated, a \a std::invalid_argument exception is thrown.
*/
template< AlignmentFlag AF    // Alignment flag
        , size_t K            // Index of the first page
        , size_t I            // Index of the first row
        , size_t J            // Index of the first column
        , size_t O            // Number of pages
        , size_t M            // Number of rows
        , size_t N            // Number of columns
        , typename TT         // Type of the dense tensor
        , typename... RSAs >  // Option subtensor arguments
inline decltype(auto) subtensor( const Tensor<TT>& tensor, RSAs... args )
{
   BLAZE_FUNCTION_TRACE;

   using ReturnType = const Subtensor_<const TT,AF,K,I,J,O,M,N>;
   return ReturnType( ~tensor, args... );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Creating a view on a specific subtensor of the given temporary tensor.
// \ingroup subtensor
//
// \param tensor The temporary tensor containing the subtensor.
// \param args Optional subtensor arguments.
// \return View on the specific subtensor of the tensor.
// \exception std::invalid_argument Invalid subtensor specification.
//
// This function returns an expression representing an aligned or unaligned subtensor of the
// given temporary dense or sparse tensor, based on the specified alignment flag \a AF. In
// case the subtensor is not properly specified (i.e. if the specified row or column is larger
// than the total number of rows or columns of the given tensor or the subtensor is specified
// beyond the number of rows or columns of the tensor) or any alignment restrictions are
// violated, a \a std::invalid_argument exception is thrown.
*/
template< AlignmentFlag AF    // Alignment flag
        , size_t K            // Index of the first page
        , size_t I            // Index of the first row
        , size_t J            // Index of the first column
        , size_t O            // Number of pages
        , size_t M            // Number of rows
        , size_t N            // Number of columns
        , typename TT         // Type of the dense tensor
        , typename... RSAs >  // Option subtensor arguments
inline decltype(auto) subtensor( Tensor<TT>&& tensor, RSAs... args )
{
   BLAZE_FUNCTION_TRACE;

   using ReturnType = Subtensor_<TT,AF,K,I,J,O,M,N>;
   return ReturnType( ~tensor, args... );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Creating a view on a specific subtensor of the given tensor.
// \ingroup subtensor
//
// \param tensor The tensor containing the subtensor.
// \param row The index of the first row of the subtensor.
// \param column The index of the first column of the subtensor.
// \param m The number of rows of the subtensor.
// \param n The number of columns of the subtensor.
// \param args Optional subtensor arguments.
// \return View on the specific subtensor of the tensor.
// \exception std::invalid_argument Invalid subtensor specification.
//
// This function returns an expression representing the specified subtensor of the given tensor.
// The following example demonstrates the creation of a dense and sparse subtensor:

   \code
   blaze::DynamicTensor<double,blaze::rowMajor> D;
   blaze::CompressedTensor<int,blaze::columnMajor> S;
   // ... Resizing and initialization

   // Creating a dense subtensor of size 8x4, starting in row 0 and column 16
   auto dsm = subtensor( D, 8UL, 0UL, 16UL, 4UL );

   // Creating a sparse subtensor of size 7x3, starting in row 2 and column 4
   auto ssm = subtensor( S, 7UL, 2UL, 4UL, 3UL );
   \endcode

// By default, the provided subtensor arguments are checked at runtime. In case the subtensor
// is not properly specified (i.e. if the specified row or column is larger than the total number
// of rows or columns of the given tensor or the subtensor is specified beyond the number of rows
// or columns of the tensor) a \a std::invalid_argument exception is thrown. The checks can be
// skipped by providing the optional \a blaze::unchecked argument.

   \code
   auto dsm = subtensor( D, 8UL, 0UL, 16UL, 4UL, unchecked );
   auto ssm = subtensor( S, 7UL, 2UL, 4UL, 3UL, unchecked );
   \endcode

// Please note that this function creates an unaligned dense or sparse subtensor. For instance,
// the creation of the dense subtensor is equivalent to the following function call:

   \code
   unaligned dsm = subtensor<unaligned>( D, 8UL, 0UL, 16UL, 4UL );
   \endcode

// In contrast to unaligned subtensors, which provide full flexibility, aligned subtensors pose
// additional alignment restrictions. However, especially in case of dense subtensors this may
// result in considerable performance improvements. In order to create an aligned subtensor the
// following function call has to be used:

   \code
   auto dsm = subtensor<aligned>( D, 8UL, 0UL, 16UL, 4UL );
   \endcode

// Note however that in this case the given arguments \a row, \a column, \a m, and \a n are
// subject to additional checks to guarantee proper alignment.
*/
template< typename TT         // Type of the dense tensor
        , typename... RSAs >  // Option subtensor arguments
inline decltype(auto)
   subtensor( Tensor<TT>& tensor, size_t page, size_t row, size_t column, size_t o, size_t m, size_t n, RSAs... args )
{
   BLAZE_FUNCTION_TRACE;

   return subtensor<unaligned>( ~tensor, page, row, column, o, m, n, args... );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Creating a view on a specific subtensor of the given constant tensor.
// \ingroup subtensor
//
// \param tensor The constant tensor containing the subtensor.
// \param row The index of the first row of the subtensor.
// \param column The index of the first column of the subtensor.
// \param m The number of rows of the subtensor.
// \param n The number of columns of the subtensor.
// \param args Optional subtensor arguments.
// \return View on the specific subtensor of the tensor.
// \exception std::invalid_argument Invalid subtensor specification.
//
// This function returns an expression representing the specified subtensor of the given constant
// tensor. The following example demonstrates the creation of a dense and sparse subtensor:

   \code
   const blaze::DynamicTensor<double,blaze::rowMajor> D( ... );
   const blaze::CompressedTensor<int,blaze::columnMajor> S( ... );

   // Creating a dense subtensor of size 8x4, starting in row 0 and column 16
   auto dsm = subtensor( D, 8UL, 0UL, 16UL, 4UL );

   // Creating a sparse subtensor of size 7x3, starting in row 2 and column 4
   auto ssm = subtensor( S, 7UL, 2UL, 4UL, 3UL );
   \endcode

// By default, the provided subtensor arguments are checked at runtime. In case the subtensor
// is not properly specified (i.e. if the specified row or column is larger than the total number
// of rows or columns of the given tensor or the subtensor is specified beyond the number of rows
// or columns of the tensor) a \a std::invalid_argument exception is thrown. The checks can be
// skipped by providing the optional \a blaze::unchecked argument.

   \code
   auto dsm = subtensor( D, 8UL, 0UL, 16UL, 4UL, unchecked );
   auto ssm = subtensor( S, 7UL, 2UL, 4UL, 3UL, unchecked );
   \endcode

// Please note that this function creates an unaligned dense or sparse subtensor. For instance,
// the creation of the dense subtensor is equivalent to the following three function calls:

   \code
   auto dsm = subtensor<unaligned>( D, 8UL, 0UL, 16UL, 4UL );
   \endcode

// In contrast to unaligned subtensors, which provide full flexibility, aligned subtensors pose
// additional alignment restrictions. However, especially in case of dense subtensors this may
// result in considerable performance improvements. In order to create an aligned subtensor the
// following function call has to be used:

   \code
   auto dsm = subtensor<aligned>( D, 8UL, 0UL, 16UL, 4UL );
   \endcode

// Note however that in this case the given arguments \a row, \a column, \a m, and \a n are
// subject to additional checks to guarantee proper alignment.
*/
template< typename TT         // Type of the dense tensor
        , typename... RSAs >  // Option subtensor arguments
inline decltype(auto)
   subtensor( const Tensor<TT>& tensor, size_t page, size_t row, size_t column, size_t o, size_t m, size_t n, RSAs... args )
{
   BLAZE_FUNCTION_TRACE;

   return subtensor<unaligned>( ~tensor, page, row, column, o, m, n, args... );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Creating a view on a specific subtensor of the given temporary tensor.
// \ingroup subtensor
//
// \param tensor The temporary tensor containing the subtensor.
// \param row The index of the first row of the subtensor.
// \param column The index of the first column of the subtensor.
// \param m The number of rows of the subtensor.
// \param n The number of columns of the subtensor.
// \param args Optional subtensor arguments.
// \return View on the specific subtensor of the tensor.
// \exception std::invalid_argument Invalid subtensor specification.
//
// This function returns an expression representing the specified subtensor of the given
// temporary tensor. In case the subtensor is not properly specified (i.e. if the specified
// row or column is greater than the total number of rows or columns of the given tensor or
// the subtensor is specified beyond the number of rows or columns of the tensor) a
// \a std::invalid_argument exception is thrown.
*/
template< typename TT         // Type of the dense tensor
        , typename... RSAs >  // Option subtensor arguments
inline decltype(auto)
   subtensor( Tensor<TT>&& tensor, size_t page, size_t row, size_t column, size_t o, size_t m, size_t n, RSAs... args )
{
   BLAZE_FUNCTION_TRACE;

   return subtensor<unaligned>( ~tensor, page, row, column, o, m, n, args... );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Creating a view on a specific subtensor of the given tensor.
// \ingroup subtensor
//
// \param tensor The tensor containing the subtensor.
// \param row The index of the first row of the subtensor.
// \param column The index of the first column of the subtensor.
// \param m The number of rows of the subtensor.
// \param n The number of columns of the subtensor.
// \param args Optional subtensor arguments.
// \return View on the specific subtensor of the tensor.
// \exception std::invalid_argument Invalid subtensor specification.
//
// This function returns an expression representing an aligned or unaligned subtensor of the
// given dense or sparse tensor, based on the specified alignment flag \a AF. The following
// example demonstrates the creation of both an aligned and unaligned subtensor:

   \code
   blaze::DynamicTensor<double,blaze::rowMajor> D;
   blaze::CompressedTensor<int,blaze::columnMajor> S;
   // ... Resizing and initialization

   // Creating an aligned dense subtensor of size 8x4, starting in row 0 and column 16
   auto dsm = subtensor<aligned>( D, 8UL, 0UL, 16UL, 4UL );

   // Creating an unaligned sparse subtensor of size 7x3, starting in row 2 and column 4
   auto ssm = subtensor<unaligned>( S, 7UL, 2UL, 4UL, 3UL );
   \endcode

// By default, the provided subtensor arguments are checked at runtime. In case the subtensor
// is not properly specified (i.e. if the specified row or column is larger than the total number
// of rows or columns of the given tensor or the subtensor is specified beyond the number of rows
// or columns of the tensor) a \a std::invalid_argument exception is thrown. The checks can be
// skipped by providing the optional \a blaze::unchecked argument.

   \code
   auto dsm = subtensor<aligned>( D, 8UL, 0UL, 16UL, 4UL, unchecked );
   auto ssm = subtensor<unaligned>( S, 7UL, 2UL, 4UL, 3UL, unchecked );
   \endcode

// In contrast to unaligned subtensors, which provide full flexibility, aligned subtensors pose
// additional alignment restrictions and the given \a row, and \a column arguments are subject to
// additional checks to guarantee proper alignment. However, especially in case of dense subtensors
// this may result in considerable performance improvements.
//
// The alignment restrictions refer to system dependent address restrictions for the used element
// type and the available vectorization mode (SSE, AVX, ...). In order to be properly aligned the
// first element of each row/column of the subtensor must be aligned. The following source code
// gives some examples for a double precision row-major dynamic tensor, assuming that padding is
// enabled and that AVX is available, which packs 4 \c double values into a SIMD vector:

   \code
   blaze::DynamicTensor<double,blaze::rowMajor> D( 13UL, 17UL );
   // ... Resizing and initialization

   // OK: Starts at position (0,0), i.e. the first element of each row is aligned (due to padding)
   auto dsm1 = subtensor<aligned>( D, 7UL, 0UL, 0UL, 11UL );

   // OK: First column is a multiple of 4, i.e. the first element of each row is aligned (due to padding)
   auto dsm2 = subtensor<aligned>( D, 8UL, 3UL, 12UL, 16UL );

   // OK: First column is a multiple of 4 and the subtensor includes the last row and column
   auto dsm3 = subtensor<aligned>( D, 9UL, 4UL, 0UL, 17UL );

   // Error: First column is not a multiple of 4, i.e. the first element is not aligned
   auto dsm4 = subtensor<aligned>( D, 12UL, 2UL, 3UL, 12UL );
   \endcode

// In case any alignment restrictions are violated, a \a std::invalid_argument exception is thrown.
*/
template< AlignmentFlag AF    // Alignment flag
        , typename TT         // Type of the dense tensor
        , typename... RSAs >  // Option subtensor arguments
inline decltype(auto)
   subtensor( Tensor<TT>& tensor, size_t page, size_t row, size_t column, size_t o, size_t m, size_t n, RSAs... args )
{
   BLAZE_FUNCTION_TRACE;

   using ReturnType = Subtensor_<TT,AF>;
   return ReturnType( ~tensor, page, row, column, o, m, n, args... );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Creating a view on a specific subtensor of the given constant tensor.
// \ingroup subtensor
//
// \param tensor The constant tensor containing the subtensor.
// \param row The index of the first row of the subtensor.
// \param column The index of the first column of the subtensor.
// \param m The number of rows of the subtensor.
// \param n The number of columns of the subtensor.
// \param args Optional subtensor arguments.
// \return View on the specific subtensor of the tensor.
// \exception std::invalid_argument Invalid subtensor specification.
//
// This function returns an expression representing an aligned or unaligned subtensor of the
// given dense or sparse tensor, based on the specified alignment flag \a AF. The following
// example demonstrates the creation of both an aligned and unaligned subtensor:

   \code
   const blaze::DynamicTensor<double,blaze::rowMajor> D( ... );
   const blaze::CompressedTensor<int,blaze::columnMajor> S( ... );

   // Creating an aligned dense subtensor of size 8x4, starting in row 0 and column 16
   auto dsm = subtensor<aligned>( D, 8UL, 0UL, 16UL, 4UL );

   // Creating an unaligned sparse subtensor of size 7x3, starting in row 2 and column 4
   auto ssm = subtensor<unaligned>( S, 7UL, 2UL, 4UL, 3UL );
   \endcode

// By default, the provided subtensor arguments are checked at runtime. In case the subtensor
// is not properly specified (i.e. if the specified row or column is larger than the total number
// of rows or columns of the given tensor or the subtensor is specified beyond the number of rows
// or columns of the tensor) a \a std::invalid_argument exception is thrown. The checks can be
// skipped by providing the optional \a blaze::unchecked argument.

   \code
   auto dsm = subtensor<aligned>( D, 8UL, 0UL, 16UL, 4UL, unchecked );
   auto ssm = subtensor<unaligned>( S, 7UL, 2UL, 4UL, 3UL, unchecked );
   \endcode

// In contrast to unaligned subtensors, which provide full flexibility, aligned subtensors pose
// additional alignment restrictions and the given \a row, and \a column arguments are subject to
// additional checks to guarantee proper alignment. However, especially in case of dense subtensors
// this may result in considerable performance improvements.
//
// The alignment restrictions refer to system dependent address restrictions for the used element
// type and the available vectorization mode (SSE, AVX, ...). In order to be properly aligned the
// first element of each row/column of the subtensor must be aligned. The following source code
// gives some examples for a double precision row-major dynamic tensor, assuming that padding is
// enabled and that AVX is available, which packs 4 \c double values into a SIMD vector:

   \code
   const blaze::DynamicTensor<double,blaze::rowMajor> D( ... );

   // OK: Starts at position (0,0), i.e. the first element of each row is aligned (due to padding)
   auto dsm1 = subtensor<aligned>( D, 7UL, 0UL, 0UL, 11UL );

   // OK: First column is a multiple of 4, i.e. the first element of each row is aligned (due to padding)
   auto dsm2 = subtensor<aligned>( D, 8UL, 3UL, 12UL, 16UL );

   // OK: First column is a multiple of 4 and the subtensor includes the last row and column
   auto dsm3 = subtensor<aligned>( D, 9UL, 4UL, 0UL, 17UL );

   // Error: First column is not a multiple of 4, i.e. the first element is not aligned
   auto dsm4 = subtensor<aligned>( D, 12UL, 2UL, 3UL, 12UL );
   \endcode

// In case any alignment restrictions are violated, a \a std::invalid_argument exception is thrown.
*/
template< AlignmentFlag AF    // Alignment flag
        , typename TT         // Type of the dense tensor
        , typename... RSAs >  // Option subtensor arguments
inline decltype(auto)
   subtensor( const Tensor<TT>& tensor, size_t page, size_t row, size_t column, size_t o, size_t m, size_t n, RSAs... args )
{
   BLAZE_FUNCTION_TRACE;

   using ReturnType = const Subtensor_<const TT,AF>;
   return ReturnType( ~tensor, page, row, column, o, m, n, args... );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Creating a view on a specific subtensor of the given temporary tensor.
// \ingroup subtensor
//
// \param tensor The temporary tensor containing the subtensor.
// \param row The index of the first row of the subtensor.
// \param column The index of the first column of the subtensor.
// \param m The number of rows of the subtensor.
// \param n The number of columns of the subtensor.
// \param args Optional subtensor arguments.
// \return View on the specific subtensor of the tensor.
// \exception std::invalid_argument Invalid subtensor specification.
//
// This function returns an expression representing an aligned or unaligned subtensor of the
// given temporary dense or sparse tensor, based on the specified alignment flag \a AF. In
// case the subtensor is not properly specified (i.e. if the specified row or column is larger
// than the total number of rows or columns of the given tensor or the subtensor is specified
// beyond the number of rows or columns of the tensor) or any alignment restrictions are
// violated, a \a std::invalid_argument exception is thrown.
*/
template< AlignmentFlag AF    // Alignment flag
        , typename TT         // Type of the dense tensor
        , typename... RSAs >  // Option subtensor arguments
inline decltype(auto)
   subtensor( Tensor<TT>&& tensor, size_t page, size_t row, size_t column, size_t o, size_t m, size_t n, RSAs... args )
{
   BLAZE_FUNCTION_TRACE;

   using ReturnType = Subtensor_<TT,AF>;
   return ReturnType( ~tensor, page, row, column, o, m, n, args... );
}
//*************************************************************************************************




//=================================================================================================
//
//  GLOBAL RESTRUCTURING FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific subtensor of the given tensor/tensor addition.
// \ingroup subtensor
//
// \param tensor The constant tensor/tensor addition.
// \param args The runtime subtensor arguments
// \return View on the specified subtensor of the addition.
//
// This function returns an expression representing the specified subtensor of the given
// tensor/tensor addition.
*/
template< AlignmentFlag AF    // Alignment flag
        , size_t... CSAs      // Compile time subtensor arguments
        , typename TT         // Tensor base type of the expression
        , typename... RSAs >  // Runtime subtensor arguments
inline decltype(auto) subtensor( const TensTensAddExpr<TT>& tensor, RSAs... args )
{
   BLAZE_FUNCTION_TRACE;

   return subtensor<AF,CSAs...>( (~tensor).leftOperand(), args... ) +
          subtensor<AF,CSAs...>( (~tensor).rightOperand(), args... );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific subtensor of the given tensor/tensor subtraction.
// \ingroup subtensor
//
// \param tensor The constant tensor/tensor subtraction.
// \param args The runtime subtensor arguments
// \return View on the specified subtensor of the subtraction.
//
// This function returns an expression representing the specified subtensor of the given
// tensor/tensor subtraction.
*/
template< AlignmentFlag AF    // Alignment flag
        , size_t... CSAs      // Compile time subtensor arguments
        , typename TT         // Tensor base type of the expression
        , typename... RSAs >  // Runtime subtensor arguments
inline decltype(auto) subtensor( const TensTensSubExpr<TT>& tensor, RSAs... args )
{
   BLAZE_FUNCTION_TRACE;

   return subtensor<AF,CSAs...>( (~tensor).leftOperand(), args... ) -
          subtensor<AF,CSAs...>( (~tensor).rightOperand(), args... );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific subtensor of the given Schur product.
// \ingroup subtensor
//
// \param tensor The constant Schur product.
// \param args The runtime subtensor arguments
// \return View on the specified subtensor of the Schur product.
//
// This function returns an expression representing the specified subtensor of the given Schur
// product.
*/
template< AlignmentFlag AF    // Alignment flag
        , size_t... CSAs      // Compile time subtensor arguments
        , typename TT         // Tensor base type of the expression
        , typename... RSAs >  // Runtime subtensor arguments
inline decltype(auto) subtensor( const SchurExpr<TT>& tensor, RSAs... args )
{
   BLAZE_FUNCTION_TRACE;

   return subtensor<AF,CSAs...>( (~tensor).leftOperand(), args... ) %
          subtensor<AF,CSAs...>( (~tensor).rightOperand(), args... );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific subtensor of the given tensor/tensor multiplication.
// \ingroup subtensor
//
// \param tensor The constant tensor/tensor multiplication.
// \param args The runtime subtensor arguments.
// \return View on the specified subtensor of the multiplication.
//
// This function returns an expression representing the specified subtensor of the given
// tensor/tensor multiplication.
*/
// template< AlignmentFlag AF    // Alignment flag
//         , size_t... CSAs      // Compile time subtensor arguments
//         , typename TT         // Tensor base type of the expression
//         , typename... RSAs >  // Runtime subtensor arguments
// inline decltype(auto) subtensor( const TensTensMultExpr<TT>& tensor, RSAs... args )
// {
//    BLAZE_FUNCTION_TRACE;
//
//    using TT1 = RemoveReference_t< LeftOperand_t< TensorType_t<TT> > >;
//    using TT2 = RemoveReference_t< RightOperand_t< TensorType_t<TT> > >;
//
//    const SubtensorData<CSAs...> sd( args... );
//
//    BLAZE_DECLTYPE_AUTO( left , (~tensor).leftOperand()  );
//    BLAZE_DECLTYPE_AUTO( right, (~tensor).rightOperand() );
//
//    const size_t begin( 0UL );
//    const size_t end( left.columns() );
//
//    const size_t diff( ( begin < end )?( end - begin ):( 0UL ) );
//
//    return subtensor<AF>( left, sd.row(), begin, sd.page(), sd.rows(), diff, sd.pages() ) *
//           subtensor<AF>( right, begin, sd.column(), sd.page(), diff, sd.columns(), sd.pages() );
// }
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific subtensor of the given outer product.
// \ingroup subtensor
//
// \param tensor The constant outer product.
// \return View on the specified subtensor of the outer product.
//
// This function returns an expression representing the specified subtensor of the given
// outer product.
*/
// template< AlignmentFlag AF    // Alignment flag
//         , size_t I            // Index of the first row
//         , size_t J            // Index of the first column
//         , size_t M            // Number of rows
//         , size_t N            // Number of columns
//         , typename TT         // Tensor base type of the expression
//         , typename... RSAs >  // Runtime subtensor arguments
// inline decltype(auto) subtensor( const VecTVecMultExpr<TT>& tensor, RSAs... args )
// {
//    BLAZE_FUNCTION_TRACE;
//
//    return subvector<AF,I,M>( (~tensor).leftOperand(), args... ) *
//           subvector<AF,J,N>( (~tensor).rightOperand(), args... );
// }
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific subtensor of the given outer product.
// \ingroup subtensor
//
// \param tensor The constant outer product.
// \param row The index of the first row of the subtensor.
// \param column The index of the first column of the subtensor.
// \param m The number of rows of the subtensor.
// \param n The number of columns of the subtensor.
// \return View on the specified subtensor of the outer product.
//
// This function returns an expression representing the specified subtensor of the given
// outer product.
*/
// template< AlignmentFlag AF    // Alignment flag
//         , typename TT         // Tensor base type of the expression
//         , typename... RSAs >  // Runtime subtensor arguments
// inline decltype(auto)
//    subtensor( const VecTVecMultExpr<TT>& tensor, size_t row, size_t column, size_t m, size_t n, RSAs... args )
// {
//    BLAZE_FUNCTION_TRACE;
//
//    return subvector<AF>( (~tensor).leftOperand(), row, m, args... ) *
//           subvector<AF>( (~tensor).rightOperand(), column, n, args... );
// }
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific subtensor of the given tensor/scalar multiplication.
// \ingroup subtensor
//
// \param tensor The constant tensor/scalar multiplication.
// \param args The runtime subtensor arguments.
// \return View on the specified subtensor of the multiplication.
//
// This function returns an expression representing the specified subtensor of the given
// tensor/scalar multiplication.
*/
template< AlignmentFlag AF    // Alignment flag
        , size_t... CSAs      // Compile time subtensor arguments
        , typename TT         // Tensor base type of the expression
        , typename... RSAs >  // Runtime subtensor arguments
inline decltype(auto) subtensor( const TensScalarMultExpr<TT>& tensor, RSAs... args )
{
   BLAZE_FUNCTION_TRACE;

   return subtensor<AF,CSAs...>( (~tensor).leftOperand(), args... ) * (~tensor).rightOperand();
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific subtensor of the given tensor/scalar division.
// \ingroup subtensor
//
// \param tensor The constant tensor/scalar division.
// \param args The runtime subtensor arguments.
// \return View on the specified subtensor of the division.
//
// This function returns an expression representing the specified subtensor of the given
// tensor/scalar division.
*/
template< AlignmentFlag AF    // Alignment flag
        , size_t... CSAs      // Compile time subtensor arguments
        , typename TT         // Tensor base type of the expression
        , typename... RSAs >  // Runtime subtensor arguments
inline decltype(auto) subtensor( const TensScalarDivExpr<TT>& tensor, RSAs... args )
{
   BLAZE_FUNCTION_TRACE;

   return subtensor<AF,CSAs...>( (~tensor).leftOperand(), args... ) / (~tensor).rightOperand();
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific subtensor of the given unary tensor map operation.
// \ingroup subtensor
//
// \param tensor The constant unary tensor map operation.
// \param args The runtime subtensor arguments.
// \return View on the specified subtensor of the unary map operation.
//
// This function returns an expression representing the specified subtensor of the given unary
// tensor map operation.
*/
template< AlignmentFlag AF    // Alignment flag
        , size_t... CSAs      // Compile time subtensor arguments
        , typename TT         // Tensor base type of the expression
        , typename... RSAs >  // Runtime subtensor arguments
inline decltype(auto) subtensor( const TensMapExpr<TT>& tensor, RSAs... args )
{
   BLAZE_FUNCTION_TRACE;

   return map( subtensor<AF,CSAs...>( (~tensor).operand(), args... ), (~tensor).operation() );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific subtensor of the given binary tensor map operation.
// \ingroup subtensor
//
// \param tensor The constant binary tensor map operation.
// \param args The runtime subtensor arguments.
// \return View on the specified subtensor of the binary map operation.
//
// This function returns an expression representing the specified subtensor of the given binary
// tensor map operation.
*/
template< AlignmentFlag AF    // Alignment flag
        , size_t... CSAs      // Compile time subtensor arguments
        , typename TT         // Tensor base type of the expression
        , typename... RSAs >  // Runtime subtensor arguments
inline decltype(auto) subtensor( const TensTensMapExpr<TT>& tensor, RSAs... args )
{
   BLAZE_FUNCTION_TRACE;

   return map( subtensor<AF,CSAs...>( (~tensor).leftOperand(), args... ),
               subtensor<AF,CSAs...>( (~tensor).rightOperand(), args... ),
               (~tensor).operation() );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific subtensor of the given tensor evaluation operation.
// \ingroup subtensor
//
// \param tensor The constant tensor evaluation operation.
// \param args The runtime subtensor arguments.
// \return View on the specified subtensor of the evaluation operation.
//
// This function returns an expression representing the specified subtensor of the given tensor
// evaluation operation.
*/
template< AlignmentFlag AF    // Alignment flag
        , size_t... CSAs      // Compile time subtensor arguments
        , typename TT         // Tensor base type of the expression
        , typename... RSAs >  // Runtime subtensor arguments
inline decltype(auto) subtensor( const TensEvalExpr<TT>& tensor, RSAs... args )
{
   BLAZE_FUNCTION_TRACE;

   return eval( subtensor<AF,CSAs...>( (~tensor).operand(), args... ) );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific subtensor of the given tensor serialization operation.
// \ingroup subtensor
//
// \param tensor The constant tensor serialization operation.
// \param args The runtime subtensor arguments.
// \return View on the specified subtensor of the serialization operation.
//
// This function returns an expression representing the specified subtensor of the given tensor
// serialization operation.
*/
template< AlignmentFlag AF    // Alignment flag
        , size_t... CSAs      // Compile time subtensor arguments
        , typename TT         // Tensor base type of the expression
        , typename... RSAs >  // Runtime subtensor arguments
inline decltype(auto) subtensor( const MatSerialExpr<TT>& tensor, RSAs... args )
{
   BLAZE_FUNCTION_TRACE;

   return serial( subtensor<AF,CSAs...>( (~tensor).operand(), args... ) );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific subtensor of the given tensor declaration operation.
// \ingroup subtensor
//
// \param tensor The constant tensor declaration operation.
// \param args The runtime subtensor arguments.
// \return View on the specified subtensor of the declaration operation.
//
// This function returns an expression representing the specified subtensor of the given tensor
// declaration operation.
*/
template< AlignmentFlag AF    // Alignment flag
        , size_t... CSAs      // Compile time subtensor arguments
        , typename TT         // Tensor base type of the expression
        , typename... RSAs >  // Runtime subtensor arguments
inline decltype(auto) subtensor( const DeclExpr<TT>& tensor, RSAs... args )
{
   BLAZE_FUNCTION_TRACE;

   return subtensor<AF,CSAs...>( (~tensor).operand(), args... );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific subtensor of the given tensor transpose operation.
// \ingroup subtensor
//
// \param tensor The constant tensor transpose operation.
// \param args Optional subtensor arguments.
// \return View on the specified subtensor of the transpose operation.
//
// This function returns an expression representing the specified subtensor of the given tensor
// transpose operation.
*/
template< AlignmentFlag AF    // Alignment flag
        , size_t K            // Index of the first page
        , size_t I            // Index of the first row
        , size_t J            // Index of the first column
        , size_t O            // Number of pages
        , size_t M            // Number of rows
        , size_t N            // Number of columns
        , typename TT         // Tensor base type of the expression
        , typename... RSAs >  // Optional subtensor arguments
inline decltype(auto) subtensor( const MatTransExpr<TT>& tensor, RSAs... args )
{
   BLAZE_FUNCTION_TRACE;

   return trans( subtensor<AF,K,I,J,O,M,N>( (~tensor).operand(), args... ) );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific subtensor of the given tensor transpose operation.
// \ingroup subtensor
//
// \param tensor The constant tensor transpose operation.
// \param row The index of the first row of the subtensor.
// \param column The index of the first column of the subtensor.
// \param m The number of rows of the subtensor.
// \param n The number of columns of the subtensor.
// \param args Optional subtensor arguments.
// \return View on the specified subtensor of the transpose operation.
//
// This function returns an expression representing the specified subtensor of the given tensor
// transpose operation.
*/
// template< AlignmentFlag AF    // Alignment flag
//         , typename TT         // Tensor base type of the expression
//         , typename... RSAs >  // Optional subtensor arguments
// inline decltype(auto)
//    subtensor( const TensTransExpr<TT>& tensor, size_t page, size_t row, size_t column, size_t o, size_t m, size_t n, RSAs... args )
// {
//    BLAZE_FUNCTION_TRACE;
//
//    return trans( subtensor<AF>( (~tensor).operand(), page, column, row, n, m, o, args... ) );
// }
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific subtensor of another subtensor.
// \ingroup subtensor
//
// \param sm The given subtensor
// \param args The optional subtensor arguments.
// \return View on the specified subtensor of the other subtensor.
//
// This function returns an expression representing the specified subtensor of the given subtensor.
*/
template< AlignmentFlag AF1   // Required alignment flag
        , size_t I1           // Required index of the first row
        , size_t J1           // Required index of the first column
        , size_t K1           // Required index of the first page
        , size_t M1           // Required number of rows
        , size_t N1           // Required number of columns
        , size_t O1           // Required number of pages
        , typename TT         // Type of the sparse subtensor
        , AlignmentFlag AF2   // Present alignment flag
        , size_t I2           // Present index of the first row
        , size_t J2           // Present index of the first column
        , size_t K2           // Present index of the first page
        , size_t M2           // Present number of rows
        , size_t N2           // Present number of columns
        , size_t O2           // Present number of pages
        , typename... RSAs >  // Optional subtensor arguments
inline decltype(auto) subtensor( Subtensor<TT,AF2,I2,J2,K2,M2,N2,O2>& sm, RSAs... args )
{
   BLAZE_FUNCTION_TRACE;

   BLAZE_STATIC_ASSERT_MSG( I1 + M1 <= M2, "Invalid subtensor specification" );
   BLAZE_STATIC_ASSERT_MSG( J1 + N1 <= N2, "Invalid subtensor specification" );
   BLAZE_STATIC_ASSERT_MSG( K1 + O1 <= K2, "Invalid subtensor specification" );

   return subtensor<AF1,I1+I2,J1+J2,K1+K2,M1,N1,O1>( sm.operand(), args... );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific subtensor of another constant subtensor.
// \ingroup subtensor
//
// \param sm The given constant subtensor
// \param args The optional subtensor arguments.
// \return View on the specified subtensor of the other subtensor.
//
// This function returns an expression representing the specified subtensor of the given constant
// subtensor.
*/
template< AlignmentFlag AF1   // Required alignment flag
        , size_t I1           // Required index of the first row
        , size_t J1           // Required index of the first column
        , size_t K1           // Required index of the first page
        , size_t M1           // Required number of rows
        , size_t N1           // Required number of columns
        , size_t O1           // Required number of pages
        , typename TT         // Type of the sparse subtensor
        , AlignmentFlag AF2   // Present alignment flag
        , size_t I2           // Present index of the first row
        , size_t J2           // Present index of the first column
        , size_t K2           // Present index of the first page
        , size_t M2           // Present number of rows
        , size_t N2           // Present number of columns
        , size_t O2           // Present number of pages
        , typename... RSAs >  // Optional subtensor arguments
inline decltype(auto) subtensor( const Subtensor<TT,AF2,I2,J2,K2,M2,N2,O2>& sm, RSAs... args )
{
   BLAZE_FUNCTION_TRACE;

   BLAZE_STATIC_ASSERT_MSG( I1 + M1 <= M2, "Invalid subtensor specification" );
   BLAZE_STATIC_ASSERT_MSG( J1 + N1 <= N2, "Invalid subtensor specification" );
   BLAZE_STATIC_ASSERT_MSG( K1 + O1 <= K2, "Invalid subtensor specification" );

   return subtensor<AF1,I1+I2,J1+J2,K1+K2,M1,N1,O1>( sm.operand(), args... );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific subtensor of another temporary subtensor.
// \ingroup subtensor
//
// \param sm The given temporary subtensor
// \param args The optional subtensor arguments.
// \return View on the specified subtensor of the other subtensor.
//
// This function returns an expression representing the specified subtensor of the given temporary
// subtensor.
*/
template< AlignmentFlag AF1   // Required alignment flag
        , size_t I1           // Required index of the first row
        , size_t J1           // Required index of the first column
        , size_t K1           // Required index of the first page
        , size_t M1           // Required number of rows
        , size_t N1           // Required number of columns
        , size_t O1           // Required number of pages
        , typename TT         // Type of the sparse subtensor
        , AlignmentFlag AF2   // Present alignment flag
        , size_t I2           // Present index of the first row
        , size_t J2           // Present index of the first column
        , size_t K2           // Present index of the first page
        , size_t M2           // Present number of rows
        , size_t N2           // Present number of columns
        , size_t O2           // Present number of pages
        , typename... RSAs >  // Optional subtensor arguments
inline decltype(auto) subtensor( Subtensor<TT,AF2,I2,J2,K2,M2,N2,O2>&& sm, RSAs... args )
{
   BLAZE_FUNCTION_TRACE;

   BLAZE_STATIC_ASSERT_MSG( I1 + M1 <= M2, "Invalid subtensor specification" );
   BLAZE_STATIC_ASSERT_MSG( J1 + N1 <= N2, "Invalid subtensor specification" );
   BLAZE_STATIC_ASSERT_MSG( K1 + O1 <= K2, "Invalid subtensor specification" );

   return subtensor<AF1,I1+I2,J1+J2,K1+K2,M1,N1,O1>( sm.operand(), args... );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific subtensor of another subtensor.
// \ingroup subtensor
//
// \param sm The given subtensor
// \param args The optional subtensor arguments.
// \return View on the specified subtensor of the other subtensor.
// \exception std::invalid_argument Invalid subtensor specification.
//
// This function returns an expression representing the specified subtensor of the given subtensor.
*/
template< AlignmentFlag AF1   // Required alignment flag
        , size_t K            // Index of the first page
        , size_t I            // Index of the first row
        , size_t J            // Index of the first column
        , size_t O            // Number of pages
        , size_t M            // Number of rows
        , size_t N            // Number of columns
        , typename TT         // Type of the sparse subtensor
        , AlignmentFlag AF2   // Present alignment flag
        , typename... RSAs >  // Optional subtensor arguments
inline decltype(auto) subtensor( Subtensor<TT,AF2>& sm, RSAs... args )
{
   BLAZE_FUNCTION_TRACE;

   constexpr bool isChecked( !Contains_v< TypeList<RSAs...>, Unchecked > );

   if( isChecked ) {
      if( ( I + M > sm.rows() ) || ( J + N > sm.columns() ) || ( K + O > sm.pages() ) ) {
         BLAZE_THROW_INVALID_ARGUMENT( "Invalid subtensor specification" );
      }
   }
   else {
      BLAZE_USER_ASSERT( I + M <= sm.rows()   , "Invalid subtensor specification" );
      BLAZE_USER_ASSERT( J + N <= sm.columns(), "Invalid subtensor specification" );
      BLAZE_USER_ASSERT( K + O <= sm.pages()  , "Invalid subtensor specification" );
   }

   return subtensor<AF1>( sm.operand(), sm.pages() + K, sm.row() + I, sm.column() + J, o, m, n, args... );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific subtensor of another constant subtensor.
// \ingroup subtensor
//
// \param sm The constant subtensor
// \param args The optional subtensor arguments.
// \return View on the specified subtensor of the other subtensor.
// \exception std::invalid_argument Invalid subtensor specification.
//
// This function returns an expression representing the specified subtensor of the given constant
// subtensor.
*/
template< AlignmentFlag AF1   // Required alignment flag
        , size_t K            // Index of the first page
        , size_t I            // Index of the first row
        , size_t J            // Index of the first column
        , size_t O            // Number of pages
        , size_t M            // Number of rows
        , size_t N            // Number of columns
        , typename TT         // Type of the sparse subtensor
        , AlignmentFlag AF2   // Present alignment flag
        , typename... RSAs >  // Optional subtensor arguments
inline decltype(auto) subtensor( const Subtensor<TT,AF2>& sm, RSAs... args )
{
   BLAZE_FUNCTION_TRACE;

   constexpr bool isChecked( !Contains_v< TypeList<RSAs...>, Unchecked > );

   if( isChecked ) {
      if( ( I + M > sm.rows() ) || ( J + N > sm.columns() ) || ( K + O > sm.pages() ) ) {
         BLAZE_THROW_INVALID_ARGUMENT( "Invalid subtensor specification" );
      }
   }
   else {
      BLAZE_USER_ASSERT( I + M <= sm.rows()   , "Invalid subtensor specification" );
      BLAZE_USER_ASSERT( J + N <= sm.columns(), "Invalid subtensor specification" );
      BLAZE_USER_ASSERT( K + O <= sm.pages()  , "Invalid subtensor specification" );
   }

   return subtensor<AF1>( sm.operand(), sm.pages() + K, sm.row() + I, sm.column() + J, o, m, n, args... );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific subtensor of another temporary subtensor.
// \ingroup subtensor
//
// \param sm The temporary subtensor
// \param args The optional subtensor arguments.
// \return View on the specified subtensor of the other subtensor.
// \exception std::invalid_argument Invalid subtensor specification.
//
// This function returns an expression representing the specified subtensor of the given temporary
// subtensor.
*/
template< AlignmentFlag AF1   // Required alignment flag
        , size_t K            // Index of the first page
        , size_t I            // Index of the first row
        , size_t J            // Index of the first column
        , size_t O            // Number of pages
        , size_t M            // Number of rows
        , size_t N            // Number of columns
        , typename TT         // Type of the sparse subtensor
        , AlignmentFlag AF2   // Present alignment flag
        , typename... RSAs >  // Optional subtensor arguments
inline decltype(auto) subtensor( Subtensor<TT,AF2>&& sm, RSAs... args )
{
   BLAZE_FUNCTION_TRACE;

   constexpr bool isChecked( !Contains_v< TypeList<RSAs...>, Unchecked > );

   if( isChecked ) {
      if( ( I + M > sm.rows() ) || ( J + N > sm.columns() ) || ( K + O > sm.pages() ) ) {
         BLAZE_THROW_INVALID_ARGUMENT( "Invalid subtensor specification" );
      }
   }
   else {
      BLAZE_USER_ASSERT( I + M <= sm.rows()   , "Invalid subtensor specification" );
      BLAZE_USER_ASSERT( J + N <= sm.columns(), "Invalid subtensor specification" );
      BLAZE_USER_ASSERT( K + O <= sm.pages()  , "Invalid subtensor specification" );
   }

   return subtensor<AF1>( sm.operand(), sm.pages() + K, sm.row() + I, sm.column() + J, o, m, n, args... );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific subtensor of another subtensor.
// \ingroup subtensor
//
// \param sm The given subtensor
// \param row The index of the first row of the subtensor.
// \param column The index of the first column of the subtensor.
// \param m The number of rows of the subtensor.
// \param n The number of columns of the subtensor.
// \param args The optional subtensor arguments.
// \return View on the specified subtensor of the other subtensor.
// \exception std::invalid_argument Invalid subtensor specification.
//
// This function returns an expression representing the specified subtensor of the given subtensor.
*/
template< AlignmentFlag AF1   // Required alignment flag
        , typename TT         // Type of the sparse subtensor
        , AlignmentFlag AF2   // Present alignment flag
        , size_t... CSAs      // Compile time subtensor arguments
        , typename... RSAs >  // Optional subtensor arguments
inline decltype(auto)
   subtensor( Subtensor<TT,AF2,CSAs...>& sm, size_t page, size_t row, size_t column,
              size_t o, size_t m, size_t n, RSAs... args )
{
   BLAZE_FUNCTION_TRACE;

   constexpr bool isChecked( !Contains_v< TypeList<RSAs...>, Unchecked > );

   if( isChecked ) {
      if( ( row + m > sm.rows() ) || ( column + n > sm.columns() ) || ( page + o > sm.pages() ) ) {
         BLAZE_THROW_INVALID_ARGUMENT( "Invalid subtensor specification" );
      }
   }
   else {
      BLAZE_USER_ASSERT( row    + m <= sm.rows()   , "Invalid subtensor specification" );
      BLAZE_USER_ASSERT( column + n <= sm.columns(), "Invalid subtensor specification" );
      BLAZE_USER_ASSERT( page   + o <= sm.pages()  , "Invalid subtensor specification" );
   }

   return subtensor<AF1>( sm.operand(), sm.page() + page, sm.row() + row, sm.column() + column, o, m, n, args... );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific subtensor of another constant subtensor.
// \ingroup subtensor
//
// \param sm The given constant subtensor
// \param row The index of the first row of the subtensor.
// \param column The index of the first column of the subtensor.
// \param m The number of rows of the subtensor.
// \param n The number of columns of the subtensor.
// \param args The optional subtensor arguments.
// \return View on the specified subtensor of the other subtensor.
// \exception std::invalid_argument Invalid subtensor specification.
//
// This function returns an expression representing the specified subtensor of the given constant
// subtensor.
*/
template< AlignmentFlag AF1   // Required alignment flag
        , typename TT         // Type of the sparse subtensor
        , AlignmentFlag AF2   // Present alignment flag
        , size_t... CSAs      // Compile time subtensor arguments
        , typename... RSAs >  // Optional subtensor arguments
inline decltype(auto)
   subtensor( const Subtensor<TT,AF2,CSAs...>& sm, size_t page, size_t row, size_t column,
              size_t o, size_t m, size_t n, RSAs... args )
{
   BLAZE_FUNCTION_TRACE;

   constexpr bool isChecked( !Contains_v< TypeList<RSAs...>, Unchecked > );

   if( isChecked ) {
      if( ( row + m > sm.rows() ) || ( column + n > sm.columns() ) || ( page + o > sm.pages() ) ) {
         BLAZE_THROW_INVALID_ARGUMENT( "Invalid subtensor specification" );
      }
   }
   else {
      BLAZE_USER_ASSERT( row    + m <= sm.rows()   , "Invalid subtensor specification" );
      BLAZE_USER_ASSERT( column + n <= sm.columns(), "Invalid subtensor specification" );
      BLAZE_USER_ASSERT( page   + o <= sm.pages()  , "Invalid subtensor specification" );
   }

   return subtensor<AF1>( sm.operand(), sm.page() + page, sm.row() + row, sm.column() + column, o, m, n, args... );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific subtensor of another temporary subtensor.
// \ingroup subtensor
//
// \param sm The given temporary subtensor
// \param row The index of the first row of the subtensor.
// \param column The index of the first column of the subtensor.
// \param m The number of rows of the subtensor.
// \param n The number of columns of the subtensor.
// \param args The optional subtensor arguments.
// \return View on the specified subtensor of the other subtensor.
// \exception std::invalid_argument Invalid subtensor specification.
//
// This function returns an expression representing the specified subtensor of the given temporary
// subtensor.
*/
template< AlignmentFlag AF1   // Required alignment flag
        , typename TT         // Type of the sparse subtensor
        , AlignmentFlag AF2   // Present alignment flag
        , size_t... CSAs      // Compile time subtensor arguments
        , typename... RSAs >  // Optional subtensor arguments
inline decltype(auto)
   subtensor( Subtensor<TT,AF2,CSAs...>&& sm, size_t page, size_t row, size_t column,
              size_t o, size_t m, size_t n, RSAs... args )
{
   BLAZE_FUNCTION_TRACE;

   constexpr bool isChecked( !Contains_v< TypeList<RSAs...>, Unchecked > );

   if( isChecked ) {
      if( ( row + m > sm.rows() ) || ( column + n > sm.columns() ) || ( page + o > sm.pages() ) ) {
         BLAZE_THROW_INVALID_ARGUMENT( "Invalid subtensor specification" );
      }
   }
   else {
      BLAZE_USER_ASSERT( row    + m <= sm.rows()   , "Invalid subtensor specification" );
      BLAZE_USER_ASSERT( column + n <= sm.columns(), "Invalid subtensor specification" );
      BLAZE_USER_ASSERT( page   + o <= sm.pages()  , "Invalid subtensor specification" );
   }

   return subtensor<AF1>( sm.operand(), sm.page() + page, sm.row() + row, sm.column() + column, o, m, n, args... );
}
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  GLOBAL RESTRUCTURING FUNCTIONS (SUBVECTOR)
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific subvector of the given tensor/vector multiplication.
// \ingroup subtensor
//
// \param vector The constant tensor/vector multiplication.
// \param args The runtime subvector arguments.
// \return View on the specified subvector of the multiplication.
//
// This function returns an expression representing the specified subvector of the given
// tensor/vector multiplication.
*/
// template< AlignmentFlag AF    // Alignment flag
//         , size_t... CSAs      // Compile time subvector arguments
//         , typename VT         // Vector base type of the expression
//         , typename... RSAs >  // Runtime subvector arguments
// inline decltype(auto) subvector( const MatVecMultExpr<VT>& vector, RSAs... args )
// {
//    BLAZE_FUNCTION_TRACE;
//
//    using TT = RemoveReference_t< LeftOperand_t< VectorType_t<VT> > >;
//
//    const SubvectorData<CSAs...> sd( args... );
//
//    BLAZE_DECLTYPE_AUTO( left , (~vector).leftOperand()  );
//    BLAZE_DECLTYPE_AUTO( right, (~vector).rightOperand() );
//
//    const size_t column( ( IsUpper_v<TT> )
//                         ?( ( !AF && IsStrictlyUpper_v<TT> )?( sd.offset() + 1UL ):( sd.offset() ) )
//                         :( 0UL ) );
//    const size_t n( ( IsLower_v<TT> )
//                    ?( ( IsUpper_v<TT> )?( sd.size() )
//                                        :( ( IsStrictlyLower_v<TT> && sd.size() > 0UL )
//                                           ?( sd.offset() + sd.size() - 1UL )
//                                           :( sd.offset() + sd.size() ) ) )
//                    :( ( IsUpper_v<TT> )?( left.columns() - column )
//                                        :( left.columns() ) ) );
//
//    return subtensor<AF>( left, sd.offset(), column, sd.size(), n ) * subvector<AF>( right, column, n );
// }
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific subvector of the given vector/tensor multiplication.
// \ingroup subtensor
//
// \param vector The constant vector/tensor multiplication.
// \param args The runtime subvector arguments.
// \return View on the specified subvector of the multiplication.
//
// This function returns an expression representing the specified subvector of the given
// vector/tensor multiplication.
*/
// template< AlignmentFlag AF    // Alignment flag
//         , size_t... CSAs      // Compile time subvector arguments
//         , typename VT         // Vector base type of the expression
//         , typename... RSAs >  // Runtime subvector arguments
// inline decltype(auto) subvector( const TVecMatMultExpr<VT>& vector, RSAs... args )
// {
//    BLAZE_FUNCTION_TRACE;
//
//    using TT = RemoveReference_t< RightOperand_t< VectorType_t<VT> > >;
//
//    const SubvectorData<CSAs...> sd( args... );
//
//    BLAZE_DECLTYPE_AUTO( left , (~vector).leftOperand()  );
//    BLAZE_DECLTYPE_AUTO( right, (~vector).rightOperand() );
//
//    const size_t row( ( IsLower_v<TT> )
//                      ?( ( !AF && IsStrictlyLower_v<TT> )?( sd.offset() + 1UL ):( sd.offset() ) )
//                      :( 0UL ) );
//    const size_t m( ( IsUpper_v<TT> )
//                    ?( ( IsLower_v<TT> )?( sd.size() )
//                                        :( ( IsStrictlyUpper_v<TT> && sd.size() > 0UL )
//                                           ?( sd.offset() + sd.size() - 1UL )
//                                           :( sd.offset() + sd.size() ) ) )
//                    :( ( IsLower_v<TT> )?( right.rows() - row )
//                                        :( right.rows() ) ) );
//
//    return subvector<AF>( left, row, m ) * subtensor<AF>( right, row, sd.offset(), m, sd.size() );
// }
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific subvector of the given column-wise tensor reduction operation.
// \ingroup subtensor
//
// \param vector The constant column-wise tensor reduction operation.
// \param args The runtime subvector arguments.
// \return View on the specified subvector of the reduction operation.
//
// This function returns an expression representing the specified subvector of the given
// column-wise tensor reduction operation.
*/
// template< AlignmentFlag AF    // Alignment flag
//         , size_t... CSAs      // Compile time subvector arguments
//         , typename VT         // Vector base type of the expression
//         , typename... RSAs >  // Runtime subvector arguments
// inline decltype(auto) subvector( const MatReduceExpr<VT,columnwise>& vector, RSAs... args )
// {
//    BLAZE_FUNCTION_TRACE;
//
//    const SubvectorData<CSAs...> sd( args... );
//    const size_t M( (~vector).operand().rows() );
//
//    decltype(auto) sm( subtensor<AF>( (~vector).operand(), 0UL, sd.offset(), M, sd.size() ) );
//    return reduce<columnwise>( sm, (~vector).operation() );
// }
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific subvector of the given row-wise tensor reduction operation.
// \ingroup subtensor
//
// \param vector The constant row-wise tensor reduction operation.
// \param args The runtime subvector arguments.
// \return View on the specified subvector of the reduction operation.
//
// This function returns an expression representing the specified subvector of the given row-wise
// tensor reduction operation.
*/
// template< AlignmentFlag AF    // Alignment flag
//         , size_t... CSAs      // Compile time subvector arguments
//         , typename VT         // Vector base type of the expression
//         , typename... RSAs >  // Runtime subvector arguments
// inline decltype(auto) subvector( const MatReduceExpr<VT,rowwise>& vector, RSAs... args )
// {
//    BLAZE_FUNCTION_TRACE;
//
//    const SubvectorData<CSAs...> sd( args... );
//    const size_t N( (~vector).operand().columns() );
//
//    decltype(auto) sm( subtensor<AF>( (~vector).operand(), sd.offset(), 0UL, sd.size(), N ) );
//    return reduce<rowwise>( sm, (~vector).operation() );
// }
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  GLOBAL RESTRUCTURING FUNCTIONS (ROW)
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific row of the given subtensor.
// \ingroup subtensor
//
// \param sm The subtensor containing the row.
// \param args The optional row arguments.
// \return View on the specified row of the subtensor.
//
// This function returns an expression representing the specified row of the given subtensor.
*/
// template< size_t I1           // Row index
//         , typename TT         // Type of the sparse subtensor
//         , AlignmentFlag AF    // Alignment flag
//         , bool SO             // Storage order
//         , bool DF             // Density flag
//         , size_t I2           // Index of the first row
//         , size_t J            // Index of the first column
//         , size_t K            // Index of the first page
//         , size_t M            // Number of rows
//         , size_t N            // Number of columns
//         , size_t O            // Number of pages
//         , typename... RRAs >  // Optional row arguments
// inline decltype(auto) row( Subtensor<TT,AF,I2,J,K,o,m,n>& sm, RRAs... args )
// {
//    BLAZE_FUNCTION_TRACE;
//
//    BLAZE_STATIC_ASSERT_MSG( I1 < M, "Invalid row access index" );
//
//    return subvector<J,N>( row<I1+I2>( sm.operand(), args... ), unchecked );
// }
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific row of the given constant subtensor.
// \ingroup subtensor
//
// \param sm The constant subtensor containing the row.
// \param args The optional row arguments.
// \return View on the specified row of the subtensor.
//
// This function returns an expression representing the specified row of the given constant
// subtensor.
*/
// template< size_t I1           // Row index
//         , typename TT         // Type of the sparse subtensor
//         , AlignmentFlag AF    // Alignment flag
//         , bool SO             // Storage order
//         , bool DF             // Density flag
//         , size_t I2           // Index of the first row
//         , size_t J            // Index of the first column
//         , size_t M            // Number of rows
//         , size_t N            // Number of columns
//         , typename... RRAs >  // Optional row arguments
// inline decltype(auto) row( const Subtensor<TT,AF,I2,J,M,N>& sm, RRAs... args )
// {
//    BLAZE_FUNCTION_TRACE;
//
//    BLAZE_STATIC_ASSERT_MSG( I1 < M, "Invalid row access index" );
//
//    return subvector<J,N>( row<I1+I2>( sm.operand(), args... ), unchecked );
// }
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific row of the given temporary subtensor.
// \ingroup subtensor
//
// \param sm The temporary subtensor containing the row.
// \param args The optional row arguments.
// \return View on the specified row of the subtensor.
//
// This function returns an expression representing the specified row of the given temporary
// subtensor.
*/
// template< size_t I1           // Row index
//         , typename TT         // Type of the sparse subtensor
//         , AlignmentFlag AF    // Alignment flag
//         , bool SO             // Storage order
//         , bool DF             // Density flag
//         , size_t I2           // Index of the first row
//         , size_t J            // Index of the first column
//         , size_t M            // Number of rows
//         , size_t N            // Number of columns
//         , typename... RRAs >  // Optional row arguments
// inline decltype(auto) row( Subtensor<TT,AF,I2,J,M,N>&& sm, RRAs... args )
// {
//    BLAZE_FUNCTION_TRACE;
//
//    BLAZE_STATIC_ASSERT_MSG( I1 < M, "Invalid row access index" );
//
//    return subvector<J,N>( row<I1+I2>( sm.operand(), args... ), unchecked );
// }
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific row of the given subtensor.
// \ingroup subtensor
//
// \param sm The subtensor containing the row.
// \param index The index of the row.
// \param args The optional row arguments.
// \return View on the specified row of the subtensor.
// \exception std::invalid_argument Invalid row access index.
//
// This function returns an expression representing the specified row of the given subtensor.
*/
// template< typename TT         // Type of the sparse subtensor
//         , AlignmentFlag AF    // Alignment flag
//         , bool SO             // Storage order
//         , bool DF             // Density flag
//         , size_t I            // Index of the first row
//         , size_t J            // Index of the first column
//         , size_t M            // Number of rows
//         , size_t N            // Number of columns
//         , typename... RRAs >  // Optional row arguments
// inline decltype(auto) row( Subtensor<TT,AF,K,I,J,O,M,N>& sm, size_t index, RRAs... args )
// {
//    BLAZE_FUNCTION_TRACE;
//
//    constexpr bool isChecked( !Contains_v< TypeList<RRAs...>, Unchecked > );
//
//    if( isChecked ) {
//       if( ( index >= M ) ) {
//          BLAZE_THROW_INVALID_ARGUMENT( "Invalid row access index" );
//       }
//    }
//    else {
//       BLAZE_USER_ASSERT( index < M, "Invalid row access index" );
//    }
//
//    return subvector<J,N>( row( sm.operand(), I+index, args... ), unchecked );
// }
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific row of the given constant subtensor.
// \ingroup subtensor
//
// \param sm The constant subtensor containing the row.
// \param index The index of the row.
// \param args The optional row arguments.
// \return View on the specified row of the subtensor.
// \exception std::invalid_argument Invalid row access index.
//
// This function returns an expression representing the specified row of the given constant
// subtensor.
*/
// template< typename TT         // Type of the sparse subtensor
//         , AlignmentFlag AF    // Alignment flag
//         , bool SO             // Storage order
//         , bool DF             // Density flag
//         , size_t I            // Index of the first row
//         , size_t J            // Index of the first column
//         , size_t M            // Number of rows
//         , size_t N            // Number of columns
//         , typename... RRAs >  // Optional row arguments
// inline decltype(auto) row( const Subtensor<TT,AF,K,I,J,O,M,N>& sm, size_t index, RRAs... args )
// {
//    BLAZE_FUNCTION_TRACE;
//
//    constexpr bool isChecked( !Contains_v< TypeList<RRAs...>, Unchecked > );
//
//    if( isChecked ) {
//       if( ( index >= M ) ) {
//          BLAZE_THROW_INVALID_ARGUMENT( "Invalid row access index" );
//       }
//    }
//    else {
//       BLAZE_USER_ASSERT( index < M, "Invalid row access index" );
//    }
//
//    return subvector<J,N>( row( sm.operand(), I+index, args... ), unchecked );
// }
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific row of the given temporary subtensor.
// \ingroup subtensor
//
// \param sm The temporary subtensor containing the row.
// \param index The index of the row.
// \param args The optional row arguments.
// \return View on the specified row of the subtensor.
// \exception std::invalid_argument Invalid row access index.
//
// This function returns an expression representing the specified row of the given temporary
// subtensor.
*/
// template< typename TT         // Type of the sparse subtensor
//         , AlignmentFlag AF    // Alignment flag
//         , bool SO             // Storage order
//         , bool DF             // Density flag
//         , size_t I            // Index of the first row
//         , size_t J            // Index of the first column
//         , size_t M            // Number of rows
//         , size_t N            // Number of columns
//         , typename... RRAs >  // Optional row arguments
// inline decltype(auto) row( Subtensor<TT,AF,K,I,J,O,M,N>&& sm, size_t index, RRAs... args )
// {
//    BLAZE_FUNCTION_TRACE;
//
//    constexpr bool isChecked( !Contains_v< TypeList<RRAs...>, Unchecked > );
//
//    if( isChecked ) {
//       if( ( index >= M ) ) {
//          BLAZE_THROW_INVALID_ARGUMENT( "Invalid row access index" );
//       }
//    }
//    else {
//       BLAZE_USER_ASSERT( index < M, "Invalid row access index" );
//    }
//
//    return subvector<J,N>( row( sm.operand(), I+index, args... ), unchecked );
// }
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific row of the given subtensor.
// \ingroup subtensor
//
// \param sm The subtensor containing the row.
// \param args The optional row arguments.
// \return View on the specified row of the subtensor.
// \exception std::invalid_argument Invalid row access index.
//
// This function returns an expression representing the specified row of the given subtensor.
*/
// template< size_t... CRAs      // Compile time row arguments
//         , typename TT         // Type of the sparse subtensor
//         , AlignmentFlag AF    // Alignment flag
//         , bool SO             // Storage order
//         , bool DF             // Density flag
//         , typename... RRAs >  // Runtime row arguments
// inline decltype(auto) row( Subtensor<TT,AF>& sm, RRAs... args )
// {
//    BLAZE_FUNCTION_TRACE;
//
//    const RowData<CRAs...> rd( args... );
//
//    constexpr bool isChecked( !Contains_v< TypeList<RRAs...>, Unchecked > );
//
//    if( isChecked ) {
//       if( ( rd.row() >= sm.rows() ) ) {
//          BLAZE_THROW_INVALID_ARGUMENT( "Invalid row access index" );
//       }
//    }
//    else {
//       BLAZE_USER_ASSERT( rd.row() < sm.rows(), "Invalid row access index" );
//    }
//
//    const size_t index( rd.row() + sm.row() );
//
//    return subvector( row( sm.operand(), index, args... ), sm.column(), sm.columns(), unchecked );
// }
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific row of the given constant subtensor.
// \ingroup subtensor
//
// \param sm The constant subtensor containing the row.
// \param args The optional row arguments.
// \return View on the specified row of the subtensor.
// \exception std::invalid_argument Invalid row access index.
//
// This function returns an expression representing the specified row of the given constant
// subtensor.
*/
// template< size_t... CRAs      // Compile time row arguments
//         , typename TT         // Type of the sparse subtensor
//         , AlignmentFlag AF    // Alignment flag
//         , bool SO             // Storage order
//         , bool DF             // Density flag
//         , typename... RRAs >  // Runtime row arguments
// inline decltype(auto) row( const Subtensor<TT,AF>& sm, RRAs... args )
// {
//    BLAZE_FUNCTION_TRACE;
//
//    const RowData<CRAs...> rd( args... );
//
//    constexpr bool isChecked( !Contains_v< TypeList<RRAs...>, Unchecked > );
//
//    if( isChecked ) {
//       if( ( rd.row() >= sm.rows() ) ) {
//          BLAZE_THROW_INVALID_ARGUMENT( "Invalid row access index" );
//       }
//    }
//    else {
//       BLAZE_USER_ASSERT( rd.row() < sm.rows(), "Invalid row access index" );
//    }
//
//    const size_t index( rd.row() + sm.row() );
//
//    return subvector( row( sm.operand(), index, args... ), sm.column(), sm.columns(), unchecked );
// }
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific row of the given temporary subtensor.
// \ingroup subtensor
//
// \param sm The temporary subtensor containing the row.
// \param args The optional row arguments.
// \return View on the specified row of the subtensor.
// \exception std::invalid_argument Invalid row access index.
//
// This function returns an expression representing the specified row of the given temporary
// subtensor.
*/
// template< size_t... CRAs      // Compile time row arguments
//         , typename TT         // Type of the sparse subtensor
//         , AlignmentFlag AF    // Alignment flag
//         , bool SO             // Storage order
//         , bool DF             // Density flag
//         , typename... RRAs >  // Runtime row arguments
// inline decltype(auto) row( Subtensor<TT,AF>&& sm, RRAs... args )
// {
//    BLAZE_FUNCTION_TRACE;
//
//    const RowData<CRAs...> rd( args... );
//
//    constexpr bool isChecked( !Contains_v< TypeList<RRAs...>, Unchecked > );
//
//    if( isChecked ) {
//       if( ( rd.row() >= sm.rows() ) ) {
//          BLAZE_THROW_INVALID_ARGUMENT( "Invalid row access index" );
//       }
//    }
//    else {
//       BLAZE_USER_ASSERT( rd.row() < sm.rows(), "Invalid row access index" );
//    }
//
//    const size_t index( rd.row() + sm.row() );
//
//    return subvector( row( sm.operand(), index, args... ), sm.column(), sm.columns(), unchecked );
// }
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  GLOBAL RESTRUCTURING FUNCTIONS (ROWS)
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on specific rows of the given subtensor.
// \ingroup subtensor
//
// \param sm The subtensor containing the rows.
// \param args The optional row arguments.
// \return View on the specified rows of the subtensor.
//
// This function returns an expression representing the specified rows of the given subtensor.
*/
// template< size_t I1           // First row index
//         , size_t... Is        // Remaining row indices
//         , typename TT         // Type of the sparse subtensor
//         , AlignmentFlag AF    // Alignment flag
//         , bool SO             // Storage order
//         , bool DF             // Density flag
//         , size_t I2           // Index of the first row
//         , size_t J            // Index of the first column
//         , size_t M            // Number of rows
//         , size_t N            // Number of columns
//         , typename... RRAs >  // Optional row arguments
// inline decltype(auto) rows( Subtensor<TT,AF,I2,J,M,N>& sm, RRAs... args )
// {
//    BLAZE_FUNCTION_TRACE;
//
//    return subtensor<0UL,J,sizeof...(Is)+1UL,N>(
//       rows( sm.operand(), make_shifted_index_subsequence<I2,M,I1,Is...>(), args... ), unchecked );
// }
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on specific rows of the given constant subtensor.
// \ingroup subtensor
//
// \param sm The constant subtensor containing the rows.
// \param args The optional row arguments.
// \return View on the specified rows of the subtensor.
//
// This function returns an expression representing the specified rows of the given constant
// subtensor.
*/
// template< size_t I1           // First row index
//         , size_t... Is        // Remaining row indices
//         , typename TT         // Type of the sparse subtensor
//         , AlignmentFlag AF    // Alignment flag
//         , bool SO             // Storage order
//         , bool DF             // Density flag
//         , size_t I2           // Index of the first row
//         , size_t J            // Index of the first column
//         , size_t M            // Number of rows
//         , size_t N            // Number of columns
//         , typename... RRAs >  // Optional row arguments
// inline decltype(auto) rows( const Subtensor<TT,AF,I2,J,M,N>& sm, RRAs... args )
// {
//    BLAZE_FUNCTION_TRACE;
//
//    return subtensor<0UL,J,sizeof...(Is)+1UL,N>(
//       rows( sm.operand(), make_shifted_index_subsequence<I2,M,I1,Is...>(), args... ), unchecked );
// }
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on specific rows of the given temporary subtensor.
// \ingroup subtensor
//
// \param sm The temporary subtensor containing the rows.
// \param args The optional row arguments.
// \return View on the specified rows of the subtensor.
//
// This function returns an expression representing the specified rows of the given temporary
// subtensor.
*/
// template< size_t I1           // First row index
//         , size_t... Is        // Remaining row indices
//         , typename TT         // Type of the sparse subtensor
//         , AlignmentFlag AF    // Alignment flag
//         , bool SO             // Storage order
//         , bool DF             // Density flag
//         , size_t I2           // Index of the first row
//         , size_t J            // Index of the first column
//         , size_t M            // Number of rows
//         , size_t N            // Number of columns
//         , typename... RRAs >  // Optional row arguments
// inline decltype(auto) rows( Subtensor<TT,AF,I2,J,M,N>&& sm, RRAs... args )
// {
//    BLAZE_FUNCTION_TRACE;
//
//    return subtensor<0UL,J,sizeof...(Is)+1UL,N>(
//       rows( sm.operand(), make_shifted_index_subsequence<I2,M,I1,Is...>(), args... ), unchecked );
// }
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on specific rows of the given subtensor.
// \ingroup subtensor
//
// \param sm The subtensor containing the rows.
// \param args The optional row arguments.
// \return View on the specified rows of the subtensor.
// \exception std::invalid_argument Invalid row access index.
//
// This function returns an expression representing the specified rows of the given subtensor.
*/
// template< size_t I1           // First row index
//         , size_t... Is        // Remaining row indices
//         , typename TT         // Type of the sparse subtensor
//         , AlignmentFlag AF    // Alignment flag
//         , bool SO             // Storage order
//         , bool DF             // Density flag
//         , typename... RRAs >  // Optional row arguments
// inline decltype(auto) rows( Subtensor<TT,AF>& sm, RRAs... args )
// {
//    BLAZE_FUNCTION_TRACE;
//
//    constexpr bool isChecked( !Contains_v< TypeList<RRAs...>, Unchecked > );
//
//    if( isChecked ) {
//       static constexpr size_t indices[] = { I1, Is... };
//       for( size_t i=0UL; i<sizeof...(Is)+1UL; ++i ) {
//          if( sm.rows() <= indices[i] ) {
//             BLAZE_THROW_INVALID_ARGUMENT( "Invalid row access index" );
//          }
//       }
//    }
//
//    return subtensor( rows( sm.operand(), { I1+sm.row(), Is+sm.row()... }, args... ),
//                      0UL, sm.column(), sizeof...(Is)+1UL, sm.columns(), unchecked );
// }
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on specific rows of the given constant subtensor.
// \ingroup subtensor
//
// \param sm The constant subtensor containing the rows.
// \param args The optional row arguments.
// \return View on the specified rows of the subtensor.
// \exception std::invalid_argument Invalid row access index.
//
// This function returns an expression representing the specified rows of the given constant
// subtensor.
*/
// template< size_t I1           // First row index
//         , size_t... Is        // Remaining row indices
//         , typename TT         // Type of the sparse subtensor
//         , AlignmentFlag AF    // Alignment flag
//         , bool SO             // Storage order
//         , bool DF             // Density flag
//         , typename... RRAs >  // Optional row arguments
// inline decltype(auto) rows( const Subtensor<TT,AF>& sm, RRAs... args )
// {
//    BLAZE_FUNCTION_TRACE;
//
//    constexpr bool isChecked( !Contains_v< TypeList<RRAs...>, Unchecked > );
//
//    if( isChecked ) {
//       static constexpr size_t indices[] = { I1, Is... };
//       for( size_t i=0UL; i<sizeof...(Is)+1UL; ++i ) {
//          if( sm.rows() <= indices[i] ) {
//             BLAZE_THROW_INVALID_ARGUMENT( "Invalid row access index" );
//          }
//       }
//    }
//
//    return subtensor( rows( sm.operand(), { I1+sm.row(), Is+sm.row()... }, args... ),
//                      0UL, sm.column(), sizeof...(Is)+1UL, sm.columns(), unchecked );
// }
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on specific rows of the given temporary subtensor.
// \ingroup subtensor
//
// \param sm The temporary subtensor containing the rows.
// \param args The optional row arguments.
// \return View on the specified rows of the subtensor.
// \exception std::invalid_argument Invalid row access index.
//
// This function returns an expression representing the specified rows of the given temporary
// subtensor.
*/
// template< size_t I1           // First row index
//         , size_t... Is        // Remaining row indices
//         , typename TT         // Type of the sparse subtensor
//         , AlignmentFlag AF    // Alignment flag
//         , bool SO             // Storage order
//         , bool DF             // Density flag
//         , typename... RRAs >  // Optional row arguments
// inline decltype(auto) rows( Subtensor<TT,AF>&& sm, RRAs... args )
// {
//    BLAZE_FUNCTION_TRACE;
//
//    constexpr bool isChecked( !Contains_v< TypeList<RRAs...>, Unchecked > );
//
//    if( isChecked ) {
//       static constexpr size_t indices[] = { I1, Is... };
//       for( size_t i=0UL; i<sizeof...(Is)+1UL; ++i ) {
//          if( sm.rows() <= indices[i] ) {
//             BLAZE_THROW_INVALID_ARGUMENT( "Invalid row access index" );
//          }
//       }
//    }
//
//    return subtensor( rows( sm.operand(), { I1+sm.row(), Is+sm.row()... }, args... ),
//                      0UL, sm.column(), sizeof...(Is)+1UL, sm.columns(), unchecked );
// }
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on specific rows of the given subtensor.
// \ingroup subtensor
//
// \param sm The subtensor containing the rows.
// \param indices Pointer to the first index of the selected rows.
// \param n The total number of indices.
// \param args The optional row arguments.
// \return View on the specified rows of the subtensor.
// \exception std::invalid_argument Invalid row access index.
//
// This function returns an expression representing the specified rows of the given subtensor.
*/
// template< typename TT         // Type of the sparse subtensor
//         , AlignmentFlag AF    // Alignment flag
//         , bool SO             // Storage order
//         , bool DF             // Density flag
//         , size_t... CSAs      // Compile time subtensor arguments
//         , typename T          // Type of the row indices
//         , typename... RRAs >  // Optional row arguments
// inline decltype(auto)
//    rows( Subtensor<TT,AF,CSAs...>& sm, const T* indices, size_t n, RRAs... args )
// {
//    BLAZE_FUNCTION_TRACE;
//
//    constexpr bool isChecked( !Contains_v< TypeList<RRAs...>, Unchecked > );
//
//    if( isChecked ) {
//       for( size_t i=0UL; i<n; ++i ) {
//          if( sm.rows() <= indices[i] ) {
//             BLAZE_THROW_INVALID_ARGUMENT( "Invalid row specification" );
//          }
//       }
//    }
//
//    SmallArray<size_t,128UL> newIndices( indices, indices+n );
//    std::for_each( newIndices.begin(), newIndices.end(),
//                   [row=sm.row()]( size_t& index ){ index += row; } );
//
//    return subtensor( rows( sm.operand(), newIndices.data(), n, args... ),
//                      0UL, sm.column(), n, sm.columns(), unchecked );
// }
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on specific rows of the given constant subtensor.
// \ingroup subtensor
//
// \param sm The constant subtensor containing the rows.
// \param indices Pointer to the first index of the selected rows.
// \param n The total number of indices.
// \param args The optional row arguments.
// \return View on the specified rows of the subtensor.
// \exception std::invalid_argument Invalid row access index.
//
// This function returns an expression representing the specified rows of the given constant
// subtensor.
*/
// template< typename TT         // Type of the sparse subtensor
//         , AlignmentFlag AF    // Alignment flag
//         , bool SO             // Storage order
//         , bool DF             // Density flag
//         , size_t... CSAs      // Compile time subtensor arguments
//         , typename T          // Type of the row indices
//         , typename... RRAs >  // Optional row arguments
// inline decltype(auto)
//    rows( const Subtensor<TT,AF,CSAs...>& sm, const T* indices, size_t n, RRAs... args )
// {
//    BLAZE_FUNCTION_TRACE;
//
//    constexpr bool isChecked( !Contains_v< TypeList<RRAs...>, Unchecked > );
//
//    if( isChecked ) {
//       for( size_t i=0UL; i<n; ++i ) {
//          if( sm.rows() <= indices[i] ) {
//             BLAZE_THROW_INVALID_ARGUMENT( "Invalid row specification" );
//          }
//       }
//    }
//
//    SmallArray<size_t,128UL> newIndices( indices, indices+n );
//    std::for_each( newIndices.begin(), newIndices.end(),
//                   [row=sm.row()]( size_t& index ){ index += row; } );
//
//    return subtensor( rows( sm.operand(), newIndices.data(), n, args... ),
//                      0UL, sm.column(), n, sm.columns(), unchecked );
// }
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on specific rows of the given temporary subtensor.
// \ingroup subtensor
//
// \param sm The temporary subtensor containing the rows.
// \param indices Pointer to the first index of the selected rows.
// \param n The total number of indices.
// \param args The optional row arguments.
// \return View on the specified rows of the subtensor.
// \exception std::invalid_argument Invalid row access index.
//
// This function returns an expression representing the specified rows of the given temporary
// subtensor.
*/
// template< typename TT         // Type of the sparse subtensor
//         , AlignmentFlag AF    // Alignment flag
//         , bool SO             // Storage order
//         , bool DF             // Density flag
//         , size_t... CSAs      // Compile time subtensor arguments
//         , typename T          // Type of the row indices
//         , typename... RRAs >  // Optional row arguments
// inline decltype(auto)
//    rows( Subtensor<TT,AF,CSAs...>&& sm, const T* indices, size_t n, RRAs... args )
// {
//    BLAZE_FUNCTION_TRACE;
//
//    constexpr bool isChecked( !Contains_v< TypeList<RRAs...>, Unchecked > );
//
//    if( isChecked ) {
//       for( size_t i=0UL; i<n; ++i ) {
//          if( sm.rows() <= indices[i] ) {
//             BLAZE_THROW_INVALID_ARGUMENT( "Invalid row specification" );
//          }
//       }
//    }
//
//    SmallArray<size_t,128UL> newIndices( indices, indices+n );
//    std::for_each( newIndices.begin(), newIndices.end(),
//                   [row=sm.row()]( size_t& index ){ index += row; } );
//
//    return subtensor( rows( sm.operand(), newIndices.data(), n, args... ),
//                      0UL, sm.column(), n, sm.columns(), unchecked );
// }
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  GLOBAL RESTRUCTURING FUNCTIONS (COLUMN)
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific column of the given subtensor.
// \ingroup subtensor
//
// \param sm The subtensor containing the column.
// \param args The optional column arguments.
// \return View on the specified column of the subtensor.
//
// This function returns an expression representing the specified column of the given subtensor.
*/
// template< size_t I1           // Column index
//         , typename TT         // Type of the sparse subtensor
//         , AlignmentFlag AF    // Alignment flag
//         , bool SO             // Storage order
//         , bool DF             // Density flag
//         , size_t I2           // Index of the first row
//         , size_t J            // Index of the first column
//         , size_t M            // Number of rows
//         , size_t N            // Number of columns
//         , typename... RCAs >  // Optional column arguments
// inline decltype(auto) column( Subtensor<TT,AF,I2,J,M,N>& sm, RCAs... args )
// {
//    BLAZE_FUNCTION_TRACE;
//
//    BLAZE_STATIC_ASSERT_MSG( I1 < N, "Invalid column access index" );
//
//    return subvector<I2,M>( column<I1+J>( sm.operand(), args... ), unchecked );
// }
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific column of the given constant subtensor.
// \ingroup subtensor
//
// \param sm The constant subtensor containing the column.
// \param args The optional column arguments.
// \return View on the specified column of the subtensor.
//
// This function returns an expression representing the specified column of the given constant
// subtensor.
*/
// template< size_t I1           // Column index
//         , typename TT         // Type of the sparse subtensor
//         , AlignmentFlag AF    // Alignment flag
//         , bool SO             // Storage order
//         , bool DF             // Density flag
//         , size_t I2           // Index of the first row
//         , size_t J            // Index of the first column
//         , size_t M            // Number of rows
//         , size_t N            // Number of columns
//         , typename... RCAs >  // Optional column arguments
// inline decltype(auto) column( const Subtensor<TT,AF,I2,J,M,N>& sm, RCAs... args )
// {
//    BLAZE_FUNCTION_TRACE;
//
//    BLAZE_STATIC_ASSERT_MSG( I1 < N, "Invalid column access index" );
//
//    return subvector<I2,M>( column<I1+J>( sm.operand(), args... ), unchecked );
// }
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific column of the given temporary subtensor.
// \ingroup subtensor
//
// \param sm The temporary subtensor containing the column.
// \param args The optional column arguments.
// \return View on the specified column of the subtensor.
//
// This function returns an expression representing the specified column of the given temporary
// subtensor.
*/
// template< size_t I1           // Column index
//         , typename TT         // Type of the sparse subtensor
//         , AlignmentFlag AF    // Alignment flag
//         , bool SO             // Storage order
//         , bool DF             // Density flag
//         , size_t I2           // Index of the first row
//         , size_t J            // Index of the first column
//         , size_t M            // Number of rows
//         , size_t N            // Number of columns
//         , typename... RCAs >  // Optional column arguments
// inline decltype(auto) column( Subtensor<TT,AF,I2,J,M,N>&& sm, RCAs... args )
// {
//    BLAZE_FUNCTION_TRACE;
//
//    BLAZE_STATIC_ASSERT_MSG( I1 < N, "Invalid column access index" );
//
//    return subvector<I2,M>( column<I1+J>( sm.operand(), args... ), unchecked );
// }
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific column of the given subtensor.
// \ingroup subtensor
//
// \param sm The subtensor containing the column.
// \param index The index of the column.
// \param args The optional column arguments.
// \return View on the specified column of the subtensor.
// \exception std::invalid_argument Invalid column access index.
//
// This function returns an expression representing the specified column of the given subtensor.
*/
// template< typename TT         // Type of the sparse subtensor
//         , AlignmentFlag AF    // Alignment flag
//         , bool SO             // Storage order
//         , bool DF             // Density flag
//         , size_t I            // Index of the first row
//         , size_t J            // Index of the first column
//         , size_t M            // Number of rows
//         , size_t N            // Number of columns
//         , typename... RCAs >  // Optional column arguments
// inline decltype(auto) column( Subtensor<TT,AF,K,I,J,O,M,N>& sm, size_t index, RCAs... args )
// {
//    BLAZE_FUNCTION_TRACE;
//
//    constexpr bool isChecked( !Contains_v< TypeList<RCAs...>, Unchecked > );
//
//    if( isChecked ) {
//       if( ( index >= N ) ) {
//          BLAZE_THROW_INVALID_ARGUMENT( "Invalid column access index" );
//       }
//    }
//    else {
//       BLAZE_USER_ASSERT( index < N, "Invalid column access index" );
//    }
//
//    return subvector<I,M>( column( sm.operand(), J+index, args... ), unchecked );
// }
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific column of the given constant subtensor.
// \ingroup subtensor
//
// \param sm The constant subtensor containing the column.
// \param index The index of the column.
// \param args The optional column arguments.
// \return View on the specified column of the subtensor.
// \exception std::invalid_argument Invalid column access index.
//
// This function returns an expression representing the specified column of the given constant
// subtensor.
*/
// template< typename TT         // Type of the sparse subtensor
//         , AlignmentFlag AF    // Alignment flag
//         , bool SO             // Storage order
//         , bool DF             // Density flag
//         , size_t I            // Index of the first row
//         , size_t J            // Index of the first column
//         , size_t M            // Number of rows
//         , size_t N            // Number of columns
//         , typename... RCAs >  // Optional column arguments
// inline decltype(auto) column( const Subtensor<TT,AF,K,I,J,O,M,N>& sm, size_t index, RCAs... args )
// {
//    BLAZE_FUNCTION_TRACE;
//
//    constexpr bool isChecked( !Contains_v< TypeList<RCAs...>, Unchecked > );
//
//    if( isChecked ) {
//       if( ( index >= N ) ) {
//          BLAZE_THROW_INVALID_ARGUMENT( "Invalid column access index" );
//       }
//    }
//    else {
//       BLAZE_USER_ASSERT( index < N, "Invalid column access index" );
//    }
//
//    return subvector<I,M>( column( sm.operand(), J+index, args... ), unchecked );
// }
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific column of the given temporary subtensor.
// \ingroup subtensor
//
// \param sm The temporary subtensor containing the column.
// \param index The index of the column.
// \param args The optional column arguments.
// \return View on the specified column of the subtensor.
// \exception std::invalid_argument Invalid column access index.
//
// This function returns an expression representing the specified column of the given temporary
// subtensor.
*/
// template< typename TT         // Type of the sparse subtensor
//         , AlignmentFlag AF    // Alignment flag
//         , bool SO             // Storage order
//         , bool DF             // Density flag
//         , size_t I            // Index of the first row
//         , size_t J            // Index of the first column
//         , size_t M            // Number of rows
//         , size_t N            // Number of columns
//         , typename... RCAs >  // Optional column arguments
// inline decltype(auto) column( Subtensor<TT,AF,K,I,J,O,M,N>&& sm, size_t index, RCAs... args )
// {
//    BLAZE_FUNCTION_TRACE;
//
//    constexpr bool isChecked( !Contains_v< TypeList<RCAs...>, Unchecked > );
//
//    if( isChecked ) {
//       if( ( index >= N ) ) {
//          BLAZE_THROW_INVALID_ARGUMENT( "Invalid column access index" );
//       }
//    }
//    else {
//       BLAZE_USER_ASSERT( index < N, "Invalid column access index" );
//    }
//
//    return subvector<I,M>( column( sm.operand(), J+index, args... ), unchecked );
// }
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific column of the given subtensor.
// \ingroup subtensor
//
// \param sm The subtensor containing the column.
// \param args The optional column arguments.
// \return View on the specified column of the subtensor.
// \exception std::invalid_argument Invalid column access index.
//
// This function returns an expression representing the specified column of the given subtensor.
*/
// template< size_t... CCAs      // Compile time column arguments
//         , typename TT         // Type of the sparse subtensor
//         , AlignmentFlag AF    // Alignment flag
//         , bool SO             // Storage order
//         , bool DF             // Density flag
//         , typename... RCAs >  // Runtime column arguments
// inline decltype(auto) column( Subtensor<TT,AF>& sm, RCAs... args )
// {
//    BLAZE_FUNCTION_TRACE;
//
//    const ColumnData<CCAs...> cd( args... );
//
//    constexpr bool isChecked( !Contains_v< TypeList<RCAs...>, Unchecked > );
//
//    if( isChecked ) {
//       if( ( cd.column() >= sm.columns() ) ) {
//          BLAZE_THROW_INVALID_ARGUMENT( "Invalid column access index" );
//       }
//    }
//    else {
//       BLAZE_USER_ASSERT( cd.column() < sm.columns(), "Invalid column access index" );
//    }
//
//    const size_t index( cd.column() + sm.column() );
//
//    return subvector( column( sm.operand(), index, args... ), sm.row(), sm.rows(), unchecked );
// }
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific column of the given constant subtensor.
// \ingroup subtensor
//
// \param sm The constant subtensor containing the column.
// \param args The optional column arguments.
// \return View on the specified column of the subtensor.
// \exception std::invalid_argument Invalid column access index.
//
// This function returns an expression representing the specified column of the given constant
// subtensor.
*/
// template< size_t... CCAs      // Compile time column arguments
//         , typename TT         // Type of the sparse subtensor
//         , AlignmentFlag AF    // Alignment flag
//         , bool SO             // Storage order
//         , bool DF             // Density flag
//         , typename... RCAs >  // Runtime column arguments
// inline decltype(auto) column( const Subtensor<TT,AF>& sm, RCAs... args )
// {
//    BLAZE_FUNCTION_TRACE;
//
//    const ColumnData<CCAs...> cd( args... );
//
//    constexpr bool isChecked( !Contains_v< TypeList<RCAs...>, Unchecked > );
//
//    if( isChecked ) {
//       if( ( cd.column() >= sm.columns() ) ) {
//          BLAZE_THROW_INVALID_ARGUMENT( "Invalid column access index" );
//       }
//    }
//    else {
//       BLAZE_USER_ASSERT( cd.column() < sm.columns(), "Invalid column access index" );
//    }
//
//    const size_t index( cd.column() + sm.column() );
//
//    return subvector( column( sm.operand(), index, args... ), sm.row(), sm.rows(), unchecked );
// }
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific column of the given temporary subtensor.
// \ingroup subtensor
//
// \param sm The temporary subtensor containing the column.
// \param args The optional column arguments.
// \return View on the specified column of the subtensor.
// \exception std::invalid_argument Invalid column access index.
//
// This function returns an expression representing the specified column of the given temporary
// subtensor.
*/
// template< size_t... CCAs      // Compile time column arguments
//         , typename TT         // Type of the sparse subtensor
//         , AlignmentFlag AF    // Alignment flag
//         , bool SO             // Storage order
//         , bool DF             // Density flag
//         , typename... RCAs >  // Runtime column arguments
// inline decltype(auto) column( Subtensor<TT,AF>&& sm, RCAs... args )
// {
//    BLAZE_FUNCTION_TRACE;
//
//    const ColumnData<CCAs...> cd( args... );
//
//    constexpr bool isChecked( !Contains_v< TypeList<RCAs...>, Unchecked > );
//
//    if( isChecked ) {
//       if( ( cd.column() >= sm.columns() ) ) {
//          BLAZE_THROW_INVALID_ARGUMENT( "Invalid column access index" );
//       }
//    }
//    else {
//       BLAZE_USER_ASSERT( cd.column() < sm.columns(), "Invalid column access index" );
//    }
//
//    const size_t index( cd.column() + sm.column() );
//
//    return subvector( column( sm.operand(), index, args... ), sm.row(), sm.rows(), unchecked );
// }
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  GLOBAL RESTRUCTURING FUNCTIONS (COLUMNS)
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on specific columns of the given subtensor.
// \ingroup subtensor
//
// \param sm The subtensor containing the columns.
// \param args The optional column arguments.
// \return View on the specified columns of the subtensor.
//
// This function returns an expression representing the specified columns of the given subtensor.
*/
// template< size_t I1           // First column index
//         , size_t... Is        // Remaining column indices
//         , typename TT         // Type of the sparse subtensor
//         , AlignmentFlag AF    // Alignment flag
//         , bool SO             // Storage order
//         , bool DF             // Density flag
//         , size_t I2           // Index of the first row
//         , size_t J            // Index of the first column
//         , size_t M            // Number of rows
//         , size_t N            // Number of columns
//         , typename... RCAs >  // Optional column arguments
// inline decltype(auto) columns( Subtensor<TT,AF,I2,J,M,N>& sm, RCAs... args )
// {
//    BLAZE_FUNCTION_TRACE;
//
//    return subtensor<I2,0UL,M,sizeof...(Is)+1UL>(
//       columns( sm.operand(), make_shifted_index_subsequence<J,N,I1,Is...>(), args... ), unchecked );
// }
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on specific columns of the given constant subtensor.
// \ingroup subtensor
//
// \param sm The constant subtensor containing the columns.
// \param args The optional column arguments.
// \return View on the specified columns of the subtensor.
//
// This function returns an expression representing the specified columns of the given constant
// subtensor.
*/
// template< size_t I1           // First column index
//         , size_t... Is        // Remaining column indices
//         , typename TT         // Type of the sparse subtensor
//         , AlignmentFlag AF    // Alignment flag
//         , bool SO             // Storage order
//         , bool DF             // Density flag
//         , size_t I2           // Index of the first row
//         , size_t J            // Index of the first column
//         , size_t M            // Number of rows
//         , size_t N            // Number of columns
//         , typename... RCAs >  // Optional column arguments
// inline decltype(auto) columns( const Subtensor<TT,AF,I2,J,M,N>& sm, RCAs... args )
// {
//    BLAZE_FUNCTION_TRACE;
//
//    return subtensor<I2,0UL,M,sizeof...(Is)+1UL>(
//       columns( sm.operand(), make_shifted_index_subsequence<J,N,I1,Is...>(), args... ), unchecked );
// }
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on specific columns of the given temporary subtensor.
// \ingroup subtensor
//
// \param sm The temporary subtensor containing the columns.
// \param args The optional column arguments.
// \return View on the specified columns of the subtensor.
//
// This function returns an expression representing the specified columns of the given temporary
// subtensor.
*/
// template< size_t I1           // First column index
//         , size_t... Is        // Remaining column indices
//         , typename TT         // Type of the sparse subtensor
//         , AlignmentFlag AF    // Alignment flag
//         , bool SO             // Storage order
//         , bool DF             // Density flag
//         , size_t I2           // Index of the first row
//         , size_t J            // Index of the first column
//         , size_t M            // Number of rows
//         , size_t N            // Number of columns
//         , typename... RCAs >  // Optional column arguments
// inline decltype(auto) columns( Subtensor<TT,AF,I2,J,M,N>&& sm, RCAs... args )
// {
//    BLAZE_FUNCTION_TRACE;
//
//    return subtensor<I2,0UL,M,sizeof...(Is)+1UL>(
//       columns( sm.operand(), make_shifted_index_subsequence<J,N,I1,Is...>(), args... ), unchecked );
// }
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on specific columns of the given subtensor.
// \ingroup subtensor
//
// \param sm The subtensor containing the columns.
// \param args The optional column arguments.
// \return View on the specified columns of the subtensor.
// \exception std::invalid_argument Invalid column access index.
//
// This function returns an expression representing the specified columns of the given subtensor.
*/
// template< size_t I1           // First column index
//         , size_t... Is        // Remaining column indices
//         , typename TT         // Type of the sparse subtensor
//         , AlignmentFlag AF    // Alignment flag
//         , bool SO             // Storage order
//         , bool DF             // Density flag
//         , typename... RCAs >  // Optional column arguments
// inline decltype(auto) columns( Subtensor<TT,AF>& sm, RCAs... args )
// {
//    BLAZE_FUNCTION_TRACE;
//
//    constexpr bool isChecked( !Contains_v< TypeList<RCAs...>, Unchecked > );
//
//    if( isChecked ) {
//       static constexpr size_t indices[] = { I1, Is... };
//       for( size_t j=0UL; j<sizeof...(Is)+1UL; ++j ) {
//          if( sm.columns() <= indices[j] ) {
//             BLAZE_THROW_INVALID_ARGUMENT( "Invalid column access index" );
//          }
//       }
//    }
//
//    return subtensor( columns( sm.operand(), { I1+sm.column(), Is+sm.column()... }, args... ),
//                      sm.row(), 0UL, sm.rows(), sizeof...(Is)+1UL, unchecked );
// }
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on specific columns of the given constant subtensor.
// \ingroup subtensor
//
// \param sm The constant subtensor containing the columns.
// \param args The optional column arguments.
// \return View on the specified columns of the subtensor.
// \exception std::invalid_argument Invalid column access index.
//
// This function returns an expression representing the specified columns of the given constant
// subtensor.
*/
// template< size_t I1           // First column index
//         , size_t... Is        // Remaining column indices
//         , typename TT         // Type of the sparse subtensor
//         , AlignmentFlag AF    // Alignment flag
//         , bool SO             // Storage order
//         , bool DF             // Density flag
//         , typename... RCAs >  // Optional column arguments
// inline decltype(auto) columns( const Subtensor<TT,AF>& sm, RCAs... args )
// {
//    BLAZE_FUNCTION_TRACE;
//
//    constexpr bool isChecked( !Contains_v< TypeList<RCAs...>, Unchecked > );
//
//    if( isChecked ) {
//       static constexpr size_t indices[] = { I1, Is... };
//       for( size_t j=0UL; j<sizeof...(Is)+1UL; ++j ) {
//          if( sm.columns() <= indices[j] ) {
//             BLAZE_THROW_INVALID_ARGUMENT( "Invalid column access index" );
//          }
//       }
//    }
//
//    return subtensor( columns( sm.operand(), { I1+sm.column(), Is+sm.column()... }, args... ),
//                      sm.row(), 0UL, sm.rows(), sizeof...(Is)+1UL, unchecked );
// }
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on specific columns of the given temporary subtensor.
// \ingroup subtensor
//
// \param sm The temporary subtensor containing the columns.
// \param args The optional column arguments.
// \return View on the specified columns of the subtensor.
// \exception std::invalid_argument Invalid column access index.
//
// This function returns an expression representing the specified columns of the given temporary
// subtensor.
*/
// template< size_t I1           // First column index
//         , size_t... Is        // Remaining column indices
//         , typename TT         // Type of the sparse subtensor
//         , AlignmentFlag AF    // Alignment flag
//         , bool SO             // Storage order
//         , bool DF             // Density flag
//         , typename... RCAs >  // Optional column arguments
// inline decltype(auto) columns( Subtensor<TT,AF>&& sm, RCAs... args )
// {
//    BLAZE_FUNCTION_TRACE;
//
//    constexpr bool isChecked( !Contains_v< TypeList<RCAs...>, Unchecked > );
//
//    if( isChecked ) {
//       static constexpr size_t indices[] = { I1, Is... };
//       for( size_t j=0UL; j<sizeof...(Is)+1UL; ++j ) {
//          if( sm.columns() <= indices[j] ) {
//             BLAZE_THROW_INVALID_ARGUMENT( "Invalid column access index" );
//          }
//       }
//    }
//
//    return subtensor( columns( sm.operand(), { I1+sm.column(), Is+sm.column()... }, args... ),
//                      sm.row(), 0UL, sm.rows(), sizeof...(Is)+1UL, unchecked );
// }
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on specific columns of the given subtensor.
// \ingroup subtensor
//
// \param sm The subtensor containing the columns.
// \param indices Pointer to the first index of the selected columns.
// \param n The total number of indices.
// \param args The optional column arguments.
// \return View on the specified columns of the subtensor.
// \exception std::invalid_argument Invalid column access index.
//
// This function returns an expression representing the specified columns of the given subtensor.
*/
// template< typename TT         // Type of the sparse subtensor
//         , AlignmentFlag AF    // Alignment flag
//         , bool SO             // Storage order
//         , bool DF             // Density flag
//         , size_t... CSAs      // Compile time subtensor arguments
//         , typename T          // Type of the column indices
//         , typename... RCAs >  // Optional column arguments
// inline decltype(auto)
//    columns( Subtensor<TT,AF,CSAs...>& sm, const T* indices, size_t n, RCAs... args )
// {
//    BLAZE_FUNCTION_TRACE;
//
//    constexpr bool isChecked( !Contains_v< TypeList<RCAs...>, Unchecked > );
//
//    if( isChecked ) {
//       for( size_t j=0UL; j<n; ++j ) {
//          if( sm.columns() <= indices[j] ) {
//             BLAZE_THROW_INVALID_ARGUMENT( "Invalid column specification" );
//          }
//       }
//    }
//
//    SmallArray<size_t,128UL> newIndices( indices, indices+n );
//    std::for_each( newIndices.begin(), newIndices.end(),
//                   [column=sm.column()]( size_t& index ){ index += column; } );
//
//    return subtensor( columns( sm.operand(), newIndices.data(), n, args... ),
//                      sm.row(), 0UL, sm.rows(), n, unchecked );
// }
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on specific columns of the given constant subtensor.
// \ingroup subtensor
//
// \param sm The constant subtensor containing the columns.
// \param indices Pointer to the first index of the selected columns.
// \param n The total number of indices.
// \param args The optional column arguments.
// \return View on the specified columns of the subtensor.
// \exception std::invalid_argument Invalid column access index.
//
// This function returns an expression representing the specified columns of the given constant
// subtensor.
*/
// template< typename TT         // Type of the sparse subtensor
//         , AlignmentFlag AF    // Alignment flag
//         , bool SO             // Storage order
//         , bool DF             // Density flag
//         , size_t... CSAs      // Compile time subtensor arguments
//         , typename T          // Type of the column indices
//         , typename... RCAs >  // Optional column arguments
// inline decltype(auto)
//    columns( const Subtensor<TT,AF,CSAs...>& sm, const T* indices, size_t n, RCAs... args )
// {
//    BLAZE_FUNCTION_TRACE;
//
//    constexpr bool isChecked( !Contains_v< TypeList<RCAs...>, Unchecked > );
//
//    if( isChecked ) {
//       for( size_t j=0UL; j<n; ++j ) {
//          if( sm.columns() <= indices[j] ) {
//             BLAZE_THROW_INVALID_ARGUMENT( "Invalid column specification" );
//          }
//       }
//    }
//
//    SmallArray<size_t,128UL> newIndices( indices, indices+n );
//    std::for_each( newIndices.begin(), newIndices.end(),
//                   [column=sm.column()]( size_t& index ){ index += column; } );
//
//    return subtensor( columns( sm.operand(), newIndices.data(), n, args... ),
//                      sm.row(), 0UL, sm.rows(), n, unchecked );
// }
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on specific columns of the given temporary subtensor.
// \ingroup subtensor
//
// \param sm The temporary subtensor containing the columns.
// \param indices Pointer to the first index of the selected columns.
// \param n The total number of indices.
// \param args The optional column arguments.
// \return View on the specified columns of the subtensor.
// \exception std::invalid_argument Invalid column access index.
//
// This function returns an expression representing the specified columns of the given temporary
// subtensor.
*/
// template< typename TT         // Type of the sparse subtensor
//         , AlignmentFlag AF    // Alignment flag
//         , bool SO             // Storage order
//         , bool DF             // Density flag
//         , size_t... CSAs      // Compile time subtensor arguments
//         , typename T          // Type of the column indices
//         , typename... RCAs >  // Optional column arguments
// inline decltype(auto)
//    columns( Subtensor<TT,AF,CSAs...>&& sm, const T* indices, size_t n, RCAs... args )
// {
//    BLAZE_FUNCTION_TRACE;
//
//    constexpr bool isChecked( !Contains_v< TypeList<RCAs...>, Unchecked > );
//
//    if( isChecked ) {
//       for( size_t j=0UL; j<n; ++j ) {
//          if( sm.columns() <= indices[j] ) {
//             BLAZE_THROW_INVALID_ARGUMENT( "Invalid column specification" );
//          }
//       }
//    }
//
//    SmallArray<size_t,128UL> newIndices( indices, indices+n );
//    std::for_each( newIndices.begin(), newIndices.end(),
//                   [column=sm.column()]( size_t& index ){ index += column; } );
//
//    return subtensor( columns( sm.operand(), newIndices.data(), n, args... ),
//                      sm.row(), 0UL, sm.rows(), n, unchecked );
// }
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  SUBMATRIX OPERATORS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Resetting the given subtensor.
// \ingroup subtensor
//
// \param sm The subtensor to be resetted.
// \return void
*/
template< typename TT       // Type of the tensor
        , AlignmentFlag AF  // Alignment flag
        , size_t... CSAs >  // Compile time subtensor arguments
inline void reset( Subtensor<TT,AF,CSAs...>& sm )
{
   sm.reset();
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Resetting the given temporary subtensor.
// \ingroup subtensor
//
// \param sm The temporary subtensor to be resetted.
// \return void
*/
template< typename TT       // Type of the tensor
        , AlignmentFlag AF  // Alignment flag
        , size_t... CSAs >  // Compile time subtensor arguments
inline void reset( Subtensor<TT,AF,CSAs...>&& sm )
{
   sm.reset();
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Reset the specified row/column of the given subtensor.
// \ingroup subtensor
//
// \param sm The subtensor to be resetted.
// \param i The index of the row/column to be resetted.
// \return void
//
// This function resets the values in the specified row/column of the given subtensor to their
// default value. In case the given subtensor is a \a rowMajor tensor the function resets the
// values in row \a i, if it is a \a columnMajor tensor the function resets the values in column
// \a i. Note that the capacity of the row/column remains unchanged.
*/
template< typename TT       // Type of the tensor
        , AlignmentFlag AF  // Alignment flag
        , size_t... CSAs >  // Compile time subtensor arguments
inline void reset( Subtensor<TT,AF,CSAs...>& sm, size_t i, size_t k )
{
   sm.reset( i, k );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Clearing the given tensor.
// \ingroup subtensor
//
// \param sm The tensor to be cleared.
// \return void
//
// Clearing a subtensor is equivalent to resetting it via the reset() function.
*/
template< typename TT       // Type of the tensor
        , AlignmentFlag AF  // Alignment flag
        , size_t... CSAs >  // Compile time subtensor arguments
inline void clear( Subtensor<TT,AF,CSAs...>& sm )
{
   sm.reset();
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Clearing the given temporary tensor.
// \ingroup subtensor
//
// \param sm The temporary tensor to be cleared.
// \return void
//
// Clearing a subtensor is equivalent to resetting it via the reset() function.
*/
template< typename TT       // Type of the tensor
        , AlignmentFlag AF  // Alignment flag
        , size_t... CSAs >  // Compile time subtensor arguments
inline void clear( Subtensor<TT,AF,CSAs...>&& sm )
{
   sm.reset();
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns whether the given dense subtensor is in default state.
// \ingroup subtensor
//
// \param sm The dense subtensor to be tested for its default state.
// \return \a true in case the given dense subtensor is component-wise zero, \a false otherwise.
//
// This function checks whether the dense subtensor is in default state. For instance, in case
// the subtensor is instantiated for a built-in integral or floating point data type, the function
// returns \a true in case all subtensor elements are 0 and \a false in case any subtensor element
// is not 0. The following example demonstrates the use of the \a isDefault function:

   \code
   blaze::DynamicTensor<double,rowMajor> A;
   // ... Resizing and initialization
   if( isDefault( subtensor( A, 22UL, 12UL, 13UL, 33UL ) ) ) { ... }
   \endcode

// Optionally, it is possible to switch between strict semantics (blaze::strict) and relaxed
// semantics (blaze::relaxed):

   \code
   if( isDefault<relaxed>( subtensor( A, 22UL, 12UL, 13UL, 33UL ) ) ) { ... }
   \endcode
*/
template< bool RF           // Relaxation flag
        , typename TT       // Type of the dense tensor
        , AlignmentFlag AF  // Alignment flag
        , size_t... CSAs >  // Compile time subtensor arguments
inline bool isDefault( const Subtensor<TT,AF,CSAs...>& sm )
{
   using blaze::isDefault;

   for( size_t k=0UL; k<(~sm).pages(); ++k )
      for( size_t i=0UL; i<(~sm).rows(); ++i )
         for( size_t j=0UL; j<(~sm).columns(); ++j )
            if( !isDefault<RF>( (~sm)(k,i,j) ) )
               return false;

   return true;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns whether the invariants of the given subtensor are intact.
// \ingroup subtensor
//
// \param sm The subtensor to be tested.
// \return \a true in case the given subtensor's invariants are intact, \a false otherwise.
//
// This function checks whether the invariants of the subtensor are intact, i.e. if its
// state is valid. In case the invariants are intact, the function returns \a true, else it
// will return \a false. The following example demonstrates the use of the \a isIntact()
// function:

   \code
   blaze::DynamicTensor<double,rowMajor> A;
   // ... Resizing and initialization
   if( isIntact( subtensor( A, 22UL, 12UL, 13UL, 33UL ) ) ) { ... }
   \endcode
*/
template< typename TT       // Type of the tensor
        , AlignmentFlag AF  // Alignment flag
        , size_t... CSAs >  // Compile time subtensor arguments
inline bool isIntact( const Subtensor<TT,AF,CSAs...>& sm ) noexcept
{
   return ( sm.row() + sm.rows() <= sm.operand().rows() &&
            sm.column() + sm.columns() <= sm.operand().columns() &&
            sm.page() + sm.pages() <= sm.operand().pages() &&
            isIntact( sm.operand() ) );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Checks if the given subtensor is symmetric.
// \ingroup subtensor
//
// \param sm The subtensor to be checked.
// \return \a true if the subtensor is symmetric, \a false if not.
//
// This function checks if the given subtensor is symmetric. The subtensor is considered to
// be symmetric if it is a square tensor whose transpose is equal to itself (\f$ A = A^T \f$). The
// following code example demonstrates the use of the function:

   \code
   blaze::DynamicTensor<int,blaze::rowMajor> A( 32UL, 16UL );
   // ... Initialization

   auto sm = subtensor( A, 16UL, 8UL, 8UL, 16UL );

   if( isSymmetric( sm ) ) { ... }
   \endcode
*/
template< typename TT       // Type of the tensor
        , AlignmentFlag AF  // Alignment flag
        , bool SO           // Storage order
        , bool DF           // Density flag
        , size_t... CSAs >  // Compile time subtensor arguments
inline bool isSymmetric( const Subtensor<TT,AF,CSAs...>& sm )
{
   using BaseType = BaseType_t< Subtensor<TT,AF,CSAs...> >;

   return isSymmetric( static_cast<const BaseType&>( sm ) );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Checks if the given subtensor is Hermitian.
// \ingroup subtensor
//
// \param sm The subtensor to be checked.
// \return \a true if the subtensor is Hermitian, \a false if not.
//
// This function checks if the given subtensor is Hermitian. The subtensor is considered to
// be Hermitian if it is a square tensor whose transpose is equal to its conjugate transpose
// (\f$ A = \overline{A^T} \f$). The following code example demonstrates the use of the function:

   \code
   blaze::DynamicTensor<int,blaze::rowMajor> A( 32UL, 16UL );
   // ... Initialization

   auto sm = subtensor( A, 16UL, 8UL, 8UL, 16UL );

   if( isHermitian( sm ) ) { ... }
   \endcode
*/
template< typename TT       // Type of the tensor
        , AlignmentFlag AF  // Alignment flag
        , size_t... CSAs >  // Compile time subtensor arguments
inline bool isHermitian( const Subtensor<TT,AF,CSAs...>& sm )
{
   using BaseType = BaseType_t< Subtensor<TT,AF,CSAs...> >;

   return isHermitian( static_cast<const BaseType&>( sm ) );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Checks if the given subtensor is a lower triangular tensor.
// \ingroup subtensor
//
// \param sm The subtensor to be checked.
// \return \a true if the subtensor is a lower triangular tensor, \a false if not.
//
// This function checks if the given subtensor is a lower triangular tensor. The tensor is
// considered to be lower triangular if it is a square tensor of the form

                        \f[\left(\begin{array}{*{5}{c}}
                        l_{0,0} & 0       & 0       & \cdots & 0       \\
                        l_{1,0} & l_{1,1} & 0       & \cdots & 0       \\
                        l_{2,0} & l_{2,1} & l_{2,2} & \cdots & 0       \\
                        \vdots  & \vdots  & \vdots  & \ddots & \vdots  \\
                        l_{N,0} & l_{N,1} & l_{N,2} & \cdots & l_{N,N} \\
                        \end{array}\right).\f]

// \f$ 0 \times 0 \f$ or \f$ 1 \times 1 \f$ tensors are considered as trivially lower triangular.
// The following code example demonstrates the use of the function:

   \code
   blaze::DynamicTensor<int,blaze::rowMajor> A( 32UL, 16UL );
   // ... Initialization

   auto sm = subtensor( A, 16UL, 8UL, 8UL, 16UL );

   if( isLower( sm ) ) { ... }
   \endcode
*/
template< typename TT       // Type of the tensor
        , AlignmentFlag AF  // Alignment flag
        , size_t... CSAs >  // Compile time subtensor arguments
inline bool isLower( const Subtensor<TT,AF,CSAs...>& sm )
{
   using BaseType = BaseType_t< Subtensor<TT,AF,CSAs...> >;

   return isLower( static_cast<const BaseType&>( sm ) );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Checks if the given subtensor is a lower unitriangular tensor.
// \ingroup subtensor
//
// \param sm The subtensor to be checked.
// \return \a true if the subtensor is a lower unitriangular tensor, \a false if not.
//
// This function checks if the given subtensor is a lower unitriangular tensor. The tensor is
// considered to be lower triangular if it is a square tensor of the form

                        \f[\left(\begin{array}{*{5}{c}}
                        1       & 0       & 0       & \cdots & 0      \\
                        l_{1,0} & 1       & 0       & \cdots & 0      \\
                        l_{2,0} & l_{2,1} & 1       & \cdots & 0      \\
                        \vdots  & \vdots  & \vdots  & \ddots & \vdots \\
                        l_{N,0} & l_{N,1} & l_{N,2} & \cdots & 1      \\
                        \end{array}\right).\f]

// The following code example demonstrates the use of the function:

   \code
   blaze::DynamicTensor<int,blaze::rowMajor> A( 32UL, 16UL );
   // ... Initialization

   auto sm = subtensor( A, 16UL, 8UL, 8UL, 16UL );

   if( isUniLower( sm ) ) { ... }
   \endcode
*/
template< typename TT       // Type of the tensor
        , AlignmentFlag AF  // Alignment flag
        , size_t... CSAs >  // Compile time subtensor arguments
inline bool isUniLower( const Subtensor<TT,AF,CSAs...>& sm )
{
   using BaseType = BaseType_t< Subtensor<TT,AF,CSAs...> >;

   return isUniLower( static_cast<const BaseType&>( sm ) );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Checks if the given subtensor is a strictly lower triangular tensor.
// \ingroup subtensor
//
// \param sm The subtensor to be checked.
// \return \a true if the subtensor is a strictly lower triangular tensor, \a false if not.
//
// This function checks if the given subtensor is a strictly lower triangular tensor. The
// tensor is considered to be lower triangular if it is a square tensor of the form

                        \f[\left(\begin{array}{*{5}{c}}
                        0       & 0       & 0       & \cdots & 0      \\
                        l_{1,0} & 0       & 0       & \cdots & 0      \\
                        l_{2,0} & l_{2,1} & 0       & \cdots & 0      \\
                        \vdots  & \vdots  & \vdots  & \ddots & \vdots \\
                        l_{N,0} & l_{N,1} & l_{N,2} & \cdots & 0      \\
                        \end{array}\right).\f]

// The following code example demonstrates the use of the function:

   \code
   blaze::DynamicTensor<int,blaze::rowMajor> A( 32UL, 16UL );
   // ... Initialization

   auto sm = subtensor( A, 16UL, 8UL, 8UL, 16UL );

   if( isStrictlyLower( sm ) ) { ... }
   \endcode
*/
template< typename TT       // Type of the tensor
        , AlignmentFlag AF  // Alignment flag
        , size_t... CSAs >  // Compile time subtensor arguments
inline bool isStrictlyLower( const Subtensor<TT,AF,CSAs...>& sm )
{
   using BaseType = BaseType_t< Subtensor<TT,AF,CSAs...> >;

   return isStrictlyLower( static_cast<const BaseType&>( sm ) );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Checks if the given subtensor is an upper triangular tensor.
// \ingroup subtensor
//
// \param sm The subtensor to be checked.
// \return \a true if the subtensor is an upper triangular tensor, \a false if not.
//
// This function checks if the given sparse subtensor is an upper triangular tensor. The tensor
// is considered to be upper triangular if it is a square tensor of the form

                        \f[\left(\begin{array}{*{5}{c}}
                        u_{0,0} & u_{0,1} & u_{0,2} & \cdots & u_{0,N} \\
                        0       & u_{1,1} & u_{1,2} & \cdots & u_{1,N} \\
                        0       & 0       & u_{2,2} & \cdots & u_{2,N} \\
                        \vdots  & \vdots  & \vdots  & \ddots & \vdots  \\
                        0       & 0       & 0       & \cdots & u_{N,N} \\
                        \end{array}\right).\f]

// \f$ 0 \times 0 \f$ or \f$ 1 \times 1 \f$ tensors are considered as trivially upper triangular.
// The following code example demonstrates the use of the function:

   \code
   blaze::DynamicTensor<int,blaze::rowMajor> A( 32UL, 16UL );
   // ... Initialization

   auto sm = subtensor( A, 16UL, 8UL, 8UL, 16UL );

   if( isUpper( sm ) ) { ... }
   \endcode
*/
template< typename TT       // Type of the tensor
        , AlignmentFlag AF  // Alignment flag
        , size_t... CSAs >  // Compile time subtensor arguments
inline bool isUpper( const Subtensor<TT,AF,CSAs...>& sm )
{
   using BaseType = BaseType_t< Subtensor<TT,AF,CSAs...> >;

   return isUpper( static_cast<const BaseType&>( sm ) );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Checks if the given subtensor is an upper unitriangular tensor.
// \ingroup subtensor
//
// \param sm The subtensor to be checked.
// \return \a true if the subtensor is an upper unitriangular tensor, \a false if not.
//
// This function checks if the given sparse subtensor is an upper triangular tensor. The tensor
// is considered to be upper triangular if it is a square tensor of the form

                        \f[\left(\begin{array}{*{5}{c}}
                        1      & u_{0,1} & u_{0,2} & \cdots & u_{0,N} \\
                        0      & 1       & u_{1,2} & \cdots & u_{1,N} \\
                        0      & 0       & 1       & \cdots & u_{2,N} \\
                        \vdots & \vdots  & \vdots  & \ddots & \vdots  \\
                        0      & 0       & 0       & \cdots & 1       \\
                        \end{array}\right).\f]

// The following code example demonstrates the use of the function:

   \code
   blaze::DynamicTensor<int,blaze::rowMajor> A( 32UL, 16UL );
   // ... Initialization

   auto sm = subtensor( A, 16UL, 8UL, 8UL, 16UL );

   if( isUniUpper( sm ) ) { ... }
   \endcode
*/
template< typename TT       // Type of the tensor
        , AlignmentFlag AF  // Alignment flag
        , size_t... CSAs >  // Compile time subtensor arguments
inline bool isUniUpper( const Subtensor<TT,AF,CSAs...>& sm )
{
   using BaseType = BaseType_t< Subtensor<TT,AF,CSAs...> >;

   return isUniUpper( static_cast<const BaseType&>( sm ) );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Checks if the given subtensor is a strictly upper triangular tensor.
// \ingroup subtensor
//
// \param sm The subtensor to be checked.
// \return \a true if the subtensor is a strictly upper triangular tensor, \a false if not.
//
// This function checks if the given sparse subtensor is a strictly upper triangular tensor. The
// tensor is considered to be upper triangular if it is a square tensor of the form

                        \f[\left(\begin{array}{*{5}{c}}
                        0      & u_{0,1} & u_{0,2} & \cdots & u_{0,N} \\
                        0      & 0       & u_{1,2} & \cdots & u_{1,N} \\
                        0      & 0       & 0       & \cdots & u_{2,N} \\
                        \vdots & \vdots  & \vdots  & \ddots & \vdots  \\
                        0      & 0       & 0       & \cdots & 0       \\
                        \end{array}\right).\f]

// The following code example demonstrates the use of the function:

   \code
   blaze::DynamicTensor<int,blaze::rowMajor> A( 32UL, 16UL );
   // ... Initialization

   auto sm = subtensor( A, 16UL, 8UL, 8UL, 16UL );

   if( isStrictlyUpper( sm ) ) { ... }
   \endcode
*/
template< typename TT       // Type of the tensor
        , AlignmentFlag AF  // Alignment flag
        , size_t... CSAs >  // Compile time subtensor arguments
inline bool isStrictlyUpper( const Subtensor<TT,AF,CSAs...>& sm )
{
   using BaseType = BaseType_t< Subtensor<TT,AF,CSAs...> >;

   return isStrictlyUpper( static_cast<const BaseType&>( sm ) );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns whether the given tensor and subtensor represent the same observable state.
// \ingroup subtensor
//
// \param a The subtensor to be tested for its state.
// \param b The tensor to be tested for its state.
// \return \a true in case the subtensor and tensor share a state, \a false otherwise.
//
// This overload of the isSame function tests if the given subtensor refers to the full given
// tensor and by that represents the same observable state. In this case, the function returns
// \a true, otherwise it returns \a false.
*/
template< typename TT       // Type of the tensor
        , AlignmentFlag AF  // Alignment flag
        , size_t... CSAs >  // Compile time subtensor arguments
inline bool isSame( const Subtensor<TT,AF,CSAs...>& a, const Tensor<TT>& b ) noexcept
{
   return ( isSame( a.operand(), ~b ) &&
            ( a.rows() == (~b).rows() ) &&
            ( a.columns() == (~b).columns() ) &&
            ( a.pages() == (~b).pages() ) );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns whether the given tensor and subtensor represent the same observable state.
// \ingroup subtensor
//
// \param a The tensor to be tested for its state.
// \param b The subtensor to be tested for its state.
// \return \a true in case the tensor and subtensor share a state, \a false otherwise.
//
// This overload of the isSame function tests if the given subtensor refers to the full given
// tensor and by that represents the same observable state. In this case, the function returns
// \a true, otherwise it returns \a false.
*/
template< typename TT       // Type of the tensor
        , AlignmentFlag AF  // Alignment flag
        , size_t... CSAs >  // Compile time subtensor arguments
inline bool isSame( const Tensor<TT>& a, const Subtensor<TT,AF,CSAs...>& b ) noexcept
{
   return ( isSame( ~a, b.operand() ) &&
            ( (~a).rows() == b.rows() ) &&
            ( (~a).columns() == b.columns() ) &&
            ( (~a).pages() == (~b).pages() ) );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns whether the two given subtensors represent the same observable state.
// \ingroup subtensor
//
// \param a The first subtensor to be tested for its state.
// \param b The second subtensor to be tested for its state.
// \return \a true in case the two subtensors share a state, \a false otherwise.
//
// This overload of the isSame function tests if the two given subtensors refer to exactly the
// same part of the same tensor. In case both subtensors represent the same observable state,
// the function returns \a true, otherwise it returns \a false.
*/
template< typename TT1       // Type of the tensor of the left-hand side subtensor
        , AlignmentFlag AF1  // Alignment flag of the left-hand side subtensor
        , size_t... CSAs1    // Compile time subtensor arguments of the left-hand side subtensor
        , typename TT2       // Type of the tensor of the right-hand side subtensor
        , AlignmentFlag AF2  // Alignment flag of the right-hand side subtensor
        , size_t... CSAs2 >  // Compile time subtensor arguments of the right-hand side subtensor
inline bool isSame( const Subtensor<TT1,AF1,CSAs1...>& a,
                    const Subtensor<TT2,AF2,CSAs2...>& b ) noexcept
{
   return ( isSame( a.operand(), b.operand() ) &&
            ( a.row() == b.row() ) && ( a.column() == b.column() ) &&
            ( a.rows() == b.rows() ) && ( a.columns() == b.columns() ) &&
            ( a.page() == b.page() ) && ( a.pages() == b.pages() ) );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief In-place inversion of the given dense subtensor.
// \ingroup subtensor
//
// \param sm The dense subtensor to be inverted.
// \return void
// \exception std::invalid_argument Invalid non-square tensor provided.
// \exception std::runtime_error Inversion of singular tensor failed.
//
// This function inverts the given dense subtensor by means of the specified tensor type or tensor
// inversion algorithm \c IF (see the InversionFlag documentation):

   \code
   invert<asLower>( A );     // Inversion of a lower triangular tensor
   invert<asUniUpper>( A );  // Inversion of an upper unitriangular tensor
   invert<byLU>( A );        // Inversion by means of an LU decomposition
   invert<byLLH>( A );       // Inversion by means of a Cholesky decomposition
   ...
   \endcode

// The tensor inversion fails if ...
//
//  - ... the given subtensor is not a square tensor;
//  - ... the given subtensor is singular and not invertible.
//
// In all failure cases either a compilation error is created if the failure can be predicted at
// compile time or an exception is thrown.
//
// \note The tensor inversion can only be used for dense tensors with \c float, \c double,
// \c complex<float> or \c complex<double> element type. The attempt to call the function with
// tensors of any other element type results in a compile time error!
//
// \note This function can only be used if a fitting LAPACK library is available and linked to
// the executable. Otherwise a linker error will be created.
//
// \note This function does only provide the basic exception safety guarantee, i.e. in case of an
// exception \a sm may already have been modified.
*/
template< InversionFlag IF  // Inversion algorithm
        , typename TT       // Type of the dense tensor
        , AlignmentFlag AF  // Alignment flag
        , size_t... CSAs >  // Compile time subtensor arguments
inline auto invert( Subtensor<TT,AF,CSAs...>& sm )
   -> DisableIf_t< HasMutableDataAccess_v<TT> >
{
   using RT = ResultType_t< Subtensor<TT,AF,CSAs...> >;

   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION  ( RT );
   BLAZE_CONSTRAINT_MUST_HAVE_MUTABLE_DATA_ACCESS( RT );

   RT tmp( sm );
   invert<IF>( tmp );
   sm = tmp;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Predict invariant violations by setting a single element of a subtensor.
// \ingroup subtensor
//
// \param sm The target subtensor.
// \param i The row index of the element to be set.
// \param j The column index of the element to be set.
// \param value The value to be set to the element.
// \return \a true in case the operation would be successful, \a false if not.
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename TT       // Type of the tensor
        , AlignmentFlag AF  // Alignment flag
        , size_t... CSAs    // Compile time subtensor arguments
        , typename ET >     // Type of the element
inline bool trySet( const Subtensor<TT,AF,CSAs...>& sm, size_t k, size_t i, size_t j, const ET& value )
{
   BLAZE_INTERNAL_ASSERT( i < sm.rows(), "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( j < sm.columns(), "Invalid column access index" );
   BLAZE_INTERNAL_ASSERT( k < sm.pages(), "Invalid column access index" );

   return trySet( sm.operand(), sm.row()+i, sm.column()+j, sm.page()+k, value );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Predict invariant violations by adding to a single element of a subtensor.
// \ingroup subtensor
//
// \param sm The target subtensor.
// \param i The row index of the element to be modified.
// \param j The column index of the element to be modified.
// \param value The value to be added to the element.
// \return \a true in case the operation would be successful, \a false if not.
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename TT       // Type of the tensor
        , AlignmentFlag AF  // Alignment flag
        , size_t... CSAs    // Compile time subtensor arguments
        , typename ET >     // Type of the element
inline bool tryAdd( const Subtensor<TT,AF,CSAs...>& sm, size_t k, size_t i, size_t j, const ET& value )
{
   BLAZE_INTERNAL_ASSERT( i < sm.rows(), "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( j < sm.columns(), "Invalid column access index" );
   BLAZE_INTERNAL_ASSERT( k < sm.pages(), "Invalid column access index" );

   return tryAdd( sm.operand(), sm.row()+i, sm.column()+j, sm.pages()+k, value );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Predict invariant violations by subtracting from a single element of a subtensor.
// \ingroup subtensor
//
// \param sm The target subtensor.
// \param i The row index of the element to be modified.
// \param j The column index of the element to be modified.
// \param value The value to be subtracted from the element.
// \return \a true in case the operation would be successful, \a false if not.
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename TT       // Type of the tensor
        , AlignmentFlag AF  // Alignment flag
        , size_t... CSAs    // Compile time subtensor arguments
        , typename ET >     // Type of the element
inline bool trySub( const Subtensor<TT,AF,CSAs...>& sm, size_t k, size_t i, size_t j, const ET& value )
{
   BLAZE_INTERNAL_ASSERT( i < sm.rows(), "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( j < sm.columns(), "Invalid column access index" );
   BLAZE_INTERNAL_ASSERT( k < sm.pages(), "Invalid column access index" );

   return trySub( sm.operand(), sm.row()+i, sm.column()+j, sm.pages()+k, value );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Predict invariant violations by scaling a single element of a subtensor.
// \ingroup subtensor
//
// \param sm The target subtensor.
// \param i The row index of the element to be modified.
// \param j The column index of the element to be modified.
// \param value The factor for the element.
// \return \a true in case the operation would be successful, \a false if not.
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename TT       // Type of the tensor
        , AlignmentFlag AF  // Alignment flag
        , size_t... CSAs    // Compile time subtensor arguments
        , typename ET >     // Type of the element
inline bool tryMult( const Subtensor<TT,AF,CSAs...>& sm, size_t k, size_t i, size_t j, const ET& value )
{
   BLAZE_INTERNAL_ASSERT( i < sm.rows(), "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( j < sm.columns(), "Invalid column access index" );
   BLAZE_INTERNAL_ASSERT( k < sm.pages(), "Invalid column access index" );

   return tryMult( sm.operand(), sm.row()+i, sm.column()+j, sm.pages()+k, value );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Predict invariant violations by scaling a range of elements of a subtensor.
// \ingroup subtensor
//
// \param sm The target subtensor.
// \param row The index of the first row of the range to be modified.
// \param column The index of the first column of the range to be modified.
// \param m The number of rows of the range to be modified.
// \param n The number of columns of the range to be modified.
// \param value The factor for the elements.
// \return \a true in case the operation would be successful, \a false if not.
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename TT       // Type of the tensor
        , AlignmentFlag AF  // Alignment flag
        , size_t... CSAs    // Compile time subtensor arguments
        , typename ET >     // Type of the element
BLAZE_ALWAYS_INLINE bool
   tryMult( const Subtensor<TT,AF,CSAs...>& sm, size_t row, size_t column, size_t page, size_t m, size_t n, size_t o, const ET& value )
{
   UNUSED_PARAMETER( column );

   BLAZE_INTERNAL_ASSERT( row <= (~sm).rows(), "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( column <= (~sm).columns(), "Invalid column access index" );
   BLAZE_INTERNAL_ASSERT( page <= (~sm).pages(), "Invalid page access index" );
   BLAZE_INTERNAL_ASSERT( row + m <= (~sm).rows(), "Invalid number of rows" );
   BLAZE_INTERNAL_ASSERT( column + n <= (~sm).columns(), "Invalid number of columns" );
   BLAZE_INTERNAL_ASSERT( page + o <= (~sm).pages(), "Invalid number of pages" );

   return tryMult( sm.operand(), sm.row()+row, sm.column(), sm.page(), m, n, o, value );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Predict invariant violations by scaling a single element of a subtensor.
// \ingroup subtensor
//
// \param sm The target subtensor.
// \param i The row index of the element to be modified.
// \param j The column index of the element to be modified.
// \param value The divisor for the element.
// \return \a true in case the operation would be successful, \a false if not.
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename TT       // Type of the tensor
        , AlignmentFlag AF  // Alignment flag
        , size_t... CSAs    // Compile time subtensor arguments
        , typename ET >     // Type of the element
inline bool tryDiv( const Subtensor<TT,AF,CSAs...>& sm, size_t k, size_t i, size_t j, const ET& value )
{
   BLAZE_INTERNAL_ASSERT( i < sm.rows(), "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( j < sm.columns(), "Invalid column access index" );
   BLAZE_INTERNAL_ASSERT( k < sm.pages(), "Invalid column access index" );

   return tryDiv( sm.operand(), sm.row()+i, sm.column()+j, sm.pages()+k, value );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Predict invariant violations by scaling a range of elements of a subtensor.
// \ingroup subtensor
//
// \param sm The target subtensor.
// \param row The index of the first row of the range to be modified.
// \param column The index of the first column of the range to be modified.
// \param m The number of rows of the range to be modified.
// \param n The number of columns of the range to be modified.
// \param value The divisor for the elements.
// \return \a true in case the operation would be successful, \a false if not.
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename TT       // Type of the tensor
        , AlignmentFlag AF  // Alignment flag
        , size_t... CSAs    // Compile time subtensor arguments
        , typename ET >     // Type of the element
BLAZE_ALWAYS_INLINE bool
   tryDiv( const Subtensor<TT,AF,CSAs...>& sm, size_t row, size_t column, size_t page, size_t m, size_t n, size_t o, const ET& value )
{
   UNUSED_PARAMETER( column );

   BLAZE_INTERNAL_ASSERT( row <= (~sm).rows(), "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( column <= (~sm).columns(), "Invalid column access index" );
   BLAZE_INTERNAL_ASSERT( page <= (~sm).pages(), "Invalid page access index" );
   BLAZE_INTERNAL_ASSERT( row + m <= (~sm).rows(), "Invalid number of rows" );
   BLAZE_INTERNAL_ASSERT( column + n <= (~sm).columns(), "Invalid number of columns" );
   BLAZE_INTERNAL_ASSERT( page + o <= (~sm).pages(), "Invalid number of pages" );

   return tryDiv( sm.operand(), sm.row()+row, sm.column(), sm.page(), m, n, o, value );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Predict invariant violations by the assignment of a matrix to a subtensor.
// \ingroup subtensor
//
// \param lhs The target left-hand side subtensor.
// \param rhs The right-hand side matrix to be assigned.
// \param row The row index of the first element to be modified.
// \param column The column index of the first element to be modified.
// \return \a true in case the assignment would be successful, \a false if not.
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename TT       // Type of the tensor
        , AlignmentFlag AF  // Alignment flag
        , size_t... CSAs    // Compile time subtensor arguments
        , typename VT >      // Type of the right-hand side vector
inline bool tryAssign( const Subtensor<TT,AF,CSAs...>& lhs,
                       const Matrix<VT,false>& rhs, size_t row, size_t column, size_t page )
{
   BLAZE_INTERNAL_ASSERT( row <= lhs.rows(), "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( column <= lhs.columns(), "Invalid column access index" );
   BLAZE_INTERNAL_ASSERT( page <= lhs.pages(), "Invalid page access index" );
   BLAZE_INTERNAL_ASSERT( row + (~rhs).rows() <= lhs.rows(), "Invalid number of rows" );
   BLAZE_INTERNAL_ASSERT( column + (~rhs).columns() <= lhs.columns(), "Invalid number of columns" );

   return tryAssign( lhs.operand(), ~rhs, lhs.row() + row, lhs.column() + column, lhs.page() + page );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Predict invariant violations by the assignment of a matrix to the band (pageslices) of a subtensor.
// \ingroup subtensor
//
// \param lhs The target left-hand side subtensor.
// \param rhs The right-hand side vector to be assigned.
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
template< typename TT       // Type of the tensor
        , AlignmentFlag AF  // Alignment flag
        , size_t... CSAs    // Compile time subtensor arguments
        , typename VT >      // Type of the right-hand side vector
inline bool tryAssign( const Subtensor<TT,AF,CSAs...>& lhs,
                       const Matrix<VT,false>& rhs, ptrdiff_t band, size_t row, size_t column, size_t page )
{
   BLAZE_INTERNAL_ASSERT( row <= lhs.rows(), "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( column <= lhs.columns(), "Invalid column access index" );
   BLAZE_INTERNAL_ASSERT( page <= lhs.pages(), "Invalid page access index" );
   BLAZE_INTERNAL_ASSERT( row + (~rhs).rows() <= lhs.rows(), "Invalid number of rows" );
   BLAZE_INTERNAL_ASSERT( column + (~rhs).columns() <= lhs.columns(), "Invalid number of columns" );

   return tryAssign( lhs.operand(), ~rhs, band + ptrdiff_t( lhs.column() - lhs.row() ),
                     lhs.row() + row, lhs.column() + column, lhs.page() + page );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Predict invariant violations by the assignment of a tensor to a subtensor.
// \ingroup subtensor
//
// \param lhs The target left-hand side subtensor.
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
template< typename TT1      // Type of the tensor
        , AlignmentFlag AF  // Alignment flag
        , size_t... CSAs    // Compile time subtensor arguments
        , typename TT2 >     // Type of the right-hand side tensor
inline bool tryAssign( const Subtensor<TT1,AF,CSAs...>& lhs,
                       const Tensor<TT2>& rhs, size_t row, size_t column, size_t page )
{
   BLAZE_INTERNAL_ASSERT( row <= lhs.rows(), "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( column <= lhs.columns(), "Invalid column access index" );
   BLAZE_INTERNAL_ASSERT( page <= lhs.pages(), "Invalid page access index" );
   BLAZE_INTERNAL_ASSERT( row + (~rhs).rows() <= lhs.rows(), "Invalid number of rows" );
   BLAZE_INTERNAL_ASSERT( column + (~rhs).columns() <= lhs.columns(), "Invalid number of columns" );
   BLAZE_INTERNAL_ASSERT( page + (~rhs).pages() <= lhs.pages(), "Invalid number of pages" );

   return tryAssign( lhs.operand(), ~rhs, lhs.row() + row, lhs.column() + column, lhs.page() + page );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Predict invariant violations by the addition assignment of a vector to a subtensor.
// \ingroup subtensor
//
// \param lhs The target left-hand side subtensor.
// \param rhs The right-hand side vector to be added.
// \param row The row index of the first element to be modified.
// \param column The column index of the first element to be modified.
// \return \a true in case the assignment would be successful, \a false if not.
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename TT       // Type of the tensor
        , AlignmentFlag AF  // Alignment flag
        , size_t... CSAs    // Compile time subtensor arguments
        , typename VT >      // Type of the right-hand side vector
inline bool tryAddAssign( const Subtensor<TT,AF,CSAs...>& lhs,
                          const Matrix<VT,false>& rhs, size_t row, size_t column, size_t page )
{
   BLAZE_INTERNAL_ASSERT( row <= lhs.rows(), "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( column <= lhs.columns(), "Invalid column access index" );
   BLAZE_INTERNAL_ASSERT( page <= lhs.pages(), "Invalid page access index" );
   BLAZE_INTERNAL_ASSERT( row + (~rhs).rows() <= lhs.rows(), "Invalid number of rows" );
   BLAZE_INTERNAL_ASSERT( column + (~rhs).columns() <= lhs.columns(), "Invalid number of columns" );

   return tryAddAssign( lhs.operand(), ~rhs, lhs.row() + row, lhs.column() + column, lhs.page() + page );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Predict invariant violations by the addition assignment of a vector to the band of
//        a subtensor.
// \ingroup subtensor
//
// \param lhs The target left-hand side subtensor.
// \param rhs The right-hand side vector to be added.
// \param band The index of the band the right-hand side vector is assigned to.
// \param row The row index of the first element to be modified.
// \param column The column index of the first element to be modified.
// \return \a true in case the assignment would be successful, \a false if not.
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename TT       // Type of the tensor
        , AlignmentFlag AF  // Alignment flag
        , size_t... CSAs    // Compile time subtensor arguments
        , typename VT >      // Type of the right-hand side vector
inline bool tryAddAssign( const Subtensor<TT,AF,CSAs...>& lhs,
                          const Matrix<VT,false>& rhs, ptrdiff_t band, size_t row, size_t column, size_t page )
{
   BLAZE_INTERNAL_ASSERT( row <= lhs.rows(), "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( column <= lhs.columns(), "Invalid column access index" );
   BLAZE_INTERNAL_ASSERT( page <= lhs.pages(), "Invalid page access index" );
   BLAZE_INTERNAL_ASSERT( row + (~rhs).rows() <= lhs.rows(), "Invalid number of rows" );
   BLAZE_INTERNAL_ASSERT( column + (~rhs).columns() <= lhs.columns(), "Invalid number of columns" );

   return tryAddAssign( lhs.operand(), ~rhs, band + ptrdiff_t( lhs.column() - lhs.row() ),
                        lhs.row() + row, lhs.column() + column, lhs.page() + page );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Predict invariant violations by the addition assignment of a tensor to a subtensor.
// \ingroup subtensor
//
// \param lhs The target left-hand side subtensor.
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
template< typename TT1      // Type of the tensor
        , AlignmentFlag AF  // Alignment flag
        , size_t... CSAs    // Compile time subtensor arguments
        , typename TT2 >     // Type of the right-hand side tensor
inline bool tryAddAssign( const Subtensor<TT1,AF,CSAs...>& lhs,
                          const Tensor<TT2>& rhs, size_t row, size_t column, size_t page )
{
   BLAZE_INTERNAL_ASSERT( row <= lhs.rows(), "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( column <= lhs.columns(), "Invalid column access index" );
   BLAZE_INTERNAL_ASSERT( page <= lhs.pages(), "Invalid page access index" );
   BLAZE_INTERNAL_ASSERT( row + (~rhs).rows() <= lhs.rows(), "Invalid number of rows" );
   BLAZE_INTERNAL_ASSERT( column + (~rhs).columns() <= lhs.columns(), "Invalid number of columns" );
   BLAZE_INTERNAL_ASSERT( page + (~rhs).pages() <= lhs.pages(), "Invalid number of pages" );

   return tryAddAssign( lhs.operand(), ~rhs, lhs.row() + row, lhs.column() + column, lhs.page() + page );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Predict invariant violations by the subtraction assignment of a vector to a subtensor.
// \ingroup subtensor
//
// \param lhs The target left-hand side subtensor.
// \param rhs The right-hand side vector to be subtracted.
// \param row The row index of the first element to be modified.
// \param column The column index of the first element to be modified.
// \return \a true in case the assignment would be successful, \a false if not.
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename TT       // Type of the tensor
        , AlignmentFlag AF  // Alignment flag
        , bool SO           // Storage order
        , bool DF           // Density flag
        , size_t... CSAs    // Compile time subtensor arguments
        , typename VT       // Type of the right-hand side vector
        , bool TF >         // Transpose flag of the right-hand side vector
inline bool trySubAssign( const Subtensor<TT,AF,CSAs...>& lhs,
                          const Vector<VT,TF>& rhs, size_t row, size_t column )
{
   BLAZE_INTERNAL_ASSERT( row <= lhs.rows(), "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( column <= lhs.columns(), "Invalid column access index" );
   BLAZE_INTERNAL_ASSERT( TF || ( row + (~rhs).size() <= lhs.rows() ), "Invalid number of rows" );
   BLAZE_INTERNAL_ASSERT( !TF || ( column + (~rhs).size() <= lhs.columns() ), "Invalid number of columns" );

   return trySubAssign( lhs.operand(), ~rhs, lhs.row() + row, lhs.column() + column );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Predict invariant violations by the subtraction assignment of a vector to the band of
//        a subtensor.
// \ingroup subtensor
//
// \param lhs The target left-hand side subtensor.
// \param rhs The right-hand side vector to be subtracted.
// \param band The index of the band the right-hand side vector is assigned to.
// \param row The row index of the first element to be modified.
// \param column The column index of the first element to be modified.
// \return \a true in case the assignment would be successful, \a false if not.
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename TT       // Type of the tensor
        , AlignmentFlag AF  // Alignment flag
        , bool SO           // Storage order
        , bool DF           // Density flag
        , size_t... CSAs    // Compile time subtensor arguments
        , typename VT       // Type of the right-hand side vector
        , bool TF >         // Transpose flag of the right-hand side vector
inline bool trySubAssign( const Subtensor<TT,AF,CSAs...>& lhs,
                          const Vector<VT,TF>& rhs, ptrdiff_t band, size_t row, size_t column )
{
   BLAZE_INTERNAL_ASSERT( row <= lhs.rows(), "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( column <= lhs.columns(), "Invalid column access index" );
   BLAZE_INTERNAL_ASSERT( row + (~rhs).size() <= lhs.rows(), "Invalid number of rows" );
   BLAZE_INTERNAL_ASSERT( column + (~rhs).size() <= lhs.columns(), "Invalid number of columns" );

   return trySubAssign( lhs.operand(), ~rhs, band + ptrdiff_t( lhs.column() - lhs.row() ),
                        lhs.row() + row, lhs.column() + column );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Predict invariant violations by the subtraction assignment of a tensor to a subtensor.
// \ingroup subtensor
//
// \param lhs The target left-hand side subtensor.
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
template< typename TT1      // Type of the tensor
        , AlignmentFlag AF  // Alignment flag
        , bool DF           // Density flag
        , size_t... CSAs    // Compile time subtensor arguments
        , typename TT2 >     // Type of the right-hand side tensor
inline bool trySubAssign( const Subtensor<TT1,AF,CSAs...>& lhs,
                          const Tensor<TT2>& rhs, size_t row, size_t column )
{
   BLAZE_INTERNAL_ASSERT( row <= lhs.rows(), "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( column <= lhs.columns(), "Invalid column access index" );
   BLAZE_INTERNAL_ASSERT( row + (~rhs).rows() <= lhs.rows(), "Invalid number of rows" );
   BLAZE_INTERNAL_ASSERT( column + (~rhs).columns() <= lhs.columns(), "Invalid number of columns" );

   return trySubAssign( lhs.operand(), ~rhs, lhs.row() + row, lhs.column() + column );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Predict invariant violations by the multiplication assignment of a vector to a subtensor.
// \ingroup subtensor
//
// \param lhs The target left-hand side subtensor.
// \param rhs The right-hand side vector to be multiplied.
// \param row The row index of the first element to be modified.
// \param column The column index of the first element to be modified.
// \return \a true in case the assignment would be successful, \a false if not.
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename TT       // Type of the tensor
        , AlignmentFlag AF  // Alignment flag
        , size_t... CSAs    // Compile time subtensor arguments
        , typename VT       // Type of the right-hand side vector
        , bool TF >         // Transpose flag of the right-hand side vector
inline bool tryMultAssign( const Subtensor<TT,AF,CSAs...>& lhs,
                           const Vector<VT,TF>& rhs, size_t row, size_t column )
{
   BLAZE_INTERNAL_ASSERT( row <= lhs.rows(), "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( column <= lhs.columns(), "Invalid column access index" );
   BLAZE_INTERNAL_ASSERT( TF || ( row + (~rhs).size() <= lhs.rows() ), "Invalid number of rows" );
   BLAZE_INTERNAL_ASSERT( !TF || ( column + (~rhs).size() <= lhs.columns() ), "Invalid number of columns" );

   return tryMultAssign( lhs.operand(), ~rhs, lhs.row() + row, lhs.column() + column );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Predict invariant violations by the multiplication assignment of a vector to the band
//        of a subtensor.
// \ingroup subtensor
//
// \param lhs The target left-hand side subtensor.
// \param rhs The right-hand side vector to be multiplied.
// \param band The index of the band the right-hand side vector is assigned to.
// \param row The row index of the first element to be modified.
// \param column The column index of the first element to be modified.
// \return \a true in case the assignment would be successful, \a false if not.
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename TT       // Type of the tensor
        , AlignmentFlag AF  // Alignment flag
        , size_t... CSAs    // Compile time subtensor arguments
        , typename VT       // Type of the right-hand side vector
        , bool TF >         // Transpose flag of the right-hand side vector
inline bool tryMultAssign( const Subtensor<TT,AF,CSAs...>& lhs,
                           const Vector<VT,TF>& rhs, ptrdiff_t band, size_t row, size_t column )
{
   BLAZE_INTERNAL_ASSERT( row <= lhs.rows(), "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( column <= lhs.columns(), "Invalid column access index" );
   BLAZE_INTERNAL_ASSERT( row + (~rhs).size() <= lhs.rows(), "Invalid number of rows" );
   BLAZE_INTERNAL_ASSERT( column + (~rhs).size() <= lhs.columns(), "Invalid number of columns" );

   return tryMultAssign( lhs.operand(), ~rhs, band + ptrdiff_t( lhs.column() - lhs.row() ),
                         lhs.row() + row, lhs.column() + column );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Predict invariant violations by the Schur product assignment of a tensor to a subtensor.
// \ingroup subtensor
//
// \param lhs The target left-hand side subtensor.
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
template< typename TT1      // Type of the tensor
        , AlignmentFlag AF  // Alignment flag
        , size_t... CSAs    // Compile time subtensor arguments
        , typename TT2 >     // Type of the right-hand side tensor
inline bool trySchurAssign( const Subtensor<TT1,AF,CSAs...>& lhs,
                            const Tensor<TT2>& rhs, size_t row, size_t column )
{
   BLAZE_INTERNAL_ASSERT( row <= lhs.rows(), "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( column <= lhs.columns(), "Invalid column access index" );
   BLAZE_INTERNAL_ASSERT( row + (~rhs).rows() <= lhs.rows(), "Invalid number of rows" );
   BLAZE_INTERNAL_ASSERT( column + (~rhs).columns() <= lhs.columns(), "Invalid number of columns" );

   return trySchurAssign( lhs.operand(), ~rhs, lhs.row() + row, lhs.column() + column );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Predict invariant violations by the division assignment of a vector to a subtensor.
// \ingroup subtensor
//
// \param lhs The target left-hand side subtensor.
// \param rhs The right-hand side vector divisor.
// \param row The row index of the first element to be modified.
// \param column The column index of the first element to be modified.
// \return \a true in case the assignment would be successful, \a false if not.
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename TT       // Type of the tensor
        , AlignmentFlag AF  // Alignment flag
        , size_t... CSAs    // Compile time subtensor arguments
        , typename VT >      // Type of the right-hand side vector
inline bool tryDivAssign( const Subtensor<TT,AF,CSAs...>& lhs,
                          const Matrix<VT,false>& rhs, size_t row, size_t column, size_t page )
{
   BLAZE_INTERNAL_ASSERT( row <= lhs.rows(), "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( column <= lhs.columns(), "Invalid column access index" );
   BLAZE_INTERNAL_ASSERT( TF || ( row + (~rhs).size() <= lhs.rows() ), "Invalid number of rows" );
   BLAZE_INTERNAL_ASSERT( !TF || ( column + (~rhs).size() <= lhs.columns() ), "Invalid number of columns" );

   return tryDivAssign( lhs.operand(), ~rhs, lhs.row() + row, lhs.column() + column, lhs.page() + page );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Predict invariant violations by the division assignment of a vector to the band of
//        a subtensor.
// \ingroup subtensor
//
// \param lhs The target left-hand side subtensor.
// \param rhs The right-hand side vector divisor.
// \param band The index of the band the right-hand side vector is assigned to.
// \param row The row index of the first element to be modified.
// \param column The column index of the first element to be modified.
// \return \a true in case the assignment would be successful, \a false if not.
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename TT       // Type of the tensor
        , AlignmentFlag AF  // Alignment flag
        , size_t... CSAs    // Compile time subtensor arguments
        , typename VT >      // Type of the right-hand side vector
inline bool tryDivAssign( const Subtensor<TT,AF,CSAs...>& lhs,
                          const Matrix<VT,false>& rhs, ptrdiff_t band, size_t row, size_t column, size_t page )
{
   BLAZE_INTERNAL_ASSERT( row <= lhs.rows(), "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( column <= lhs.columns(), "Invalid column access index" );
   BLAZE_INTERNAL_ASSERT( row + (~rhs).size() <= lhs.rows(), "Invalid number of rows" );
   BLAZE_INTERNAL_ASSERT( column + (~rhs).size() <= lhs.columns(), "Invalid number of columns" );

   return tryDivAssign( lhs.operand(), ~rhs, band + ptrdiff_t( lhs.column() - lhs.row() ),
                        lhs.row() + row, lhs.column() + column, lhs.page() + page );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Removal of all restrictions on the data access to the given subtensor.
// \ingroup subtensor
//
// \param dm The subtensor to be derestricted.
// \return Subtensor without access restrictions.
//
// This function removes all restrictions on the data access to the given subtensor. It returns a
// subtensor that does provide the same interface but does not have any restrictions on the data
// access.\n
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in the violation of invariants, erroneous results and/or in compilation errors.
*/
template< typename TT       // Type of the tensor
        , AlignmentFlag AF  // Alignment flag
        , size_t K          // Index of the first page
        , size_t I          // Index of the first row
        , size_t J          // Index of the first column
        , size_t O          // Number of pages
        , size_t M          // Number of rows
        , size_t N >        // Number of columns
inline decltype(auto) derestrict( Subtensor<TT,AF,K,I,J,O,M,N>& dm )
{
   return subtensor<AF,K,I,J,O,M,N>( derestrict( dm.operand() ), unchecked );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Removal of all restrictions on the data access to the given temporary subtensor.
// \ingroup subtensor
//
// \param dm The temporary subtensor to be derestricted.
// \return Subtensor without access restrictions.
//
// This function removes all restrictions on the data access to the given temporary subtensor. It
// returns a subtensor that does provide the same interface but does not have any restrictions on
// the data access.\n
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in the violation of invariants, erroneous results and/or in compilation errors.
*/
template< typename TT       // Type of the tensor
        , AlignmentFlag AF  // Alignment flag
        , size_t K          // Index of the first page
        , size_t I          // Index of the first row
        , size_t J          // Index of the first column
        , size_t O          // Number of pages
        , size_t M          // Number of rows
        , size_t N >        // Number of columns
inline decltype(auto) derestrict( Subtensor<TT,AF,K,I,J,O,M,N>&& dm )
{
   return subtensor<AF,K,I,J,O,M,N>( derestrict( dm.operand() ), unchecked );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Removal of all restrictions on the data access to the given subtensor.
// \ingroup subtensor
//
// \param dm The subtensor to be derestricted.
// \return Subtensor without access restrictions.
//
// This function removes all restrictions on the data access to the given subtensor. It returns a
// subtensor that does provide the same interface but does not have any restrictions on the data
// access.\n
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in the violation of invariants, erroneous results and/or in compilation errors.
*/
template< typename TT       // Type of the tensor
        , AlignmentFlag AF > // Alignment flag
inline decltype(auto) derestrict( Subtensor<TT,AF>& dm )
{
   return subtensor<AF>( derestrict( dm.operand() ), dm.row(), dm.column(), dm.page(), dm.rows(), dm.columns(), dm.pages(), unchecked );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Removal of all restrictions on the data access to the given temporary subtensor.
// \ingroup subtensor
//
// \param dm The temporary subtensor to be derestricted.
// \return Subtensor without access restrictions.
//
// This function removes all restrictions on the data access to the given temporary subtensor. It
// returns a subtensor that does provide the same interface but does not have any restrictions on
// the data access.\n
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in the violation of invariants, erroneous results and/or in compilation errors.
*/
template< typename TT       // Type of the tensor
        , AlignmentFlag AF > // Alignment flag
inline decltype(auto) derestrict( Subtensor<TT,AF>&& dm )
{
   return subtensor<AF>( derestrict( dm.operand() ), dm.row(), dm.column(), dm.page(), dm.rows(), dm.columns(), dm.pages(), unchecked );
}
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  SIZE SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename TT, AlignmentFlag AF, size_t K, size_t I, size_t J, size_t O, size_t M, size_t N >
struct Size< Subtensor<TT,AF,K,I,J,O,M,N>, 0UL >
   : public PtrdiffT<M>
{};

template< typename TT, AlignmentFlag AF, size_t K, size_t I, size_t J, size_t O, size_t M, size_t N >
struct Size< Subtensor<TT,AF,K,I,J,O,M,N>, 1UL >
   : public PtrdiffT<N>
{};

template< typename TT, AlignmentFlag AF, size_t K, size_t I, size_t J, size_t O, size_t M, size_t N >
struct Size< Subtensor<TT,AF,K,I,J,O,M,N>, 2UL >
   : public PtrdiffT<O>
{};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  MAXSIZE SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename TT, AlignmentFlag AF, size_t K, size_t I, size_t J, size_t O, size_t M, size_t N >
struct MaxSize< Subtensor<TT,AF,K,I,J,O,M,N>, 0UL >
   : public PtrdiffT<M>
{};

template< typename TT, AlignmentFlag AF, size_t K, size_t I, size_t J, size_t O, size_t M, size_t N >
struct MaxSize< Subtensor<TT,AF,K,I,J,O,M,N>, 1UL >
   : public PtrdiffT<N>
{};

template< typename TT, AlignmentFlag AF, size_t K, size_t I, size_t J, size_t O, size_t M, size_t N >
struct MaxSize< Subtensor<TT,AF,K,I,J,O,M,N>, 2UL >
   : public PtrdiffT<O>
{};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  ISRESTRICTED SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename TT, AlignmentFlag AF, size_t... CSAs >
struct IsRestricted< Subtensor<TT,AF,CSAs...> >
   : public IsRestricted<TT>
{};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  HASCONSTDATAACCESS SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename TT, AlignmentFlag AF, size_t... CSAs >
struct HasConstDataAccess< Subtensor<TT,AF,CSAs...> >
   : public HasConstDataAccess<TT>
{};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  HASMUTABLEDATAACCESS SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename TT, AlignmentFlag AF, size_t... CSAs >
struct HasMutableDataAccess< Subtensor<TT,AF,CSAs...> >
   : public HasMutableDataAccess<TT>
{};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  ISALIGNED SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename TT, size_t... CSAs >
struct IsAligned< Subtensor<TT,aligned,CSAs...> >
   : public TrueType
{};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  ISCONTIGUOUS SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename TT, AlignmentFlag AF, size_t... CSAs >
struct IsContiguous< Subtensor<TT,AF,CSAs...> >
   : public IsContiguous<TT>
{};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  ISSYMMETRIC SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename TT, AlignmentFlag AF, size_t K, size_t I, size_t J, size_t O, size_t M, size_t N >
struct IsSymmetric< Subtensor<TT,AF,K,I,J,O,M,N> >
   : public BoolConstant< ( IsSymmetric_v<TT> && I == J && M == N ) >
{};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  ISHERMITIAN SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename TT, AlignmentFlag AF, size_t K, size_t I, size_t J, size_t O, size_t M, size_t N >
struct IsHermitian< Subtensor<TT,AF,K,I,J,O,M,N> >
   : public BoolConstant< ( IsHermitian_v<TT> && I == J && M == N ) >
{};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  ISLOWER SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename TT, AlignmentFlag AF, size_t K, size_t I, size_t J, size_t O, size_t M, size_t N >
struct IsLower< Subtensor<TT,AF,K,I,J,O,M,N> >
   : public BoolConstant< ( IsLower_v<TT> && I == J && M == N ) ||
                          ( IsStrictlyLower_v<TT> && I == J+1UL && M == N ) >
{};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  ISUNILOWER SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename TT, AlignmentFlag AF, size_t K, size_t I, size_t J, size_t O, size_t M, size_t N >
struct IsUniLower< Subtensor<TT,AF,K,I,J,O,M,N> >
   : public BoolConstant< ( IsUniLower_v<TT> && I == J && M == N ) >
{};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  ISSTRICTLYLOWER SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename TT, AlignmentFlag AF, size_t K, size_t I, size_t J, size_t O, size_t M, size_t N >
struct IsStrictlyLower< Subtensor<TT,AF,K,I,J,O,M,N> >
   : public BoolConstant< ( IsLower_v<TT> && I < J && M == N ) ||
                          ( IsStrictlyLower_v<TT> && I == J && M == N ) >
{};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  ISUPPER SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename TT, AlignmentFlag AF, size_t K, size_t I, size_t J, size_t O, size_t M, size_t N >
struct IsUpper< Subtensor<TT,AF,K,I,J,O,M,N> >
   : public BoolConstant< ( IsUpper_v<TT> && I == J && M == N ) ||
                          ( IsStrictlyUpper_v<TT> && I+1UL == J && M == N ) >
{};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  ISUNIUPPER SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename TT, AlignmentFlag AF, size_t K, size_t I, size_t J, size_t O, size_t M, size_t N >
struct IsUniUpper< Subtensor<TT,AF,K,I,J,O,M,N> >
   : public BoolConstant< ( IsUniUpper_v<TT> && I == J && M == N ) >
{};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  ISSTRICTLYUPPER SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename TT, AlignmentFlag AF, size_t K, size_t I, size_t J, size_t O, size_t M, size_t N >
struct IsStrictlyUpper< Subtensor<TT,AF,K,I,J,O,M,N> >
   : public BoolConstant< ( IsUpper_v<TT> && I > J && M == N ) ||
                          ( IsStrictlyUpper_v<TT> && I == J && M == N ) >
{};
/*! \endcond */
//*************************************************************************************************

} // namespace blaze

#endif
