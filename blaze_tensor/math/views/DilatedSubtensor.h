//=================================================================================================
/*!
//  \file blaze_tensor/math/views/DilatedSubtensor.h
//  \brief Header file for the implementation of the DilatedSubtensor view
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

#ifndef _BLAZE_TENSOR_MATH_VIEWS_DILATEDSUBTENSOR_H_
#define _BLAZE_TENSOR_MATH_VIEWS_DILATEDSUBTENSOR_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <blaze/math/Aliases.h>
#include <blaze/math/constraints/MutableDataAccess.h>
#include <blaze/math/constraints/RequiresEvaluation.h>
#include <blaze/math/expressions/DeclExpr.h>
#include <blaze/math/expressions/Matrix.h>
#include <blaze/math/expressions/SchurExpr.h>
//#include <blaze/math/expressions/TVecMatMultExpr.h>
//#include <blaze/math/expressions/VecTVecMultExpr.h>
#include <blaze/math/InversionFlag.h>
#include <blaze/math/ReductionFlag.h>
#include <blaze/math/shims/IsDefault.h>
#include <blaze/math/shims/Serial.h>
#include <blaze/math/typetraits/HasConstDataAccess.h>
#include <blaze/math/typetraits/HasMutableDataAccess.h>
#include <blaze/math/typetraits/IsHermitian.h>
#include <blaze/math/typetraits/IsLower.h>
#include <blaze/math/typetraits/IsRestricted.h>
#include <blaze/math/typetraits/IsStrictlyLower.h>
#include <blaze/math/typetraits/IsStrictlyUpper.h>
#include <blaze/math/typetraits/IsSymmetric.h>
#include <blaze/math/typetraits/IsUniLower.h>
#include <blaze/math/typetraits/IsUniUpper.h>
#include <blaze/math/typetraits/IsUpper.h>
#include <blaze/math/typetraits/Size.h>
#include <blaze/math/typetraits/StorageOrder.h>
#include <blaze/math/typetraits/TransposeFlag.h>
#include <blaze/math/views/Check.h>
#include <blaze/math/views/column/ColumnData.h>
#include <blaze/math/views/row/RowData.h>
#include <blaze/math/views/Subvector.h>
#include <blaze/util/algorithms/Max.h>
#include <blaze/util/algorithms/Min.h>
#include <blaze/util/Assert.h>
#include <blaze/util/EnableIf.h>
#include <blaze/util/FunctionTrace.h>
#include <blaze/util/IntegralConstant.h>
#include <blaze/util/MaybeUnused.h>
#include <blaze/util/SmallArray.h>
#include <blaze/util/StaticAssert.h>
#include <blaze/util/TypeList.h>
#include <blaze/util/Types.h>
#include <blaze/util/typetraits/IsPointer.h>
#include <blaze/util/typetraits/RemoveReference.h>

#include <blaze_tensor/math/Aliases.h>
#include <blaze_tensor/math/ReductionFlag.h>
#include <blaze_tensor/math/expressions/Forward.h>
#include <blaze_tensor/math/expressions/MatExpandExpr.h>
#include <blaze_tensor/math/expressions/TensEvalExpr.h>
#include <blaze_tensor/math/expressions/TensMapExpr.h>
#include <blaze_tensor/math/expressions/TensTensAddExpr.h>
#include <blaze_tensor/math/expressions/TensTensMapExpr.h>
#include <blaze_tensor/math/expressions/TensTensSubExpr.h>
#include <blaze_tensor/math/expressions/TensReduceExpr.h>
#include <blaze_tensor/math/expressions/TensScalarDivExpr.h>
#include <blaze_tensor/math/expressions/TensScalarMultExpr.h>
#include <blaze_tensor/math/expressions/TensSerialExpr.h>
#include <blaze_tensor/math/expressions/TensTransExpr.h>
#include <blaze_tensor/math/expressions/TensVecMultExpr.h>
#include <blaze_tensor/math/expressions/Tensor.h>
#include <blaze_tensor/math/IntegerSequence.h>
#include <blaze_tensor/math/views/DilatedSubmatrix.h>
#include <blaze_tensor/math/views/DilatedSubvector.h>
#include <blaze_tensor/math/views/Forward.h>
#include <blaze_tensor/math/views/columnslice/ColumnSliceData.h>
#include <blaze_tensor/math/views/dilatedsubtensor/BaseTemplate.h>
#include <blaze_tensor/math/views/dilatedsubtensor/Dense.h>
#include <blaze_tensor/math/views/pageslice/PageSliceData.h>
#include <blaze_tensor/math/views/rowslice/RowSliceData.h>
// #include <blaze_tensor/math/views/DilatedSubtensor/Sparse.h>

namespace blaze {

//=================================================================================================
//
//  GLOBAL FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Creating a view on a specific DilatedSubtensor of the given tensor.
// \ingroup DilatedSubtensor
//
// \param tensor The tensor containing the DilatedSubtensor.
// \param args Optional DilatedSubtensor arguments.
// \return View on the specific dilatedDilatedSubtensor of the tensor.
// \exception std::invalid_argument Invalid dilatedDilatedSubtensor specification.
//
// This function returns an expression representing the specified dilatedDilatedSubtensor of the given tensor.
// The following example demonstrates the creation of a dense and sparse dilatedDilatedSubtensor:

   \code
   blaze::DynamicTensor<double,blaze::rowMajor> D;
   blaze::CompressedTensor<int,blaze::columnMajor> S;
   // ... Resizing and initialization

   // Creating a dense DilatedSubtensor of size 8x4, starting in row 0 and column 16
   auto dsm = DilatedSubtensor<0UL,16UL,8UL,4UL>( D );

   // Creating a sparse DilatedSubtensor of size 7x3, starting in row 2 and column 4
   auto ssm = DilatedSubtensor<2UL,4UL,7UL,3UL>( S );
   \endcode

// By default, the provided DilatedSubtensor arguments are checked at runtime. In case the DilatedSubtensor
// is not properly specified (i.e. if the specified row or column is larger than the total number
// of rows or columns of the given tensor or the DilatedSubtensor is specified beyond the number of rows
// or columns of the tensor) a \a std::invalid_argument exception is thrown. The checks can be
// skipped by providing the optional \a blaze::unchecked argument.

   \code
   auto dsm = DilatedSubtensor<0UL,16UL,8UL,4UL>( D, unchecked );
   auto ssm = DilatedSubtensor<2UL,4UL,7UL,3UL>( S, unchecked );
   \endcode

// Please note that this function creates an unaligned dense or sparse DilatedSubtensor. For instance,
// the creation of the dense DilatedSubtensor is equivalent to the following function call:

   \code
   auto dsm = DilatedSubtensor<unaligned,0UL,16UL,8UL,4UL>( D );
   \endcode

// In contrast to unaligned subtensors, which provide full flexibility, aligned subtensors pose
// additional alignment restrictions. However, especially in case of dense subtensors this may
// result in considerable performance improvements. In order to create an aligned DilatedSubtensor the
// following function call has to be used:

   \code
   auto dsm = DilatedSubtensor<aligned,0UL,16UL,8UL,4UL>( D );
   \endcode

// Note however that in this case the given compile time arguments \a I, \a J, \a M, and \a N are
// subject to additional checks to guarantee proper alignment.
*/
template< size_t I               // Index of the first row
        , size_t J               // Index of the first column
        , size_t K               // Index of the first page
        , size_t M               // Number of rows
        , size_t N               // Number of columns
        , size_t O               // Number of pages
        , size_t RowDilation     // The page step-size of the dilatedsubtensor
        , size_t ColumnDilation  // The row step-size of the dilatedsubtensor
        , size_t PageDilation    // The column step-size of the dilatedsubtensor
        , typename TT            // Type of the dense tensor
        , typename... RSAs >     // Optional DilatedSubtensor arguments
inline decltype(auto) dilatedsubtensor( Tensor<TT>& tensor, RSAs... args )
{
   BLAZE_FUNCTION_TRACE;

   return dilatedsubtensor<K,I,J,O,M,N,PageDilation,RowDilation,ColumnDilation>( ~tensor, args... );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Creating a view on a specific DilatedSubtensor of the given constant tensor.
// \ingroup DilatedSubtensor
//
// \param tensor The constant tensor containing the DilatedSubtensor.
// \param args Optional DilatedSubtensor arguments.
// \return View on the specific DilatedSubtensor of the tensor.
// \exception std::invalid_argument Invalid DilatedSubtensor specification.
//
// This function returns an expression representing the specified DilatedSubtensor of the given constant
// tensor. The following example demonstrates the creation of a dense and sparse DilatedSubtensor:

   \code
   const blaze::DynamicTensor<double,blaze::rowMajor> D( ... );
   const blaze::CompressedTensor<int,blaze::columnMajor> S( ... );

   // Creating a dense DilatedSubtensor of size 8x4, starting in row 0 and column 16
   auto dsm = DilatedSubtensor<0UL,16UL,8UL,4UL>( D );

   // Creating a sparse DilatedSubtensor of size 7x3, starting in row 2 and column 4
   auto ssm = DilatedSubtensor<2UL,4UL,7UL,3UL>( S );
   \endcode

// By default, the provided DilatedSubtensor arguments are checked at runtime. In case the DilatedSubtensor
// is not properly specified (i.e. if the specified row or column is larger than the total number
// of rows or columns of the given tensor or the DilatedSubtensor is specified beyond the number of rows
// or columns of the tensor) a \a std::invalid_argument exception is thrown. The checks can be
// skipped by providing the optional \a blaze::unchecked argument.

   \code
   auto dsm = DilatedSubtensor<0UL,16UL,8UL,4UL>( D, unchecked );
   auto ssm = DilatedSubtensor<2UL,4UL,7UL,3UL>( S, unchecked );
   \endcode

// Please note that this function creates an unaligned dense or sparse DilatedSubtensor. For instance,
// the creation of the dense DilatedSubtensor is equivalent to the following three function calls:

   \code
   auto dsm = DilatedSubtensor<unaligned,0UL,16UL,8UL,4UL>( D );
   \endcode

// In contrast to unaligned subtensors, which provide full flexibility, aligned subtensors pose
// additional alignment restrictions. However, especially in case of dense subtensors this may
// result in considerable performance improvements. In order to create an aligned DilatedSubtensor the
// following function call has to be used:

   \code
   auto dsm = DilatedSubtensor<aligned,0UL,16UL,8UL,4UL>( D );
   \endcode

// Note however that in this case the given compile time arguments \a I, \a J, \a M, and \a N are
// subject to additional checks to guarantee proper alignment.
*/
template< size_t I               // Index of the first row
        , size_t J               // Index of the first column
        , size_t K               // Index of the first page
        , size_t M               // Number of rows
        , size_t N               // Number of columns
        , size_t O               // Number of pages
        , size_t RowDilation     // The row step-size of the dilatedsubtensor
        , size_t ColumnDilation  // The column step-size of the dilatedsubtensor
        , size_t PageDilation    // The page step-size of the dilatedsubtensor
        , typename TT            // Type of the dense tensor
        , typename... RSAs >     // Option DilatedSubtensor arguments
inline decltype(auto) dilatedsubtensor( const Tensor<TT>& tensor, RSAs... args )
{
   BLAZE_FUNCTION_TRACE;

   return dilatedsubtensor<K,I,J,O,M,N,PageDilation,RowDilation,ColumnDilation>( ~tensor, args... );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Creating a view on a specific DilatedSubtensor of the given temporary tensor.
// \ingroup DilatedSubtensor
//
// \param tensor The temporary tensor containing the DilatedSubtensor.
// \param args Optional DilatedSubtensor arguments.
// \return View on the specific DilatedSubtensor of the tensor.
// \exception std::invalid_argument Invalid DilatedSubtensor specification.
//
// This function returns an expression representing the specified DilatedSubtensor of the given
// temporary tensor. In case the DilatedSubtensor is not properly specified (i.e. if the specified
// row or column is greater than the total number of rows or columns of the given tensor or
// the DilatedSubtensor is specified beyond the number of rows or columns of the tensor) a
// \a std::invalid_argument exception is thrown.
*/
template< size_t I               // Index of the first row
        , size_t J               // Index of the first column
        , size_t K               // Index of the first page
        , size_t M               // Number of rows
        , size_t N               // Number of columns
        , size_t O               // Number of pages
        , size_t RowDilation     // The row step-size of the dilatedsubtensor
        , size_t ColumnDilation  // The column step-size of the dilatedsubtensor
        , size_t PageDilation    // The page step-size of the dilatedsubtensor
        , typename TT            // Type of the dense tensor
        , typename... RSAs >     // Option DilatedSubtensor arguments
inline decltype(auto) dilatedsubtensor( Tensor<TT>&& tensor, RSAs... args )
{
   BLAZE_FUNCTION_TRACE;

   return dilatedsubtensor<K,I,J,O,M,N,PageDilation,RowDilation,ColumnDilation>( ~tensor, args... );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Creating a view on a specific DilatedSubtensor of the given tensor.
// \ingroup DilatedSubtensor
//
// \param tensor The tensor containing the DilatedSubtensor.
// \param args Optional DilatedSubtensor arguments.
// \return View on the specific DilatedSubtensor of the tensor.
// \exception std::invalid_argument Invalid DilatedSubtensor specification.
//
// This function returns an expression representing an aligned or unaligned DilatedSubtensor of the
// given dense or sparse tensor, based on the specified alignment flag \a AF. The following
// example demonstrates the creation of both an aligned and unaligned DilatedSubtensor:

   \code
   blaze::DynamicTensor<double,blaze::rowMajor> D;
   blaze::CompressedTensor<int,blaze::columnMajor> S;
   // ... Resizing and initialization

   // Creating an aligned dense DilatedSubtensor of size 8x4, starting in row 0 and column 16
   auto dsm = DilatedSubtensor<aligned,0UL,16UL,8UL,4UL>( D );

   // Creating an unaligned sparse DilatedSubtensor of size 7x3, starting in row 2 and column 4
   auto ssm = DilatedSubtensor<unaligned,2UL,4UL,7UL,3UL>( S );
   \endcode

// By default, the provided DilatedSubtensor arguments are checked at runtime. In case the DilatedSubtensor
// is not properly specified (i.e. if the specified row or column is larger than the total number
// of rows or columns of the given tensor or the DilatedSubtensor is specified beyond the number of rows
// or columns of the tensor) a \a std::invalid_argument exception is thrown. The checks can be
// skipped by providing the optional \a blaze::unchecked argument.

   \code
   auto dsm = DilatedSubtensor<aligned,0UL,16UL,8UL,4UL>( D, unchecked );
   auto ssm = DilatedSubtensor<unaligned,2UL,4UL,7UL,3UL>( S, unchecked );
   \endcode

// Please note that this function creates an unaligned dense or sparse DilatedSubtensor. For instance,
// the creation of the dense DilatedSubtensor is equivalent to the following function call:

*/
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Creating a view on a specific DilatedSubtensor of the given tensor.
// \ingroup DilatedSubtensor
//
// \param tensor The tensor containing the DilatedSubtensor.
// \param row The index of the first row of the DilatedSubtensor.
// \param column The index of the first column of the DilatedSubtensor.
// \param m The number of rows of the DilatedSubtensor.
// \param n The number of columns of the DilatedSubtensor.
// \param args Optional DilatedSubtensor arguments.
// \return View on the specific DilatedSubtensor of the tensor.
// \exception std::invalid_argument Invalid DilatedSubtensor specification.
//
// This function returns an expression representing the specified DilatedSubtensor of the given tensor.
// The following example demonstrates the creation of a dense and sparse DilatedSubtensor:

   \code
   blaze::DynamicTensor<double,blaze::rowMajor> D;
   blaze::CompressedTensor<int,blaze::columnMajor> S;
   // ... Resizing and initialization

   // Creating a dense DilatedSubtensor of size 8x4, starting in row 0 and column 16
   auto dsm = DilatedSubtensor( D, 0UL, 16UL, 8UL, 4UL );

   // Creating a sparse DilatedSubtensor of size 7x3, starting in row 2 and column 4
   auto ssm = DilatedSubtensor( S, 2UL, 4UL, 7UL, 3UL );
   \endcode

// By default, the provided DilatedSubtensor arguments are checked at runtime. In case the DilatedSubtensor
// is not properly specified (i.e. if the specified row or column is larger than the total number
// of rows or columns of the given tensor or the DilatedSubtensor is specified beyond the number of rows
// or columns of the tensor) a \a std::invalid_argument exception is thrown. The checks can be
// skipped by providing the optional \a blaze::unchecked argument.

   \code
   auto dsm = DilatedSubtensor( D, 0UL, 16UL, 8UL, 4UL, unchecked );
   auto ssm = DilatedSubtensor( S, 2UL, 4UL, 7UL, 3UL, unchecked );
   \endcode

// Please note that this function creates an unaligned dense or sparse DilatedSubtensor. For instance,
// the creation of the dense DilatedSubtensor is equivalent to the following function call:

   \code
   unaligned dsm = DilatedSubtensor<unaligned>( D, 0UL, 16UL, 8UL, 4UL );
   \endcode

// In contrast to unaligned subtensors, which provide full flexibility, aligned subtensors pose
// additional alignment restrictions. However, especially in case of dense subtensors this may
// result in considerable performance improvements. In order to create an aligned DilatedSubtensor the
// following function call has to be used:

   \code
   auto dsm = DilatedSubtensor<aligned>( D, 0UL, 16UL, 8UL, 4UL );
   \endcode

// Note however that in this case the given arguments \a row, \a column, \a m, and \a n are
// subject to additional checks to guarantee proper alignment.
*/
template< typename TT         // Type of the dense tensor
        , typename... RSAs >  // Option DilatedSubtensor arguments
inline decltype(auto)
   dilatedsubtensor( Tensor<TT>& tensor, size_t page, size_t row, size_t column, size_t o, size_t m, size_t n,
      size_t pagedilation, size_t rowdilation, size_t columndilation, RSAs... args )
{
   BLAZE_FUNCTION_TRACE;

   using ReturnType = DilatedSubtensor_<TT>;
   return ReturnType( ~tensor, page, row, column, o, m, n, pagedilation, rowdilation, columndilation, args... );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Creating a view on a specific DilatedSubtensor of the given constant tensor.
// \ingroup DilatedSubtensor
//
// \param tensor The constant tensor containing the DilatedSubtensor.
// \param row The index of the first row of the DilatedSubtensor.
// \param column The index of the first column of the DilatedSubtensor.
// \param m The number of rows of the DilatedSubtensor.
// \param n The number of columns of the DilatedSubtensor.
// \param args Optional DilatedSubtensor arguments.
// \return View on the specific DilatedSubtensor of the tensor.
// \exception std::invalid_argument Invalid DilatedSubtensor specification.
//
// This function returns an expression representing the specified DilatedSubtensor of the given constant
// tensor. The following example demonstrates the creation of a dense and sparse DilatedSubtensor:

   \code
   const blaze::DynamicTensor<double,blaze::rowMajor> D( ... );
   const blaze::CompressedTensor<int,blaze::columnMajor> S( ... );

   // Creating a dense DilatedSubtensor of size 8x4, starting in row 0 and column 16
   auto dsm = DilatedSubtensor( D, 0UL, 16UL, 8UL, 4UL );

   // Creating a sparse DilatedSubtensor of size 7x3, starting in row 2 and column 4
   auto ssm = DilatedSubtensor( S, 2UL, 4UL, 7UL, 3UL );
   \endcode

// By default, the provided DilatedSubtensor arguments are checked at runtime. In case the DilatedSubtensor
// is not properly specified (i.e. if the specified row or column is larger than the total number
// of rows or columns of the given tensor or the DilatedSubtensor is specified beyond the number of rows
// or columns of the tensor) a \a std::invalid_argument exception is thrown. The checks can be
// skipped by providing the optional \a blaze::unchecked argument.

   \code
   auto dsm = DilatedSubtensor( D, 0UL, 16UL, 8UL, 4UL, unchecked );
   auto ssm = DilatedSubtensor( S, 2UL, 4UL, 7UL, 3UL, unchecked );
   \endcode

// Please note that this function creates an unaligned dense or sparse DilatedSubtensor. For instance,
// the creation of the dense DilatedSubtensor is equivalent to the following three function calls:

   \code
   auto dsm = DilatedSubtensor<unaligned>( D, 0UL, 16UL, 8UL, 4UL );
   \endcode

// In contrast to unaligned subtensors, which provide full flexibility, aligned subtensors pose
// additional alignment restrictions. However, especially in case of dense subtensors this may
// result in considerable performance improvements. In order to create an aligned DilatedSubtensor the
// following function call has to be used:

   \code
   auto dsm = DilatedSubtensor<aligned>( D, 0UL, 16UL, 8UL, 4UL );
   \endcode

// Note however that in this case the given arguments \a row, \a column, \a m, and \a n are
// subject to additional checks to guarantee proper alignment.
*/
template< typename TT         // Type of the dense tensor
        , typename... RSAs >  // Option DilatedSubtensor arguments
inline decltype(auto)
   dilatedsubtensor( const Tensor<TT>& tensor, size_t page, size_t row, size_t column, size_t o, size_t m, size_t n,
      size_t pagedilation, size_t rowdilation, size_t columndilation, RSAs... args )
{
   BLAZE_FUNCTION_TRACE;

   using ReturnType = const DilatedSubtensor_<const TT>;
   return ReturnType( ~tensor, page, row, column, o, m, n, pagedilation, rowdilation, columndilation, args... );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Creating a view on a specific DilatedSubtensor of the given temporary tensor.
// \ingroup DilatedSubtensor
//
// \param tensor The temporary tensor containing the DilatedSubtensor.
// \param row The index of the first row of the DilatedSubtensor.
// \param column The index of the first column of the DilatedSubtensor.
// \param m The number of rows of the DilatedSubtensor.
// \param n The number of columns of the DilatedSubtensor.
// \param args Optional DilatedSubtensor arguments.
// \return View on the specific DilatedSubtensor of the tensor.
// \exception std::invalid_argument Invalid DilatedSubtensor specification.
//
// This function returns an expression representing the specified DilatedSubtensor of the given
// temporary tensor. In case the DilatedSubtensor is not properly specified (i.e. if the specified
// row or column is greater than the total number of rows or columns of the given tensor or
// the DilatedSubtensor is specified beyond the number of rows or columns of the tensor) a
// \a std::invalid_argument exception is thrown.
*/
template< typename TT         // Type of the dense tensor
        , typename... RSAs >  // Option DilatedSubtensor arguments
inline decltype(auto)
   dilatedsubtensor( Tensor<TT>&& tensor, size_t page, size_t row, size_t column, size_t o, size_t m, size_t n,
      size_t pagedilation, size_t rowdilation, size_t columndilation, RSAs... args )
{
   BLAZE_FUNCTION_TRACE;

   using ReturnType = DilatedSubtensor_<TT>;
   return ReturnType( ~tensor, page, row, column, o, m, n, pagedilation, rowdilation, columndilation, args... );
}

//=================================================================================================
//
//  GLOBAL RESTRUCTURING FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific DilatedSubtensor of the given tensor/tensor addition.
// \ingroup DilatedSubtensor
//
// \param tensor The constant tensor/tensor addition.
// \param args The runtime DilatedSubtensor arguments
// \return View on the specified DilatedSubtensor of the addition.
//
// This function returns an expression representing the specified DilatedSubtensor of the given
// tensor/tensor addition.
*/
template< size_t... CSAs      // Compile time DilatedSubtensor arguments
        , typename TT         // Tensor base type of the expression
        , typename... RSAs >  // Runtime DilatedSubtensor arguments
inline decltype(auto) dilatedsubtensor( const TensTensAddExpr<TT>& tensor, RSAs... args )
{
   BLAZE_FUNCTION_TRACE;

   return dilatedsubtensor<CSAs...>( (~tensor).leftOperand(), args... ) +
          dilatedsubtensor<CSAs...>( (~tensor).rightOperand(), args... );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific DilatedSubtensor of the given tensor/tensor subtraction.
// \ingroup DilatedSubtensor
//
// \param tensor The constant tensor/tensor subtraction.
// \param args The runtime DilatedSubtensor arguments
// \return View on the specified DilatedSubtensor of the subtraction.
//
// This function returns an expression representing the specified DilatedSubtensor of the given
// tensor/tensor subtraction.
*/
template< size_t... CSAs      // Compile time DilatedSubtensor arguments
        , typename TT         // Tensor base type of the expression
        , typename... RSAs >  // Runtime DilatedSubtensor arguments
inline decltype(auto) dilatedsubtensor( const TensTensSubExpr<TT>& tensor, RSAs... args )
{
   BLAZE_FUNCTION_TRACE;

   return dilatedsubtensor<CSAs...>( (~tensor).leftOperand(), args... ) -
          dilatedsubtensor<CSAs...>( (~tensor).rightOperand(), args... );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific DilatedSubtensor of the given Schur product.
// \ingroup DilatedSubtensor
//
// \param tensor The constant Schur product.
// \param args The runtime DilatedSubtensor arguments
// \return View on the specified DilatedSubtensor of the Schur product.
//
// This function returns an expression representing the specified DilatedSubtensor of the given Schur
// product.
*/
template< size_t... CSAs      // Compile time DilatedSubtensor arguments
        , typename TT         // Tensor base type of the expression
        , typename... RSAs >  // Runtime DilatedSubtensor arguments
inline decltype(auto) dilatedsubtensor( const SchurExpr<TT>& tensor, RSAs... args )
{
   BLAZE_FUNCTION_TRACE;

   return dilatedsubtensor<CSAs...>( (~tensor).leftOperand(), args... ) %
          dilatedsubtensor<CSAs...>( (~tensor).rightOperand(), args... );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific DilatedSubtensor of the given tensor/scalar multiplication.
// \ingroup DilatedSubtensor
//
// \param tensor The constant tensor/scalar multiplication.
// \param args The runtime DilatedSubtensor arguments.
// \return View on the specified DilatedSubtensor of the multiplication.
//
// This function returns an expression representing the specified DilatedSubtensor of the given
// tensor/scalar multiplication.
*/
template< size_t... CSAs      // Compile time DilatedSubtensor arguments
        , typename TT         // Tensor base type of the expression
        , typename... RSAs >  // Runtime DilatedSubtensor arguments
inline decltype(auto) dilatedsubtensor( const TensScalarMultExpr<TT>& tensor, RSAs... args )
{
   BLAZE_FUNCTION_TRACE;

   return dilatedsubtensor<CSAs...>( (~tensor).leftOperand(), args... ) * (~tensor).rightOperand();
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific DilatedSubtensor of the given tensor/scalar division.
// \ingroup DilatedSubtensor
//
// \param tensor The constant tensor/scalar division.
// \param args The runtime DilatedSubtensor arguments.
// \return View on the specified DilatedSubtensor of the division.
//
// This function returns an expression representing the specified DilatedSubtensor of the given
// tensor/scalar division.
*/
template< size_t... CSAs      // Compile time DilatedSubtensor arguments
        , typename TT         // Tensor base type of the expression
        , typename... RSAs >  // Runtime DilatedSubtensor arguments
inline decltype(auto) dilatedsubtensor( const TensScalarDivExpr<TT>& tensor, RSAs... args )
{
   BLAZE_FUNCTION_TRACE;

   return dilatedsubtensor<CSAs...>( (~tensor).leftOperand(), args... ) / (~tensor).rightOperand();
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific DilatedSubtensor of the given unary tensor map operation.
// \ingroup DilatedSubtensor
//
// \param tensor The constant unary tensor map operation.
// \param args The runtime DilatedSubtensor arguments.
// \return View on the specified DilatedSubtensor of the unary map operation.
//
// This function returns an expression representing the specified DilatedSubtensor of the given unary
// tensor map operation.
*/
template< size_t... CSAs      // Compile time DilatedSubtensor arguments
        , typename TT         // Tensor base type of the expression
        , typename... RSAs >  // Runtime DilatedSubtensor arguments
inline decltype(auto) dilatedsubtensor( const TensMapExpr<TT>& tensor, RSAs... args )
{
   BLAZE_FUNCTION_TRACE;

   return map( dilatedsubtensor<CSAs...>( (~tensor).operand(), args... ), (~tensor).operation() );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific DilatedSubtensor of the given binary tensor map operation.
// \ingroup DilatedSubtensor
//
// \param tensor The constant binary tensor map operation.
// \param args The runtime DilatedSubtensor arguments.
// \return View on the specified DilatedSubtensor of the binary map operation.
//
// This function returns an expression representing the specified DilatedSubtensor of the given binary
// tensor map operation.
*/
template< size_t... CSAs      // Compile time DilatedSubtensor arguments
        , typename TT         // Tensor base type of the expression
        , typename... RSAs >  // Runtime DilatedSubtensor arguments
inline decltype(auto) dilatedsubtensor( const TensTensMapExpr<TT>& tensor, RSAs... args )
{
   BLAZE_FUNCTION_TRACE;

   return map( dilatedsubtensor<CSAs...>( (~tensor).leftOperand(), args... ),
               dilatedsubtensor<CSAs...>( (~tensor).rightOperand(), args... ),
               (~tensor).operation() );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific DilatedSubtensor of the given tensor evaluation operation.
// \ingroup DilatedSubtensor
//
// \param tensor The constant tensor evaluation operation.
// \param args The runtime DilatedSubtensor arguments.
// \return View on the specified DilatedSubtensor of the evaluation operation.
//
// This function returns an expression representing the specified DilatedSubtensor of the given tensor
// evaluation operation.
*/
template< size_t... CSAs      // Compile time DilatedSubtensor arguments
        , typename TT         // Tensor base type of the expression
        , typename... RSAs >  // Runtime DilatedSubtensor arguments
inline decltype(auto) dilatedsubtensor( const TensEvalExpr<TT>& tensor, RSAs... args )
{
   BLAZE_FUNCTION_TRACE;

   return eval( dilatedsubtensor<CSAs...>( (~tensor).operand(), args... ) );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific DilatedSubtensor of the given tensor serialization operation.
// \ingroup DilatedSubtensor
//
// \param tensor The constant tensor serialization operation.
// \param args The runtime DilatedSubtensor arguments.
// \return View on the specified DilatedSubtensor of the serialization operation.
//
// This function returns an expression representing the specified DilatedSubtensor of the given tensor
// serialization operation.
*/
template< size_t... CSAs      // Compile time DilatedSubtensor arguments
        , typename TT         // Tensor base type of the expression
        , typename... RSAs >  // Runtime DilatedSubtensor arguments
inline decltype(auto) dilatedsubtensor( const TensSerialExpr<TT>& tensor, RSAs... args )
{
   BLAZE_FUNCTION_TRACE;

   return serial( dilatedsubtensor<CSAs...>( (~tensor).operand(), args... ) );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific DilatedSubtensor of the given tensor transpose operation.
// \ingroup DilatedSubtensor
//
// \param tensor The constant tensor transpose operation.
// \param args Optional DilatedSubtensor arguments.
// \return View on the specified DilatedSubtensor of the transpose operation.
//
// This function returns an expression representing the specified DilatedSubtensor of the given tensor
// transpose operation.
*/
//template< size_t K               // Index of the first page
//        , size_t I               // Index of the first row
//        , size_t J               // Index of the first column
//        , size_t O               // Number of pages
//        , size_t M               // Number of rows
//        , size_t N               // Number of columns
//        , size_t PageDilation    // The page step-size of the dilatedsubtensor
//        , size_t RowDilation     // The row step-size of the dilatedsubtensor
//        , size_t ColumnDilation  // The column step-size of the dilatedsubtensor
//        , typename TT            // Tensor base type of the expression
//        , typename... RSAs >     // Optional DilatedSubtensor arguments
//inline decltype(auto) dilatedsubtensor( const TensTransExpr<TT>& tensor, RSAs... args )
//{
//   BLAZE_FUNCTION_TRACE;
//
//   return trans( dilatedsubtensor<J,I,K,N,M,O,ColumnDilation,RowDilation,PageDilation>( (~tensor).operand(), args... ) );
//}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific DilatedSubtensor of the given tensor transpose operation.
// \ingroup DilatedSubtensor
//
// \param tensor The constant tensor transpose operation.
// \param row The index of the first row of the DilatedSubtensor.
// \param column The index of the first column of the DilatedSubtensor.
// \param m The number of rows of the DilatedSubtensor.
// \param n The number of columns of the DilatedSubtensor.
// \param args Optional DilatedSubtensor arguments.
// \return View on the specified DilatedSubtensor of the transpose operation.
//
// This function returns an expression representing the specified DilatedSubtensor of the given tensor
// transpose operation.
*/
template< typename TT         // Tensor base type of the expression
        , typename... RSAs >  // Optional DilatedSubtensor arguments
inline decltype(auto)
   dilatedsubtensor( const TensTransExpr<TT>& tensor, size_t page, size_t row, size_t column, size_t o, size_t m, size_t n,
      size_t pagedilation, size_t rowdilation, size_t columndilation, RSAs... args )
{
   BLAZE_FUNCTION_TRACE;

   return trans( dilatedsubtensor( (~tensor).operand(), page, column, row, n, m, o, pagedilation, columndilation, rowdilation, args... ) );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific DilatedSubtensor of the given vector expansion operation.
// \ingroup DilatedSubtensor
//
// \param tensor The constant vector expansion operation.
// \param args Optional DilatedSubtensor arguments.
// \return View on the specified DilatedSubtensor of the expansion operation.
//
// This function returns an expression representing the specified DilatedSubtensor of the given vector
// expansion operation.
*/
template< size_t K               // Index of the first page
        , size_t I               // Index of the first row
        , size_t J               // Index of the first column
        , size_t O               // Number of pages
        , size_t M               // Number of rows
        , size_t N               // Number of columns
        , size_t PageDilation    // The page step-size of the dilatedsubtensor
        , size_t RowDilation     // The row step-size of the dilatedsubtensor
        , size_t ColumnDilation  // The column step-size of the dilatedsubtensor
        , typename TT         // Tensor base type of the expression
        , size_t... CEAs      // Compile time expansion arguments
        , typename... RSAs >  // Optional DilatedSubtensor arguments
inline decltype(auto) dilatedsubtensor( const MatExpandExpr<TT,CEAs...>& tensor, RSAs... args )
{
   BLAZE_FUNCTION_TRACE;

   using MT = MatrixType_t< RemoveReference_t< decltype( (~tensor).operand() ) > >;

   constexpr bool SO( StorageOrder_v<MT> );

   constexpr size_t row      ( SO ? J : I );
   constexpr size_t column   ( SO ? I : J );
   constexpr size_t rows     ( SO ? N : M );
   constexpr size_t columns  ( SO ? M : N );
   constexpr size_t rowdilation   ( SO ? ColumnDilation : RowDilation );
   constexpr size_t columndilation( SO ? RowDilation : ColumnDilation );

   return expand<O>( dilatedsubmatrix<row,column,rows,columns,rowdilation,columndilation>( (~tensor).operand(), args... ) );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific DilatedSubtensor of the given vector expansion operation.
// \ingroup DilatedSubtensor
//
// \param tensor The constant vector expansion operation.
// \param row The index of the first row of the DilatedSubtensor.
// \param column The index of the first column of the DilatedSubtensor.
// \param m The number of rows of the DilatedSubtensor.
// \param n The number of columns of the DilatedSubtensor.
// \param args Optional DilatedSubtensor arguments.
// \return View on the specified DilatedSubtensor of the expansion operation.
//
// This function returns an expression representing the specified DilatedSubtensor of the given vector
// expansion operation.
*/
template< typename TT         // Tensor base type of the expression
        , size_t... CEAs      // Compile time expansion arguments
        , typename... RSAs >  // Optional DilatedSubtensor arguments
inline decltype(auto)
   dilatedsubtensor( const MatExpandExpr<TT,CEAs...>& tensor,
              size_t page, size_t row, size_t column, size_t o, size_t m, size_t n,
              size_t pagedilation, size_t rowdilation, size_t columndilation, RSAs... args )
{
   BLAZE_FUNCTION_TRACE;

   using MT = MatrixType_t< RemoveReference_t< decltype( (~tensor).operand() ) > >;

   constexpr bool SO( StorageOrder_v<MT> );

   constexpr size_t I   ( SO ? column : row    );
   constexpr size_t J   ( SO ? row    : column );
   constexpr size_t M   ( SO ? n      : m    );
   constexpr size_t N   ( SO ? m      : n );
   constexpr size_t RowDilation   ( SO ? columndilation : rowdilation );
   constexpr size_t ColumnDilation( SO ? rowdilation : columndilation );

   return expand( dilatedsubmatrix( (~tensor).operand(), I,J,M,N,RowDilation,ColumnDilation, args... ), o );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific DilatedSubtensor of another DilatedSubtensor.
// \ingroup DilatedSubtensor
//
// \param sm The given DilatedSubtensor
// \param args The optional DilatedSubtensor arguments.
// \return View on the specified DilatedSubtensor of the other DilatedSubtensor.
//
// This function returns an expression representing the specified DilatedSubtensor of the given DilatedSubtensor.
*/
template< size_t I1               // Index of the first row
        , size_t J1               // Index of the first column
        , size_t K1               // Index of the first page
        , size_t M1               // Number of rows
        , size_t N1               // Number of columns
        , size_t O1               // Number of pages
        , size_t RowDilation1     // The row step-size of the dilatedsubtensor
        , size_t ColumnDilation1  // The column step-size of the dilatedsubtensor
        , size_t PageDilation1    // The page step-size of the dilatedsubtensor
        , typename TT             // Type of the sparse DilatedSubtensor
        , bool DF                 // Density flag
        , size_t I2               // Index of the first row
        , size_t J2               // Index of the first column
        , size_t K2               // Index of the first page
        , size_t M2               // Number of rows
        , size_t N2               // Number of columns
        , size_t O2               // Number of pages
        , size_t RowDilation2     // The row step-size of the dilatedsubtensor
        , size_t ColumnDilation2  // The column step-size of the dilatedsubtensor
        , size_t PageDilation2    // The page step-size of the dilatedsubtensor
        , typename... RSAs >      // Optional DilatedSubtensor arguments
inline decltype(auto) dilatedsubtensor( DilatedSubtensor<TT,DF,I2,J2,K2,M2,N2,O2,RowDilation2,ColumnDilation2,PageDilation2>& st, RSAs... args )
{
   BLAZE_FUNCTION_TRACE;

   BLAZE_STATIC_ASSERT_MSG( K1 + O1 * PageDilation1 <= O2 * PageDilation2, "Invalid dilatedsubtensor specification" );
   BLAZE_STATIC_ASSERT_MSG( I1 + M1 * RowDilation1 <= M2 * RowDilation2, "Invalid dilatedsubtensor specification" );
   BLAZE_STATIC_ASSERT_MSG( J1 + N1 * ColumnDilation1 <= N2 * ColumnDilation2, "Invalid dilatedsubtensor specification" );

   return dilatedsubtensor<I1*RowDilation2+I2,J1*ColumnDilation2+J2,K1*PageDilation2+K2,M1,N1,O1,
      RowDilation1*RowDilation2, ColumnDilation1*ColumnDilation2,PageDilation1*PageDilation2>( st.operand(), args... );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific DilatedSubtensor of another constant DilatedSubtensor.
// \ingroup DilatedSubtensor
//
// \param sm The given constant DilatedSubtensor
// \param args The optional DilatedSubtensor arguments.
// \return View on the specified DilatedSubtensor of the other DilatedSubtensor.
//
// This function returns an expression representing the specified DilatedSubtensor of the given constant
// DilatedSubtensor.
*/
template< size_t I1               // Index of the first row
        , size_t J1               // Index of the first column
        , size_t K1               // Index of the first page
        , size_t M1               // Number of rows
        , size_t N1               // Number of columns
        , size_t O1               // Number of pages
        , size_t RowDilation1     // The row step-size of the dilatedsubtensor
        , size_t ColumnDilation1  // The column step-size of the dilatedsubtensor
        , size_t PageDilation1    // The page step-size of the dilatedsubtensor
        , typename TT             // Type of the sparse DilatedSubtensor
        , bool DF                 // Density flag
        , size_t I2               // Index of the first row
        , size_t J2               // Index of the first column
        , size_t K2               // Index of the first page
        , size_t M2               // Number of rows
        , size_t N2               // Number of columns
        , size_t O2               // Number of pages
        , size_t RowDilation2     // The row step-size of the dilatedsubtensor
        , size_t ColumnDilation2  // The column step-size of the dilatedsubtensor
        , size_t PageDilation2    // The page step-size of the dilatedsubtensor
        , typename... RSAs >  // Optional DilatedSubtensor arguments
inline decltype(auto) dilatedsubtensor( const DilatedSubtensor<TT,DF,I2,J2,K2,M2,N2,O2,RowDilation2,ColumnDilation2,PageDilation2>& st, RSAs... args )
{
   BLAZE_FUNCTION_TRACE;

   BLAZE_STATIC_ASSERT_MSG( K1 + O1 * PageDilation1 <= O2 * PageDilation2, "Invalid dilatedsubtensor specification" );
   BLAZE_STATIC_ASSERT_MSG( I1 + M1 * RowDilation1 <= M2 * RowDilation2, "Invalid dilatedsubtensor specification" );
   BLAZE_STATIC_ASSERT_MSG( J1 + N1 * ColumnDilation1 <= N2 * ColumnDilation2, "Invalid dilatedsubtensor specification" );

   return dilatedsubtensor<I1*RowDilation2+I2,J1*ColumnDilation2+J2,K1*PageDilation2+K2,M1,N1,O1,
      RowDilation1*RowDilation2, ColumnDilation1*ColumnDilation2,PageDilation1*PageDilation2>( st.operand(), args... );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific DilatedSubtensor of another temporary DilatedSubtensor.
// \ingroup DilatedSubtensor
//
// \param sm The given temporary DilatedSubtensor
// \param args The optional DilatedSubtensor arguments.
// \return View on the specified DilatedSubtensor of the other DilatedSubtensor.
//
// This function returns an expression representing the specified DilatedSubtensor of the given temporary
// DilatedSubtensor.
*/
template< size_t I1               // Index of the first row
        , size_t J1               // Index of the first column
        , size_t K1               // Index of the first page
        , size_t M1               // Number of rows
        , size_t N1               // Number of columns
        , size_t O1               // Number of pages
        , size_t RowDilation1     // The row step-size of the dilatedsubtensor
        , size_t ColumnDilation1  // The column step-size of the dilatedsubtensor
        , size_t PageDilation1    // The page step-size of the dilatedsubtensor
        , typename TT             // Type of the sparse DilatedSubtensor
        , bool DF                 // Density flag
        , size_t I2               // Index of the first row
        , size_t J2               // Index of the first column
        , size_t K2               // Index of the first page
        , size_t M2               // Number of rows
        , size_t N2               // Number of columns
        , size_t O2               // Number of pages
        , size_t RowDilation2     // The row step-size of the dilatedsubtensor
        , size_t ColumnDilation2  // The column step-size of the dilatedsubtensor
        , size_t PageDilation2    // The page step-size of the dilatedsubtensor
        , typename... RSAs >  // Optional DilatedSubtensor arguments
inline decltype(auto) dilatedsubtensor( DilatedSubtensor<TT,DF,I2,J2,K2,M2,N2,O2,RowDilation2,ColumnDilation2,PageDilation2>&& st, RSAs... args )
{
   BLAZE_FUNCTION_TRACE;

   BLAZE_STATIC_ASSERT_MSG( K1 + O1 * PageDilation1 <= O2 * PageDilation2, "Invalid dilatedsubtensor specification" );
   BLAZE_STATIC_ASSERT_MSG( I1 + M1 * RowDilation1 <= M2 * RowDilation2, "Invalid dilatedsubtensor specification" );
   BLAZE_STATIC_ASSERT_MSG( J1 + N1 * ColumnDilation1 <= N2 * ColumnDilation2, "Invalid dilatedsubtensor specification" );

   return dilatedsubtensor<I1*RowDilation2+I2,J1*ColumnDilation2+J2,K1*PageDilation2+K2,M1,N1,O1,
      RowDilation1*RowDilation2, ColumnDilation1*ColumnDilation2,PageDilation1*PageDilation2>( st.operand(), args... );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific DilatedSubtensor of another DilatedSubtensor.
// \ingroup DilatedSubtensor
//
// \param sm The given DilatedSubtensor
// \param args The optional DilatedSubtensor arguments.
// \return View on the specified DilatedSubtensor of the other DilatedSubtensor.
// \exception std::invalid_argument Invalid DilatedSubtensor specification.
//
// This function returns an expression representing the specified DilatedSubtensor of the given DilatedSubtensor.
*/
template< size_t K               // Index of the first page
        , size_t I               // Index of the first row
        , size_t J               // Index of the first column
        , size_t O               // Number of pages
        , size_t M               // Number of rows
        , size_t N               // Number of columns
        , size_t PageDilation    // The page step-size of the dilatedsubtensor
        , size_t RowDilation     // The row step-size of the dilatedsubtensor
        , size_t ColumnDilation  // The column step-size of the dilatedsubtensor
        , typename TT         // Type of the sparse DilatedSubtensor
        , bool DF             // Density flag
        , typename... RSAs >  // Optional DilatedSubtensor arguments
inline decltype(auto) dilatedsubtensor( DilatedSubtensor<TT,DF>& st, RSAs... args )
{
   BLAZE_FUNCTION_TRACE;

   constexpr bool isChecked( !Contains_v< TypeList<RSAs...>, Unchecked > );

   if( isChecked ) {
      if( ( K + O * PageDilation > st.pages() * st.pagedilation() ) ||
          ( I + M * RowDilation > st.rows() * st.rowdilation() ) ||
          ( J + N * ColumnDilation > st.columns() * st.columndilation() ) )
      {
         BLAZE_THROW_INVALID_ARGUMENT( "Invalid dilatedsubtensor specification" );
      }
   }
   else {
      BLAZE_USER_ASSERT( K + O * PageDilation <= st.pages() * st.pagedilation(),
         "Invalid dilatedsubtensor specification" );
      BLAZE_USER_ASSERT( I + M * RowDilation <= st.rows() * st.rowdilation(),
         "Invalid dilatedsubtensor specification" );
      BLAZE_USER_ASSERT(
         J + N * ColumnDilation <= st.columns() * st.columndilation(),
         "Invalid dilatedsubtensor specification" );
   }

   return dilatedsubtensor( st.operand(), st.page() + K * st.pagedilation() ,st.row() + I * st.rowdilation(),
      st.column() + J * st.columndilation(), O * st.pagedilation(), M * st.rowdilation(),
      N * st.columndilation(), PageDilation * st.pagedilation(), RowDilation * st.rowdilation(),
      ColumnDilation * st.columndilation(), args... );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific DilatedSubtensor of another constant DilatedSubtensor.
// \ingroup DilatedSubtensor
//
// \param sm The constant DilatedSubtensor
// \param args The optional DilatedSubtensor arguments.
// \return View on the specified DilatedSubtensor of the other DilatedSubtensor.
// \exception std::invalid_argument Invalid DilatedSubtensor specification.
//
// This function returns an expression representing the specified DilatedSubtensor of the given constant
// DilatedSubtensor.
*/
template< size_t K               // Index of the first page
        , size_t I               // Index of the first row
        , size_t J               // Index of the first column
        , size_t O               // Number of pages
        , size_t M               // Number of rows
        , size_t N               // Number of columns
        , size_t PageDilation    // The page step-size of the dilatedsubtensor
        , size_t RowDilation     // The row step-size of the dilatedsubtensor
        , size_t ColumnDilation  // The column step-size of the dilatedsubtensor
        , typename TT            // Type of the sparse DilatedSubtensor
        , bool DF                // Density flag
        , typename... RSAs >     // Optional DilatedSubtensor arguments
inline decltype(auto) dilatedsubtensor( const DilatedSubtensor<TT,DF>& st, RSAs... args )
{
   BLAZE_FUNCTION_TRACE;

   constexpr bool isChecked( !Contains_v< TypeList<RSAs...>, Unchecked > );

   if( isChecked ) {
      if(( K + O * PageDilation > st.pages( ) * st.pagedilation( ) ) ||
         ( I + M * RowDilation > st.rows( ) * st.rowdilation( ) ) ||
         ( J + N * ColumnDilation > st.columns( ) * st.columndilation( ) ) ) {
         BLAZE_THROW_INVALID_ARGUMENT( "Invalid dilatedsubtensor specification" );
      }
   }
   else {
      BLAZE_USER_ASSERT( K + O * PageDilation <= st.pages() * st.pagedilation(), "Invalid dilatedsubtensor specification" );
      BLAZE_USER_ASSERT( I + M * RowDilation <= st.rows( ) * st.rowdilation( ) , "Invalid dilatedsubtensor specification" );
      BLAZE_USER_ASSERT( J + N * ColumnDilation <= st.columns( ) * st.columndilation( ) , "Invalid dilatedsubtensor specification" );
   }

   return dilatedsubtensor( st.operand(), st.page() + K * st.pagedilation() ,st.row() + I * st.rowdilation(),
      st.column() + J * st.columndilation(), O * st.pagedilation(), M * st.rowdilation(),
      N * st.columndilation(), PageDilation * st.pagedilation(), RowDilation * st.rowdilation(),
      ColumnDilation * st.columndilation(), args... );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific DilatedSubtensor of another temporary DilatedSubtensor.
// \ingroup DilatedSubtensor
//
// \param sm The temporary DilatedSubtensor
// \param args The optional DilatedSubtensor arguments.
// \return View on the specified DilatedSubtensor of the other DilatedSubtensor.
// \exception std::invalid_argument Invalid DilatedSubtensor specification.
//
// This function returns an expression representing the specified DilatedSubtensor of the given temporary
// DilatedSubtensor.
*/
template< size_t K               // Index of the first page
        , size_t I               // Index of the first row
        , size_t J               // Index of the first column
        , size_t O               // Number of pages
        , size_t M               // Number of rows
        , size_t N               // Number of columns
        , size_t PageDilation    // The page step-size of the dilatedsubtensor
        , size_t RowDilation     // The row step-size of the dilatedsubtensor
        , size_t ColumnDilation  // The column step-size of the dilatedsubtensor
        , typename TT            // Type of the sparse DilatedSubtensor
        , bool DF                // Density flag
        , typename... RSAs >     // Optional DilatedSubtensor arguments
inline decltype(auto) dilatedsubtensor( DilatedSubtensor<TT,DF>&& st, RSAs... args )
{
   BLAZE_FUNCTION_TRACE;

   constexpr bool isChecked( !Contains_v< TypeList<RSAs...>, Unchecked > );

   if( isChecked ) {
      if(( K + O * PageDilation > st.pages( ) * st.pagedilation( ) ) ||
         ( I + M * RowDilation > st.rows( ) * st.rowdilation( ) ) ||
         ( J + N * ColumnDilation > st.columns( ) * st.columndilation( ) ) ) {
         BLAZE_THROW_INVALID_ARGUMENT( "Invalid dilatedsubtensor specification" );
      }
   }
   else {
      BLAZE_USER_ASSERT( K + O * PageDilation <= st.pages() * st.pagedilation(), "Invalid dilatedsubtensor specification" );
      BLAZE_USER_ASSERT( I + M * RowDilation <= st.rows( ) * st.rowdilation( ) , "Invalid dilatedsubtensor specification" );
      BLAZE_USER_ASSERT( J + N * ColumnDilation <= st.columns( ) * st.columndilation( ) , "Invalid dilatedsubtensor specification" );
   }

   return dilatedsubtensor( st.operand(), st.page() + K * st.pagedilation() ,st.row() + I * st.rowdilation(),
      st.column() + J * st.columndilation(), O * st.pagedilation(), M * st.rowdilation(),
      N * st.columndilation(), PageDilation * st.pagedilation(), RowDilation * st.rowdilation(),
      ColumnDilation * st.columndilation(), args... );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific DilatedSubtensor of another DilatedSubtensor.
// \ingroup DilatedSubtensor
//
// \param sm The given DilatedSubtensor
// \param row The index of the first row of the DilatedSubtensor.
// \param column The index of the first column of the DilatedSubtensor.
// \param m The number of rows of the DilatedSubtensor.
// \param n The number of columns of the DilatedSubtensor.
// \param args The optional DilatedSubtensor arguments.
// \return View on the specified DilatedSubtensor of the other DilatedSubtensor.
// \exception std::invalid_argument Invalid DilatedSubtensor specification.
//
// This function returns an expression representing the specified DilatedSubtensor of the given DilatedSubtensor.
*/
template< typename TT         // Type of the sparse DilatedSubtensor
        , bool DF             // Density flag
        , size_t... CSAs      // Compile time DilatedSubtensor arguments
        , typename... RSAs >  // Optional DilatedSubtensor arguments
inline decltype(auto)
   dilatedsubtensor( DilatedSubtensor<TT,DF,CSAs...>& st, size_t page, size_t row, size_t column,
              size_t o, size_t m, size_t n, size_t pagedilation, size_t rowdilation, size_t columndilation, RSAs... args )
{
   BLAZE_FUNCTION_TRACE;

   constexpr bool isChecked( !Contains_v< TypeList<RSAs...>, Unchecked > );

   if( isChecked ) {
      if(( page + o * pagedilation     > st.pages() * st.pagedilation() ) ||
         ( row + m * rowdilation       > st.rows() * st.rowdilation() ) ||
         ( column + n * columndilation > st.columns() * st.columndilation() ) )
      {
         BLAZE_THROW_INVALID_ARGUMENT( "Invalid dilatedsubtensor specification" );
      }
   }
   else {
      BLAZE_USER_ASSERT( page   + o * pagedilation <= st.pages() * st.pagedilation()      , "Invalid dilatedsubtensor specification" );
      BLAZE_USER_ASSERT( row    + m * rowdilation <= st.rows() * st.rowdilation()         , "Invalid dilatedsubtensor specification" );
      BLAZE_USER_ASSERT( column + n * columndilation <= st.columns() * st.columndilation(), "Invalid dilatedsubtensor specification" );
   }

   return dilatedsubtensor( st.operand(), st.page() + page * st.pagedilation(), st.row() + row * st.rowdilation(),
      st.column() + column * st.columndilation(), o, m, n, pagedilation * st.pagedilation(),
      rowdilation * st.rowdilation(), columndilation * st.columndilation(), args... );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific DilatedSubtensor of another constant DilatedSubtensor.
// \ingroup DilatedSubtensor
//
// \param sm The given constant DilatedSubtensor
// \param row The index of the first row of the DilatedSubtensor.
// \param column The index of the first column of the DilatedSubtensor.
// \param m The number of rows of the DilatedSubtensor.
// \param n The number of columns of the DilatedSubtensor.
// \param args The optional DilatedSubtensor arguments.
// \return View on the specified DilatedSubtensor of the other DilatedSubtensor.
// \exception std::invalid_argument Invalid DilatedSubtensor specification.
//
// This function returns an expression representing the specified DilatedSubtensor of the given constant
// DilatedSubtensor.
*/
template< typename TT         // Type of the sparse DilatedSubtensor
        , bool DF             // Density flag
        , size_t... CSAs      // Compile time DilatedSubtensor arguments
        , typename... RSAs >  // Optional DilatedSubtensor arguments
inline decltype(auto)
   dilatedsubtensor( const DilatedSubtensor<TT,DF,CSAs...>& st, size_t page, size_t row, size_t column,
              size_t o, size_t m, size_t n, size_t pagedilation, size_t rowdilation, size_t columndilation, RSAs... args )
{
   BLAZE_FUNCTION_TRACE;

   constexpr bool isChecked( !Contains_v< TypeList<RSAs...>, Unchecked > );

   if( isChecked ) {
      if(( page + o * pagedilation     > st.pages() * st.pagedilation() ) ||
         ( row + m * rowdilation       > st.rows() * st.rowdilation() ) ||
         ( column + n * columndilation > st.columns() * st.columndilation() ) )
      {
         BLAZE_THROW_INVALID_ARGUMENT( "Invalid dilatedsubtensor specification" );
      }
   }
   else {
      BLAZE_USER_ASSERT( page   + o * pagedilation <= st.pages() * st.pagedilation()      , "Invalid dilatedsubtensor specification" );
      BLAZE_USER_ASSERT( row    + m * rowdilation <= st.rows() * st.rowdilation()         , "Invalid dilatedsubtensor specification" );
      BLAZE_USER_ASSERT( column + n * columndilation <= st.columns() * st.columndilation(), "Invalid dilatedsubtensor specification" );
   }

   return dilatedsubtensor( st.operand(), st.page() + page * st.pagedilation(), st.row() + row * st.rowdilation(),
      st.column() + column * st.columndilation(), o, m, n, pagedilation * st.pagedilation(),
      rowdilation * st.rowdilation(), columndilation * st.columndilation(), args... );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific DilatedSubtensor of another temporary DilatedSubtensor.
// \ingroup DilatedSubtensor
//
// \param sm The given temporary DilatedSubtensor
// \param row The index of the first row of the DilatedSubtensor.
// \param column The index of the first column of the DilatedSubtensor.
// \param m The number of rows of the DilatedSubtensor.
// \param n The number of columns of the DilatedSubtensor.
// \param args The optional DilatedSubtensor arguments.
// \return View on the specified DilatedSubtensor of the other DilatedSubtensor.
// \exception std::invalid_argument Invalid DilatedSubtensor specification.
//
// This function returns an expression representing the specified DilatedSubtensor of the given temporary
// DilatedSubtensor.
*/
template< typename TT         // Type of the sparse DilatedSubtensor
        , bool DF             // Density flag
        , size_t... CSAs      // Compile time DilatedSubtensor arguments
        , typename... RSAs >  // Optional DilatedSubtensor arguments
inline decltype(auto)
   dilatedsubtensor( DilatedSubtensor<TT,DF,CSAs...>&& st, size_t page, size_t row, size_t column,
              size_t o, size_t m, size_t n, size_t pagedilation, size_t rowdilation, size_t columndilation, RSAs... args )
{
   BLAZE_FUNCTION_TRACE;

   constexpr bool isChecked( !Contains_v< TypeList<RSAs...>, Unchecked > );

   if( isChecked ) {
      if(( page + o * pagedilation     > st.pages() * st.pagedilation() ) ||
         ( row + m * rowdilation       > st.rows() * st.rowdilation() ) ||
         ( column + n * columndilation > st.columns() * st.columndilation() ) )
      {
         BLAZE_THROW_INVALID_ARGUMENT( "Invalid dilatedsubtensor specification" );
      }
   }
   else {
      BLAZE_USER_ASSERT( page   + o * pagedilation <= st.pages() * st.pagedilation()      , "Invalid dilatedsubtensor specification" );
      BLAZE_USER_ASSERT( row    + m * rowdilation <= st.rows() * st.rowdilation()         , "Invalid dilatedsubtensor specification" );
      BLAZE_USER_ASSERT( column + n * columndilation <= st.columns() * st.columndilation(), "Invalid dilatedsubtensor specification" );
   }

   return dilatedsubtensor( st.operand(), st.page() + page * st.pagedilation(), st.row() + row * st.rowdilation(),
      st.column() + column * st.columndilation(), o, m, n, pagedilation * st.pagedilation(),
      rowdilation * st.rowdilation(), columndilation * st.columndilation(), args... );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific DilatedSubtensor of another DilatedSubtensor.
// \ingroup DilatedSubtensor
//
// \param sm The given DilatedSubtensor
// \param row The index of the first row of the DilatedSubtensor.
// \param column The index of the first column of the DilatedSubtensor.
// \param m The number of rows of the DilatedSubtensor.
// \param n The number of columns of the DilatedSubtensor.
// \param args The optional DilatedSubtensor arguments.
// \return View on the specified DilatedSubtensor of the other DilatedSubtensor.
// \exception std::invalid_argument Invalid DilatedSubtensor specification.
//
// This function returns an expression representing the specified DilatedSubtensor of the given DilatedSubtensor.
*/
template< typename TT         // Type of the sparse DilatedSubtensor
        , AlignmentFlag AF    // Alignment Flag
        , bool DF             // Density flag
        , size_t... CSAs      // Compile time DilatedSubtensor arguments
        , typename... RSAs >  // Optional DilatedSubtensor arguments
inline decltype(auto)
   dilatedsubtensor( Subtensor<TT,AF,DF,CSAs...>& st, size_t page, size_t row, size_t column,
              size_t o, size_t m, size_t n, size_t pagedilation, size_t rowdilation, size_t columndilation, RSAs... args )
{
   BLAZE_FUNCTION_TRACE;

   constexpr bool isChecked( !Contains_v< TypeList<RSAs...>, Unchecked > );

   if( isChecked ) {
      if(( page + ( o - 1 ) * pagedilation + 1     > st.pages() ) ||
         ( row + ( m - 1 ) * rowdilation + 1       > st.rows() ) ||
         ( column + ( n - 1 ) * columndilation + 1 > st.columns() ) )
      {
         BLAZE_THROW_INVALID_ARGUMENT( "Invalid dilatedsubtensor specification" );
      }
   }
   else {
      BLAZE_USER_ASSERT( page   + ( o - 1 ) * pagedilation + 1 <= st.pages(),     "Invalid dilatedsubtensor specification" );
      BLAZE_USER_ASSERT( row    + ( m - 1 ) * rowdilation + 1 <= st.rows(),       "Invalid dilatedsubtensor specification" );
      BLAZE_USER_ASSERT( column + ( n - 1 ) * columndilation + 1 <= st.columns(), "Invalid dilatedsubtensor specification" );
   }

   return dilatedsubtensor( st.operand( ), st.page( ) + page, st.row( ) + row, st.column( ) + column, o, m, n,
      pagedilation, rowdilation, columndilation, args... );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific DilatedSubtensor of another constant DilatedSubtensor.
// \ingroup DilatedSubtensor
//
// \param sm The given constant DilatedSubtensor
// \param row The index of the first row of the DilatedSubtensor.
// \param column The index of the first column of the DilatedSubtensor.
// \param m The number of rows of the DilatedSubtensor.
// \param n The number of columns of the DilatedSubtensor.
// \param args The optional DilatedSubtensor arguments.
// \return View on the specified DilatedSubtensor of the other DilatedSubtensor.
// \exception std::invalid_argument Invalid DilatedSubtensor specification.
//
// This function returns an expression representing the specified DilatedSubtensor of the given constant
// DilatedSubtensor.
*/
template< typename TT         // Type of the sparse DilatedSubtensor
        , AlignmentFlag AF    // Alignment Flag
        , bool DF             // Density flag
        , size_t... CSAs      // Compile time DilatedSubtensor arguments
        , typename... RSAs >  // Optional DilatedSubtensor arguments
inline decltype(auto)
   dilatedsubtensor( const Subtensor<TT,AF,DF,CSAs...>& st, size_t page, size_t row, size_t column,
              size_t o, size_t m, size_t n, size_t pagedilation, size_t rowdilation, size_t columndilation, RSAs... args )
{
   BLAZE_FUNCTION_TRACE;

   constexpr bool isChecked( !Contains_v< TypeList<RSAs...>, Unchecked > );

   if( isChecked ) {
      if(( page + ( o - 1 ) * pagedilation + 1     > st.pages() ) ||
         ( row + ( m - 1 ) * rowdilation + 1       > st.rows() ) ||
         ( column + ( n - 1 ) * columndilation + 1 > st.columns() ) )
      {
         BLAZE_THROW_INVALID_ARGUMENT( "Invalid dilatedsubtensor specification" );
      }
   }
   else {
      BLAZE_USER_ASSERT( page   + ( o - 1 ) * pagedilation + 1 <= st.pages(),     "Invalid dilatedsubtensor specification" );
      BLAZE_USER_ASSERT( row    + ( m - 1 ) * rowdilation + 1 <= st.rows(),       "Invalid dilatedsubtensor specification" );
      BLAZE_USER_ASSERT( column + ( n - 1 ) * columndilation + 1 <= st.columns(), "Invalid dilatedsubtensor specification" );
   }

   return dilatedsubtensor( st.operand( ), st.page( ) + page, st.row( ) + row, st.column( ) + column, o, m, n,
      pagedilation, rowdilation, columndilation, args... );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific DilatedSubtensor of another temporary DilatedSubtensor.
// \ingroup DilatedSubtensor
//
// \param sm The given temporary DilatedSubtensor
// \param row The index of the first row of the DilatedSubtensor.
// \param column The index of the first column of the DilatedSubtensor.
// \param m The number of rows of the DilatedSubtensor.
// \param n The number of columns of the DilatedSubtensor.
// \param args The optional DilatedSubtensor arguments.
// \return View on the specified DilatedSubtensor of the other DilatedSubtensor.
// \exception std::invalid_argument Invalid DilatedSubtensor specification.
//
// This function returns an expression representing the specified DilatedSubtensor of the given temporary
// DilatedSubtensor.
*/
template< typename TT         // Type of the sparse DilatedSubtensor
        , AlignmentFlag AF    // Alignment Flag
        , bool DF             // Density flag
        , size_t... CSAs      // Compile time DilatedSubtensor arguments
        , typename... RSAs >  // Optional DilatedSubtensor arguments
inline decltype(auto)
   dilatedsubtensor( Subtensor<TT,AF,DF,CSAs...>&& st, size_t page, size_t row, size_t column,
              size_t o, size_t m, size_t n, size_t pagedilation, size_t rowdilation, size_t columndilation, RSAs... args )
{
   BLAZE_FUNCTION_TRACE;

   constexpr bool isChecked( !Contains_v< TypeList<RSAs...>, Unchecked > );

   if( isChecked ) {
      if(( page + ( o - 1 ) * pagedilation + 1     > st.pages() ) ||
         ( row + ( m - 1 ) * rowdilation + 1       > st.rows() ) ||
         ( column + ( n - 1 ) * columndilation + 1 > st.columns() ) )
      {
         BLAZE_THROW_INVALID_ARGUMENT( "Invalid dilatedsubtensor specification" );
      }
   }
   else {
      BLAZE_USER_ASSERT( page   + ( o - 1 ) * pagedilation + 1 <= st.pages(),     "Invalid dilatedsubtensor specification" );
      BLAZE_USER_ASSERT( row    + ( m - 1 ) * rowdilation + 1 <= st.rows(),       "Invalid dilatedsubtensor specification" );
      BLAZE_USER_ASSERT( column + ( n - 1 ) * columndilation + 1 <= st.columns(), "Invalid dilatedsubtensor specification" );
   }

   return dilatedsubtensor( st.operand( ), st.page( ) + page, st.row( ) + row, st.column( ) + column, o, m, n,
      pagedilation, rowdilation, columndilation, args... );
}
/*! \endcond */
//*************************************************************************************************



//=================================================================================================
//
//  GLOBAL RESTRUCTURING FUNCTIONS (DILATEDSUBVECTOR)
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific dilatedsubvector of the given tensor/vector multiplication.
// \ingroup DilatedSubtensor
//
// \param vector The constant tensor/vector multiplication.
// \param args The runtime dilatedsubvector arguments.
// \return View on the specified dilatedsubvector of the multiplication.
//
// This function returns an expression representing the specified dilatedsubvector of the given
// tensor/vector multiplication.
*/
//template< size_t... CSAs      // Compile time dilatedsubvector arguments
//        , typename MT         // Matrix base type of the expression
//        , typename... RSAs >  // Runtime dilatedsubvector arguments
//inline decltype(auto) dilatedsubmatrix( const TensVecMultExpr<MT>& matrix, RSAs... args )
//{
//   BLAZE_FUNCTION_TRACE;
//
//   using TT = RemoveReference_t< LeftOperand_t< MatrixType_t<MT> > >;
//
//   const DilatedSubmatrixData<CSAs...> sm( args... );
//
//   BLAZE_DECLTYPE_AUTO( left , (~matrix).leftOperand()  );
//   BLAZE_DECLTYPE_AUTO( right, (~matrix).rightOperand() );
//
//   const size_t column( 0UL );
//   const size_t n( left.columns() );
//
//   return dilatedsubtensor( left, sm.row(), column, sm.size(), n, sm.dilation(), 1UL ) *
//      dilatedsubvector( right, column, n, sm.columndilation() );
//}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific dilatedsubvector of the given vector/tensor multiplication.
// \ingroup DilatedSubtensor
//
// \param vector The constant vector/tensor multiplication.
// \param args The runtime dilatedsubvector arguments.
// \return View on the specified dilatedsubvector of the multiplication.
//
// This function returns an expression representing the specified dilatedsubvector of the given
// vector/tensor multiplication.
*/
//template< size_t... CSAs      // Compile time dilatedsubvector arguments
//        , typename VT         // Vector base type of the expression
//        , typename... RSAs >  // Runtime dilatedsubvector arguments
//inline decltype(auto) dilatedsubvector( const TVecMatMultExpr<VT>& vector, RSAs... args )
//{
//   BLAZE_FUNCTION_TRACE;
//
//   using TT = RemoveReference_t< RightOperand_t< VectorType_t<VT> > >;
//
//   const DilatedSubvectorData<CSAs...> sd( args... );
//
//   BLAZE_DECLTYPE_AUTO( left , (~vector).leftOperand()  );
//   BLAZE_DECLTYPE_AUTO( right, (~vector).rightOperand() );
//
//   const size_t row( ( IsLower_v<TT> )
//                     ?( ( !IsStrictlyLower_v<TT> )?( sd.offset() + 1UL ):( sd.offset() ) )
//                     :( 0UL ) );
//   const size_t m( ( IsUpper_v<TT> )
//                   ?( ( IsLower_v<TT> )?( sd.size() )
//                                       :( ( IsStrictlyUpper_v<TT> && sd.size() > 0UL )
//                                          ?( sd.offset() + sd.size() - 1UL )
//                                          :( sd.offset() + sd.size() ) ) )
//                   :( ( IsLower_v<TT> )?( right.rows() - row )
//                                       :( right.rows() ) ) );
//
//   return dilatedsubvector( left, row, m, sd.dilation() ) * dilatedsubtensor( right, row, sd.offset(), m, sd.size(), 1UL, sd.dilation() );
//}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific submatrix of the given page-wise tensor reduction operation.
// \ingroup subtensor
//
// \param matrix The constant page-wise tensor reduction operation.
// \param args The runtime subvector arguments.
// \return View on the specified submatrix of the reduction operation.
//
// This function returns an expression representing the specified submatrix of the given page-wise
// tensor reduction operation.
*/
template< size_t... CSAs      // Compile time subvector arguments
        , typename MT         // Vector base type of the expression
        , typename... RSAs >  // Runtime submatrix arguments
inline decltype(auto) dilatedsubmatrix( const TensReduceExpr<MT,pagewise>& matrix, RSAs... args )
{
   BLAZE_FUNCTION_TRACE;

   const DilatedSubmatrixData<CSAs...> sm( args... );
   const size_t M( (~matrix).operand().rows() );

   decltype(auto) st( dilatedsubtensor( (~matrix).operand(), sm.row(), 0UL, sm.column(), sm.rows(), M, sm.columns(), sm.rowdilation(), 1UL, sm.columndilation() ) );
   return reduce<pagewise>( st, (~matrix).operation() );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific dilatedsubvector of the given column-wise tensor reduction operation.
// \ingroup DilatedSubtensor
//
// \param vector The constant column-wise tensor reduction operation.
// \param args The runtime dilatedsubvector arguments.
// \return View on the specified dilatedsubvector of the reduction operation.
//
// This function returns an expression representing the specified dilatedsubvector of the given
// column-wise tensor reduction operation.
*/
template< size_t... CSAs      // Compile time dilatedsubvector arguments
        , typename MT         // Matrix base type of the expression
        , typename... RSAs >  // Runtime dilatedsubvector arguments
inline decltype(auto) dilatedsubmatrix( const TensReduceExpr<MT,columnwise>& matrix, RSAs... args )
{
   BLAZE_FUNCTION_TRACE;

   const DilatedSubmatrixData<CSAs...> sm( args... );
   const size_t O( (~matrix).operand().rows() );

   decltype(auto) st( dilatedsubtensor( (~matrix).operand(), 0UL, sm.row(), sm.column(), O, sm.rows(),
      sm.columns(), 1UL, sm.rowdilation(), sm.columndilation() ) );
   return reduce<columnwise>( st, (~matrix).operation() );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific dilatedsubvector of the given row-wise tensor reduction operation.
// \ingroup DilatedSubtensor
//
// \param vector The constant row-wise tensor reduction operation.
// \param args The runtime dilatedsubvector arguments.
// \return View on the specified dilatedsubvector of the reduction operation.
//
// This function returns an expression representing the specified dilatedsubvector of the given row-wise
// tensor reduction operation.
*/
template< size_t... CSAs      // Compile time dilatedsubvector arguments
        , typename MT         // Vector base type of the expression
        , typename... RSAs >  // Runtime dilatedsubvector arguments
inline decltype(auto) dilatedsubmatrix( const TensReduceExpr<MT,rowwise>& matrix, RSAs... args )
{
   BLAZE_FUNCTION_TRACE;

   const DilatedSubmatrixData<CSAs...> sm( args... );
   const size_t N( (~matrix).operand().rows() );

   decltype(auto) st( dilatedsubtensor( (~matrix).operand(), sm.row(), sm.column(), 0UL, sm.rows(), sm.columns(), N,
      sm.rowdilation(), sm.columndilation(), 1UL ) );
   return reduce<rowwise>( st, (~matrix).operation() );
}
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  GLOBAL RESTRUCTURING FUNCTIONS (PAGESLICE)
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific row of the given DilatedSubtensor.
// \ingroup DilatedSubtensor
//
// \param sm The DilatedSubtensor containing the row.
// \param args The optional row arguments.
// \return View on the specified row of the DilatedSubtensor.
//
// This function returns an expression representing the specified row of the given DilatedSubtensor.
*/
template< size_t K1              // Page index
        , typename TT            // Type of the sparse DilatedSubtensor
        , bool DF                // Density flag
        , size_t K2              // Index of the first page
        , size_t I               // Index of the first row
        , size_t J               // Index of the first column
        , size_t O               // Number of pages
        , size_t M               // Number of rows
        , size_t N               // Number of columns
        , size_t PageDilation    // Step between pages of the DilatedSubtensor
        , size_t RowDilation     // Step between rows of the DilatedSubtensor
        , size_t ColumnDilation  // Step between columns of the DilatedSubtensor
        , typename... RRAs >  // Optional row arguments
inline decltype(auto) pageslice( DilatedSubtensor<TT,DF,K2,I,J,O,M,N,PageDilation,RowDilation,ColumnDilation>& st, RRAs... args )
{
   BLAZE_FUNCTION_TRACE;

   BLAZE_STATIC_ASSERT_MSG( K1 < O, "Invalid page access index" );

   return dilatedsubmatrix<I,J,M,N,RowDilation,ColumnDilation>( pageslice<K1*PageDilation+K2>( st.operand(), args... ), unchecked );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific row of the given constant DilatedSubtensor.
// \ingroup DilatedSubtensor
//
// \param sm The constant DilatedSubtensor containing the row.
// \param args The optional row arguments.
// \return View on the specified row of the DilatedSubtensor.
//
// This function returns an expression representing the specified row of the given constant
// DilatedSubtensor.
*/
template< size_t K1              // Page index
        , typename TT            // Type of the sparse DilatedSubtensor
        , bool DF                // Density flag
        , size_t K2              // Index of the first page
        , size_t I               // Index of the first row
        , size_t J               // Index of the first column
        , size_t O               // Number of pages
        , size_t M               // Number of rows
        , size_t N               // Number of columns
        , size_t PageDilation    // Step between pages of the DilatedSubtensor
        , size_t RowDilation     // Step between rows of the DilatedSubtensor
        , size_t ColumnDilation  // Step between columns of the DilatedSubtensor
        , typename... RRAs >  // Optional row arguments
inline decltype(auto) pageslice( const DilatedSubtensor<TT,DF,K2,I,J,O,M,N,PageDilation,RowDilation,ColumnDilation>& st, RRAs... args )
{
   BLAZE_FUNCTION_TRACE;

   BLAZE_STATIC_ASSERT_MSG( K1 < O, "Invalid page access index" );

   return dilatedsubmatrix<I,J,M,N,RowDilation,ColumnDilation>( pageslice<K1*PageDilation+K2>( st.operand(), args... ), unchecked );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific row of the given temporary DilatedSubtensor.
// \ingroup DilatedSubtensor
//
// \param sm The temporary DilatedSubtensor containing the row.
// \param args The optional row arguments.
// \return View on the specified row of the DilatedSubtensor.
//
// This function returns an expression representing the specified row of the given temporary
// DilatedSubtensor.
*/
template< size_t K1              // Page index
        , typename TT            // Type of the sparse DilatedSubtensor
        , bool DF                // Density flag
        , size_t K2              // Index of the first page
        , size_t I               // Index of the first row
        , size_t J               // Index of the first column
        , size_t O               // Number of pages
        , size_t M               // Number of rows
        , size_t N               // Number of columns
        , size_t PageDilation    // Step between pages of the DilatedSubtensor
        , size_t RowDilation     // Step between rows of the DilatedSubtensor
        , size_t ColumnDilation  // Step between columns of the DilatedSubtensor
        , typename... RRAs >  // Optional row arguments
inline decltype(auto) pageslice( DilatedSubtensor<TT,DF,K2,I,J,O,M,N,PageDilation,RowDilation,ColumnDilation>&& st, RRAs... args )
{
   BLAZE_FUNCTION_TRACE;

   BLAZE_STATIC_ASSERT_MSG( K1 < O, "Invalid page access index" );

   return dilatedsubmatrix<I,J,M,N,RowDilation,ColumnDilation>( pageslice<K1*PageDilation+K2>( st.operand(), args... ), unchecked );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific row of the given DilatedSubtensor.
// \ingroup DilatedSubtensor
//
// \param sm The DilatedSubtensor containing the row.
// \param index The index of the row.
// \param args The optional row arguments.
// \return View on the specified row of the DilatedSubtensor.
// \exception std::invalid_argument Invalid row access index.
//
// This function returns an expression representing the specified row of the given DilatedSubtensor.
*/
template< typename TT            // Type of the sparse DilatedSubtensor
        , bool DF                // Density flag
        , size_t K               // Index of the first page
        , size_t I              // Index of the first row
        , size_t J               // Index of the first column
        , size_t O               // Number of pages
        , size_t M               // Number of rows
        , size_t N               // Number of columns
        , size_t PageDilation    // Step between pages of the DilatedSubtensor
        , size_t RowDilation     // Step between rows of the DilatedSubtensor
        , size_t ColumnDilation  // Step between columns of the DilatedSubtensor
        , typename... RRAs >  // Optional row arguments
inline decltype(auto) pageslice( DilatedSubtensor<TT,DF,K,I,J,O,M,N,PageDilation,RowDilation,ColumnDilation>& st, size_t index, RRAs... args )
{
   BLAZE_FUNCTION_TRACE;

   constexpr bool isChecked( !Contains_v< TypeList<RRAs...>, Unchecked > );

   if( isChecked ) {
      if( ( index >= O ) ) {
         BLAZE_THROW_INVALID_ARGUMENT( "Invalid row access index" );
      }
   }
   else {
      BLAZE_USER_ASSERT( index < O, "Invalid row access index" );
   }

   return dilatedsubmatrix<I,J,M,N,RowDilation,ColumnDilation>( pageslice( st.operand(), I+index*PageDilation, args... ), unchecked );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific row of the given constant DilatedSubtensor.
// \ingroup DilatedSubtensor
//
// \param sm The constant DilatedSubtensor containing the row.
// \param index The index of the row.
// \param args The optional row arguments.
// \return View on the specified row of the DilatedSubtensor.
// \exception std::invalid_argument Invalid row access index.
//
// This function returns an expression representing the specified row of the given constant
// DilatedSubtensor.
*/
template< typename TT            // Type of the sparse DilatedSubtensor
        , bool DF                // Density flag
        , size_t K               // Index of the first page
        , size_t I              // Index of the first row
        , size_t J               // Index of the first column
        , size_t O               // Number of pages
        , size_t M               // Number of rows
        , size_t N               // Number of columns
        , size_t PageDilation    // Step between pages of the DilatedSubtensor
        , size_t RowDilation     // Step between rows of the DilatedSubtensor
        , size_t ColumnDilation  // Step between columns of the DilatedSubtensor
        , typename... RRAs >  // Optional row arguments
inline decltype(auto) pageslice( const DilatedSubtensor<TT,DF,K,I,J,O,M,N,PageDilation,RowDilation,ColumnDilation>& st, size_t index, RRAs... args )
{
   BLAZE_FUNCTION_TRACE;

   constexpr bool isChecked( !Contains_v< TypeList<RRAs...>, Unchecked > );

   if( isChecked ) {
      if( ( index >= O ) ) {
         BLAZE_THROW_INVALID_ARGUMENT( "Invalid row access index" );
      }
   }
   else {
      BLAZE_USER_ASSERT( index < O, "Invalid row access index" );
   }

   return dilatedsubmatrix<I,J,M,N,RowDilation,ColumnDilation>( pageslice( st.operand(), K+index*PageDilation, args... ), unchecked );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific row of the given temporary DilatedSubtensor.
// \ingroup DilatedSubtensor
//
// \param sm The temporary DilatedSubtensor containing the row.
// \param index The index of the row.
// \param args The optional row arguments.
// \return View on the specified row of the DilatedSubtensor.
// \exception std::invalid_argument Invalid row access index.
//
// This function returns an expression representing the specified row of the given temporary
// DilatedSubtensor.
*/
template< typename TT            // Type of the sparse DilatedSubtensor
        , bool DF                // Density flag
        , size_t K               // Index of the first page
        , size_t I              // Index of the first row
        , size_t J               // Index of the first column
        , size_t O               // Number of pages
        , size_t M               // Number of rows
        , size_t N               // Number of columns
        , size_t PageDilation    // Step between pages of the DilatedSubtensor
        , size_t RowDilation     // Step between rows of the DilatedSubtensor
        , size_t ColumnDilation  // Step between columns of the DilatedSubtensor
        , typename... RRAs >  // Optional row arguments
inline decltype(auto) pageslice( DilatedSubtensor<TT,DF,K,I,J,O,M,N,PageDilation,RowDilation,ColumnDilation>&& st, size_t index, RRAs... args )
{
   BLAZE_FUNCTION_TRACE;

   constexpr bool isChecked( !Contains_v< TypeList<RRAs...>, Unchecked > );

   if( isChecked ) {
      if( ( index >= O ) ) {
         BLAZE_THROW_INVALID_ARGUMENT( "Invalid row access index" );
      }
   }
   else {
      BLAZE_USER_ASSERT( index < O , "Invalid row access index" );
   }

   return dilatedsubmatrix<I,J,M,N,RowDilation,ColumnDilation>( pageslice( st.operand(), K+index*PageDilation, args... ), unchecked );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific row of the given DilatedSubtensor.
// \ingroup DilatedSubtensor
//
// \param sm The DilatedSubtensor containing the row.
// \param args The optional row arguments.
// \return View on the specified row of the DilatedSubtensor.
// \exception std::invalid_argument Invalid row access index.
//
// This function returns an expression representing the specified row of the given DilatedSubtensor.
*/
template< size_t... CRAs      // Compile time row arguments
        , typename TT         // Type of the sparse DilatedSubtensor
        , bool DF             // Density flag
        , typename... RRAs >  // Runtime row arguments
inline decltype(auto) pageslice( DilatedSubtensor<TT,DF>& st, RRAs... args )
{
   BLAZE_FUNCTION_TRACE;

   const PageSliceData<CRAs...> pd( args... );

   constexpr bool isChecked( !Contains_v< TypeList<RRAs...>, Unchecked > );

   if( isChecked ) {
      if( ( pd.page() >= st.pages() ) ) {
         BLAZE_THROW_INVALID_ARGUMENT( "Invalid page access index" );
      }
   }
   else {
      BLAZE_USER_ASSERT( pd.page() < st.pages() , "Invalid page access index" );
   }

   const size_t index( pd.page() * st.pagedilation() + st.page() );

  return dilatedsubmatrix( pageslice(st.operand(), index, args...), st.row(), st.column(), st.rows(),
      st.columns(),st.rowdilation(), st.columndilation(), unchecked );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific row of the given constant DilatedSubtensor.
// \ingroup DilatedSubtensor
//
// \param sm The constant DilatedSubtensor containing the row.
// \param args The optional row arguments.
// \return View on the specified row of the DilatedSubtensor.
// \exception std::invalid_argument Invalid row access index.
//
// This function returns an expression representing the specified row of the given constant
// DilatedSubtensor.
*/
template< size_t... CRAs      // Compile time row arguments
        , typename TT         // Type of the sparse DilatedSubtensor
        , bool DF             // Density flag
        , typename... RRAs >  // Runtime row arguments
inline decltype(auto) pageslice( const DilatedSubtensor<TT,DF>& st, RRAs... args )
{
   BLAZE_FUNCTION_TRACE;

   const PageSliceData<CRAs...> pd( args... );

   constexpr bool isChecked( !Contains_v< TypeList<RRAs...>, Unchecked > );

   if( isChecked ) {
      if( ( pd.page() >= st.pages() ) ) {
         BLAZE_THROW_INVALID_ARGUMENT( "Invalid page access index" );
      }
   }
   else {
      BLAZE_USER_ASSERT( pd.page() < st.pages() , "Invalid page access index" );
   }

   const size_t index( pd.page() * st.pagedilation() + st.page() );

   return dilatedsubmatrix( pageslice( st.operand(), index, args... ), st.row(), st.column(), st.rows(),
      st.columns(), st.rowdilation(), st.columndilation(), unchecked );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific row of the given temporary DilatedSubtensor.
// \ingroup DilatedSubtensor
//
// \param sm The temporary DilatedSubtensor containing the row.
// \param args The optional row arguments.
// \return View on the specified row of the DilatedSubtensor.
// \exception std::invalid_argument Invalid row access index.
//
// This function returns an expression representing the specified row of the given temporary
// DilatedSubtensor.
*/
template< size_t... CRAs      // Compile time row arguments
        , typename TT         // Type of the sparse DilatedSubtensor
        , bool DF             // Density flag
        , typename... RRAs >  // Runtime row arguments
inline decltype(auto) pageslice( DilatedSubtensor<TT,DF>&& st, RRAs... args )
{
   BLAZE_FUNCTION_TRACE;

   const PageSliceData<CRAs...> pd( args... );

   constexpr bool isChecked( !Contains_v< TypeList<RRAs...>, Unchecked > );

   if( isChecked ) {
      if( ( pd.page() >= st.pages() ) ) {
         BLAZE_THROW_INVALID_ARGUMENT( "Invalid page access index" );
      }
   }
   else {
      BLAZE_USER_ASSERT( pd.page() < st.pages() , "Invalid page access index" );
   }

   const size_t index( pd.page() * st.pagedilation() + st.page() );

   return dilatedsubmatrix( pageslice( st.operand(), index, args... ), st.row(), st.column(), st.rows(),
      st.columns(),st.rowdilation(), st.columndilation(), unchecked );
}
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  GLOBAL RESTRUCTURING FUNCTIONS (ROWSLICE)
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific row of the given DilatedSubtensor.
// \ingroup DilatedSubtensor
//
// \param sm The DilatedSubtensor containing the row.
// \param args The optional row arguments.
// \return View on the specified row of the DilatedSubtensor.
//
// This function returns an expression representing the specified row of the given DilatedSubtensor.
*/
template< size_t I1              // Row index
        , typename TT            // Type of the sparse DilatedSubtensor
        , bool DF                // Density flag
        , size_t K               // Index of the first page
        , size_t I2              // Index of the first row
        , size_t J               // Index of the first column
        , size_t O               // Number of pages
        , size_t M               // Number of rows
        , size_t N               // Number of columns
        , size_t PageDilation    // Step between pages of the DilatedSubtensor
        , size_t RowDilation     // Step between rows of the DilatedSubtensor
        , size_t ColumnDilation  // Step between columns of the DilatedSubtensor
        , typename... RRAs >  // Optional row arguments
inline decltype(auto) rowslice( DilatedSubtensor<TT,DF,K,I2,J,O,M,N,PageDilation,RowDilation,ColumnDilation>& st, RRAs... args )
{
   BLAZE_FUNCTION_TRACE;

   BLAZE_STATIC_ASSERT_MSG( I1 < M, "Invalid row access index" );

   return dilatedsubmatrix<J,K,N,O,ColumnDilation,PageDilation>( rowslice<I1*RowDilation+I2>( st.operand(), args... ), unchecked );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific row of the given constant DilatedSubtensor.
// \ingroup DilatedSubtensor
//
// \param sm The constant DilatedSubtensor containing the row.
// \param args The optional row arguments.
// \return View on the specified row of the DilatedSubtensor.
//
// This function returns an expression representing the specified row of the given constant
// DilatedSubtensor.
*/
template< size_t I1              // Row index
        , typename TT            // Type of the sparse DilatedSubtensor
        , bool DF                // Density flag
        , size_t K               // Index of the first page
        , size_t I2              // Index of the first row
        , size_t J               // Index of the first column
        , size_t O               // Number of pages
        , size_t M               // Number of rows
        , size_t N               // Number of columns
        , size_t PageDilation    // Step between pages of the DilatedSubtensor
        , size_t RowDilation     // Step between rows of the DilatedSubtensor
        , size_t ColumnDilation  // Step between columns of the DilatedSubtensor
        , typename... RRAs >  // Optional row arguments
inline decltype(auto) rowslice( const DilatedSubtensor<TT,DF,K,I2,J,O,M,N,PageDilation,RowDilation,ColumnDilation>& st, RRAs... args )
{
   BLAZE_FUNCTION_TRACE;

   BLAZE_STATIC_ASSERT_MSG( I1 < M, "Invalid row access index" );

   return dilatedsubmatrix<J,K,N,O,ColumnDilation,PageDilation>( rowslice<I1*RowDilation+I2>( st.operand(), args... ), unchecked );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific row of the given temporary DilatedSubtensor.
// \ingroup DilatedSubtensor
//
// \param sm The temporary DilatedSubtensor containing the row.
// \param args The optional row arguments.
// \return View on the specified row of the DilatedSubtensor.
//
// This function returns an expression representing the specified row of the given temporary
// DilatedSubtensor.
*/
template< size_t I1              // Row index
        , typename TT            // Type of the sparse DilatedSubtensor
        , bool DF                // Density flag
        , size_t K               // Index of the first page
        , size_t I2              // Index of the first row
        , size_t J               // Index of the first column
        , size_t O               // Number of pages
        , size_t M               // Number of rows
        , size_t N               // Number of columns
        , size_t PageDilation    // Step between pages of the DilatedSubtensor
        , size_t RowDilation     // Step between rows of the DilatedSubtensor
        , size_t ColumnDilation  // Step between columns of the DilatedSubtensor
        , typename... RRAs >  // Optional row arguments
inline decltype(auto) rowslice( DilatedSubtensor<TT,DF,K,I2,J,O,M,N,PageDilation,RowDilation,ColumnDilation>&& st, RRAs... args )
{
   BLAZE_FUNCTION_TRACE;

   BLAZE_STATIC_ASSERT_MSG( I1 < M , "Invalid row access index" );

   return dilatedsubvector<J,K,N,O,ColumnDilation,PageDilation>( rowslice<I1*RowDilation+I2>( st.operand(), args... ), unchecked );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific row of the given DilatedSubtensor.
// \ingroup DilatedSubtensor
//
// \param sm The DilatedSubtensor containing the row.
// \param index The index of the row.
// \param args The optional row arguments.
// \return View on the specified row of the DilatedSubtensor.
// \exception std::invalid_argument Invalid row access index.
//
// This function returns an expression representing the specified row of the given DilatedSubtensor.
*/
template< typename TT            // Type of the sparse DilatedSubtensor
        , bool DF                // Density flag
        , size_t K               // Index of the first page
        , size_t I              // Index of the first row
        , size_t J               // Index of the first column
        , size_t O               // Number of pages
        , size_t M               // Number of rows
        , size_t N               // Number of columns
        , size_t PageDilation    // Step between pages of the DilatedSubtensor
        , size_t RowDilation     // Step between rows of the DilatedSubtensor
        , size_t ColumnDilation  // Step between columns of the DilatedSubtensor
        , typename... RRAs >  // Optional row arguments
inline decltype(auto) rowslice( DilatedSubtensor<TT,DF,K,I,J,O,M,N,PageDilation,RowDilation,ColumnDilation>& st, size_t index, RRAs... args )
{
   BLAZE_FUNCTION_TRACE;

   constexpr bool isChecked( !Contains_v< TypeList<RRAs...>, Unchecked > );

   if( isChecked ) {
      if( ( index >= M ) ) {
         BLAZE_THROW_INVALID_ARGUMENT( "Invalid row access index" );
      }
   }
   else {
      BLAZE_USER_ASSERT( index < M, "Invalid row access index" );
   }

   return dilatedsubmatrix<J,K,N,O,ColumnDilation,PageDilation>( rowslice( st.operand(), I+index*RowDilation, args... ), unchecked );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific row of the given constant DilatedSubtensor.
// \ingroup DilatedSubtensor
//
// \param sm The constant DilatedSubtensor containing the row.
// \param index The index of the row.
// \param args The optional row arguments.
// \return View on the specified row of the DilatedSubtensor.
// \exception std::invalid_argument Invalid row access index.
//
// This function returns an expression representing the specified row of the given constant
// DilatedSubtensor.
*/
template< typename TT            // Type of the sparse DilatedSubtensor
        , bool DF                // Density flag
        , size_t K               // Index of the first page
        , size_t I              // Index of the first row
        , size_t J               // Index of the first column
        , size_t O               // Number of pages
        , size_t M               // Number of rows
        , size_t N               // Number of columns
        , size_t PageDilation    // Step between pages of the DilatedSubtensor
        , size_t RowDilation     // Step between rows of the DilatedSubtensor
        , size_t ColumnDilation  // Step between columns of the DilatedSubtensor
        , typename... RRAs >  // Optional row arguments
inline decltype(auto) rowslice( const DilatedSubtensor<TT,DF,K,I,J,O,M,N,PageDilation,RowDilation,ColumnDilation>& st, size_t index, RRAs... args )
{
   BLAZE_FUNCTION_TRACE;

   constexpr bool isChecked( !Contains_v< TypeList<RRAs...>, Unchecked > );

   if( isChecked ) {
      if( ( index >= M ) ) {
         BLAZE_THROW_INVALID_ARGUMENT( "Invalid row access index" );
      }
   }
   else {
      BLAZE_USER_ASSERT( index < M, "Invalid row access index" );
   }

   return dilatedsubmatrix<J,K,N,O,ColumnDilation,PageDilation>( rowslice( st.operand(), I+index*RowDilation, args... ), unchecked );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific row of the given temporary DilatedSubtensor.
// \ingroup DilatedSubtensor
//
// \param sm The temporary DilatedSubtensor containing the row.
// \param index The index of the row.
// \param args The optional row arguments.
// \return View on the specified row of the DilatedSubtensor.
// \exception std::invalid_argument Invalid row access index.
//
// This function returns an expression representing the specified row of the given temporary
// DilatedSubtensor.
*/
template< typename TT            // Type of the sparse DilatedSubtensor
        , bool DF                // Density flag
        , size_t K               // Index of the first page
        , size_t I              // Index of the first row
        , size_t J               // Index of the first column
        , size_t O               // Number of pages
        , size_t M               // Number of rows
        , size_t N               // Number of columns
        , size_t PageDilation    // Step between pages of the DilatedSubtensor
        , size_t RowDilation     // Step between rows of the DilatedSubtensor
        , size_t ColumnDilation  // Step between columns of the DilatedSubtensor
        , typename... RRAs >  // Optional row arguments
inline decltype(auto) rowslice( DilatedSubtensor<TT,DF,K,I,J,O,M,N,PageDilation,RowDilation,ColumnDilation>&& st, size_t index, RRAs... args )
{
   BLAZE_FUNCTION_TRACE;

   constexpr bool isChecked( !Contains_v< TypeList<RRAs...>, Unchecked > );

   if( isChecked ) {
      if( ( index >= M ) ) {
         BLAZE_THROW_INVALID_ARGUMENT( "Invalid row access index" );
      }
   }
   else {
      BLAZE_USER_ASSERT( index < M , "Invalid row access index" );
   }

   return dilatedsubvector<J,K,N,O,ColumnDilation,PageDilation>( rowslice( st.operand(), I+index*RowDilation, args... ), unchecked );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific row of the given DilatedSubtensor.
// \ingroup DilatedSubtensor
//
// \param sm The DilatedSubtensor containing the row.
// \param args The optional row arguments.
// \return View on the specified row of the DilatedSubtensor.
// \exception std::invalid_argument Invalid row access index.
//
// This function returns an expression representing the specified row of the given DilatedSubtensor.
*/
template< size_t... CRAs      // Compile time row arguments
        , typename TT         // Type of the sparse DilatedSubtensor
        , bool DF             // Density flag
        , typename... RRAs >  // Runtime row arguments
inline decltype(auto) rowslice( DilatedSubtensor<TT,DF>& st, RRAs... args )
{
   BLAZE_FUNCTION_TRACE;

   const RowSliceData<CRAs...> rd( args... );

   constexpr bool isChecked( !Contains_v< TypeList<RRAs...>, Unchecked > );

   if( isChecked ) {
      if( ( rd.row() >= st.rows() ) ) {
         BLAZE_THROW_INVALID_ARGUMENT( "Invalid row access index" );
      }
   }
   else {
      BLAZE_USER_ASSERT( rd.row() < st.rows() , "Invalid row access index" );
   }

   const size_t index( rd.row() * st.rowdilation() + st.row() );

   return dilatedsubmatrix( rowslice( st.operand(), index, args... ), st.column(), st.page(), st.columns(),
      st.pages(), st.columndilation(), st.pagedilation(), unchecked );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific row of the given constant DilatedSubtensor.
// \ingroup DilatedSubtensor
//
// \param sm The constant DilatedSubtensor containing the row.
// \param args The optional row arguments.
// \return View on the specified row of the DilatedSubtensor.
// \exception std::invalid_argument Invalid row access index.
//
// This function returns an expression representing the specified row of the given constant
// DilatedSubtensor.
*/
template< size_t... CRAs      // Compile time row arguments
        , typename TT         // Type of the sparse DilatedSubtensor
        , bool DF             // Density flag
        , typename... RRAs >  // Runtime row arguments
inline decltype(auto) rowslice( const DilatedSubtensor<TT,DF>& st, RRAs... args )
{
   BLAZE_FUNCTION_TRACE;

   const RowSliceData<CRAs...> rd( args... );

   constexpr bool isChecked( !Contains_v< TypeList<RRAs...>, Unchecked > );

   if( isChecked ) {
      if( ( rd.row() >= st.rows() ) ) {
         BLAZE_THROW_INVALID_ARGUMENT( "Invalid row access index" );
      }
   }
   else {
      BLAZE_USER_ASSERT( rd.row() < st.rows() , "Invalid row access index" );
   }

   const size_t index( rd.row() * st.rowdilation() + st.row() );

   return dilatedsubmatrix( rowslice( st.operand(), index, args... ), st.column(), st.page(), st.columns(),
      st.pages(), st.columndilation(), st.pagedilation(), unchecked );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific row of the given temporary DilatedSubtensor.
// \ingroup DilatedSubtensor
//
// \param sm The temporary DilatedSubtensor containing the row.
// \param args The optional row arguments.
// \return View on the specified row of the DilatedSubtensor.
// \exception std::invalid_argument Invalid row access index.
//
// This function returns an expression representing the specified row of the given temporary
// DilatedSubtensor.
*/
template< size_t... CRAs      // Compile time row arguments
        , typename TT         // Type of the sparse DilatedSubtensor
        , bool DF             // Density flag
        , typename... RRAs >  // Runtime row arguments
inline decltype(auto) rowslice( DilatedSubtensor<TT,DF>&& st, RRAs... args )
{
   BLAZE_FUNCTION_TRACE;

   const RowSliceData<CRAs...> rd( args... );

   constexpr bool isChecked( !Contains_v< TypeList<RRAs...>, Unchecked > );

   if( isChecked ) {
      if( ( rd.row() >= st.rows() ) ) {
         BLAZE_THROW_INVALID_ARGUMENT( "Invalid row access index" );
      }
   }
   else {
      BLAZE_USER_ASSERT( rd.row() < st.rows() , "Invalid row access index" );
   }

   const size_t index( rd.row() * st.rowdilation() + st.row() );

   return dilatedsubmatrix( rowslice( st.operand(), index, args... ), st.column(), st.page(), st.columns(),
      st.pages(), st.columndilation(), st.pagedilation(), unchecked );
}
/*! \endcond */
//*************************************************************************************************



//=================================================================================================
//
//  GLOBAL RESTRUCTURING FUNCTIONS (COLUMNSLICE)
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific column of the given DilatedSubtensor.
// \ingroup DilatedSubtensor
//
// \param sm The DilatedSubtensor containing the column.
// \param args The optional column arguments.
// \return View on the specified column of the DilatedSubtensor.
//
// This function returns an expression representing the specified column of the given DilatedSubtensor.
*/
template< size_t J1              // Column index
        , typename TT            // Type of the sparse DilatedSubtensor
        , bool DF                // Density flag
        , size_t K               // Index of the first page
        , size_t I               // Index of the first row
        , size_t J2              // Index of the first column
        , size_t O               // Number of pages
        , size_t M               // Number of rows
        , size_t N               // Number of columns
        , size_t PageDilation    // Step between rows of the DilatedSubtensor
        , size_t RowDilation     // Step between rows of the DilatedSubtensor
        , size_t ColumnDilation  // Step between columns of the DilatedSubtensor
        , typename... RCAs >     // Optional column arguments
inline decltype(auto) columnslice( DilatedSubtensor<TT,DF,K,I,J2,O,M,N,PageDilation,RowDilation,ColumnDilation>& st, RCAs... args )
{
   BLAZE_FUNCTION_TRACE;

   BLAZE_STATIC_ASSERT_MSG( J1 < N, "Invalid column access index" );

   return dilatedsubmatrix<K,I,O,M,PageDilation,RowDilation>( columnslice<J1*ColumnDilation+J2>( st.operand(), args... ), unchecked );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific column of the given constant DilatedSubtensor.
// \ingroup DilatedSubtensor
//
// \param sm The constant DilatedSubtensor containing the column.
// \param args The optional column arguments.
// \return View on the specified column of the DilatedSubtensor.
//
// This function returns an expression representing the specified column of the given constant
// DilatedSubtensor.
*/
template< size_t J1              // Column index
        , typename TT            // Type of the sparse DilatedSubtensor
        , bool DF                // Density flag
        , size_t K               // Index of the first page
        , size_t I               // Index of the first row
        , size_t J2              // Index of the first column
        , size_t O               // Number of pages
        , size_t M               // Number of rows
        , size_t N               // Number of columns
        , size_t PageDilation    // Step between rows of the DilatedSubtensor
        , size_t RowDilation     // Step between rows of the DilatedSubtensor
        , size_t ColumnDilation  // Step between columns of the DilatedSubtensor
        , typename... RCAs >  // Optional column arguments
inline decltype(auto) columnslice( const DilatedSubtensor<TT,DF,K,I,J2,O,M,N,PageDilation,RowDilation,ColumnDilation>& st, RCAs... args )
{
   BLAZE_FUNCTION_TRACE;

   BLAZE_STATIC_ASSERT_MSG( J1 < N, "Invalid column access index" );

   return dilatedsubmatrix<K,I,O,M,PageDilation,RowDilation>( columnslice<J1*ColumnDilation+J2>( st.operand(), args... ), unchecked );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific column of the given temporary DilatedSubtensor.
// \ingroup DilatedSubtensor
//
// \param sm The temporary DilatedSubtensor containing the column.
// \param args The optional column arguments.
// \return View on the specified column of the DilatedSubtensor.
//
// This function returns an expression representing the specified column of the given temporary
// DilatedSubtensor.
*/
template< size_t J1              // Column index
        , typename TT            // Type of the sparse DilatedSubtensor
        , bool DF                // Density flag
        , size_t K               // Index of the first page
        , size_t I               // Index of the first row
        , size_t J2              // Index of the first column
        , size_t O               // Number of pages
        , size_t M               // Number of rows
        , size_t N               // Number of columns
        , size_t PageDilation    // Step between rows of the DilatedSubtensor
        , size_t RowDilation     // Step between rows of the DilatedSubtensor
        , size_t ColumnDilation  // Step between columns of the DilatedSubtensor
        , typename... RCAs >  // Optional column arguments
inline decltype(auto) columnslice( DilatedSubtensor<TT,DF,K,I,J2,O,M,N,PageDilation,RowDilation,ColumnDilation>&& st, RCAs... args )
{
   BLAZE_FUNCTION_TRACE;

   BLAZE_STATIC_ASSERT_MSG( J1 < N, "Invalid column access index" );

   return dilatedsubmatrix<K,I,O,M,PageDilation,RowDilation>( columnslice<J1*ColumnDilation+J2>( st.operand(), args... ), unchecked );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific column of the given DilatedSubtensor.
// \ingroup DilatedSubtensor
//
// \param sm The DilatedSubtensor containing the column.
// \param index The index of the column.
// \param args The optional column arguments.
// \return View on the specified column of the DilatedSubtensor.
// \exception std::invalid_argument Invalid column access index.
//
// This function returns an expression representing the specified column of the given DilatedSubtensor.
*/
template< typename TT            // Type of the sparse DilatedSubtensor
        , bool DF                // Density flag
        , size_t K               // Index of the first page
        , size_t I               // Index of the first row
        , size_t J               // Index of the first column
        , size_t O               // Number of pages
        , size_t M               // Number of rows
        , size_t N               // Number of columns
        , size_t PageDilation    // Step between rows of the DilatedSubtensor
        , size_t RowDilation     // Step between rows of the DilatedSubtensor
        , size_t ColumnDilation  // Step between columns of the DilatedSubtensor
        , typename... RCAs >  // Optional column arguments
inline decltype(auto) columnslice( DilatedSubtensor<TT,DF,K,I,J,O,M,N,PageDilation,RowDilation,ColumnDilation>& st, size_t index, RCAs... args )
{
   BLAZE_FUNCTION_TRACE;

   constexpr bool isChecked( !Contains_v< TypeList<RCAs...>, Unchecked > );

   if( isChecked ) {
      if( ( index >= N ) ) {
         BLAZE_THROW_INVALID_ARGUMENT( "Invalid column access index" );
      }
   }
   else {
      BLAZE_USER_ASSERT( index < N, "Invalid column access index" );
   }

   return dilatedsubmatrix<K,I,O,M,PageDilation,RowDilation>( columnslice( st.operand(), J+index*ColumnDilation, args... ), unchecked );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific column of the given constant DilatedSubtensor.
// \ingroup DilatedSubtensor
//
// \param sm The constant DilatedSubtensor containing the column.
// \param index The index of the column.
// \param args The optional column arguments.
// \return View on the specified column of the DilatedSubtensor.
// \exception std::invalid_argument Invalid column access index.
//
// This function returns an expression representing the specified column of the given constant
// DilatedSubtensor.
*/
template< typename TT            // Type of the sparse DilatedSubtensor
        , bool DF                // Density flag
        , size_t K               // Index of the first page
        , size_t I               // Index of the first row
        , size_t J               // Index of the first column
        , size_t O               // Number of pages
        , size_t M               // Number of rows
        , size_t N               // Number of columns
        , size_t PageDilation    // Step between rows of the DilatedSubtensor
        , size_t RowDilation     // Step between rows of the DilatedSubtensor
        , size_t ColumnDilation  // Step between columns of the DilatedSubtensor
        , typename... RCAs >  // Optional column arguments
inline decltype(auto) columnslice( const DilatedSubtensor<TT,DF,K,I,J,O,M,N,PageDilation,RowDilation,ColumnDilation>& st, size_t index, RCAs... args )
{
   BLAZE_FUNCTION_TRACE;

   constexpr bool isChecked( !Contains_v< TypeList<RCAs...>, Unchecked > );

   if( isChecked ) {
      if( ( index >= N ) ) {
         BLAZE_THROW_INVALID_ARGUMENT( "Invalid column access index" );
      }
   }
   else {
      BLAZE_USER_ASSERT( index < N, "Invalid column access index" );
   }

   return dilatedsubmatrix<K,I,O,M,PageDilation,RowDilation>( columnslice( st.operand(), J+index*ColumnDilation, args... ), unchecked );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific column of the given temporary DilatedSubtensor.
// \ingroup DilatedSubtensor
//
// \param sm The temporary DilatedSubtensor containing the column.
// \param index The index of the column.
// \param args The optional column arguments.
// \return View on the specified column of the DilatedSubtensor.
// \exception std::invalid_argument Invalid column access index.
//
// This function returns an expression representing the specified column of the given temporary
// DilatedSubtensor.
*/
template< typename TT            // Type of the sparse DilatedSubtensor
        , bool DF                // Density flag
        , size_t K               // Index of the first page
        , size_t I               // Index of the first row
        , size_t J               // Index of the first column
        , size_t O               // Number of pages
        , size_t M               // Number of rows
        , size_t N               // Number of columns
        , size_t PageDilation    // Step between rows of the DilatedSubtensor
        , size_t RowDilation     // Step between rows of the DilatedSubtensor
        , size_t ColumnDilation  // Step between columns of the DilatedSubtensor
        , typename... RCAs >  // Optional column arguments
inline decltype(auto) columnslice( DilatedSubtensor<TT,DF,K,I,J,O,M,N,PageDilation,RowDilation,ColumnDilation>&& st, size_t index, RCAs... args )
{
   BLAZE_FUNCTION_TRACE;

   constexpr bool isChecked( !Contains_v< TypeList<RCAs...>, Unchecked > );

   if( isChecked ) {
      if( ( index >= N ) ) {
         BLAZE_THROW_INVALID_ARGUMENT( "Invalid column access index" );
      }
   }
   else {
      BLAZE_USER_ASSERT( index < N, "Invalid column access index" );
   }

   return dilatedsubmatrix<K,I,O,M,PageDilation,RowDilation>( columnslice( st.operand(), J+index*ColumnDilation, args... ), unchecked );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific column of the given DilatedSubtensor.
// \ingroup DilatedSubtensor
//
// \param sm The DilatedSubtensor containing the column.
// \param args The optional column arguments.
// \return View on the specified column of the DilatedSubtensor.
// \exception std::invalid_argument Invalid column access index.
//
// This function returns an expression representing the specified column of the given DilatedSubtensor.
*/
template< size_t... CCAs      // Compile time column arguments
        , typename TT         // Type of the sparse DilatedSubtensor
        , bool DF             // Density flag
        , typename... RCAs >  // Runtime column arguments
inline decltype(auto) columnslice( DilatedSubtensor<TT,DF>& st, RCAs... args )
{
   BLAZE_FUNCTION_TRACE;

   const ColumnSliceData<CCAs...> cd( args... );

   constexpr bool isChecked( !Contains_v< TypeList<RCAs...>, Unchecked > );

   if( isChecked ) {
      if( ( cd.column() >= st.columns() ) ) {
         BLAZE_THROW_INVALID_ARGUMENT( "Invalid column access index" );
      }
   }
   else {
      BLAZE_USER_ASSERT( cd.column() < st.columns(), "Invalid column access index" );
   }

   const size_t index( cd.column() * st.columndilation() + st.column() );

   return dilatedsubmatrix( columnslice( st.operand(), index, args... ), st.page(), st.row(), st.pages(), st.rows(),
      st.pagedilation(),st.rowdilation(), unchecked );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific column of the given constant DilatedSubtensor.
// \ingroup DilatedSubtensor
//
// \param sm The constant DilatedSubtensor containing the column.
// \param args The optional column arguments.
// \return View on the specified column of the DilatedSubtensor.
// \exception std::invalid_argument Invalid column access index.
//
// This function returns an expression representing the specified column of the given constant
// DilatedSubtensor.
*/
template< size_t... CCAs      // Compile time column arguments
        , typename TT         // Type of the sparse DilatedSubtensor
        , bool DF             // Density flag
        , typename... RCAs >  // Runtime column arguments
inline decltype(auto) columnslice( const DilatedSubtensor<TT,DF>& st, RCAs... args )
{
   BLAZE_FUNCTION_TRACE;

   const ColumnSliceData<CCAs...> cd( args... );

   constexpr bool isChecked( !Contains_v< TypeList<RCAs...>, Unchecked > );

   if( isChecked ) {
      if( ( cd.column() >= st.columns() ) ) {
         BLAZE_THROW_INVALID_ARGUMENT( "Invalid column access index" );
      }
   }
   else {
      BLAZE_USER_ASSERT( cd.column() < st.columns(), "Invalid column access index" );
   }

   const size_t index( cd.column() * st.columndilation() + st.column() );

   return dilatedsubmatrix( columnslice( st.operand(), index, args... ), st.page(), st.row(), st.pages(), st.rows(),
      st.pagedilation(),st.rowdilation(), unchecked );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific column of the given temporary DilatedSubtensor.
// \ingroup DilatedSubtensor
//
// \param sm The temporary DilatedSubtensor containing the column.
// \param args The optional column arguments.
// \return View on the specified column of the DilatedSubtensor.
// \exception std::invalid_argument Invalid column access index.
//
// This function returns an expression representing the specified column of the given temporary
// DilatedSubtensor.
*/
template< size_t... CCAs      // Compile time column arguments
        , typename TT         // Type of the sparse DilatedSubtensor
        , bool DF             // Density flag
        , typename... RCAs >  // Runtime column arguments
inline decltype(auto) columnslice( DilatedSubtensor<TT,DF>&& st, RCAs... args )
{
   BLAZE_FUNCTION_TRACE;

   const ColumnSliceData<CCAs...> cd( args... );

   constexpr bool isChecked( !Contains_v< TypeList<RCAs...>, Unchecked > );

   if( isChecked ) {
      if( ( cd.column() >= st.columns() ) ) {
         BLAZE_THROW_INVALID_ARGUMENT( "Invalid column access index" );
      }
   }
   else {
      BLAZE_USER_ASSERT( cd.column() < st.columns(), "Invalid column access index" );
   }

   const size_t index( cd.column() * st.columndilation() + st.column() );

   return dilatedsubmatrix( columnslice( st.operand(), index, args... ), st.page(), st.row(), st.pages(), st.rows(),
      st.pagedilation(),st.rowdilation(), unchecked );
}
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  DILATEDSUBTENSOR OPERATORS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Resetting the given DilatedSubtensor.
// \ingroup DilatedSubtensor
//
// \param sm The DilatedSubtensor to be resetted.
// \return void
*/
template< typename TT       // Type of the tensor
        , bool DF           // Density flag
        , size_t... CSAs >  // Compile time DilatedSubtensor arguments
inline void reset( DilatedSubtensor<TT,DF,CSAs...>& st )
{
   st.reset();
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Resetting the given temporary DilatedSubtensor.
// \ingroup DilatedSubtensor
//
// \param sm The temporary DilatedSubtensor to be resetted.
// \return void
*/
template< typename TT       // Type of the tensor
        , bool DF           // Density flag
        , size_t... CSAs >  // Compile time DilatedSubtensor arguments
inline void reset( DilatedSubtensor<TT,DF,CSAs...>&& st )
{
   st.reset();
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Reset the specified row/column of the given DilatedSubtensor.
// \ingroup DilatedSubtensor
//
// \param sm The DilatedSubtensor to be resetted.
// \param i The index of the row/column to be resetted.
// \return void
//
// This function resets the values in the specified row/column of the given DilatedSubtensor to their
// default value. In case the given DilatedSubtensor is a \a rowMajor tensor the function resets the
// values in row \a i, if it is a \a columnMajor tensor the function resets the values in column
// \a i. Note that the capacity of the row/column remains unchanged.
*/
template< typename TT       // Type of the tensor
        , bool DF           // Density flag
        , size_t... CSAs >  // Compile time DilatedSubtensor arguments
inline void reset( DilatedSubtensor<TT,DF,CSAs...>& st, size_t i, size_t k )
{
   st.reset( i, k );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Clearing the given tensor.
// \ingroup DilatedSubtensor
//
// \param sm The tensor to be cleared.
// \return void
//
// Clearing a DilatedSubtensor is equivalent to resetting it via the reset() function.
*/
template< typename TT       // Type of the tensor
        , bool DF           // Density flag
        , size_t... CSAs >  // Compile time DilatedSubtensor arguments
inline void clear( DilatedSubtensor<TT,DF,CSAs...>& st )
{
   st.reset();
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Clearing the given temporary tensor.
// \ingroup DilatedSubtensor
//
// \param sm The temporary tensor to be cleared.
// \return void
//
// Clearing a DilatedSubtensor is equivalent to resetting it via the reset() function.
*/
template< typename TT       // Type of the tensor
        , bool DF           // Density flag
        , size_t... CSAs >  // Compile time DilatedSubtensor arguments
inline void clear( DilatedSubtensor<TT,DF,CSAs...>&& st )
{
   st.reset();
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns whether the given dense DilatedSubtensor is in default state.
// \ingroup DilatedSubtensor
//
// \param sm The dense DilatedSubtensor to be tested for its default state.
// \return \a true in case the given dense DilatedSubtensor is component-wise zero, \a false otherwise.
//
// This function checks whether the dense DilatedSubtensor is in default state. For instance, in case
// the DilatedSubtensor is instantiated for a built-in integral or floating point data type, the function
// returns \a true in case all DilatedSubtensor elements are 0 and \a false in case any DilatedSubtensor element
// is not 0. The following example demonstrates the use of the \a isDefault function:

   \code
   blaze::DynamicTensor<double,rowMajor> A;
   // ... Resizing and initialization
   if( isDefault( DilatedSubtensor( A, 12UL, 13UL, 22UL, 33UL ) ) ) { ... }
   \endcode

// Optionally, it is possible to switch between strict semantics (blaze::strict) and relaxed
// semantics (blaze::relaxed):

   \code
   if( isDefault<relaxed>( DilatedSubtensor( A, 12UL, 13UL, 22UL, 33UL ) ) ) { ... }
   \endcode
*/
template< bool RF           // Relaxation flag
        , typename TT       // Type of the dense tensor
        , size_t... CSAs >  // Compile time DilatedSubtensor arguments
inline bool isDefault( const DilatedSubtensor<TT,true,CSAs...>& st )
{
   using blaze::isDefault;

   for( size_t k=0UL; k<(~st).pages(); ++k )
      for( size_t i=0UL; i<(~st).rows(); ++i )
         for( size_t j=0UL; j<(~st).columns(); ++j )
            if( !isDefault<RF>( (~st)(k,i,j) ) )
               return false;

   return true;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns whether the invariants of the given DilatedSubtensor are intact.
// \ingroup DilatedSubtensor
//
// \param sm The DilatedSubtensor to be tested.
// \return \a true in case the given DilatedSubtensor's invariants are intact, \a false otherwise.
//
// This function checks whether the invariants of the DilatedSubtensor are intact, i.e. if its
// state is valid. In case the invariants are intact, the function returns \a true, else it
// will return \a false. The following example demonstrates the use of the \a isIntact()
// function:

   \code
   blaze::DynamicTensor<double,rowMajor> A;
   // ... Resizing and initialization
   if( isIntact( DilatedSubtensor( A, 12UL, 13UL, 22UL, 33UL ) ) ) { ... }
   \endcode
*/
template< typename TT       // Type of the tensor
        , bool DF           // Density flag
        , size_t... CSAs >  // Compile time DilatedSubtensor arguments
inline bool isIntact( const DilatedSubtensor<TT,DF,CSAs...>& st ) noexcept
{
   return ( st.page() + st.pages()*st.pagedilation() <= st.operand().pages() &&
            st.row() + st.rows()*st.rowdilation() <= st.operand().rows() &&
            st.column() + st.columns()*st.columndilation() <= st.operand().columns() &&
            isIntact( st.operand() ) );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Checks if the given DilatedSubtensor is symmetric.
// \ingroup DilatedSubtensor
//
// \param sm The DilatedSubtensor to be checked.
// \return \a true if the DilatedSubtensor is symmetric, \a false if not.
//
// This function checks if the given DilatedSubtensor is symmetric. The DilatedSubtensor is considered to
// be symmetric if it is a square tensor whose transpose is equal to itself (\f$ A = A^T \f$). The
// following code example demonstrates the use of the function:

   \code
   blaze::DynamicTensor<int,blaze::rowMajor> A( 32UL, 16UL );
   // ... Initialization

   auto sm = DilatedSubtensor( A, 8UL, 8UL, 16UL, 16UL );

   if( isSymmetric( sm ) ) { ... }
   \endcode
*/
template< typename TT       // Type of the tensor
        , bool DF           // Density flag
        , size_t... CSAs >  // Compile time DilatedSubtensor arguments
inline bool isSymmetric( const DilatedSubtensor<TT,DF,CSAs...>& st )
{
   using BaseType = BaseType_t< DilatedSubtensor<TT,DF,CSAs...> >;

   return isSymmetric( static_cast< const BaseType& >( st ) );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Checks if the given DilatedSubtensor is Hermitian.
// \ingroup DilatedSubtensor
//
// \param sm The DilatedSubtensor to be checked.
// \return \a true if the DilatedSubtensor is Hermitian, \a false if not.
//
// This function checks if the given DilatedSubtensor is Hermitian. The DilatedSubtensor is considered to
// be Hermitian if it is a square tensor whose transpose is equal to its conjugate transpose
// (\f$ A = \overline{A^T} \f$). The following code example demonstrates the use of the function:

   \code
   blaze::DynamicTensor<int,blaze::rowMajor> A( 32UL, 16UL );
   // ... Initialization

   auto sm = DilatedSubtensor( A, 8UL, 8UL, 16UL, 16UL );

   if( isHermitian( sm ) ) { ... }
   \endcode
*/
template< typename TT       // Type of the tensor
        , bool DF           // Density flag
        , size_t... CSAs >  // Compile time DilatedSubtensor arguments
inline bool isHermitian( const DilatedSubtensor<TT,DF,CSAs...>& st )
{
   using BaseType = BaseType_t< DilatedSubtensor<TT,DF,CSAs...> >;

   return isHermitian( static_cast< const BaseType& >( st ) );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Checks if the given DilatedSubtensor is a lower triangular tensor.
// \ingroup DilatedSubtensor
//
// \param sm The DilatedSubtensor to be checked.
// \return \a true if the DilatedSubtensor is a lower triangular tensor, \a false if not.
//
// This function checks if the given DilatedSubtensor is a lower triangular tensor. The tensor is
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

   auto sm = DilatedSubtensor( A, 8UL, 8UL, 16UL, 16UL );

   if( isLower( sm ) ) { ... }
   \endcode
*/
template< typename TT       // Type of the tensor
        , bool DF           // Density flag
        , size_t... CSAs >  // Compile time DilatedSubtensor arguments
inline bool isLower( const DilatedSubtensor<TT,DF,CSAs...>& st )
{
   using BaseType = BaseType_t< DilatedSubtensor<TT,DF,CSAs...> >;

   return isLower( static_cast<const BaseType&>( st ) );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Checks if the given DilatedSubtensor is a lower unitriangular tensor.
// \ingroup DilatedSubtensor
//
// \param sm The DilatedSubtensor to be checked.
// \return \a true if the DilatedSubtensor is a lower unitriangular tensor, \a false if not.
//
// This function checks if the given DilatedSubtensor is a lower unitriangular tensor. The tensor is
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

   auto sm = DilatedSubtensor( A, 8UL, 8UL, 16UL, 16UL );

   if( isUniLower( sm ) ) { ... }
   \endcode
*/
template< typename TT       // Type of the tensor
        , bool DF           // Density flag
        , size_t... CSAs >  // Compile time DilatedSubtensor arguments
inline bool isUniLower( const DilatedSubtensor<TT,DF,CSAs...>& st )
{
   using BaseType = BaseType_t< DilatedSubtensor<TT,DF,CSAs...> >;

   return isUniLower( static_cast<const BaseType&>( st ) );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Checks if the given DilatedSubtensor is a strictly lower triangular tensor.
// \ingroup DilatedSubtensor
//
// \param sm The DilatedSubtensor to be checked.
// \return \a true if the DilatedSubtensor is a strictly lower triangular tensor, \a false if not.
//
// This function checks if the given DilatedSubtensor is a strictly lower triangular tensor. The
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

   auto sm = DilatedSubtensor( A, 8UL, 8UL, 16UL, 16UL );

   if( isStrictlyLower( sm ) ) { ... }
   \endcode
*/
template< typename TT       // Type of the tensor
        , bool DF           // Density flag
        , size_t... CSAs >  // Compile time DilatedSubtensor arguments
inline bool isStrictlyLower( const DilatedSubtensor<TT,DF,CSAs...>& st )
{
   using BaseType = BaseType_t< DilatedSubtensor<TT,DF,CSAs...> >;

   return isStrictlyLower( static_cast<const BaseType&>( st ) );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Checks if the given DilatedSubtensor is an upper triangular tensor.
// \ingroup DilatedSubtensor
//
// \param sm The DilatedSubtensor to be checked.
// \return \a true if the DilatedSubtensor is an upper triangular tensor, \a false if not.
//
// This function checks if the given sparse DilatedSubtensor is an upper triangular tensor. The tensor
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

   auto sm = DilatedSubtensor( A, 8UL, 8UL, 16UL, 16UL );

   if( isUpper( sm ) ) { ... }
   \endcode
*/
template< typename TT       // Type of the tensor
        , bool DF           // Density flag
        , size_t... CSAs >  // Compile time DilatedSubtensor arguments
inline bool isUpper( const DilatedSubtensor<TT,DF,CSAs...>& st )
{
   using BaseType = BaseType_t< DilatedSubtensor<TT,DF,CSAs...> >;

   return isUpper( static_cast<const BaseType&>( st ) );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Checks if the given DilatedSubtensor is an upper unitriangular tensor.
// \ingroup DilatedSubtensor
//
// \param sm The DilatedSubtensor to be checked.
// \return \a true if the DilatedSubtensor is an upper unitriangular tensor, \a false if not.
//
// This function checks if the given sparse DilatedSubtensor is an upper triangular tensor. The tensor
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

   auto sm = DilatedSubtensor( A, 8UL, 8UL, 16UL, 16UL );

   if( isUniUpper( sm ) ) { ... }
   \endcode
*/
template< typename TT       // Type of the tensor
        , bool DF           // Density flag
        , size_t... CSAs >  // Compile time DilatedSubtensor arguments
inline bool isUniUpper( const DilatedSubtensor<TT,DF,CSAs...>& st )
{
   using BaseType = BaseType_t< DilatedSubtensor<TT,DF,CSAs...> >;

   return isUniUpper( static_cast<const BaseType&>( st ) );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Checks if the given DilatedSubtensor is a strictly upper triangular tensor.
// \ingroup DilatedSubtensor
//
// \param sm The DilatedSubtensor to be checked.
// \return \a true if the DilatedSubtensor is a strictly upper triangular tensor, \a false if not.
//
// This function checks if the given sparse DilatedSubtensor is a strictly upper triangular tensor. The
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

   auto sm = DilatedSubtensor( A, 8UL, 8UL, 16UL, 16UL );

   if( isStrictlyUpper( sm ) ) { ... }
   \endcode
*/
template< typename TT       // Type of the tensor
        , bool DF           // Density flag
        , size_t... CSAs >  // Compile time DilatedSubtensor arguments
inline bool isStrictlyUpper( const DilatedSubtensor<TT,DF,CSAs...>& st )
{
   using BaseType = BaseType_t< DilatedSubtensor<TT,DF,CSAs...> >;

   return isStrictlyUpper( static_cast<const BaseType&>( st ) );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns whether the given tensor and DilatedSubtensor represent the same observable state.
// \ingroup DilatedSubtensor
//
// \param a The DilatedSubtensor to be tested for its state.
// \param b The tensor to be tested for its state.
// \return \a true in case the DilatedSubtensor and tensor share a state, \a false otherwise.
//
// This overload of the isSame function tests if the given DilatedSubtensor refers to the full given
// tensor and by that represents the same observable state. In this case, the function returns
// \a true, otherwise it returns \a false.
*/
template< typename TT       // Type of the tensor
        , bool DF           // Density flag
        , size_t... CSAs >  // Compile time DilatedSubtensor arguments
inline bool isSame( const DilatedSubtensor<TT,DF,CSAs...>& a, const Tensor<TT>& b ) noexcept
{
   return ( isSame( a.operand(), ~b ) && ( a.pages() == ( ~b ).pages() ) && ( a.rows() == ( ~b ).rows() ) &&
      ( a.columns() == ( ~b ).columns() ) && ( a.pagedilation() == 1UL ) && ( a.rowdilation() == 1UL ) &&
      ( a.columndilation() == 1UL ) );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns whether the given tensor and DilatedSubtensor represent the same observable state.
// \ingroup DilatedSubtensor
//
// \param a The tensor to be tested for its state.
// \param b The DilatedSubtensor to be tested for its state.
// \return \a true in case the tensor and DilatedSubtensor share a state, \a false otherwise.
//
// This overload of the isSame function tests if the given DilatedSubtensor refers to the full given
// tensor and by that represents the same observable state. In this case, the function returns
// \a true, otherwise it returns \a false.
*/
template< typename TT       // Type of the tensor
        , bool DF           // Density flag
        , size_t... CSAs >  // Compile time DilatedSubtensor arguments
inline bool isSame( const Tensor<TT>& a, const DilatedSubtensor<TT,DF,CSAs...>& b ) noexcept
{
   return isSame(b, a);
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns whether the two given subtensors represent the same observable state.
// \ingroup DilatedSubtensor
//
// \param a The first DilatedSubtensor to be tested for its state.
// \param b The second DilatedSubtensor to be tested for its state.
// \return \a true in case the two subtensors share a state, \a false otherwise.
//
// This overload of the isSame function tests if the two given subtensors refer to exactly the
// same part of the same tensor. In case both subtensors represent the same observable state,
// the function returns \a true, otherwise it returns \a false.
*/
template< typename TT1       // Type of the tensor of the left-hand side DilatedSubtensor
        , bool DF1           // Density flag of the left-hand side DilatedSubtensor
        , size_t... CSAs1    // Compile time DilatedSubtensor arguments of the left-hand side DilatedSubtensor
        , typename TT2       // Type of the tensor of the right-hand side DilatedSubtensor
        , bool DF2           // Density flag of the right-hand side DilatedSubtensor
        , size_t... CSAs2 >  // Compile time DilatedSubtensor arguments of the right-hand side DilatedSubtensor
inline bool isSame( const DilatedSubtensor<TT1,DF1,CSAs1...>& a,
                    const DilatedSubtensor<TT2,DF2,CSAs2...>& b ) noexcept
{
   return ( isSame( a.operand(), b.operand() ) &&
            ( a.page() == b.page() ) && ( a.pages() == b.pages() ) &&
            ( a.row() == b.row() ) && ( a.column() == b.column() ) &&
            ( a.rows() == b.rows() ) && ( a.columns() == b.columns() ) &&
            ( a.pagedilation() == b.pagedilation() ) &&
            ( a.rowdilation() == b.rowdilation() ) &&
            ( a.columndilation() == b.columndilation() ) );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief In-place inversion of the given dense DilatedSubtensor.
// \ingroup DilatedSubtensor
//
// \param sm The dense DilatedSubtensor to be inverted.
// \return void
// \exception std::invalid_argument Invalid non-square tensor provided.
// \exception std::runtime_error Inversion of singular tensor failed.
//
// This function inverts the given dense DilatedSubtensor by means of the specified tensor type or tensor
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
//  - ... the given DilatedSubtensor is not a square tensor;
//  - ... the given DilatedSubtensor is singular and not invertible.
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
        , bool DF           // Density flag
        , size_t... CSAs >  // Compile time DilatedSubtensor arguments
inline auto invert( DilatedSubtensor<TT,DF,CSAs...>& st )
   -> EnableIf_t< !HasMutableDataAccess_v<TT> >
{
   using RT = ResultType_t< DilatedSubtensor<TT,DF,CSAs...> >;

   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION  ( RT );
   BLAZE_CONSTRAINT_MUST_HAVE_MUTABLE_DATA_ACCESS( RT );

   RT tmp( st );
   invert<IF>( tmp );
   st = tmp;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Predict invariant violations by setting a single element of a DilatedSubtensor.
// \ingroup DilatedSubtensor
//
// \param sm The target DilatedSubtensor.
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
        , bool DF           // Density flag
        , size_t... CSAs    // Compile time DilatedSubtensor arguments
        , typename ET >     // Type of the element
inline bool trySet( const DilatedSubtensor<TT,DF,CSAs...>& st, size_t k, size_t i, size_t j, const ET& value )
{
   BLAZE_INTERNAL_ASSERT( k < st.pages(),   "Invalid page access index"  );
   BLAZE_INTERNAL_ASSERT( i < st.rows(),    "Invalid row access index"   );
   BLAZE_INTERNAL_ASSERT( j < st.columns(), "Invalid column access index");

   return trySet( st.operand(), st.row()+i*st.rowdilation(), st.column()+j*st.columndilation(), st.page()+k*st.pagedilation(), value );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Predict invariant violations by adding to a single element of a DilatedSubtensor.
// \ingroup DilatedSubtensor
//
// \param sm The target DilatedSubtensor.
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
        , bool DF           // Density flag
        , size_t... CSAs    // Compile time DilatedSubtensor arguments
        , typename ET >     // Type of the element
inline bool tryAdd( const DilatedSubtensor<TT,DF,CSAs...>& st, size_t k, size_t i, size_t j, const ET& value )
{
   BLAZE_INTERNAL_ASSERT( k < st.pages(),   "Invalid page access index"  );
   BLAZE_INTERNAL_ASSERT( i < st.rows(),    "Invalid row access index"   );
   BLAZE_INTERNAL_ASSERT( j < st.columns(), "Invalid column access index");

   return tryAdd( st.operand(), st.row()+i*st.rowdilation(), st.column()+j*st.columndilation(), st.page()+k*st.pagedilation(), value );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Predict invariant violations by subtracting from a single element of a DilatedSubtensor.
// \ingroup DilatedSubtensor
//
// \param sm The target DilatedSubtensor.
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
        , bool DF           // Density flag
        , size_t... CSAs    // Compile time DilatedSubtensor arguments
        , typename ET >     // Type of the element
inline bool trySub( const DilatedSubtensor<TT,DF,CSAs...>& st, size_t k, size_t i, size_t j, const ET& value )
{
   BLAZE_INTERNAL_ASSERT( k < st.pages(),   "Invalid page access index"  );
   BLAZE_INTERNAL_ASSERT( i < st.rows(),    "Invalid row access index"   );
   BLAZE_INTERNAL_ASSERT( j < st.columns(), "Invalid column access index");

   return trySub( st.operand(), st.row()+i*st.rowdilation(), st.column()+j*st.columndilation(), st.page()+k*st.pagedilation(), value );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Predict invariant violations by scaling a single element of a DilatedSubtensor.
// \ingroup DilatedSubtensor
//
// \param sm The target DilatedSubtensor.
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
        , bool DF           // Density flag
        , size_t... CSAs    // Compile time DilatedSubtensor arguments
        , typename ET >     // Type of the element
inline bool tryMult( const DilatedSubtensor<TT,DF,CSAs...>& st, size_t k, size_t i, size_t j, const ET& value )
{
   BLAZE_INTERNAL_ASSERT( k < st.pages(),   "Invalid page access index"  );
   BLAZE_INTERNAL_ASSERT( i < st.rows(),    "Invalid row access index"   );
   BLAZE_INTERNAL_ASSERT( j < st.columns(), "Invalid column access index");

   return tryMult( st.operand(), st.row()+i*st.rowdilation(), st.column()+j*st.columndilation(), st.page()+k*st.pagedilation(), value );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Predict invariant violations by scaling a range of elements of a DilatedSubtensor.
// \ingroup DilatedSubtensor
//
// \param sm The target DilatedSubtensor.
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
        , bool DF           // Density flag
        , size_t... CSAs    // Compile time DilatedSubtensor arguments
        , typename ET >     // Type of the element
BLAZE_ALWAYS_INLINE bool
   tryMult( const DilatedSubtensor<TT,DF,CSAs...>& st, size_t row, size_t column, size_t page, size_t m, size_t n, size_t o, const ET& value )
{
   MAYBE_UNUSED( column );

   BLAZE_INTERNAL_ASSERT( page <= (~st).pages(),         "Invalid page access index"  );
   BLAZE_INTERNAL_ASSERT( row <= (~st).rows(),           "Invalid row access index"   );
   BLAZE_INTERNAL_ASSERT( column <= (~st).columns(),     "Invalid column access index");
   BLAZE_INTERNAL_ASSERT( page + o <= (~st).pages(),     "Invalid number of pages"    );
   BLAZE_INTERNAL_ASSERT( row + m <= (~st).rows(),       "Invalid number of rows"     );
   BLAZE_INTERNAL_ASSERT( column + n <= (~st).columns(), "Invalid number of columns"  );


   return tryMult( st.operand(), st.row()+row*st.rowdilation(), st.column(), st.page(), m*st.rowdilation(), n, o, value );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Predict invariant violations by scaling a single element of a DilatedSubtensor.
// \ingroup DilatedSubtensor
//
// \param sm The target DilatedSubtensor.
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
        , bool DF           // Density flag
        , size_t... CSAs    // Compile time DilatedSubtensor arguments
        , typename ET >     // Type of the element
inline bool tryDiv( const DilatedSubtensor<TT,DF,CSAs...>& st, size_t k, size_t i, size_t j, const ET& value )
{
   BLAZE_INTERNAL_ASSERT( k < st.pages(),   "Invalid page access index"  );
   BLAZE_INTERNAL_ASSERT( i < st.rows(),    "Invalid row access index"   );
   BLAZE_INTERNAL_ASSERT( j < st.columns(), "Invalid column access index");

   return tryDiv( st.operand(), st.row()+i*st.rowdilation(), st.column()+j*st.columndilation(), st.page()+k*st.pagedilation(), value );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Predict invariant violations by scaling a range of elements of a DilatedSubtensor.
// \ingroup DilatedSubtensor
//
// \param sm The target DilatedSubtensor.
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
        , bool DF           // Density flag
        , size_t... CSAs    // Compile time DilatedSubtensor arguments
        , typename ET >     // Type of the element
BLAZE_ALWAYS_INLINE bool
   tryDiv( const DilatedSubtensor<TT,DF,CSAs...>& st, size_t row, size_t column, size_t page, size_t m, size_t n, size_t o, const ET& value )
{
   MAYBE_UNUSED( column );

   BLAZE_INTERNAL_ASSERT( page <= (~st).pages(),         "Invalid page access index"  );
   BLAZE_INTERNAL_ASSERT( row <= (~st).rows(),           "Invalid row access index"   );
   BLAZE_INTERNAL_ASSERT( column <= (~st).columns(),     "Invalid column access index");
   BLAZE_INTERNAL_ASSERT( page + o <= (~st).pages(),     "Invalid number of pages"    );
   BLAZE_INTERNAL_ASSERT( row + m <= (~st).rows(),       "Invalid number of rows"     );
   BLAZE_INTERNAL_ASSERT( column + n <= (~st).columns(), "Invalid number of columns"  );

   return tryDiv( st.operand(), st.row()+row*st.rowdilation(), st.column(), st.page(), m*st.rowdilation(), n, o, value );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Predict invariant violations by the assignment of a vector to a DilatedSubtensor.
// \ingroup DilatedSubtensor
//
// \param lhs The target left-hand side DilatedSubtensor.
// \param rhs The right-hand side vector to be assigned.
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
        , bool DF           // Density flag
        , size_t... CSAs    // Compile time DilatedSubtensor arguments
        , typename MT  >    // Type of the right-hand side vector
inline bool tryAssign( const DilatedSubtensor<TT,DF,CSAs...>& lhs,
                       const Matrix<MT,false>& rhs, size_t row, size_t column, size_t page )
{
   BLAZE_INTERNAL_ASSERT( page <= lhs.pages(),    "Invalid page access index"  );
   BLAZE_INTERNAL_ASSERT( row <= lhs.rows(),      "Invalid row access index"   );
   BLAZE_INTERNAL_ASSERT( column <= lhs.columns(),"Invalid column access index");

   BLAZE_INTERNAL_ASSERT( row + (~rhs).rows() <= lhs.rows(), "Invalid number of rows" );
   BLAZE_INTERNAL_ASSERT( column + (~rhs).columns() <= lhs.columns(), "Invalid number of columns" );

   return tryAssign( lhs.operand(), ~rhs, lhs.row() + row * lhs.rowdilation(),
                    lhs.column() + column * lhs.columndilation(), lhs.page() + page * lhs.pagedilation() );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Predict invariant violations by the assignment of a vector to the band of a DilatedSubtensor.
// \ingroup DilatedSubtensor
//
// \param lhs The target left-hand side DilatedSubtensor.
// \param rhs The right-hand side vector to be assigned.
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
        , bool DF           // Density flag
        , size_t... CSAs    // Compile time DilatedSubtensor arguments
        , typename MT >     // Type of the right-hand side vector
inline bool tryAssign( const DilatedSubtensor<TT,DF,CSAs...>& lhs,
                       const Matrix<MT,false>& rhs, ptrdiff_t band, size_t row, size_t column, size_t page )
{
   BLAZE_INTERNAL_ASSERT( page <= lhs.pages(),    "Invalid page access index"  );
   BLAZE_INTERNAL_ASSERT( row <= lhs.rows(),      "Invalid row access index"   );
   BLAZE_INTERNAL_ASSERT( column <= lhs.columns(),"Invalid column access index");

   BLAZE_INTERNAL_ASSERT( row + (~rhs).rows() <= lhs.rows(), "Invalid number of rows" );
   BLAZE_INTERNAL_ASSERT( column + (~rhs).columns() <= lhs.columns(), "Invalid number of columns" );

   return tryAssign( lhs.operand(), ~rhs, band + ptrdiff_t( lhs.column() - lhs.row() ),
                     lhs.row() + row * lhs.rowdilation(), lhs.column() + column * lhs.columndilation(), lhs.page() + page * lhs.pagedilation() );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Predict invariant violations by the assignment of a tensor to a DilatedSubtensor.
// \ingroup DilatedSubtensor
//
// \param lhs The target left-hand side DilatedSubtensor.
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
        , bool DF           // Density flag
        , size_t... CSAs    // Compile time DilatedSubtensor arguments
        , typename TT2 >    // Type of the right-hand side tensor
inline bool tryAssign( const DilatedSubtensor<TT1,DF,CSAs...>& lhs,
                       const Tensor<TT2>& rhs, size_t row, size_t column, size_t page )
{
   BLAZE_INTERNAL_ASSERT( row <= lhs.rows(), "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( column <= lhs.columns(), "Invalid column access index" );
   BLAZE_INTERNAL_ASSERT( page <= lhs.pages(), "Invalid page access index" );
   BLAZE_INTERNAL_ASSERT( row + (~rhs).rows() <= lhs.rows(), "Invalid number of rows" );
   BLAZE_INTERNAL_ASSERT( column + (~rhs).columns() <= lhs.columns(), "Invalid number of columns" );
   BLAZE_INTERNAL_ASSERT( page + (~rhs).pages() <= lhs.pages(), "Invalid number of pages" );

   return tryAssign( lhs.operand(), ~rhs, lhs.row() + row * lhs.rowdilation(), lhs.column() + column * lhs.columndilation(), lhs.page() + page * lhs.pagedilation() );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Predict invariant violations by the addition assignment of a vector to a DilatedSubtensor.
// \ingroup DilatedSubtensor
//
// \param lhs The target left-hand side DilatedSubtensor.
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
        , bool DF           // Density flag
        , size_t... CSAs    // Compile time DilatedSubtensor arguments
        , typename MT >     // Type of the right-hand side matrix
inline bool tryAddAssign( const DilatedSubtensor<TT,DF,CSAs...>& lhs,
                          const Matrix<MT,false>& rhs, size_t row, size_t column, size_t page )
{
   BLAZE_INTERNAL_ASSERT( row <= lhs.rows(), "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( column <= lhs.columns(), "Invalid column access index" );
   BLAZE_INTERNAL_ASSERT( page <= lhs.pages(), "Invalid page access index" );
   BLAZE_INTERNAL_ASSERT( row + (~rhs).rows() <= lhs.rows(), "Invalid number of rows" );
   BLAZE_INTERNAL_ASSERT( column + (~rhs).columns() <= lhs.columns(), "Invalid number of columns" );

   return tryAddAssign( lhs.operand(), ~rhs, lhs.row() + row * lhs.rowdilation(), lhs.column() + column * lhs.columndilation(), lhs.page() + page * lhs.pagedilation() );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Predict invariant violations by the addition assignment of a vector to the band of
//        a DilatedSubtensor.
// \ingroup DilatedSubtensor
//
// \param lhs The target left-hand side DilatedSubtensor.
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
        , bool DF           // Density flag
        , size_t... CSAs    // Compile time DilatedSubtensor arguments
        , typename MT >     // Type of the right-hand side vector
inline bool tryAddAssign( const DilatedSubtensor<TT,DF,CSAs...>& lhs,
                          const Matrix<MT,false>& rhs, ptrdiff_t band, size_t row, size_t column, size_t page )
{
   BLAZE_INTERNAL_ASSERT( row <= lhs.rows(), "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( column <= lhs.columns(), "Invalid column access index" );
   BLAZE_INTERNAL_ASSERT( page <= lhs.pages(), "Invalid page access index" );
   BLAZE_INTERNAL_ASSERT( row + (~rhs).rows() <= lhs.rows(), "Invalid number of rows" );
   BLAZE_INTERNAL_ASSERT( column + (~rhs).columns() <= lhs.columns(), "Invalid number of columns" );

   return tryAddAssign( lhs.operand(), ~rhs, band + ptrdiff_t( lhs.column() - lhs.row() ),
                        lhs.row() + row * lhs.rowdilation(), lhs.column() + column * lhs.columndilation(), lhs.page() + page * lhs.pagedilation() );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Predict invariant violations by the addition assignment of a tensor to a DilatedSubtensor.
// \ingroup DilatedSubtensor
//
// \param lhs The target left-hand side DilatedSubtensor.
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
        , bool DF           // Density flag
        , size_t... CSAs    // Compile time DilatedSubtensor arguments
        , typename TT2 >    // Type of the right-hand side tensor
inline bool tryAddAssign( const DilatedSubtensor<TT1,DF,CSAs...>& lhs,
                          const Tensor<TT2>& rhs, size_t row, size_t column, size_t page )
{
   BLAZE_INTERNAL_ASSERT( row <= lhs.rows(), "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( column <= lhs.columns(), "Invalid column access index" );
   BLAZE_INTERNAL_ASSERT( page <= lhs.pages(), "Invalid page access index" );
   BLAZE_INTERNAL_ASSERT( row + (~rhs).rows() <= lhs.rows(), "Invalid number of rows" );
   BLAZE_INTERNAL_ASSERT( column + (~rhs).columns() <= lhs.columns(), "Invalid number of columns" );
   BLAZE_INTERNAL_ASSERT( page + (~rhs).pages() <= lhs.pages(), "Invalid number of pages" );

   return tryAddAssign( lhs.operand(), ~rhs, lhs.row() + row * lhs.rowdilation(), lhs.column() + column * lhs.columndilation(), lhs.page() + page * lhs.pagedilation() );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Predict invariant violations by the subtraction assignment of a vector to a DilatedSubtensor.
// \ingroup DilatedSubtensor
//
// \param lhs The target left-hand side DilatedSubtensor.
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
        , bool DF           // Density flag
        , size_t... CSAs    // Compile time DilatedSubtensor arguments
        , typename MT >     // Type of the right-hand side vector
inline bool trySubAssign( const DilatedSubtensor<TT,DF,CSAs...>& lhs,
                          const Matrix<MT,false>& rhs, size_t row, size_t column, size_t page )
{
   BLAZE_INTERNAL_ASSERT( row <= lhs.rows(), "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( column <= lhs.columns(), "Invalid column access index" );
   BLAZE_INTERNAL_ASSERT( page <= lhs.pages(), "Invalid page access index" );
   BLAZE_INTERNAL_ASSERT( row + (~rhs).rows() <= lhs.rows(), "Invalid number of rows" );
   BLAZE_INTERNAL_ASSERT( column + (~rhs).columns() <= lhs.columns(), "Invalid number of columns" );

   return trySubAssign( lhs.operand(), ~rhs, lhs.row() + row * lhs.rowdilation(), lhs.column() + column * lhs.columndilation(), lhs.page() + page * lhs.pagedilation() );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Predict invariant violations by the subtraction assignment of a vector to the band of
//        a DilatedSubtensor.
// \ingroup DilatedSubtensor
//
// \param lhs The target left-hand side DilatedSubtensor.
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
        , bool DF           // Density flag
        , size_t... CSAs    // Compile time DilatedSubtensor arguments
        , typename MT >     // Type of the right-hand side vector
inline bool trySubAssign( const DilatedSubtensor<TT,DF,CSAs...>& lhs,
                          const Matrix<MT,false>& rhs, ptrdiff_t band, size_t row, size_t column, size_t page )
{
   BLAZE_INTERNAL_ASSERT( row <= lhs.rows(), "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( column <= lhs.columns(), "Invalid column access index" );
   BLAZE_INTERNAL_ASSERT( page <= lhs.pages(), "Invalid page access index" );
   BLAZE_INTERNAL_ASSERT( row + (~rhs).rows() <= lhs.rows(), "Invalid number of rows" );
   BLAZE_INTERNAL_ASSERT( column + (~rhs).columns() <= lhs.columns(), "Invalid number of columns" );

   return trySubAssign( lhs.operand(), ~rhs, band + ptrdiff_t( lhs.column() - lhs.row() ),
                        lhs.row() + row * lhs.rowdilation(), lhs.column() + column * lhs.columndilation(), lhs.page() + page * lhs.pagedilation() );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Predict invariant violations by the subtraction assignment of a tensor to a DilatedSubtensor.
// \ingroup DilatedSubtensor
//
// \param lhs The target left-hand side DilatedSubtensor.
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
        , bool DF           // Density flag
        , size_t... CSAs    // Compile time DilatedSubtensor arguments
        , typename TT2 >    // Type of the right-hand side tensor
inline bool trySubAssign( const DilatedSubtensor<TT1,DF,CSAs...>& lhs,
                          const Tensor<TT2>& rhs, size_t row, size_t column, size_t page )
{
   BLAZE_INTERNAL_ASSERT( row <= lhs.rows(), "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( column <= lhs.columns(), "Invalid column access index" );
   BLAZE_INTERNAL_ASSERT( page <= lhs.pages(), "Invalid page access index" );
   BLAZE_INTERNAL_ASSERT( row + (~rhs).rows() <= lhs.rows(), "Invalid number of rows" );
   BLAZE_INTERNAL_ASSERT( column + (~rhs).columns() <= lhs.columns(), "Invalid number of columns" );
   BLAZE_INTERNAL_ASSERT( page + (~rhs).pages() <= lhs.pages(), "Invalid number of pages" );

   return trySubAssign( lhs.operand(), ~rhs, lhs.row() + row * lhs.rowdilation(), lhs.column() + column * lhs.columndilation(), lhs.page() + page * lhs.pagedilation() );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Predict invariant violations by the multiplication assignment of a vector to a DilatedSubtensor.
// \ingroup DilatedSubtensor
//
// \param lhs The target left-hand side DilatedSubtensor.
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
//template< typename TT       // Type of the tensor
//        , bool SO           // Storage order
//        , bool DF           // Density flag
//        , size_t... CSAs    // Compile time DilatedSubtensor arguments
//        , typename VT       // Type of the right-hand side vector
//        , bool TF >         // Transpose flag of the right-hand side vector
//inline bool tryMultAssign( const DilatedSubtensor<TT,DF,CSAs...>& lhs,
//                           const Vector<VT,TF>& rhs, size_t row, size_t column )
//{
//   BLAZE_INTERNAL_ASSERT( row <= lhs.rows(), "Invalid row access index" );
//   BLAZE_INTERNAL_ASSERT( column <= lhs.columns(), "Invalid column access index" );
//   BLAZE_INTERNAL_ASSERT( TF || ( row + (~rhs).size() <= lhs.rows() ), "Invalid number of rows" );
//   BLAZE_INTERNAL_ASSERT( !TF || ( column + (~rhs).size() <= lhs.columns() ), "Invalid number of columns" );
//
//   return tryMultAssign( lhs.operand(), ~rhs, lhs.row() + row * lhs.rowdilation(), lhs.column() + column * lhs.columndilation() );
//}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Predict invariant violations by the multiplication assignment of a vector to the band
//        of a DilatedSubtensor.
// \ingroup DilatedSubtensor
//
// \param lhs The target left-hand side DilatedSubtensor.
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
//template< typename TT       // Type of the tensor
//        , bool SO           // Storage order
//        , bool DF           // Density flag
//        , size_t... CSAs    // Compile time DilatedSubtensor arguments
//        , typename VT       // Type of the right-hand side vector
//        , bool TF >         // Transpose flag of the right-hand side vector
//inline bool tryMultAssign( const DilatedSubtensor<TT,DF,CSAs...>& lhs,
//                           const Vector<VT,TF>& rhs, ptrdiff_t band, size_t row, size_t column )
//{
//   BLAZE_INTERNAL_ASSERT( row <= lhs.rows(), "Invalid row access index" );
//   BLAZE_INTERNAL_ASSERT( column <= lhs.columns(), "Invalid column access index" );
//   BLAZE_INTERNAL_ASSERT( row + (~rhs).size() <= lhs.rows(), "Invalid number of rows" );
//   BLAZE_INTERNAL_ASSERT( column + (~rhs).size() <= lhs.columns(), "Invalid number of columns" );
//
//   return tryMultAssign( lhs.operand(), ~rhs, band + ptrdiff_t( lhs.column() - lhs.row() ),
//                         lhs.row() + row * lhs.rowdilation(), lhs.column() + column * lhs.columndilation() );
//}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Predict invariant violations by the Schur product assignment of a tensor to a DilatedSubtensor.
// \ingroup DilatedSubtensor
//
// \param lhs The target left-hand side DilatedSubtensor.
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
//template< typename TT1      // Type of the tensor
//        , bool SO1          // Storage order
//        , bool DF           // Density flag
//        , size_t... CSAs    // Compile time DilatedSubtensor arguments
//        , typename TT2      // Type of the right-hand side tensor
//        , bool SO2 >        // Storage order of the right-hand side tensor
//inline bool trySchurAssign( const DilatedSubtensor<TT1,SO1,DF,CSAs...>& lhs,
//                            const Tensor<TT2,SO2>& rhs, size_t row, size_t column )
//{
//   BLAZE_INTERNAL_ASSERT( row <= lhs.rows(), "Invalid row access index" );
//   BLAZE_INTERNAL_ASSERT( column <= lhs.columns(), "Invalid column access index" );
//   BLAZE_INTERNAL_ASSERT( row + (~rhs).rows() <= lhs.rows(), "Invalid number of rows" );
//   BLAZE_INTERNAL_ASSERT( column + (~rhs).columns() <= lhs.columns(), "Invalid number of columns" );
//
//   return trySchurAssign( lhs.operand(), ~rhs, lhs.row() + row * lhs.rowdilation(), lhs.column() + column * lhs.columndilation() );
//}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Predict invariant violations by the division assignment of a vector to a DilatedSubtensor.
// \ingroup DilatedSubtensor
//
// \param lhs The target left-hand side DilatedSubtensor.
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
//template< typename TT       // Type of the tensor
//        , bool SO           // Storage order
//        , bool DF           // Density flag
//        , size_t... CSAs    // Compile time DilatedSubtensor arguments
//        , typename VT       // Type of the right-hand side vector
//        , bool TF >         // Transpose flag of the right-hand side vector
//inline bool tryDivAssign( const DilatedSubtensor<TT,DF,CSAs...>& lhs,
//                          const Vector<VT,TF>& rhs, size_t row, size_t column )
//{
//   BLAZE_INTERNAL_ASSERT( row <= lhs.rows(), "Invalid row access index" );
//   BLAZE_INTERNAL_ASSERT( column <= lhs.columns(), "Invalid column access index" );
//   BLAZE_INTERNAL_ASSERT( TF || ( row + (~rhs).size() <= lhs.rows() ), "Invalid number of rows" );
//   BLAZE_INTERNAL_ASSERT( !TF || ( column + (~rhs).size() <= lhs.columns() ), "Invalid number of columns" );
//
//   return tryDivAssign( lhs.operand(), ~rhs, lhs.row() + row * lhs.rowdilation(), lhs.column() + column * lhs.columndilation() );
//}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Predict invariant violations by the division assignment of a vector to the band of
//        a DilatedSubtensor.
// \ingroup DilatedSubtensor
//
// \param lhs The target left-hand side DilatedSubtensor.
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
//template< typename TT       // Type of the tensor
//        , bool SO           // Storage order
//        , bool DF           // Density flag
//        , size_t... CSAs    // Compile time DilatedSubtensor arguments
//        , typename VT       // Type of the right-hand side vector
//        , bool TF >         // Transpose flag of the right-hand side vector
//inline bool tryDivAssign( const DilatedSubtensor<TT,DF,CSAs...>& lhs,
//                          const Vector<VT,TF>& rhs, ptrdiff_t band, size_t row, size_t column )
//{
//   BLAZE_INTERNAL_ASSERT( row <= lhs.rows(), "Invalid row access index" );
//   BLAZE_INTERNAL_ASSERT( column <= lhs.columns(), "Invalid column access index" );
//   BLAZE_INTERNAL_ASSERT( row + (~rhs).size() <= lhs.rows(), "Invalid number of rows" );
//   BLAZE_INTERNAL_ASSERT( column + (~rhs).size() <= lhs.columns(), "Invalid number of columns" );
//
//   return tryDivAssign( lhs.operand(), ~rhs, band + ptrdiff_t( lhs.column() - lhs.row() ),
//                        lhs.row() + row * lhs.rowdilation(), lhs.column() + column * lhs.columndilation() );
//}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Removal of all restrictions on the data access to the given DilatedSubtensor.
// \ingroup DilatedSubtensor
//
// \param dm The DilatedSubtensor to be derestricted.
// \return DilatedSubtensor without access restrictions.
//
// This function removes all restrictions on the data access to the given DilatedSubtensor. It returns a
// DilatedSubtensor that does provide the same interface but does not have any restrictions on the data
// access.\n
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in the violation of invariants, erroneous results and/or in compilation errors.
*/
template< typename TT       // Type of the tensor
        , bool DF           // Density flag
        , size_t K          // Index of the first page
        , size_t I          // Index of the first row
        , size_t J          // Index of the first column
        , size_t O          // Number of pages
        , size_t M          // Number of rows
        , size_t N          // Number of columns
        , size_t PageDilation    // Step between pages of the DilatedSubtensor
        , size_t RowDilation     // Step between rows of the DilatedSubtensor
        , size_t ColumnDilation >// Step between columns of the DilatedSubtensor
inline decltype(auto) derestrict( DilatedSubtensor<TT,DF,K,I,J,O,M,N,PageDilation,RowDilation,ColumnDilation>& dm )
{
   return dilatedsubtensor<K,I,J,O,M,N,PageDilation,RowDilation,ColumnDilation>( derestrict( dm.operand() ), unchecked );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Removal of all restrictions on the data access to the given temporary DilatedSubtensor.
// \ingroup DilatedSubtensor
//
// \param dm The temporary DilatedSubtensor to be derestricted.
// \return DilatedSubtensor without access restrictions.
//
// This function removes all restrictions on the data access to the given temporary DilatedSubtensor. It
// returns a DilatedSubtensor that does provide the same interface but does not have any restrictions on
// the data access.\n
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in the violation of invariants, erroneous results and/or in compilation errors.
*/
template< typename TT       // Type of the tensor
        , bool DF           // Density flag
        , size_t K          // Index of the first page
        , size_t I          // Index of the first row
        , size_t J          // Index of the first column
        , size_t O          // Number of pages
        , size_t M          // Number of rows
        , size_t N          // Number of columns
        , size_t PageDilation    // Step between pages of the DilatedSubtensor
        , size_t RowDilation     // Step between rows of the DilatedSubtensor
        , size_t ColumnDilation >// Step between columns of the DilatedSubtensor
inline decltype(auto) derestrict( DilatedSubtensor<TT,DF,K,I,J,O,M,N,PageDilation,RowDilation,ColumnDilation>&& dm )
{
   return dilatedsubtensor<K,I,J,O,M,N,PageDilation,RowDilation,ColumnDilation>( derestrict( dm.operand() ), unchecked );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Removal of all restrictions on the data access to the given DilatedSubtensor.
// \ingroup DilatedSubtensor
//
// \param dm The DilatedSubtensor to be derestricted.
// \return DilatedSubtensor without access restrictions.
//
// This function removes all restrictions on the data access to the given DilatedSubtensor. It returns a
// DilatedSubtensor that does provide the same interface but does not have any restrictions on the data
// access.\n
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in the violation of invariants, erroneous results and/or in compilation errors.
*/
template< typename TT       // Type of the tensor
        , bool DF >         // Density flag
inline decltype(auto) derestrict( DilatedSubtensor<TT,DF>& dm )
{
   return dilatedsubtensor( derestrict( dm.operand() ), dm.page(), dm.row(), dm.column(), dm.pages(), dm.rows(), dm.columns(),
      dm.pagedilation(), dm.rowdilation(), dm.columndilation(), unchecked );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Removal of all restrictions on the data access to the given temporary DilatedSubtensor.
// \ingroup DilatedSubtensor
//
// \param dm The temporary DilatedSubtensor to be derestricted.
// \return DilatedSubtensor without access restrictions.
//
// This function removes all restrictions on the data access to the given temporary DilatedSubtensor. It
// returns a DilatedSubtensor that does provide the same interface but does not have any restrictions on
// the data access.\n
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in the violation of invariants, erroneous results and/or in compilation errors.
*/
template< typename TT       // Type of the tensor
        , bool DF >         // Density flag
inline decltype(auto) derestrict( DilatedSubtensor<TT,DF>&& dm )
{
   return dilatedsubtensor( derestrict( dm.operand() ), dm.page(), dm.row(), dm.column(), dm.pages(), dm.rows(), dm.columns(),
      dm.pagedilation(), dm.rowdilation(), dm.columndilation(), unchecked );
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
template< typename TT, bool DF, size_t K, size_t I, size_t J, size_t O, size_t M, size_t N, size_t PageDilation, size_t RowDilation, size_t ColumnDilation >
struct Size< DilatedSubtensor<TT,DF,K,I,J,O,M,N,PageDilation,RowDilation,ColumnDilation>, 0UL >
   : public Ptrdiff_t<M>
{};

template< typename TT, bool DF, size_t K, size_t I, size_t J, size_t O, size_t M, size_t N, size_t PageDilation, size_t RowDilation, size_t ColumnDilation >
struct Size< DilatedSubtensor<TT,DF,K,I,J,O,M,N,PageDilation,RowDilation,ColumnDilation>, 1UL >
   : public Ptrdiff_t<N>
{};

template< typename TT, bool DF, size_t K, size_t I, size_t J, size_t O, size_t M, size_t N, size_t PageDilation, size_t RowDilation, size_t ColumnDilation >
struct Size< DilatedSubtensor<TT,DF,K,I,J,O,M,N,PageDilation,RowDilation,ColumnDilation>, 2UL >
   : public Ptrdiff_t<O>
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
template< typename TT, bool DF, size_t K, size_t I, size_t J, size_t O, size_t M, size_t N, size_t PageDilation, size_t RowDilation, size_t ColumnDilation >
struct MaxSize< DilatedSubtensor<TT,DF,K,I,J,O,M,N,PageDilation,RowDilation,ColumnDilation>, 0UL >
   : public Ptrdiff_t<M>
{};

template< typename TT, bool DF, size_t K, size_t I, size_t J, size_t O, size_t M, size_t N, size_t PageDilation, size_t RowDilation, size_t ColumnDilation >
struct MaxSize< DilatedSubtensor<TT,DF,K,I,J,O,M,N,PageDilation,RowDilation,ColumnDilation>, 1UL >
   : public Ptrdiff_t<N>
{};

template< typename TT, bool DF, size_t K, size_t I, size_t J, size_t O, size_t M, size_t N, size_t PageDilation, size_t RowDilation, size_t ColumnDilation >
struct MaxSize< DilatedSubtensor<TT,DF,K,I,J,O,M,N,PageDilation,RowDilation,ColumnDilation>, 2UL >
   : public Ptrdiff_t<O>
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
template< typename TT, bool DF, size_t... CSAs >
struct IsRestricted< DilatedSubtensor<TT,DF,CSAs...> >
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
template< typename TT, size_t... CSAs >
struct HasConstDataAccess< DilatedSubtensor<TT,true,CSAs...> >
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
template< typename TT, size_t... CSAs >
struct HasMutableDataAccess< DilatedSubtensor<TT,true,CSAs...> >
   : public HasMutableDataAccess<TT>
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
template< typename TT, bool DF, size_t K, size_t I, size_t J, size_t O, size_t M, size_t N, size_t PageDilation, size_t RowDilation, size_t ColumnDilation >
struct IsSymmetric< DilatedSubtensor<TT,DF,K,I,J,O,M,N,PageDilation,RowDilation,ColumnDilation> >
   : public BoolConstant< ( IsSymmetric_v<TT> && K == I && I == J && O == M && M == N && PageDilation == RowDilation && RowDilation == ColumnDilation) >
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
template< typename TT, bool DF, size_t K, size_t I, size_t J, size_t O, size_t M, size_t N, size_t PageDilation, size_t RowDilation, size_t ColumnDilation >
struct IsHermitian< DilatedSubtensor<TT,DF,K,I,J,O,M,N,PageDilation,RowDilation,ColumnDilation> >
   : public BoolConstant< ( IsHermitian_v<TT> && K == I && I == J && O == M && M == N && PageDilation == RowDilation && RowDilation == ColumnDilation) >
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
template< typename TT, bool DF, size_t K, size_t I, size_t J, size_t O, size_t M, size_t N, size_t PageDilation, size_t RowDilation, size_t ColumnDilation >
struct IsLower< DilatedSubtensor<TT,DF,K,I,J,O,M,N,PageDilation,RowDilation,ColumnDilation> >
   : public BoolConstant< ( IsLower_v<TT> && I == J && M == N && RowDilation == ColumnDilation ) ||
                          ( IsStrictlyLower_v<TT> && I == J+1UL && M == N && RowDilation == ColumnDilation) >
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
template< typename TT, bool DF,size_t K, size_t I, size_t J, size_t O, size_t M, size_t N, size_t PageDilation, size_t RowDilation, size_t ColumnDilation >
struct IsUniLower< DilatedSubtensor<TT,DF,K,I,J,O,M,N,PageDilation,RowDilation,ColumnDilation> >
   : public BoolConstant< ( IsUniLower_v<TT> && I == J && M == N && RowDilation == ColumnDilation) >
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
template< typename TT, bool DF, size_t K, size_t I, size_t J, size_t O, size_t M, size_t N, size_t PageDilation, size_t RowDilation, size_t ColumnDilation >
struct IsStrictlyLower< DilatedSubtensor<TT,DF,K,I,J,O,M,N,PageDilation,RowDilation,ColumnDilation> >
   : public BoolConstant< ( IsLower_v<TT> && I < J && M == N && RowDilation == ColumnDilation) ||
                          ( IsStrictlyLower_v<TT> && I == J && M == N && RowDilation == ColumnDilation ) >
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
template< typename TT, bool DF, size_t K, size_t I, size_t J, size_t O, size_t M, size_t N, size_t PageDilation, size_t RowDilation, size_t ColumnDilation >
struct IsUpper< DilatedSubtensor<TT,DF,K,I,J,O,M,N,PageDilation,RowDilation,ColumnDilation> >
   : public BoolConstant< ( IsUpper_v<TT> && I == J && M == N && RowDilation == ColumnDilation) ||
                          ( IsStrictlyUpper_v<TT> && I+1UL == J && M == N && RowDilation == ColumnDilation ) >
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
template< typename TT, bool DF, size_t K, size_t I, size_t J, size_t O, size_t M, size_t N, size_t PageDilation, size_t RowDilation, size_t ColumnDilation >
struct IsUniUpper< DilatedSubtensor<TT,DF,K,I,J,O,M,N,PageDilation,RowDilation,ColumnDilation> >
   : public BoolConstant< ( IsUniUpper_v<TT> && I == J && M == N && RowDilation == ColumnDilation ) >
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
template< typename TT, bool DF, size_t K, size_t I, size_t J, size_t O, size_t M, size_t N, size_t PageDilation, size_t RowDilation, size_t ColumnDilation>
struct IsStrictlyUpper< DilatedSubtensor<TT,DF,K,I,J,O,M,N,PageDilation,RowDilation,ColumnDilation> >
   : public BoolConstant< ( IsUpper_v<TT> && I > J && M == N && RowDilation == ColumnDilation) ||
                          ( IsStrictlyUpper_v<TT> && I == J && M == N && RowDilation == ColumnDilation) >
{};
/*! \endcond */
//*************************************************************************************************

} // namespace blaze

#endif
