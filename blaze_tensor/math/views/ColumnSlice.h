//=================================================================================================
/*!
//  \file blaze_tensor/math/views/ColumnSlice.h
//  \brief Header file for the implementation of the ColumnSlice view
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

#ifndef _BLAZE_TENSOR_MATH_VIEWS_COLUMNSLICE_H_
#define _BLAZE_TENSOR_MATH_VIEWS_COLUMNSLICE_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <blaze/math/expressions/DVecExpandExpr.h>
#include <blaze/math/expressions/Forward.h>
#include <blaze/math/expressions/SchurExpr.h>
#include <blaze/math/expressions/TransExpr.h>
#include <blaze/math/views/Column.h>
#include <blaze/math/views/Row.h>

#include <blaze_tensor/math/expressions/MatExpandExpr.h>
#include <blaze_tensor/math/expressions/TensEvalExpr.h>
#include <blaze_tensor/math/expressions/TensMapExpr.h>
#include <blaze_tensor/math/expressions/TensReduceExpr.h>
#include <blaze_tensor/math/expressions/TensScalarDivExpr.h>
#include <blaze_tensor/math/expressions/TensScalarMultExpr.h>
#include <blaze_tensor/math/expressions/TensSerialExpr.h>
#include <blaze_tensor/math/expressions/TensTensAddExpr.h>
#include <blaze_tensor/math/expressions/TensTensMapExpr.h>
#include <blaze_tensor/math/expressions/TensTensMultExpr.h>
#include <blaze_tensor/math/expressions/TensTensSubExpr.h>
#include <blaze_tensor/math/expressions/TensTransExpr.h>
#include <blaze_tensor/math/expressions/Tensor.h>
#include <blaze_tensor/math/views/Forward.h>
#include <blaze_tensor/math/views/columnslice/BaseTemplate.h>
#include <blaze_tensor/math/views/columnslice/Dense.h>

namespace blaze {

//=================================================================================================
//
//  GLOBAL FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Creating a view on a specific columnslice of the given tensor.
// \ingroup columnslice
//
// \param tensor The tensor containing the columnslice.
// \param args Optional columnslice arguments.
// \return View on the specified columnslice of the tensor.
// \exception std::invalid_argument Invalid columnslice access index.
//
// This function returns an expression representing the specified columnslice of the given tensor.

   \code
   blaze::DynamicTensor<double> D;
   blaze::CompressedTensor<double> S;
   // ... Resizing and initialization

   // Creating a view on the 3rd columnslice of the dense tensor D
   auto columnslice3 = columnslice<3UL>( D );

   // Creating a view on the 4th columnslice of the sparse tensor S
   auto columnslice4 = columnslice<4UL>( S );
   \page()

// By default, the provided columnslice arguments are checked at runtime. In case the columnslice is not properly
// specified (i.e. if the specified index is greater than or equal to the total number of the columnslices
// in the given tensor) a \a std::invalid_argument exception is thrown. The checks can be skipped
// by providing the optional \a blaze::unchecked argument.

   \code
   auto columnslice3 = columnslice<3UL>( D, unchecked );
   auto columnslice4 = columnslice<4UL>( S, unchecked );
   \page()
*/
template< size_t I            // ColumnSlice index
        , typename MT         // Type of the tensor
        , typename... RRAs >  // Optional columnslice arguments
inline decltype(auto) columnslice( Tensor<MT>& tensor, RRAs... args )
{
   BLAZE_FUNCTION_TRACE;

   using ReturnType = ColumnSlice_<MT,I>;
   return ReturnType( ~tensor, args... );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Creating a view on a specific columnslice of the given constant tensor.
// \ingroup columnslice
//
// \param tensor The constant tensor containing the columnslice.
// \param args Optional columnslice arguments.
// \return View on the specified columnslice of the tensor.
// \exception std::invalid_argument Invalid columnslice access index.
//
// This function returns an expression representing the specified columnslice of the given constant
// tensor.

   \code

   const blaze::DynamicTensor<double> D( ... );
   const blaze::CompressedTensor<double> S( ... );

   // Creating a view on the 3rd columnslice of the dense tensor D
   auto columnslice3 = columnslice<3UL>( D );

   // Creating a view on the 4th columnslice of the sparse tensor S
   auto columnslice4 = columnslice<4UL>( S );
   \page()

// By default, the provided columnslice arguments are checked at runtime. In case the columnslice is not properly
// specified (i.e. if the specified index is greater than or equal to the total number of the columnslices
// in the given tensor) a \a std::invalid_argument exception is thrown. The checks can be skipped
// by providing the optional \a blaze::unchecked argument.

   \code
   auto columnslice3 = columnslice<3UL>( D, unchecked );
   auto columnslice4 = columnslice<4UL>( S, unchecked );
   \page()
*/
template< size_t I            // ColumnSlice index
        , typename MT         // Type of the tensor
        , typename... RRAs >  // Optional columnslice arguments
inline decltype(auto) columnslice( const Tensor<MT>& tensor, RRAs... args )
{
   BLAZE_FUNCTION_TRACE;

   using ReturnType = const ColumnSlice_<const MT,I>;
   return ReturnType( ~tensor, args... );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Creating a view on a specific columnslice of the given temporary tensor.
// \ingroup columnslice
//
// \param tensor The temporary tensor containing the columnslice.
// \param args Optional columnslice arguments.
// \return View on the specified columnslice of the tensor.
// \exception std::invalid_argument Invalid columnslice access index.
//
// This function returns an expression representing the specified columnslice of the given temporary
// tensor. In case the columnslice is not properly specified (i.e. if the specified index is greater
// than or equal to the total number of the columnslices in the given tensor) a \a std::invalid_argument
// exception is thrown.
*/
template< size_t I            // ColumnSlice index
        , typename MT         // Type of the tensor
        , typename... RRAs >  // Optional columnslice arguments
inline decltype(auto) columnslice( Tensor<MT>&& tensor, RRAs... args )
{
   BLAZE_FUNCTION_TRACE;

   using ReturnType = ColumnSlice_<MT,I>;
   return ReturnType( ~tensor, args... );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Creating a view on a specific columnslice of the given tensor.
// \ingroup columnslice
//
// \param tensor The tensor containing the columnslice.
// \param index The index of the columnslice.
// \param args Optional columnslice arguments.
// \return View on the specified columnslice of the tensor.
// \exception std::invalid_argument Invalid columnslice access index.
//
// This function returns an expression representing the specified columnslice of the given tensor.

   \code
   blaze::DynamicTensor<double> D;
   blaze::CompressedTensor<double> S;
   // ... Resizing and initialization

   // Creating a view on the 3rd columnslice of the dense tensor D
   auto columnslice3 = columnslice( D, 3UL );

   // Creating a view on the 4th columnslice of the sparse tensor S
   auto columnslice4 = columnslice( S, 4UL );
   \page()

// By default, the provided columnslice arguments are checked at runtime. In case the columnslice is not properly
// specified (i.e. if the specified index is greater than or equal to the total number of the columnslices
// in the given tensor) a \a std::invalid_argument exception is thrown. The checks can be skipped
// by providing the optional \a blaze::unchecked argument.

   \code
   auto columnslice3 = columnslice( D, 3UL, unchecked );
   auto columnslice4 = columnslice( S, 4UL, unchecked );
   \page()
*/
template< typename MT         // Type of the tensor
        , typename... RRAs >  // Optional columnslice arguments
inline decltype(auto) columnslice( Tensor<MT>& tensor, size_t index, RRAs... args )
{
   BLAZE_FUNCTION_TRACE;

   using ReturnType = ColumnSlice_<MT>;
   return ReturnType( ~tensor, index, args... );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Creating a view on a specific columnslice of the given constant tensor.
// \ingroup columnslice
//
// \param tensor The constant tensor containing the columnslice.
// \param index The index of the columnslice.
// \param args Optional columnslice arguments.
// \return View on the specified columnslice of the tensor.
// \exception std::invalid_argument Invalid columnslice access index.
//
// This function returns an expression representing the specified columnslice of the given constant
// tensor.

   \code
   const blaze::DynamicTensor<double> D( ... );
   const blaze::CompressedTensor<double> S( ... );

   // Creating a view on the 3rd columnslice of the dense tensor D
   auto columnslice3 = columnslice( D, 3UL );

   // Creating a view on the 4th columnslice of the sparse tensor S
   auto columnslice4 = columnslice( S, 4UL );
   \page()

// By default, the provided columnslice arguments are checked at runtime. In case the columnslice is not properly
// specified (i.e. if the specified index is greater than or equal to the total number of the columnslices
// in the given tensor) a \a std::invalid_argument exception is thrown. The checks can be skipped
// by providing the optional \a blaze::unchecked argument.

   \code
   auto columnslice3 = columnslice( D, 3UL, unchecked );
   auto columnslice4 = columnslice( S, 4UL, unchecked );
   \page()
*/
template< typename MT         // Type of the tensor
        , typename... RRAs >  // Optional columnslice arguments
inline decltype(auto) columnslice( const Tensor<MT>& tensor, size_t index, RRAs... args )
{
   BLAZE_FUNCTION_TRACE;

   using ReturnType = const ColumnSlice_<const MT>;
   return ReturnType( ~tensor, index, args... );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Creating a view on a specific columnslice of the given temporary tensor.
// \ingroup columnslice
//
// \param tensor The temporary tensor containing the columnslice.
// \param index The index of the columnslice.
// \param args Optional columnslice arguments.
// \return View on the specified columnslice of the tensor.
// \exception std::invalid_argument Invalid columnslice access index.
//
// This function returns an expression representing the specified columnslice of the given temporary
// tensor. In case the columnslice is not properly specified (i.e. if the specified index is greater
// than or equal to the total number of the columnslices in the given tensor) a \a std::invalid_argument
// exception is thrown.
*/
template< typename MT         // Type of the tensor
        , typename... RRAs >  // Optional columnslice arguments
inline decltype(auto) columnslice( Tensor<MT>&& tensor, size_t index, RRAs... args )
{
   BLAZE_FUNCTION_TRACE;

   using ReturnType = ColumnSlice_<MT>;
   return ReturnType( ~tensor, index, args... );
}
//*************************************************************************************************




//=================================================================================================
//
//  GLOBAL RESTRUCTURING FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific columnslice of the given tensor/tensor addition.
// \ingroup columnslice
//
// \param tensor The constant tensor/tensor addition.
// \param args The runtime columnslice arguments.
// \return View on the specified columnslice of the addition.
//
// This function returns an expression representing the specified columnslice of the given tensor/tensor
// addition.
*/
template< size_t... CRAs      // Compile time columnslice arguments
        , typename MT         // Tensor base type of the expression
        , typename... RRAs >  // Runtime columnslice arguments
inline decltype(auto) columnslice( const TensTensAddExpr<MT>& tensor, RRAs... args )
{
   BLAZE_FUNCTION_TRACE;

   return columnslice<CRAs...>( (~tensor).leftOperand(), args... ) +
          columnslice<CRAs...>( (~tensor).rightOperand(), args... );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific columnslice of the given tensor/tensor subtraction.
// \ingroup columnslice
//
// \param tensor The constant tensor/tensor subtraction.
// \param args The runtime columnslice arguments.
// \return View on the specified columnslice of the subtraction.
//
// This function returns an expression representing the specified columnslice of the given tensor/tensor
// subtraction.
*/
template< size_t... CRAs      // Compile time columnslice arguments
        , typename MT         // Tensor base type of the expression
        , typename... RRAs >  // Runtime columnslice arguments
inline decltype(auto) columnslice( const TensTensSubExpr<MT>& tensor, RRAs... args )
{
   BLAZE_FUNCTION_TRACE;

   return columnslice<CRAs...>( (~tensor).leftOperand(), args... ) -
          columnslice<CRAs...>( (~tensor).rightOperand(), args... );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific columnslice of the given Schur product.
// \ingroup columnslice
//
// \param tensor The constant Schur product.
// \param args The runtime columnslice arguments.
// \return View on the specified columnslice of the Schur product.
//
// This function returns an expression representing the specified columnslice of the given Schur product.
*/
template< size_t... CRAs      // Compile time columnslice arguments
        , typename MT         // Tensor base type of the expression
        , typename... RRAs >  // Runtime columnslice arguments
inline decltype(auto) columnslice( const SchurExpr<MT>& tensor, RRAs... args )
{
   BLAZE_FUNCTION_TRACE;

   return columnslice<CRAs...>( (~tensor).leftOperand(), args... ) *
          columnslice<CRAs...>( (~tensor).rightOperand(), args... );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific columnslice of the given tensor/tensor multiplication.
// \ingroup columnslice
//
// \param tensor The constant tensor/tensor multiplication.
// \param args The runtime columnslice arguments
// \return View on the specified columnslice of the multiplication.
//
// This function returns an expression representing the specified columnslice of the given tensor/tensor
// multiplication.
*/
template< size_t... CRAs      // Compile time columnslice arguments
        , typename MT         // Tensor base type of the expression
        , typename... RRAs >  // Runtime columnslice arguments
inline decltype(auto) columnslice( const TensTensMultExpr<MT>& tensor, RRAs... args )
{
   BLAZE_FUNCTION_TRACE;

   return columnslice<CRAs...>( (~tensor).leftOperand(), args... ) * (~tensor).rightOperand();
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific columnslice of the given outer product.
// \ingroup columnslice
//
// \param tensor The constant outer product.
// \param args Optional columnslice arguments.
// \return View on the specified columnslice of the outer product.
// \exception std::invalid_argument Invalid columnslice access index.
//
// This function returns an expression representing the specified columnslice of the given outer product.
*/
// template< size_t I            // ColumnSlice index
//         , typename MT         // Tensor base type of the expression
//         , typename... RRAs >  // Optional columnslice arguments
// inline decltype(auto) columnslice( const VecTVecMultExpr<MT>& tensor, RRAs... args )
// {
//    BLAZE_FUNCTION_TRACE;
//
//    UNUSED_PARAMETER( args... );
//
//    if( !Contains_v< TypeList<RRAs...>, Unchecked > ) {
//       if( (~tensor).columnslices() <= I ) {
//          BLAZE_THCOLUMNSLICE_INVALID_ARGUMENT( "Invalid columnslice access index" );
//       }
//    }
//
//    return (~tensor).leftOperand()[I] * (~tensor).rightOperand();
// }
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific columnslice of the given outer product.
// \ingroup columnslice
//
// \param tensor The constant outer product.
// \param index The index of the columnslice.
// \param args Optional columnslice arguments.
// \return View on the specified columnslice of the outer product.
// \exception std::invalid_argument Invalid columnslice access index.
//
// This function returns an expression representing the specified columnslice of the given outer product.
*/
// template< typename MT         // Tensor base type of the expression
//         , typename... RRAs >  // Optional columnslice arguments
// inline decltype(auto) columnslice( const VecTVecMultExpr<MT>& tensor, size_t index, RRAs... args )
// {
//    BLAZE_FUNCTION_TRACE;
//
//    UNUSED_PARAMETER( args... );
//
//    if( !Contains_v< TypeList<RRAs...>, Unchecked > ) {
//       if( (~tensor).columnslices() <= index ) {
//          BLAZE_THCOLUMNSLICE_INVALID_ARGUMENT( "Invalid columnslice access index" );
//       }
//    }
//
//    return (~tensor).leftOperand()[index] * (~tensor).rightOperand();
// }
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific columnslice of the given tensor/scalar multiplication.
// \ingroup columnslice
//
// \param tensor The constant tensor/scalar multiplication.
// \param args The runtime columnslice arguments
// \return View on the specified columnslice of the multiplication.
//
// This function returns an expression representing the specified columnslice of the given tensor/scalar
// multiplication.
*/
template< size_t... CRAs      // Compile time columnslice arguments
        , typename MT         // Tensor base type of the expression
        , typename... RRAs >  // Runtime columnslice arguments
inline decltype(auto) columnslice( const TensScalarMultExpr<MT>& tensor, RRAs... args )
{
   BLAZE_FUNCTION_TRACE;

   return columnslice<CRAs...>( (~tensor).leftOperand(), args... ) * (~tensor).rightOperand();
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific columnslice of the given tensor/scalar division.
// \ingroup columnslice
//
// \param tensor The constant tensor/scalar division.
// \param args The runtime columnslice arguments
// \return View on the specified columnslice of the division.
//
// This function returns an expression representing the specified columnslice of the given tensor/scalar
// division.
*/
template< size_t... CRAs      // Compile time columnslice arguments
        , typename MT         // Tensor base type of the expression
        , typename... RRAs >  // Runtime columnslice arguments
inline decltype(auto) columnslice( const TensScalarDivExpr<MT>& tensor, RRAs... args )
{
   BLAZE_FUNCTION_TRACE;

   return columnslice<CRAs...>( (~tensor).leftOperand(), args... ) / (~tensor).rightOperand();
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific columnslice of the given unary tensor map operation.
// \ingroup columnslice
//
// \param tensor The constant unary tensor map operation.
// \param args The runtime columnslice arguments
// \return View on the specified columnslice of the unary map operation.
//
// This function returns an expression representing the specified columnslice of the given unary tensor
// map operation.
*/
template< size_t... CRAs      // Compile time columnslice arguments
        , typename MT         // Tensor base type of the expression
        , typename... RRAs >  // Runtime columnslice arguments
inline decltype(auto) columnslice( const TensMapExpr<MT>& tensor, RRAs... args )
{
   BLAZE_FUNCTION_TRACE;

   return map( columnslice<CRAs...>( (~tensor).operand(), args... ), (~tensor).operation() );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific columnslice of the given binary tensor map operation.
// \ingroup columnslice
//
// \param tensor The constant binary tensor map operation.
// \param args The runtime columnslice arguments
// \return View on the specified columnslice of the binary map operation.
//
// This function returns an expression representing the specified columnslice of the given binary tensor
// map operation.
*/
template< size_t... CRAs      // Compile time columnslice arguments
        , typename MT         // Tensor base type of the expression
        , typename... RRAs >  // Runtime columnslice arguments
inline decltype(auto) columnslice( const TensTensMapExpr<MT>& tensor, RRAs... args )
{
   BLAZE_FUNCTION_TRACE;

   return map( columnslice<CRAs...>( (~tensor).leftOperand(), args... ),
               columnslice<CRAs...>( (~tensor).rightOperand(), args... ),
               (~tensor).operation() );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific columnslice of the given tensor evaluation operation.
// \ingroup columnslice
//
// \param tensor The constant tensor evaluation operation.
// \param args The runtime columnslice arguments
// \return View on the specified columnslice of the evaluation operation.
//
// This function returns an expression representing the specified columnslice of the given tensor
// evaluation operation.
*/
template< size_t... CRAs      // Compile time columnslice arguments
        , typename MT         // Tensor base type of the expression
        , typename... RRAs >  // Runtime columnslice arguments
inline decltype(auto) columnslice( const TensEvalExpr<MT>& tensor, RRAs... args )
{
   BLAZE_FUNCTION_TRACE;

   return eval( columnslice<CRAs...>( (~tensor).operand(), args... ) );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific columnslice of the given tensor serialization operation.
// \ingroup columnslice
//
// \param tensor The constant tensor serialization operation.
// \param args The runtime columnslice arguments
// \return View on the specified columnslice of the serialization operation.
//
// This function returns an expression representing the specified columnslice of the given tensor
// serialization operation.
*/
template< size_t... CRAs      // Compile time columnslice arguments
        , typename MT         // Tensor base type of the expression
        , typename... RRAs >  // Runtime columnslice arguments
inline decltype(auto) columnslice( const TensSerialExpr<MT>& tensor, RRAs... args )
{
   BLAZE_FUNCTION_TRACE;

   return serial( columnslice<CRAs...>( (~tensor).operand(), args... ) );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific columnslice of the given tensor declaration operation.
// \ingroup columnslice
//
// \param tensor The constant tensor declaration operation.
// \param args The runtime columnslice arguments
// \return View on the specified columnslice of the declaration operation.
//
// This function returns an expression representing the specified columnslice of the given tensor
// declaration operation.
*/
template< size_t... CRAs      // Compile time columnslice arguments
        , typename MT         // Tensor base type of the expression
        , typename... RRAs >  // Runtime columnslice arguments
inline decltype(auto) columnslice( const DeclExpr<MT>& tensor, RRAs... args )
{
   BLAZE_FUNCTION_TRACE;

   return columnslice<CRAs...>( (~tensor).operand(), args... );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific columnslice of the given tensor transpose operation.
// \ingroup columnslice
//
// \param tensor The constant tensor transpose operation.
// \param args The runtime columnslice arguments
// \return View on the specified columnslice of the transpose operation.
//
// This function returns an expression representing the specified columnslice of the given tensor
// transpose operation.
*/
template< size_t MK           // Compile time pageslice arguments
        , size_t MI
        , size_t MJ
        , typename MT         // Tensor base type of the expression
        , typename... RTAs >  // Runtime columnslice arguments
inline decltype(auto) columnslice( const TensTransExpr<MT>& tensor, size_t index, RTAs... args )
{
   BLAZE_FUNCTION_TRACE;

   return columnslice<MK, MI, MJ>( evaluate( ~tensor ), index, args... );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific columnslice of the given tensor transpose operation.
// \ingroup columnslice
//
// \param tensor The constant tensor transpose operation.
// \param args The runtime columnslice arguments
// \return View on the specified columnslice of the transpose operation.
//
// This function returns an expression representing the specified columnslice of the given tensor
// transpose operation.
*/
template< typename MT         // Tensor base type of the expression
        , typename... RTAs >  // Runtime columnslice arguments
inline decltype(auto) columnslice( const TensTransExpr<MT>& tensor, size_t index, RTAs... args )
{
   BLAZE_FUNCTION_TRACE;

   return columnslice( evaluate( ~tensor ), index, args... );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific columnslice of the given matrix expansion operation.
// \ingroup subtensor
//
// \param tensor The constant matrix expansion operation.
// \param args Optional pageslice arguments.
// \return View on the specified columnslice of the expansion operation.
//
// This function returns an expression representing the specified columnslice of the given matrix
// expansion operation.
*/
template< size_t... CCAs      // Compile time columnslice arguments
        , typename TT         // Matrix base type of the expression
        , size_t... CEAs      // Compile time expansion arguments
        , typename... CSAs >  // Runtime pageslice arguments
inline decltype(auto) columnslice( const MatExpandExpr<TT,CEAs...>& tensor, CSAs... args )
{
   BLAZE_FUNCTION_TRACE;

   UNUSED_PARAMETER( args... );

   return expand( trans( column( (~tensor).operand(), 0UL ) ), (~tensor).expansion() );
}
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  COLUMNSLICE OPERATORS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Resetting the given columnslice.
// \ingroup columnslice
//
// \param columnslice The columnslice to be resetted.
// \return void
*/
template< typename MT       // Type of the tensor
                   // Density flag
                   // Symmetry flag
        , size_t... CRAs >  // Compile time columnslice arguments
inline void reset( ColumnSlice<MT,CRAs...>& columnslice )
{
   columnslice.reset();
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Resetting the given temporary columnslice.
// \ingroup columnslice
//
// \param columnslice The temporary columnslice to be resetted.
// \return void
*/
template< typename MT       // Type of the tensor
                   // Density flag
                   // Symmetry flag
        , size_t... CRAs >  // Compile time columnslice arguments
inline void reset( ColumnSlice<MT,CRAs...>&& columnslice )
{
   columnslice.reset();
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Clearing the given columnslice.
// \ingroup columnslice
//
// \param columnslice The columnslice to be cleared.
// \return void
//
// Clearing a columnslice is equivalent to resetting it via the reset() function.
*/
template< typename MT       // Type of the tensor
                   // Density flag
                   // Symmetry flag
        , size_t... CRAs >  // Compile time columnslice arguments
inline void clear( ColumnSlice<MT,CRAs...>& columnslice )
{
   columnslice.reset();
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Clearing the given temporary columnslice.
// \ingroup columnslice
//
// \param columnslice The temporary columnslice to be cleared.
// \return void
//
// Clearing a columnslice is equivalent to resetting it via the reset() function.
*/
template< typename MT       // Type of the tensor
                   // Density flag
                   // Symmetry flag
        , size_t... CRAs >  // Compile time columnslice arguments
inline void clear( ColumnSlice<MT,CRAs...>&& columnslice )
{
   columnslice.reset();
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns whether the given dense columnslice is in default state.
// \ingroup columnslice
//
// \param columnslice The dense columnslice to be tested for its default state.
// \return \a true in case the given dense columnslice is component-wise zero, \a false otherwise.
//
// This function checks whether the dense columnslice is in default state. For instance, in case the
// columnslice is instantiated for a built-in integral or floating point data type, the function returns
// \a true in case all columnslice elements are 0 and \a false in case any columnslice element is not 0. The
// following example demonstrates the use of the \a isDefault function:

   \code
   blaze::DynamicTensor<int> A;
   // ... Resizing and initialization
   if( isDefault( columnslice( A, 0UL ) ) ) { ... }
   \page()

// Optionally, it is possible to switch between strict semantics (blaze::strict) and relaxed
// semantics (blaze::relaxed):

   \code
   if( isDefault<relaxed>( columnslice( A, 0UL ) ) ) { ... }
   \page()
*/
template< bool RF           // Relaxation flag
        , typename MT       // Type of the tensor
        , size_t... CRAs >  // Compile time columnslice arguments
inline bool isDefault( const ColumnSlice<MT,CRAs...>& columnslice )
{
   using blaze::isDefault;

   for( size_t i=0UL; i<columnslice.rows(); ++i )
      for( size_t j=0UL; j<columnslice.columns(); ++j )
         if( !isDefault<RF>( columnslice(i, j) ) ) return false;
   return true;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns whether the invariants of the given columnslice are intact.
// \ingroup columnslice
//
// \param columnslice The columnslice to be tested.
// \return \a true in case the given columnslice's invariants are intact, \a false otherwise.
//
// This function checks whether the invariants of the columnslice are intact, i.e. if its state is valid.
// In case the invariants are intact, the function returns \a true, else it will return \a false.
// The following example demonstrates the use of the \a isIntact() function:

   \code
   blaze::DynamicTensor<int> A;
   // ... Resizing and initialization
   if( isIntact( columnslice( A, 0UL ) ) ) { ... }
   \page()
*/
template< typename MT       // Type of the tensor
        , size_t... CRAs >  // Compile time columnslice arguments
inline bool isIntact( const ColumnSlice<MT,CRAs...>& columnslice ) noexcept
{
   return ( columnslice.page() < columnslice.operand().columns() &&
            isIntact( columnslice.operand() ) );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns whether the two given columnslices represent the same observable state.
// \ingroup columnslice
//
// \param a The first columnslice to be tested for its state.
// \param b The second columnslice to be tested for its state.
// \return \a true in case the two columnslices share a state, \a false otherwise.
//
// This overload of the isSame() function tests if the two given columnslices refer to exactly the same
// range of the same tensor. In case both columnslices represent the same observable state, the function
// returns \a true, otherwise it returns \a false.
*/
template< typename MT1       // Type of the tensor of the left-hand side columnslice
        , size_t... CRAs1    // Compile time columnslice arguments of the left-hand side columnslice
        , typename MT2       // Type of the tensor of the right-hand side columnslice
        , size_t... CRAs2 >  // Compile time columnslice arguments of the right-hand side columnslice
inline bool isSame( const ColumnSlice<MT1,CRAs1...>& a,
                    const ColumnSlice<MT2,CRAs2...>& b ) noexcept
{
   return ( isSame( a.operand(), b.operand() ) && ( a.column() == b.column() ) );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Predict invariant violations by setting a single element of a columnslice.
// \ingroup columnslice
//
// \param columnslice The target columnslice.
// \param i The row to be set.
// \param j The column to be set.
// \param value The value to be set to the element.
// \return \a true in case the operation would be successful, \a false if not.
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename MT     // Type of the tensor
        , size_t... CRAs  // Compile time columnslice arguments
        , typename ET >   // Type of the element
inline bool trySet( const ColumnSlice<MT,CRAs...>& columnslice, size_t i, size_t k, const ET& value )
{
   BLAZE_INTERNAL_ASSERT( i < columnslice.rows(), "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( j < columnslice.columns(), "Invalid column access index" );

   return trySet( columnslice.operand(), i, columnslice.column(), k, value );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Predict invariant violations by adding to a single element of a columnslice.
// \ingroup columnslice
//
// \param columnslice The target columnslice.
// \param i The row to be modified.
// \param j The column to be modified.
// \param value The value to be added to the element.
// \return \a true in case the operation would be successful, \a false if not.
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename MT     // Type of the tensor
        , size_t... CRAs  // Compile time columnslice arguments
        , typename ET >   // Type of the element
inline bool tryAdd( const ColumnSlice<MT,CRAs...>& columnslice, size_t i, size_t k, const ET& value )
{
   BLAZE_INTERNAL_ASSERT( i < columnslice.rows(), "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( j < columnslice.columns(), "Invalid column access index" );

   return tryAdd( columnslice.operand(), i, columnslice.column(), k, value );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Predict invariant violations by subtracting from a single element of a columnslice.
// \ingroup columnslice
//
// \param columnslice The target columnslice.
// \param i The row to be modified.
// \param j The column to be modified.
// \param value The value to be subtracted from the element.
// \return \a true in case the operation would be successful, \a false if not.
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename MT     // Type of the tensor
        , size_t... CRAs  // Compile time columnslice arguments
        , typename ET >   // Type of the element
inline bool trySub( const ColumnSlice<MT,CRAs...>& columnslice, size_t i, size_t k, const ET& value )
{
   BLAZE_INTERNAL_ASSERT( i < columnslice.rows(), "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( j < columnslice.columns(), "Invalid column access index" );

   return trySub( columnslice.operand(), i, columnslice.column(), k, value );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Predict invariant violations by scaling a single element of a columnslice.
// \ingroup columnslice
//
// \param columnslice The target columnslice.
// \param i The row to be modified.
// \param j The column to be modified.
// \param value The factor for the element.
// \return \a true in case the operation would be successful, \a false if not.
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename MT     // Type of the tensor
        , size_t... CRAs  // Compile time columnslice arguments
        , typename ET >   // Type of the element
inline bool tryMult( const ColumnSlice<MT,CRAs...>& columnslice, size_t i, size_t k, const ET& value )
{
   BLAZE_INTERNAL_ASSERT( i < columnslice.rows(), "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( j < columnslice.columns(), "Invalid column access index" );

   return tryMult( columnslice.operand(), i, columnslice.column(), k, value );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Predict invariant violations by scaling a range of elements of a columnslice.
// \ingroup columnslice
//
// \param columnslice The target columnslice.
// \param index The index of the first element of the range to be modified.
// \param size The number of elements of the range to be modified.
// \param value The factor for the elements.
// \return \a true in case the operation would be successful, \a false if not.
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename MT     // Type of the tensor
        , size_t... CRAs  // Compile time columnslice arguments
        , typename ET >  // Type of the element
BLAZE_ALWAYS_INLINE bool
   tryMult( const ColumnSlice<MT,CRAs...>& columnslice, size_t row, size_t col, size_t rows, size_t cols, const ET& value )
{
   BLAZE_INTERNAL_ASSERT( row <= (~columnslice).rows(), "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( row + rows <= (~columnslice).rows(), "Invalid rows range size" );
   BLAZE_INTERNAL_ASSERT( col <= (~columnslice).columns(), "Invalid column access index" );
   BLAZE_INTERNAL_ASSERT( col + cols <= (~columnslice).columns(), "Invalid columns range size" );

   return tryMult( columnslice.operand(), row, columnslice.column(), col, rows, 1UL, cols, value );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Predict invariant violations by scaling a single element of a columnslice.
// \ingroup columnslice
//
// \param columnslice The target columnslice.
// \param index The index of the element to be modified.
// \param value The divisor for the element.
// \return \a true in case the operation would be successful, \a false if not.
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename MT     // Type of the tensor
        , size_t... CRAs  // Compile time columnslice arguments
        , typename ET >   // Type of the element
inline bool tryDiv( const ColumnSlice<MT,CRAs...>& columnslice, size_t i, size_t k, const ET& value )
{
   BLAZE_INTERNAL_ASSERT( i < columnslice.rows(), "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( j < columnslice.columns(), "Invalid column access index" );

   return tryDiv( columnslice.operand(), i, columnslice.column(), k, value );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Predict invariant violations by scaling a range of elements of a columnslice.
// \ingroup columnslice
//
// \param columnslice The target columnslice.
// \param index The index of the first element of the range to be modified.
// \param size The number of elements of the range to be modified.
// \param value The divisor for the elements.
// \return \a true in case the operation would be successful, \a false if not.
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename MT     // Type of the tensor
        , size_t... CRAs  // Compile time columnslice arguments
        , typename ET >  // Type of the element
BLAZE_ALWAYS_INLINE bool
   tryDiv( const ColumnSlice<MT,CRAs...>& columnslice, size_t row, size_t col, size_t rows, size_t cols, const ET& value )
{
   BLAZE_INTERNAL_ASSERT( row <= (~columnslice).rows(), "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( row + rows <= (~columnslice).rows(), "Invalid rows range size" );
   BLAZE_INTERNAL_ASSERT( col <= (~columnslice).columns(), "Invalid column access index" );
   BLAZE_INTERNAL_ASSERT( col + cols <= (~columnslice).columns(), "Invalid columns range size" );

   return tryDiv( columnslice.operand(), row, columnslice.column(), col, rows, 1UL, cols, value );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Predict invariant violations by the assignment of a matrix to a columnslice.
// \ingroup columnslice
//
// \param lhs The target left-hand side columnslice.
// \param rhs The right-hand side vector to be assigned.
// \param index The index of the first element to be modified.
// \return \a true in case the assignment would be successful, \a false if not.
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename MT     // Type of the tensor
        , size_t... CRAs  // Compile time columnslice arguments
        , typename VT >   // Type of the right-hand side matrix
inline bool tryAssign( const ColumnSlice<MT,CRAs...>& lhs,
                       const Matrix<VT,false>& rhs, size_t i, size_t j )
{
   BLAZE_INTERNAL_ASSERT( i <= lhs.rows(), "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( i + (~rhs).rows() <= lhs.rows(), "Invalid rows range size" );
   BLAZE_INTERNAL_ASSERT( j <= lhs.columns(), "Invalid column access index" );
   BLAZE_INTERNAL_ASSERT( j + (~rhs).columns() <= lhs.columns(), "Invalid columns range size" );

   return tryAssign(lhs.operand(), ~rhs, j, lhs.column(), i);
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Predict invariant violations by the addition assignment of a vector to a columnslice.
// \ingroup columnslice
//
// \param lhs The target left-hand side columnslice.
// \param rhs The right-hand side vector to be added.
// \param index The index of the first element to be modified.
// \return \a true in case the assignment would be successful, \a false if not.
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename MT     // Type of the tensor
        , size_t... CRAs  // Compile time columnslice arguments
        , typename VT >   // Type of the right-hand side matrix
inline bool tryAddAssign( const ColumnSlice<MT,CRAs...>& lhs,
                          const Matrix<VT,false>& rhs, size_t i, size_t j )
{
   BLAZE_INTERNAL_ASSERT( i <= lhs.rows(), "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( i + (~rhs).rows() <= lhs.rows(), "Invalid rows range size" );
   BLAZE_INTERNAL_ASSERT( j <= lhs.columns(), "Invalid column access index" );
   BLAZE_INTERNAL_ASSERT( j + (~rhs).columns() <= lhs.columns(), "Invalid columns range size" );

   return tryAddAssign(lhs.operand(), ~rhs, j, lhs.column(), i);
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Predict invariant violations by the subtraction assignment of a vector to a columnslice.
// \ingroup columnslice
//
// \param lhs The target left-hand side columnslice.
// \param rhs The right-hand side vector to be subtracted.
// \param index The index of the first element to be modified.
// \return \a true in case the assignment would be successful, \a false if not.
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename MT     // Type of the tensor
        , size_t... CRAs  // Compile time columnslice arguments
        , typename VT >   // Type of the right-hand side matrix
inline bool trySubAssign( const ColumnSlice<MT,CRAs...>& lhs,
                          const Matrix<VT,false>& rhs, size_t i, size_t j )
{
   BLAZE_INTERNAL_ASSERT( i <= lhs.rows(), "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( i + (~rhs).rows() <= lhs.rows(), "Invalid rows range size" );
   BLAZE_INTERNAL_ASSERT( j <= lhs.columns(), "Invalid column access index" );
   BLAZE_INTERNAL_ASSERT( j + (~rhs).columns() <= lhs.columns(), "Invalid columns range size" );

   return trySubAssign(lhs.operand(), ~rhs, j, lhs.column(), i);
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Predict invariant violations by the multiplication assignment of a vector to a columnslice.
// \ingroup columnslice
//
// \param lhs The target left-hand side columnslice.
// \param rhs The right-hand side vector to be multiplied.
// \param index The index of the first element to be modified.
// \return \a true in case the assignment would be successful, \a false if not.
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename MT     // Type of the tensor
        , size_t... CRAs  // Compile time columnslice arguments
        , typename VT >   // Type of the right-hand side matrix
inline bool tryMultAssign( const ColumnSlice<MT,CRAs...>& lhs,
                           const Vector<VT,true>& rhs, size_t i, size_t j )
{
   BLAZE_INTERNAL_ASSERT( i <= lhs.rows(), "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( i + (~rhs).rows() <= lhs.rows(), "Invalid rows range size" );
   BLAZE_INTERNAL_ASSERT( j <= lhs.columns(), "Invalid column access index" );
   BLAZE_INTERNAL_ASSERT( j + (~rhs).columns() <= lhs.columns(), "Invalid columns range size" );

   return tryMultAssign(lhs.operand(), ~rhs, j, lhs.column(), i);
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Predict invariant violations by the division assignment of a vector to a columnslice.
// \ingroup columnslice
//
// \param lhs The target left-hand side columnslice.
// \param rhs The right-hand side vector divisor.
// \param index The index of the first element to be modified.
// \return \a true in case the assignment would be successful, \a false if not.
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename MT     // Type of the tensor
        , size_t... CRAs  // Compile time columnslice arguments
        , typename VT >   // Type of the right-hand side matrix
inline bool tryDivAssign( const ColumnSlice<MT,CRAs...>& lhs,
                          const Matrix<VT,false>& rhs, size_t i, size_t j )
{
   BLAZE_INTERNAL_ASSERT( i <= lhs.rows(), "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( i + (~rhs).rows() <= lhs.rows(), "Invalid rows range size" );
   BLAZE_INTERNAL_ASSERT( j <= lhs.columns(), "Invalid column access index" );
   BLAZE_INTERNAL_ASSERT( j + (~rhs).columns() <= lhs.columns(), "Invalid columns range size" );

   return tryDivAssign(lhs.operand(), ~rhs, j, lhs.column(), i);
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Removal of all restrictions on the data access to the given columnslice.
// \ingroup columnslice
//
// \param r The columnslice to be derestricted.
// \return ColumnSlice without access restrictions.
//
// This function removes all restrictions on the data access to the given columnslice. It returns a columnslice
// object that does provide the same interface but does not have any restrictions on the data
// access.\n
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in the violation of invariants, erroneous results and/or in compilation errors.
*/
template< typename MT  // Type of the tensor
              // Storage order
              // Density flag
              // Symmetry flag
        , size_t I >   // ColumnSlice index
inline decltype(auto) derestrict( ColumnSlice<MT,I>& r )
{
   return columnslice<I>( derestrict( r.operand() ), unchecked );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Removal of all restrictions on the data access to the given temporary columnslice.
// \ingroup columnslice
//
// \param r The temporary columnslice to be derestricted.
// \return ColumnSlice without access restrictions.
//
// This function removes all restrictions on the data access to the given temporary columnslice. It
// returns a columnslice object that does provide the same interface but does not have any restrictions
// on the data access.\n
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in the violation of invariants, erroneous results and/or in compilation errors.
*/
template< typename MT  // Type of the tensor
              // Storage order
              // Density flag
              // Symmetry flag
        , size_t I >   // ColumnSlice index
inline decltype(auto) derestrict( ColumnSlice<MT,I>&& r )
{
   return columnslice<I>( derestrict( r.operand() ), unchecked );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Removal of all restrictions on the data access to the given columnslice.
// \ingroup columnslice
//
// \param r The columnslice to be derestricted.
// \return ColumnSlice without access restrictions.
//
// This function removes all restrictions on the data access to the given columnslice. It returns a columnslice
// object that does provide the same interface but does not have any restrictions on the data
// access.\n
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in the violation of invariants, erroneous results and/or in compilation errors.
*/
template< typename MT  // Type of the tensor
              // Storage order
              // Density flag
         >    // Symmetry flag
inline decltype(auto) derestrict( ColumnSlice<MT>& r )
{
   return columnslice( derestrict( r.operand() ), r.column(), unchecked );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Removal of all restrictions on the data access to the given temporary columnslice.
// \ingroup columnslice
//
// \param r The temporary columnslice to be derestricted.
// \return ColumnSlice without access restrictions.
//
// This function removes all restrictions on the data access to the given temporary columnslice. It
// returns a columnslice object that does provide the same interface but does not have any restrictions
// on the data access.\n
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in the violation of invariants, erroneous results and/or in compilation errors.
*/
template< typename MT  // Type of the tensor
              // Storage order
              // Density flag
         >    // Symmetry flag
inline decltype(auto) derestrict( ColumnSlice<MT>&& r )
{
   return columnslice( derestrict( r.operand() ), r.column(), unchecked );
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
template< typename MT, size_t... CRAs >
struct Size< ColumnSlice<MT,CRAs...>, 0UL >
   : public Size<MT,0UL>
{};

template< typename MT, size_t... CRAs >
struct Size< ColumnSlice<MT,CRAs...>, 1UL >
   : public Size<MT,1UL>
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
template< typename MT, size_t... CRAs >
struct MaxSize< ColumnSlice<MT,CRAs...>, 0UL >
   : public MaxSize<MT,0UL>
{};
template< typename MT, size_t... CRAs >
struct MaxSize< ColumnSlice<MT,CRAs...>, 1UL >
   : public MaxSize<MT,1UL>
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
template< typename MT, size_t... CRAs >
struct IsRestricted< ColumnSlice<MT,CRAs...> >
   : public IsRestricted<MT>
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
template< typename MT, size_t... CRAs >
struct HasConstDataAccess< ColumnSlice<MT,CRAs...> >
   : public HasConstDataAccess<MT>
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
template< typename MT, size_t... CRAs >
struct HasMutableDataAccess< ColumnSlice<MT,CRAs...> >
   : public HasMutableDataAccess<MT>
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
template< typename MT, size_t... CRAs >
struct IsAligned< ColumnSlice<MT,CRAs...> >
   : public BoolConstant< IsAligned_v<MT> >
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
template< typename MT, size_t... CRAs >
struct IsContiguous< ColumnSlice<MT,CRAs...> >
   : public IsContiguous<MT>
{};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  ISPADDED SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename MT, size_t... CRAs >
struct IsPadded< ColumnSlice<MT,CRAs...> >
   : public BoolConstant< IsPadded_v<MT> >
{};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  ISOPPOSEDVIEW SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename MT, size_t... CRAs >
struct IsOpposedView< ColumnSlice<MT,CRAs...> >
   : public TrueType
{};
/*! \endcond */
//*************************************************************************************************

} // namespace blaze

#endif
