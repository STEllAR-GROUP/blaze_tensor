//=================================================================================================
/*!
//  \file blaze_tensor/math/views/RowSlice.h
//  \brief Header file for the implementation of the RowSlice view
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

#ifndef _BLAZE_TENSOR_MATH_VIEWS_ROWSLICE_H_
#define _BLAZE_TENSOR_MATH_VIEWS_ROWSLICE_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <blaze/math/expressions/DVecExpandExpr.h>
#include <blaze/math/expressions/Forward.h>
#include <blaze/math/expressions/SchurExpr.h>
#include <blaze/math/expressions/TransExpr.h>
#include <blaze/math/views/Row.h>

#include <blaze_tensor/math/expressions/MatExpandExpr.h>
#include <blaze_tensor/math/expressions/TensEvalExpr.h>
#include <blaze_tensor/math/expressions/TensMapExpr.h>
#include <blaze_tensor/math/expressions/TensMatSchurExpr.h>
#include <blaze_tensor/math/expressions/TensReduceExpr.h>
#include <blaze_tensor/math/expressions/TensScalarDivExpr.h>
#include <blaze_tensor/math/expressions/TensScalarMultExpr.h>
#include <blaze_tensor/math/expressions/TensSerialExpr.h>
#include <blaze_tensor/math/expressions/TensTensAddExpr.h>
#include <blaze_tensor/math/expressions/TensTensMapExpr.h>
#include <blaze_tensor/math/expressions/TensTensMultExpr.h>
#include <blaze_tensor/math/expressions/TensTensSubExpr.h>
#include <blaze_tensor/math/expressions/TensTransExpr.h>
#include <blaze_tensor/math/expressions/TensVecMultExpr.h>
#include <blaze_tensor/math/expressions/Tensor.h>
#include <blaze_tensor/math/views/Forward.h>
#include <blaze_tensor/math/views/rowslice/BaseTemplate.h>
#include <blaze_tensor/math/views/rowslice/Dense.h>

namespace blaze {

//=================================================================================================
//
//  GLOBAL FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Creating a view on a specific rowslice of the given tensor.
// \ingroup rowslice
//
// \param tensor The tensor containing the rowslice.
// \param args Optional rowslice arguments.
// \return View on the specified rowslice of the tensor.
// \exception std::invalid_argument Invalid rowslice access index.
//
// This function returns an expression representing the specified rowslice of the given tensor.

   \code
   blaze::DynamicTensor<double> D;
   blaze::CompressedTensor<double> S;
   // ... Resizing and initialization

   // Creating a view on the 3rd rowslice of the dense tensor D
   auto rowslice3 = rowslice<3UL>( D );

   // Creating a view on the 4th rowslice of the sparse tensor S
   auto rowslice4 = rowslice<4UL>( S );
   \page()

// By default, the provided rowslice arguments are checked at runtime. In case the rowslice is not properly
// specified (i.e. if the specified index is greater than or equal to the total number of the rowslices
// in the given tensor) a \a std::invalid_argument exception is thrown. The checks can be skipped
// by providing the optional \a blaze::unchecked argument.

   \code
   auto rowslice3 = rowslice<3UL>( D, unchecked );
   auto rowslice4 = rowslice<4UL>( S, unchecked );
   \page()
*/
template< size_t I            // RowSlice index
        , typename MT         // Type of the tensor
        , typename... RRAs >  // Optional rowslice arguments
inline decltype(auto) rowslice( Tensor<MT>& tensor, RRAs... args )
{
   BLAZE_FUNCTION_TRACE;

   using ReturnType = RowSlice_<MT,I>;
   return ReturnType( ~tensor, args... );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Creating a view on a specific rowslice of the given constant tensor.
// \ingroup rowslice
//
// \param tensor The constant tensor containing the rowslice.
// \param args Optional rowslice arguments.
// \return View on the specified rowslice of the tensor.
// \exception std::invalid_argument Invalid rowslice access index.
//
// This function returns an expression representing the specified rowslice of the given constant
// tensor.

   \code

   const blaze::DynamicTensor<double> D( ... );
   const blaze::CompressedTensor<double> S( ... );

   // Creating a view on the 3rd rowslice of the dense tensor D
   auto rowslice3 = rowslice<3UL>( D );

   // Creating a view on the 4th rowslice of the sparse tensor S
   auto rowslice4 = rowslice<4UL>( S );
   \page()

// By default, the provided rowslice arguments are checked at runtime. In case the rowslice is not properly
// specified (i.e. if the specified index is greater than or equal to the total number of the rowslices
// in the given tensor) a \a std::invalid_argument exception is thrown. The checks can be skipped
// by providing the optional \a blaze::unchecked argument.

   \code
   auto rowslice3 = rowslice<3UL>( D, unchecked );
   auto rowslice4 = rowslice<4UL>( S, unchecked );
   \page()
*/
template< size_t I            // RowSlice index
        , typename MT         // Type of the tensor
        , typename... RRAs >  // Optional rowslice arguments
inline decltype(auto) rowslice( const Tensor<MT>& tensor, RRAs... args )
{
   BLAZE_FUNCTION_TRACE;

   using ReturnType = const RowSlice_<const MT,I>;
   return ReturnType( ~tensor, args... );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Creating a view on a specific rowslice of the given temporary tensor.
// \ingroup rowslice
//
// \param tensor The temporary tensor containing the rowslice.
// \param args Optional rowslice arguments.
// \return View on the specified rowslice of the tensor.
// \exception std::invalid_argument Invalid rowslice access index.
//
// This function returns an expression representing the specified rowslice of the given temporary
// tensor. In case the rowslice is not properly specified (i.e. if the specified index is greater
// than or equal to the total number of the rowslices in the given tensor) a \a std::invalid_argument
// exception is thrown.
*/
template< size_t I            // RowSlice index
        , typename MT         // Type of the tensor
        , typename... RRAs >  // Optional rowslice arguments
inline decltype(auto) rowslice( Tensor<MT>&& tensor, RRAs... args )
{
   BLAZE_FUNCTION_TRACE;

   using ReturnType = RowSlice_<MT,I>;
   return ReturnType( ~tensor, args... );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Creating a view on a specific rowslice of the given tensor.
// \ingroup rowslice
//
// \param tensor The tensor containing the rowslice.
// \param index The index of the rowslice.
// \param args Optional rowslice arguments.
// \return View on the specified rowslice of the tensor.
// \exception std::invalid_argument Invalid rowslice access index.
//
// This function returns an expression representing the specified rowslice of the given tensor.

   \code
   blaze::DynamicTensor<double> D;
   blaze::CompressedTensor<double> S;
   // ... Resizing and initialization

   // Creating a view on the 3rd rowslice of the dense tensor D
   auto rowslice3 = rowslice( D, 3UL );

   // Creating a view on the 4th rowslice of the sparse tensor S
   auto rowslice4 = rowslice( S, 4UL );
   \page()

// By default, the provided rowslice arguments are checked at runtime. In case the rowslice is not properly
// specified (i.e. if the specified index is greater than or equal to the total number of the rowslices
// in the given tensor) a \a std::invalid_argument exception is thrown. The checks can be skipped
// by providing the optional \a blaze::unchecked argument.

   \code
   auto rowslice3 = rowslice( D, 3UL, unchecked );
   auto rowslice4 = rowslice( S, 4UL, unchecked );
   \page()
*/
template< typename MT         // Type of the tensor
        , typename... RRAs >  // Optional rowslice arguments
inline decltype(auto) rowslice( Tensor<MT>& tensor, size_t index, RRAs... args )
{
   BLAZE_FUNCTION_TRACE;

   using ReturnType = RowSlice_<MT>;
   return ReturnType( ~tensor, index, args... );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Creating a view on a specific rowslice of the given constant tensor.
// \ingroup rowslice
//
// \param tensor The constant tensor containing the rowslice.
// \param index The index of the rowslice.
// \param args Optional rowslice arguments.
// \return View on the specified rowslice of the tensor.
// \exception std::invalid_argument Invalid rowslice access index.
//
// This function returns an expression representing the specified rowslice of the given constant
// tensor.

   \code
   const blaze::DynamicTensor<double> D( ... );
   const blaze::CompressedTensor<double> S( ... );

   // Creating a view on the 3rd rowslice of the dense tensor D
   auto rowslice3 = rowslice( D, 3UL );

   // Creating a view on the 4th rowslice of the sparse tensor S
   auto rowslice4 = rowslice( S, 4UL );
   \page()

// By default, the provided rowslice arguments are checked at runtime. In case the rowslice is not properly
// specified (i.e. if the specified index is greater than or equal to the total number of the rowslices
// in the given tensor) a \a std::invalid_argument exception is thrown. The checks can be skipped
// by providing the optional \a blaze::unchecked argument.

   \code
   auto rowslice3 = rowslice( D, 3UL, unchecked );
   auto rowslice4 = rowslice( S, 4UL, unchecked );
   \page()
*/
template< typename MT         // Type of the tensor
        , typename... RRAs >  // Optional rowslice arguments
inline decltype(auto) rowslice( const Tensor<MT>& tensor, size_t index, RRAs... args )
{
   BLAZE_FUNCTION_TRACE;

   using ReturnType = const RowSlice_<const MT>;
   return ReturnType( ~tensor, index, args... );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Creating a view on a specific rowslice of the given temporary tensor.
// \ingroup rowslice
//
// \param tensor The temporary tensor containing the rowslice.
// \param index The index of the rowslice.
// \param args Optional rowslice arguments.
// \return View on the specified rowslice of the tensor.
// \exception std::invalid_argument Invalid rowslice access index.
//
// This function returns an expression representing the specified rowslice of the given temporary
// tensor. In case the rowslice is not properly specified (i.e. if the specified index is greater
// than or equal to the total number of the rowslices in the given tensor) a \a std::invalid_argument
// exception is thrown.
*/
template< typename MT         // Type of the tensor
        , typename... RRAs >  // Optional rowslice arguments
inline decltype(auto) rowslice( Tensor<MT>&& tensor, size_t index, RRAs... args )
{
   BLAZE_FUNCTION_TRACE;

   using ReturnType = RowSlice_<MT>;
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
/*!\brief Creating a view on a specific rowslice of the given tensor/tensor addition.
// \ingroup rowslice
//
// \param tensor The constant tensor/tensor addition.
// \param args The runtime rowslice arguments.
// \return View on the specified rowslice of the addition.
//
// This function returns an expression representing the specified rowslice of the given tensor/tensor
// addition.
*/
template< size_t... CRAs      // Compile time rowslice arguments
        , typename MT         // Tensor base type of the expression
        , typename... RRAs >  // Runtime rowslice arguments
inline decltype(auto) rowslice( const TensTensAddExpr<MT>& tensor, RRAs... args )
{
   BLAZE_FUNCTION_TRACE;

   return rowslice<CRAs...>( (~tensor).leftOperand(), args... ) +
          rowslice<CRAs...>( (~tensor).rightOperand(), args... );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific rowslice of the given tensor/tensor subtraction.
// \ingroup rowslice
//
// \param tensor The constant tensor/tensor subtraction.
// \param args The runtime rowslice arguments.
// \return View on the specified rowslice of the subtraction.
//
// This function returns an expression representing the specified rowslice of the given tensor/tensor
// subtraction.
*/
template< size_t... CRAs      // Compile time rowslice arguments
        , typename MT         // Tensor base type of the expression
        , typename... RRAs >  // Runtime rowslice arguments
inline decltype(auto) rowslice( const TensTensSubExpr<MT>& tensor, RRAs... args )
{
   BLAZE_FUNCTION_TRACE;

   return rowslice<CRAs...>( (~tensor).leftOperand(), args... ) -
          rowslice<CRAs...>( (~tensor).rightOperand(), args... );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific rowslice of the given Schur product.
// \ingroup rowslice
//
// \param tensor The constant Schur product.
// \param args The runtime rowslice arguments.
// \return View on the specified rowslice of the Schur product.
//
// This function returns an expression representing the specified rowslice of the given Schur product.
*/
template< size_t... CRAs      // Compile time rowslice arguments
        , typename MT         // Tensor base type of the expression
        , typename... RRAs >  // Runtime rowslice arguments
inline decltype(auto) rowslice( const SchurExpr<MT>& tensor, RRAs... args )
{
   BLAZE_FUNCTION_TRACE;

   return rowslice<CRAs...>( (~tensor).leftOperand(), args... ) %
          rowslice<CRAs...>( (~tensor).rightOperand(), args... );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific rowslice of the given Schur product.
// \ingroup rowslice
//
// \param tensor The constant Schur product.
// \param args The runtime rowslice arguments.
// \return View on the specified rowslice of the Schur product.
//
// This function returns an expression representing the specified rowslice of the given Schur product.
*/
//template< size_t... CRAs      // Compile time rowslice arguments
//        , typename TT         // Tensor base type of the expression
//        , typename... RRAs >  // Runtime rowslice arguments
//inline decltype(auto) rowslice( const TensMatSchurExpr<TT>& tensor, RRAs... args )
//{
//   BLAZE_FUNCTION_TRACE;
//
//   return rowslice<CRAs...>( (~tensor).leftOperand(), args... ) %
//          row<CRAs...>     ( (~tensor).rightOperand(), args... );
//}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific rowslice of the given tensor/tensor multiplication.
// \ingroup rowslice
//
// \param tensor The constant tensor/tensor multiplication.
// \param args The runtime rowslice arguments
// \return View on the specified rowslice of the multiplication.
//
// This function returns an expression representing the specified rowslice of the given tensor/tensor
// multiplication.
*/
template< size_t... CRAs      // Compile time rowslice arguments
        , typename MT         // Tensor base type of the expression
        , typename... RRAs >  // Runtime rowslice arguments
inline decltype(auto) rowslice( const TensTensMultExpr<MT>& tensor, RRAs... args )
{
   BLAZE_FUNCTION_TRACE;

   return rowslice<CRAs...>( (~tensor).leftOperand(), args... ) * (~tensor).rightOperand();
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific rowslice of the given outer product.
// \ingroup rowslice
//
// \param tensor The constant outer product.
// \param args Optional rowslice arguments.
// \return View on the specified rowslice of the outer product.
// \exception std::invalid_argument Invalid rowslice access index.
//
// This function returns an expression representing the specified rowslice of the given outer product.
*/
// template< size_t I            // RowSlice index
//         , typename MT         // Tensor base type of the expression
//         , typename... RRAs >  // Optional rowslice arguments
// inline decltype(auto) rowslice( const VecTVecMultExpr<MT>& tensor, RRAs... args )
// {
//    BLAZE_FUNCTION_TRACE;
//
//    MAYBE_UNUSED( args... );
//
//    if( !Contains_v< TypeList<RRAs...>, Unchecked > ) {
//       if( (~tensor).rowslices() <= I ) {
//          BLAZE_THROWSLICE_INVALID_ARGUMENT( "Invalid rowslice access index" );
//       }
//    }
//
//    return (~tensor).leftOperand()[I] * (~tensor).rightOperand();
// }
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific rowslice of the given outer product.
// \ingroup rowslice
//
// \param tensor The constant outer product.
// \param index The index of the rowslice.
// \param args Optional rowslice arguments.
// \return View on the specified rowslice of the outer product.
// \exception std::invalid_argument Invalid rowslice access index.
//
// This function returns an expression representing the specified rowslice of the given outer product.
*/
// template< typename MT         // Tensor base type of the expression
//         , typename... RRAs >  // Optional rowslice arguments
// inline decltype(auto) rowslice( const VecTVecMultExpr<MT>& tensor, size_t index, RRAs... args )
// {
//    BLAZE_FUNCTION_TRACE;
//
//    MAYBE_UNUSED( args... );
//
//    if( !Contains_v< TypeList<RRAs...>, Unchecked > ) {
//       if( (~tensor).rowslices() <= index ) {
//          BLAZE_THROWSLICE_INVALID_ARGUMENT( "Invalid rowslice access index" );
//       }
//    }
//
//    return (~tensor).leftOperand()[index] * (~tensor).rightOperand();
// }
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific rowslice of the given tensor/scalar multiplication.
// \ingroup rowslice
//
// \param tensor The constant tensor/scalar multiplication.
// \param args The runtime rowslice arguments
// \return View on the specified rowslice of the multiplication.
//
// This function returns an expression representing the specified rowslice of the given tensor/scalar
// multiplication.
*/
template< size_t... CRAs      // Compile time rowslice arguments
        , typename MT         // Tensor base type of the expression
        , typename... RRAs >  // Runtime rowslice arguments
inline decltype(auto) rowslice( const TensScalarMultExpr<MT>& tensor, RRAs... args )
{
   BLAZE_FUNCTION_TRACE;

   return rowslice<CRAs...>( (~tensor).leftOperand(), args... ) * (~tensor).rightOperand();
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific rowslice of the given tensor/scalar division.
// \ingroup rowslice
//
// \param tensor The constant tensor/scalar division.
// \param args The runtime rowslice arguments
// \return View on the specified rowslice of the division.
//
// This function returns an expression representing the specified rowslice of the given tensor/scalar
// division.
*/
template< size_t... CRAs      // Compile time rowslice arguments
        , typename MT         // Tensor base type of the expression
        , typename... RRAs >  // Runtime rowslice arguments
inline decltype(auto) rowslice( const TensScalarDivExpr<MT>& tensor, RRAs... args )
{
   BLAZE_FUNCTION_TRACE;

   return rowslice<CRAs...>( (~tensor).leftOperand(), args... ) / (~tensor).rightOperand();
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific rowslice of the given unary tensor map operation.
// \ingroup rowslice
//
// \param tensor The constant unary tensor map operation.
// \param args The runtime rowslice arguments
// \return View on the specified rowslice of the unary map operation.
//
// This function returns an expression representing the specified rowslice of the given unary tensor
// map operation.
*/
template< size_t... CRAs      // Compile time rowslice arguments
        , typename MT         // Tensor base type of the expression
        , typename... RRAs >  // Runtime rowslice arguments
inline decltype(auto) rowslice( const TensMapExpr<MT>& tensor, RRAs... args )
{
   BLAZE_FUNCTION_TRACE;

   return map( rowslice<CRAs...>( (~tensor).operand(), args... ), (~tensor).operation() );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific rowslice of the given binary tensor map operation.
// \ingroup rowslice
//
// \param tensor The constant binary tensor map operation.
// \param args The runtime rowslice arguments
// \return View on the specified rowslice of the binary map operation.
//
// This function returns an expression representing the specified rowslice of the given binary tensor
// map operation.
*/
template< size_t... CRAs      // Compile time rowslice arguments
        , typename MT         // Tensor base type of the expression
        , typename... RRAs >  // Runtime rowslice arguments
inline decltype(auto) rowslice( const TensTensMapExpr<MT>& tensor, RRAs... args )
{
   BLAZE_FUNCTION_TRACE;

   return map( rowslice<CRAs...>( (~tensor).leftOperand(), args... ),
               rowslice<CRAs...>( (~tensor).rightOperand(), args... ),
               (~tensor).operation() );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific rowslice of the given tensor evaluation operation.
// \ingroup rowslice
//
// \param tensor The constant tensor evaluation operation.
// \param args The runtime rowslice arguments
// \return View on the specified rowslice of the evaluation operation.
//
// This function returns an expression representing the specified rowslice of the given tensor
// evaluation operation.
*/
template< size_t... CRAs      // Compile time rowslice arguments
        , typename MT         // Tensor base type of the expression
        , typename... RRAs >  // Runtime rowslice arguments
inline decltype(auto) rowslice( const TensEvalExpr<MT>& tensor, RRAs... args )
{
   BLAZE_FUNCTION_TRACE;

   return eval( rowslice<CRAs...>( (~tensor).operand(), args... ) );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific rowslice of the given tensor serialization operation.
// \ingroup rowslice
//
// \param tensor The constant tensor serialization operation.
// \param args The runtime rowslice arguments
// \return View on the specified rowslice of the serialization operation.
//
// This function returns an expression representing the specified rowslice of the given tensor
// serialization operation.
*/
template< size_t... CRAs      // Compile time rowslice arguments
        , typename MT         // Tensor base type of the expression
        , typename... RRAs >  // Runtime rowslice arguments
inline decltype(auto) rowslice( const TensSerialExpr<MT>& tensor, RRAs... args )
{
   BLAZE_FUNCTION_TRACE;

   return serial( rowslice<CRAs...>( (~tensor).operand(), args... ) );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific rowslice of the given tensor declaration operation.
// \ingroup rowslice
//
// \param tensor The constant tensor declaration operation.
// \param args The runtime rowslice arguments
// \return View on the specified rowslice of the declaration operation.
//
// This function returns an expression representing the specified rowslice of the given tensor
// declaration operation.
*/
template< size_t... CRAs      // Compile time rowslice arguments
        , typename MT         // Tensor base type of the expression
        , typename... RRAs >  // Runtime rowslice arguments
inline decltype(auto) rowslice( const DeclExpr<MT>& tensor, RRAs... args )
{
   BLAZE_FUNCTION_TRACE;

   return rowslice<CRAs...>( (~tensor).operand(), args... );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific rowslice of the given tensor transpose operation.
// \ingroup rowslice
//
// \param tensor The constant tensor transpose operation.
// \param args The runtime rowslice arguments
// \return View on the specified rowslice of the transpose operation.
//
// This function returns an expression representing the specified rowslice of the given tensor
// transpose operation.
*/
template< size_t MK           // Compile time pageslice arguments
        , size_t MI
        , size_t MJ
        , typename MT         // Tensor base type of the expression
        , typename... RTAs >  // Runtime rowslice arguments
inline decltype(auto) rowslice( const TensTransExpr<MT>& tensor, size_t index, RTAs... args )
{
   BLAZE_FUNCTION_TRACE;

   return rowslice<MK, MI, MJ>( evaluate( ~tensor ), index, args... );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific rowslice of the given tensor transpose operation.
// \ingroup rowslice
//
// \param tensor The constant tensor transpose operation.
// \param args The runtime rowslice arguments
// \return View on the specified rowslice of the transpose operation.
//
// This function returns an expression representing the specified rowslice of the given tensor
// transpose operation.
*/
template< typename MT         // Tensor base type of the expression
        , typename... RTAs >  // Runtime rowslice arguments
inline decltype(auto) rowslice( const TensTransExpr<MT>& tensor, size_t index, RTAs... args )
{
   BLAZE_FUNCTION_TRACE;

   return rowslice( evaluate( ~tensor ), index, args... );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific rowslice of the given matrix expansion operation.
// \ingroup subtensor
//
// \param tensor The constant matrix expansion operation.
// \param args Optional pageslice arguments.
// \return View on the specified rowslice of the expansion operation.
//
// This function returns an expression representing the specified rowslice of the given matrix
// expansion operation.
*/
template< size_t... CCAs      // Compile time columnslice arguments
        , typename TT         // Matrix base type of the expression
        , size_t... CEAs      // Compile time expansion arguments
        , typename... CSAs >  // Runtime pageslice arguments
inline decltype(auto) rowslice( const MatExpandExpr<TT,CEAs...>& tensor, CSAs... args )
{
   BLAZE_FUNCTION_TRACE;

   MAYBE_UNUSED( args... );

   return expand( trans( row( (~tensor).operand(), 0UL ) ), (~tensor).expansion() );
}
/*! \endcond */
//*************************************************************************************************



//=================================================================================================
//
//  GLOBAL RESTRUCTURING FUNCTIONS (COLUMN)
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a selection of column of the given tensor/vector multiplication.
// \ingroup rowslice
//
// \param matrix The constant tensor/vector multiplication.
// \param args The runtime element arguments.
// \return View on the specified row of the multiplication.
//
// This function returns an expression representing the specified elements of the given
// matrix/vector multiplication.
*/
template< size_t... CEAs      // Compile time element arguments
        , typename MT         // Matrix base type of the expression
        , typename... REAs >  // Runtime element arguments
inline decltype(auto) column( const TensVecMultExpr<MT>& matrix, REAs... args )
{
   BLAZE_FUNCTION_TRACE;

   return trans(rowslice<CEAs...>( (~matrix).leftOperand(), args... )) * (~matrix).rightOperand();
}
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  ROWSLICE OPERATORS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Resetting the given rowslice.
// \ingroup rowslice
//
// \param rowslice The rowslice to be resetted.
// \return void
*/
template< typename MT       // Type of the tensor
                   // Density flag
                   // Symmetry flag
        , size_t... CRAs >  // Compile time rowslice arguments
inline void reset( RowSlice<MT,CRAs...>& rowslice )
{
   rowslice.reset();
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Resetting the given temporary rowslice.
// \ingroup rowslice
//
// \param rowslice The temporary rowslice to be resetted.
// \return void
*/
template< typename MT       // Type of the tensor
                   // Density flag
                   // Symmetry flag
        , size_t... CRAs >  // Compile time rowslice arguments
inline void reset( RowSlice<MT,CRAs...>&& rowslice )
{
   rowslice.reset();
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Clearing the given rowslice.
// \ingroup rowslice
//
// \param rowslice The rowslice to be cleared.
// \return void
//
// Clearing a rowslice is equivalent to resetting it via the reset() function.
*/
template< typename MT       // Type of the tensor
                   // Density flag
                   // Symmetry flag
        , size_t... CRAs >  // Compile time rowslice arguments
inline void clear( RowSlice<MT,CRAs...>& rowslice )
{
   rowslice.reset();
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Clearing the given temporary rowslice.
// \ingroup rowslice
//
// \param rowslice The temporary rowslice to be cleared.
// \return void
//
// Clearing a rowslice is equivalent to resetting it via the reset() function.
*/
template< typename MT       // Type of the tensor
                   // Density flag
                   // Symmetry flag
        , size_t... CRAs >  // Compile time rowslice arguments
inline void clear( RowSlice<MT,CRAs...>&& rowslice )
{
   rowslice.reset();
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns whether the given dense rowslice is in default state.
// \ingroup rowslice
//
// \param rowslice The dense rowslice to be tested for its default state.
// \return \a true in case the given dense rowslice is component-wise zero, \a false otherwise.
//
// This function checks whether the dense rowslice is in default state. For instance, in case the
// rowslice is instantiated for a built-in integral or floating point data type, the function returns
// \a true in case all rowslice elements are 0 and \a false in case any rowslice element is not 0. The
// following example demonstrates the use of the \a isDefault function:

   \code
   blaze::DynamicTensor<int> A;
   // ... Resizing and initialization
   if( isDefault( rowslice( A, 0UL ) ) ) { ... }
   \page()

// Optionally, it is possible to switch between strict semantics (blaze::strict) and relaxed
// semantics (blaze::relaxed):

   \code
   if( isDefault<relaxed>( rowslice( A, 0UL ) ) ) { ... }
   \page()
*/
template< bool RF           // Relaxation flag
        , typename MT       // Type of the tensor
        , size_t... CRAs >  // Compile time rowslice arguments
inline bool isDefault( const RowSlice<MT,CRAs...>& rowslice )
{
   using blaze::isDefault;

   for( size_t i=0UL; i<rowslice.rows(); ++i )
      for( size_t j=0UL; j<rowslice.columns(); ++j )
         if( !isDefault<RF>( rowslice(i, j) ) ) return false;
   return true;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns whether the invariants of the given rowslice are intact.
// \ingroup rowslice
//
// \param rowslice The rowslice to be tested.
// \return \a true in case the given rowslice's invariants are intact, \a false otherwise.
//
// This function checks whether the invariants of the rowslice are intact, i.e. if its state is valid.
// In case the invariants are intact, the function returns \a true, else it will return \a false.
// The following example demonstrates the use of the \a isIntact() function:

   \code
   blaze::DynamicTensor<int> A;
   // ... Resizing and initialization
   if( isIntact( rowslice( A, 0UL ) ) ) { ... }
   \page()
*/
template< typename MT       // Type of the tensor
        , size_t... CRAs >  // Compile time rowslice arguments
inline bool isIntact( const RowSlice<MT,CRAs...>& rowslice ) noexcept
{
   return ( rowslice.row() < rowslice.operand().rows() &&
            isIntact( rowslice.operand() ) );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns whether the two given rowslices represent the same observable state.
// \ingroup rowslice
//
// \param a The first rowslice to be tested for its state.
// \param b The second rowslice to be tested for its state.
// \return \a true in case the two rowslices share a state, \a false otherwise.
//
// This overload of the isSame() function tests if the two given rowslices refer to exactly the same
// range of the same tensor. In case both rowslices represent the same observable state, the function
// returns \a true, otherwise it returns \a false.
*/
template< typename MT1       // Type of the tensor of the left-hand side rowslice
        , size_t... CRAs1    // Compile time rowslice arguments of the left-hand side rowslice
        , typename MT2       // Type of the tensor of the right-hand side rowslice
        , size_t... CRAs2 >  // Compile time rowslice arguments of the right-hand side rowslice
inline bool isSame( const RowSlice<MT1,CRAs1...>& a,
                    const RowSlice<MT2,CRAs2...>& b ) noexcept
{
   return ( isSame( a.operand(), b.operand() ) && ( a.row() == b.row() ) );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Predict invariant violations by setting a single element of a rowslice.
// \ingroup rowslice
//
// \param rowslice The target rowslice.
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
        , size_t... CRAs  // Compile time rowslice arguments
        , typename ET >   // Type of the element
inline bool trySet( const RowSlice<MT,CRAs...>& rowslice, size_t i, size_t j, const ET& value )
{
   BLAZE_INTERNAL_ASSERT( i < rowslice.rows(), "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( j < rowslice.columns(), "Invalid column access index" );

   return trySet( rowslice.operand(), rowslice.row(), j, i, value );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Predict invariant violations by adding to a single element of a rowslice.
// \ingroup rowslice
//
// \param rowslice The target rowslice.
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
        , size_t... CRAs  // Compile time rowslice arguments
        , typename ET >   // Type of the element
inline bool tryAdd( const RowSlice<MT,CRAs...>& rowslice, size_t i, size_t j, const ET& value )
{
   BLAZE_INTERNAL_ASSERT( i < rowslice.rows(), "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( j < rowslice.columns(), "Invalid column access index" );

   return tryAdd( rowslice.operand(), rowslice.row(), j, i, value );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Predict invariant violations by subtracting from a single element of a rowslice.
// \ingroup rowslice
//
// \param rowslice The target rowslice.
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
        , size_t... CRAs  // Compile time rowslice arguments
        , typename ET >   // Type of the element
inline bool trySub( const RowSlice<MT,CRAs...>& rowslice, size_t i, size_t j, const ET& value )
{
   BLAZE_INTERNAL_ASSERT( i < rowslice.rows(), "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( j < rowslice.columns(), "Invalid column access index" );

   return trySub( rowslice.operand(), rowslice.row(), j, i, value );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Predict invariant violations by scaling a single element of a rowslice.
// \ingroup rowslice
//
// \param rowslice The target rowslice.
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
        , size_t... CRAs  // Compile time rowslice arguments
        , typename ET >   // Type of the element
inline bool tryMult( const RowSlice<MT,CRAs...>& rowslice, size_t i, size_t j, const ET& value )
{
   BLAZE_INTERNAL_ASSERT( i < rowslice.rows(), "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( j < rowslice.columns(), "Invalid column access index" );

   return tryMult( rowslice.operand(), rowslice.row(), j, i, value );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Predict invariant violations by scaling a range of elements of a rowslice.
// \ingroup rowslice
//
// \param rowslice The target rowslice.
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
        , size_t... CRAs  // Compile time rowslice arguments
        , typename ET >  // Type of the element
BLAZE_ALWAYS_INLINE bool
   tryMult( const RowSlice<MT,CRAs...>& rowslice, size_t row, size_t col, size_t rows, size_t cols, const ET& value )
{
   BLAZE_INTERNAL_ASSERT( row <= (~rowslice).rows(), "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( row + rows <= (~rowslice).rows(), "Invalid rows range size" );
   BLAZE_INTERNAL_ASSERT( col <= (~rowslice).columns(), "Invalid column access index" );
   BLAZE_INTERNAL_ASSERT( col + cols <= (~rowslice).columns(), "Invalid columns range size" );

   return tryMult( rowslice.operand(), rowslice.row(), col, row, 1UL, cols, rows, value );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Predict invariant violations by scaling a single element of a rowslice.
// \ingroup rowslice
//
// \param rowslice The target rowslice.
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
        , size_t... CRAs  // Compile time rowslice arguments
        , typename ET >   // Type of the element
inline bool tryDiv( const RowSlice<MT,CRAs...>& rowslice, size_t i, size_t j, const ET& value )
{
   BLAZE_INTERNAL_ASSERT( i < rowslice.rows(), "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( j < rowslice.columns(), "Invalid column access index" );

   return tryDiv( rowslice.operand(), rowslice.row(), j, i, value );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Predict invariant violations by scaling a range of elements of a rowslice.
// \ingroup rowslice
//
// \param rowslice The target rowslice.
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
        , size_t... CRAs  // Compile time rowslice arguments
        , typename ET >  // Type of the element
BLAZE_ALWAYS_INLINE bool
   tryDiv( const RowSlice<MT,CRAs...>& rowslice, size_t row, size_t col, size_t rows, size_t cols, const ET& value )
{
   BLAZE_INTERNAL_ASSERT( row <= (~rowslice).rows(), "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( row + rows <= (~rowslice).rows(), "Invalid rows range size" );
   BLAZE_INTERNAL_ASSERT( col <= (~rowslice).columns(), "Invalid column access index" );
   BLAZE_INTERNAL_ASSERT( col + cols <= (~rowslice).columns(), "Invalid columns range size" );

   return tryDiv( rowslice.operand(), rowslice.row(), col, row, 1UL, cols, rows, value );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Predict invariant violations by the assignment of a matrix to a rowslice.
// \ingroup rowslice
//
// \param lhs The target left-hand side rowslice.
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
        , size_t... CRAs  // Compile time rowslice arguments
        , typename VT >   // Type of the right-hand side matrix
inline bool tryAssign( const RowSlice<MT,CRAs...>& lhs,
                       const Matrix<VT,columnMajor>& rhs, size_t i, size_t j )
{
   BLAZE_INTERNAL_ASSERT( i <= lhs.rows(), "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( i + (~rhs).rows() <= lhs.rows(), "Invalid rows range size" );
   BLAZE_INTERNAL_ASSERT( j <= lhs.columns(), "Invalid column access index" );
   BLAZE_INTERNAL_ASSERT( j + (~rhs).columns() <= lhs.columns(), "Invalid columns range size" );

   return tryAssign( lhs.operand(), ~rhs, lhs.row(), j, i );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Predict invariant violations by the addition assignment of a vector to a rowslice.
// \ingroup rowslice
//
// \param lhs The target left-hand side rowslice.
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
        , size_t... CRAs  // Compile time rowslice arguments
        , typename VT >   // Type of the right-hand side matrix
inline bool tryAddAssign( const RowSlice<MT,CRAs...>& lhs,
                          const Matrix<VT,columnMajor>& rhs, size_t i, size_t j )
{
   BLAZE_INTERNAL_ASSERT( i <= lhs.rows(), "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( i + (~rhs).rows() <= lhs.rows(), "Invalid rows range size" );
   BLAZE_INTERNAL_ASSERT( j <= lhs.columns(), "Invalid column access index" );
   BLAZE_INTERNAL_ASSERT( j + (~rhs).columns() <= lhs.columns(), "Invalid columns range size" );

   return tryAddAssign( lhs.operand(), ~rhs, lhs.row(), j, i );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Predict invariant violations by the subtraction assignment of a vector to a rowslice.
// \ingroup rowslice
//
// \param lhs The target left-hand side rowslice.
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
        , size_t... CRAs  // Compile time rowslice arguments
        , typename VT >   // Type of the right-hand side matrix
inline bool trySubAssign( const RowSlice<MT,CRAs...>& lhs,
                          const Matrix<VT,columnMajor>& rhs, size_t i, size_t j )
{
   BLAZE_INTERNAL_ASSERT( i <= lhs.rows(), "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( i + (~rhs).rows() <= lhs.rows(), "Invalid rows range size" );
   BLAZE_INTERNAL_ASSERT( j <= lhs.columns(), "Invalid column access index" );
   BLAZE_INTERNAL_ASSERT( j + (~rhs).columns() <= lhs.columns(), "Invalid columns range size" );

   return trySubAssign( lhs.operand(), ~rhs, lhs.row(), j, i );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Predict invariant violations by the multiplication assignment of a vector to a rowslice.
// \ingroup rowslice
//
// \param lhs The target left-hand side rowslice.
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
        , size_t... CRAs  // Compile time rowslice arguments
        , typename VT >   // Type of the right-hand side matrix
inline bool tryMultAssign( const RowSlice<MT,CRAs...>& lhs,
                           const Vector<VT,true>& rhs, size_t i, size_t j )
{
   BLAZE_INTERNAL_ASSERT( i <= lhs.rows(), "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( i + (~rhs).rows() <= lhs.rows(), "Invalid rows range size" );
   BLAZE_INTERNAL_ASSERT( j <= lhs.columns(), "Invalid column access index" );
   BLAZE_INTERNAL_ASSERT( j + (~rhs).columns() <= lhs.columns(), "Invalid columns range size" );

   return tryMultAssign( lhs.operand(), ~rhs, lhs.row(), j, i );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Predict invariant violations by the division assignment of a vector to a rowslice.
// \ingroup rowslice
//
// \param lhs The target left-hand side rowslice.
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
        , size_t... CRAs  // Compile time rowslice arguments
        , typename VT >   // Type of the right-hand side matrix
inline bool tryDivAssign( const RowSlice<MT,CRAs...>& lhs,
                          const Matrix<VT,columnMajor>& rhs, size_t i, size_t j )
{
   BLAZE_INTERNAL_ASSERT( i <= lhs.rows(), "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( i + (~rhs).rows() <= lhs.rows(), "Invalid rows range size" );
   BLAZE_INTERNAL_ASSERT( j <= lhs.columns(), "Invalid column access index" );
   BLAZE_INTERNAL_ASSERT( j + (~rhs).columns() <= lhs.columns(), "Invalid columns range size" );

   return tryDivAssign( lhs.operand(), ~rhs, lhs.row(), j, i );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Removal of all restrictions on the data access to the given rowslice.
// \ingroup rowslice
//
// \param r The rowslice to be derestricted.
// \return RowSlice without access restrictions.
//
// This function removes all restrictions on the data access to the given rowslice. It returns a rowslice
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
        , size_t I >   // RowSlice index
inline decltype(auto) derestrict( RowSlice<MT,I>& r )
{
   return rowslice<I>( derestrict( r.operand() ), unchecked );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Removal of all restrictions on the data access to the given temporary rowslice.
// \ingroup rowslice
//
// \param r The temporary rowslice to be derestricted.
// \return RowSlice without access restrictions.
//
// This function removes all restrictions on the data access to the given temporary rowslice. It
// returns a rowslice object that does provide the same interface but does not have any restrictions
// on the data access.\n
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in the violation of invariants, erroneous results and/or in compilation errors.
*/
template< typename MT  // Type of the tensor
              // Storage order
              // Density flag
              // Symmetry flag
        , size_t I >   // RowSlice index
inline decltype(auto) derestrict( RowSlice<MT,I>&& r )
{
   return rowslice<I>( derestrict( r.operand() ), unchecked );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Removal of all restrictions on the data access to the given rowslice.
// \ingroup rowslice
//
// \param r The rowslice to be derestricted.
// \return RowSlice without access restrictions.
//
// This function removes all restrictions on the data access to the given rowslice. It returns a rowslice
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
inline decltype(auto) derestrict( RowSlice<MT>& r )
{
   return rowslice( derestrict( r.operand() ), r.row(), unchecked );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Removal of all restrictions on the data access to the given temporary rowslice.
// \ingroup rowslice
//
// \param r The temporary rowslice to be derestricted.
// \return RowSlice without access restrictions.
//
// This function removes all restrictions on the data access to the given temporary rowslice. It
// returns a rowslice object that does provide the same interface but does not have any restrictions
// on the data access.\n
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in the violation of invariants, erroneous results and/or in compilation errors.
*/
template< typename MT  // Type of the tensor
              // Storage order
              // Density flag
         >    // Symmetry flag
inline decltype(auto) derestrict( RowSlice<MT>&& r )
{
   return rowslice( derestrict( r.operand() ), r.row(), unchecked );
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
struct Size< RowSlice<MT,CRAs...>, 0UL >
   : public Size<MT,2UL>
{};

template< typename MT, size_t... CRAs >
struct Size< RowSlice<MT,CRAs...>, 1UL >
   : public Size<MT,0UL>
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
struct MaxSize< RowSlice<MT,CRAs...>, 0UL >
   : public MaxSize<MT,2UL>
{};
template< typename MT, size_t... CRAs >
struct MaxSize< RowSlice<MT,CRAs...>, 1UL >
   : public MaxSize<MT,0UL>
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
struct IsRestricted< RowSlice<MT,CRAs...> >
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
struct HasConstDataAccess< RowSlice<MT,CRAs...> >
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
struct HasMutableDataAccess< RowSlice<MT,CRAs...> >
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
struct IsAligned< RowSlice<MT,CRAs...> >
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
struct IsContiguous< RowSlice<MT,CRAs...> >
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
struct IsPadded< RowSlice<MT,CRAs...> >
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
struct IsOpposedView< RowSlice<MT,CRAs...> >
   : public TrueType
{};
/*! \endcond */
//*************************************************************************************************

} // namespace blaze

#endif
