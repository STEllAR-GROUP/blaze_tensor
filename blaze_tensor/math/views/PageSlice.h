//=================================================================================================
/*!
//  \file blaze_tensor/math/views/PageSlice.h
//  \brief Header file for the implementation of the PageSlice view
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

#ifndef _BLAZE_TENSOR_MATH_VIEWS_PAGESLICE_H_
#define _BLAZE_TENSOR_MATH_VIEWS_PAGESLICE_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <blaze/math/expressions/Forward.h>
#include <blaze/math/expressions/SchurExpr.h>
#include <blaze/math/views/Row.h>
#include <blaze/math/views/Submatrix.h>

#include <blaze_tensor/math/expressions/Forward.h>
#include <blaze_tensor/math/expressions/MatExpandExpr.h>
#include <blaze_tensor/math/expressions/TensEvalExpr.h>
#include <blaze_tensor/math/expressions/TensMapExpr.h>
#include <blaze_tensor/math/expressions/TensReduceExpr.h>
#include <blaze_tensor/math/expressions/TensMatSchurExpr.h>
#include <blaze_tensor/math/expressions/TensScalarDivExpr.h>
#include <blaze_tensor/math/expressions/TensScalarMultExpr.h>
#include <blaze_tensor/math/expressions/TensSerialExpr.h>
#include <blaze_tensor/math/expressions/TensTensAddExpr.h>
#include <blaze_tensor/math/expressions/TensTensMapExpr.h>
#include <blaze_tensor/math/expressions/TensTensMultExpr.h>
#include <blaze_tensor/math/expressions/TensTensSubExpr.h>
#include <blaze_tensor/math/expressions/TensVecMultExpr.h>
#include <blaze_tensor/math/expressions/Tensor.h>
#include <blaze_tensor/math/views/Forward.h>
#include <blaze_tensor/math/views/pageslice/BaseTemplate.h>
#include <blaze_tensor/math/views/pageslice/Dense.h>

namespace blaze {

//=================================================================================================
//
//  GLOBAL FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Creating a view on a specific pageslice of the given tensor.
// \ingroup pageslice
//
// \param tensor The tensor containing the pageslice.
// \param args Optional pageslice arguments.
// \return View on the specified pageslice of the tensor.
// \exception std::invalid_argument Invalid pageslice access index.
//
// This function returns an expression representing the specified pageslice of the given tensor.

   \code
   blaze::DynamicTensor<double> D;
   blaze::CompressedTensor<double> S;
   // ... Resizing and initialization

   // Creating a view on the 3rd pageslice of the dense tensor D
   auto pageslice3 = pageslice<3UL>( D );

   // Creating a view on the 4th pageslice of the sparse tensor S
   auto pageslice4 = pageslice<4UL>( S );
   \page()

// By default, the provided pageslice arguments are checked at runtime. In case the pageslice is not properly
// specified (i.e. if the specified index is greater than or equal to the total number of the pageslices
// in the given tensor) a \a std::invalid_argument exception is thrown. The checks can be skipped
// by providing the optional \a blaze::unchecked argument.

   \code
   auto pageslice3 = pageslice<3UL>( D, unchecked );
   auto pageslice4 = pageslice<4UL>( S, unchecked );
   \page()
*/
template< size_t I            // PageSlice index
        , typename MT         // Type of the tensor
        , typename... RRAs >  // Optional pageslice arguments
inline decltype(auto) pageslice( Tensor<MT>& tensor, RRAs... args )
{
   BLAZE_FUNCTION_TRACE;

   using ReturnType = PageSlice_<MT,I>;
   return ReturnType( ~tensor, args... );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Creating a view on a specific pageslice of the given constant tensor.
// \ingroup pageslice
//
// \param tensor The constant tensor containing the pageslice.
// \param args Optional pageslice arguments.
// \return View on the specified pageslice of the tensor.
// \exception std::invalid_argument Invalid pageslice access index.
//
// This function returns an expression representing the specified pageslice of the given constant
// tensor.

   \code

   const blaze::DynamicTensor<double> D( ... );
   const blaze::CompressedTensor<double> S( ... );

   // Creating a view on the 3rd pageslice of the dense tensor D
   auto pageslice3 = pageslice<3UL>( D );

   // Creating a view on the 4th pageslice of the sparse tensor S
   auto pageslice4 = pageslice<4UL>( S );
   \page()

// By default, the provided pageslice arguments are checked at runtime. In case the pageslice is not properly
// specified (i.e. if the specified index is greater than or equal to the total number of the pageslices
// in the given tensor) a \a std::invalid_argument exception is thrown. The checks can be skipped
// by providing the optional \a blaze::unchecked argument.

   \code
   auto pageslice3 = pageslice<3UL>( D, unchecked );
   auto pageslice4 = pageslice<4UL>( S, unchecked );
   \page()
*/
template< size_t I            // PageSlice index
        , typename MT         // Type of the tensor
        , typename... RRAs >  // Optional pageslice arguments
inline decltype(auto) pageslice( const Tensor<MT>& tensor, RRAs... args )
{
   BLAZE_FUNCTION_TRACE;

   using ReturnType = const PageSlice_<const MT,I>;
   return ReturnType( ~tensor, args... );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Creating a view on a specific pageslice of the given temporary tensor.
// \ingroup pageslice
//
// \param tensor The temporary tensor containing the pageslice.
// \param args Optional pageslice arguments.
// \return View on the specified pageslice of the tensor.
// \exception std::invalid_argument Invalid pageslice access index.
//
// This function returns an expression representing the specified pageslice of the given temporary
// tensor. In case the pageslice is not properly specified (i.e. if the specified index is greater
// than or equal to the total number of the pageslices in the given tensor) a \a std::invalid_argument
// exception is thrown.
*/
template< size_t I            // PageSlice index
        , typename MT         // Type of the tensor
        , typename... RRAs >  // Optional pageslice arguments
inline decltype(auto) pageslice( Tensor<MT>&& tensor, RRAs... args )
{
   BLAZE_FUNCTION_TRACE;

   using ReturnType = PageSlice_<MT,I>;
   return ReturnType( ~tensor, args... );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Creating a view on a specific pageslice of the given tensor.
// \ingroup pageslice
//
// \param tensor The tensor containing the pageslice.
// \param index The index of the pageslice.
// \param args Optional pageslice arguments.
// \return View on the specified pageslice of the tensor.
// \exception std::invalid_argument Invalid pageslice access index.
//
// This function returns an expression representing the specified pageslice of the given tensor.

   \code
   blaze::DynamicTensor<double> D;
   blaze::CompressedTensor<double> S;
   // ... Resizing and initialization

   // Creating a view on the 3rd pageslice of the dense tensor D
   auto pageslice3 = pageslice( D, 3UL );

   // Creating a view on the 4th pageslice of the sparse tensor S
   auto pageslice4 = pageslice( S, 4UL );
   \page()

// By default, the provided pageslice arguments are checked at runtime. In case the pageslice is not properly
// specified (i.e. if the specified index is greater than or equal to the total number of the pageslices
// in the given tensor) a \a std::invalid_argument exception is thrown. The checks can be skipped
// by providing the optional \a blaze::unchecked argument.

   \code
   auto pageslice3 = pageslice( D, 3UL, unchecked );
   auto pageslice4 = pageslice( S, 4UL, unchecked );
   \page()
*/
template< typename MT         // Type of the tensor
        , typename... RRAs >  // Optional pageslice arguments
inline decltype(auto) pageslice( Tensor<MT>& tensor, size_t index, RRAs... args )
{
   BLAZE_FUNCTION_TRACE;

   using ReturnType = PageSlice_<MT>;
   return ReturnType( ~tensor, index, args... );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Creating a view on a specific pageslice of the given constant tensor.
// \ingroup pageslice
//
// \param tensor The constant tensor containing the pageslice.
// \param index The index of the pageslice.
// \param args Optional pageslice arguments.
// \return View on the specified pageslice of the tensor.
// \exception std::invalid_argument Invalid pageslice access index.
//
// This function returns an expression representing the specified pageslice of the given constant
// tensor.

   \code
   const blaze::DynamicTensor<double> D( ... );
   const blaze::CompressedTensor<double> S( ... );

   // Creating a view on the 3rd pageslice of the dense tensor D
   auto pageslice3 = pageslice( D, 3UL );

   // Creating a view on the 4th pageslice of the sparse tensor S
   auto pageslice4 = pageslice( S, 4UL );
   \page()

// By default, the provided pageslice arguments are checked at runtime. In case the pageslice is not properly
// specified (i.e. if the specified index is greater than or equal to the total number of the pageslices
// in the given tensor) a \a std::invalid_argument exception is thrown. The checks can be skipped
// by providing the optional \a blaze::unchecked argument.

   \code
   auto pageslice3 = pageslice( D, 3UL, unchecked );
   auto pageslice4 = pageslice( S, 4UL, unchecked );
   \page()
*/
template< typename MT         // Type of the tensor
        , typename... RRAs >  // Optional pageslice arguments
inline decltype(auto) pageslice( const Tensor<MT>& tensor, size_t index, RRAs... args )
{
   BLAZE_FUNCTION_TRACE;

   using ReturnType = const PageSlice_<const MT>;
   return ReturnType( ~tensor, index, args... );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Creating a view on a specific pageslice of the given temporary tensor.
// \ingroup pageslice
//
// \param tensor The temporary tensor containing the pageslice.
// \param index The index of the pageslice.
// \param args Optional pageslice arguments.
// \return View on the specified pageslice of the tensor.
// \exception std::invalid_argument Invalid pageslice access index.
//
// This function returns an expression representing the specified pageslice of the given temporary
// tensor. In case the pageslice is not properly specified (i.e. if the specified index is greater
// than or equal to the total number of the pageslices in the given tensor) a \a std::invalid_argument
// exception is thrown.
*/
template< typename MT         // Type of the tensor
        , typename... RRAs >  // Optional pageslice arguments
inline decltype(auto) pageslice( Tensor<MT>&& tensor, size_t index, RRAs... args )
{
   BLAZE_FUNCTION_TRACE;

   using ReturnType = PageSlice_<MT>;
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
/*!\brief Creating a view on a specific pageslice of the given tensor/tensor addition.
// \ingroup pageslice
//
// \param tensor The constant tensor/tensor addition.
// \param args The runtime pageslice arguments.
// \return View on the specified pageslice of the addition.
//
// This function returns an expression representing the specified pageslice of the given tensor/tensor
// addition.
*/
template< size_t... CRAs      // Compile time pageslice arguments
        , typename MT         // Tensor base type of the expression
        , typename... RRAs >  // Runtime pageslice arguments
inline decltype(auto) pageslice( const TensTensAddExpr<MT>& tensor, RRAs... args )
{
   BLAZE_FUNCTION_TRACE;

   return pageslice<CRAs...>( (~tensor).leftOperand(), args... ) +
          pageslice<CRAs...>( (~tensor).rightOperand(), args... );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific pageslice of the given tensor/tensor subtraction.
// \ingroup pageslice
//
// \param tensor The constant tensor/tensor subtraction.
// \param args The runtime pageslice arguments.
// \return View on the specified pageslice of the subtraction.
//
// This function returns an expression representing the specified pageslice of the given tensor/tensor
// subtraction.
*/
template< size_t... CRAs      // Compile time pageslice arguments
        , typename MT         // Tensor base type of the expression
        , typename... RRAs >  // Runtime pageslice arguments
inline decltype(auto) pageslice( const TensTensSubExpr<MT>& tensor, RRAs... args )
{
   BLAZE_FUNCTION_TRACE;

   return pageslice<CRAs...>( (~tensor).leftOperand(), args... ) -
          pageslice<CRAs...>( (~tensor).rightOperand(), args... );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific pageslice of the given Schur product.
// \ingroup pageslice
//
// \param tensor The constant Schur product.
// \param args The runtime pageslice arguments.
// \return View on the specified pageslice of the Schur product.
//
// This function returns an expression representing the specified pageslice of the given Schur product.
*/
template< size_t... CRAs      // Compile time pageslice arguments
        , typename MT         // Tensor base type of the expression
        , typename... RRAs >  // Runtime pageslice arguments
inline decltype(auto) pageslice( const SchurExpr<MT>& tensor, RRAs... args )
{
   BLAZE_FUNCTION_TRACE;

   return pageslice<CRAs...>( (~tensor).leftOperand(), args... ) %
          pageslice<CRAs...>( (~tensor).rightOperand(), args... );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific pageslice of the given Schur product.
// \ingroup pageslice
//
// \param tensor The constant Schur product.
// \param args The runtime pageslice arguments.
// \return View on the specified pageslice of the Schur product.
//
// This function returns an expression representing the specified pageslice of the given Schur product.
*/
template< size_t... CRAs      // Compile time pageslice arguments
        , typename TT         // Tensor base type of the expression
        , typename... RRAs >  // Runtime pageslice arguments
inline decltype(auto) pageslice( const TensMatSchurExpr<TT>& tensor, RRAs... args )
{
   BLAZE_FUNCTION_TRACE;

   return pageslice<CRAs...>( (~tensor).leftOperand(), args... ) %
                              (~tensor).rightOperand();
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific pageslice of the given tensor/tensor multiplication.
// \ingroup pageslice
//
// \param tensor The constant tensor/tensor multiplication.
// \param args The runtime pageslice arguments
// \return View on the specified pageslice of the multiplication.
//
// This function returns an expression representing the specified pageslice of the given tensor/tensor
// multiplication.
*/
template< size_t... CRAs      // Compile time pageslice arguments
        , typename MT         // Tensor base type of the expression
        , typename... RRAs >  // Runtime pageslice arguments
inline decltype(auto) pageslice( const TensTensMultExpr<MT>& tensor, RRAs... args )
{
   BLAZE_FUNCTION_TRACE;

   return pageslice<CRAs...>( (~tensor).leftOperand(), args... ) * (~tensor).rightOperand();
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific pageslice of the given outer product.
// \ingroup pageslice
//
// \param tensor The constant outer product.
// \param args Optional pageslice arguments.
// \return View on the specified pageslice of the outer product.
// \exception std::invalid_argument Invalid pageslice access index.
//
// This function returns an expression representing the specified pageslice of the given outer product.
*/
// template< size_t I            // PageSlice index
//         , typename MT         // Tensor base type of the expression
//         , typename... RRAs >  // Optional pageslice arguments
// inline decltype(auto) pageslice( const VecTVecMultExpr<MT>& tensor, RRAs... args )
// {
//    BLAZE_FUNCTION_TRACE;
//
//    MAYBE_UNUSED( args... );
//
//    if( !Contains_v< TypeList<RRAs...>, Unchecked > ) {
//       if( (~tensor).pageslices() <= I ) {
//          BLAZE_THPAGESLICE_INVALID_ARGUMENT( "Invalid pageslice access index" );
//       }
//    }
//
//    return (~tensor).leftOperand()[I] * (~tensor).rightOperand();
// }
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific pageslice of the given outer product.
// \ingroup pageslice
//
// \param tensor The constant outer product.
// \param index The index of the pageslice.
// \param args Optional pageslice arguments.
// \return View on the specified pageslice of the outer product.
// \exception std::invalid_argument Invalid pageslice access index.
//
// This function returns an expression representing the specified pageslice of the given outer product.
*/
// template< typename MT         // Tensor base type of the expression
//         , typename... RRAs >  // Optional pageslice arguments
// inline decltype(auto) pageslice( const VecTVecMultExpr<MT>& tensor, size_t index, RRAs... args )
// {
//    BLAZE_FUNCTION_TRACE;
//
//    MAYBE_UNUSED( args... );
//
//    if( !Contains_v< TypeList<RRAs...>, Unchecked > ) {
//       if( (~tensor).pageslices() <= index ) {
//          BLAZE_THPAGESLICE_INVALID_ARGUMENT( "Invalid pageslice access index" );
//       }
//    }
//
//    return (~tensor).leftOperand()[index] * (~tensor).rightOperand();
// }
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific pageslice of the given tensor/scalar multiplication.
// \ingroup pageslice
//
// \param tensor The constant tensor/scalar multiplication.
// \param args The runtime pageslice arguments
// \return View on the specified pageslice of the multiplication.
//
// This function returns an expression representing the specified pageslice of the given tensor/scalar
// multiplication.
*/
template< size_t... CRAs      // Compile time pageslice arguments
        , typename MT         // Tensor base type of the expression
        , typename... RRAs >  // Runtime pageslice arguments
inline decltype(auto) pageslice( const TensScalarMultExpr<MT>& tensor, RRAs... args )
{
   BLAZE_FUNCTION_TRACE;

   return pageslice<CRAs...>( (~tensor).leftOperand(), args... ) * (~tensor).rightOperand();
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific pageslice of the given tensor/scalar division.
// \ingroup pageslice
//
// \param tensor The constant tensor/scalar division.
// \param args The runtime pageslice arguments
// \return View on the specified pageslice of the division.
//
// This function returns an expression representing the specified pageslice of the given tensor/scalar
// division.
*/
template< size_t... CRAs      // Compile time pageslice arguments
        , typename MT         // Tensor base type of the expression
        , typename... RRAs >  // Runtime pageslice arguments
inline decltype(auto) pageslice( const TensScalarDivExpr<MT>& tensor, RRAs... args )
{
   BLAZE_FUNCTION_TRACE;

   return pageslice<CRAs...>( (~tensor).leftOperand(), args... ) / (~tensor).rightOperand();
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific pageslice of the given unary tensor map operation.
// \ingroup pageslice
//
// \param tensor The constant unary tensor map operation.
// \param args The runtime pageslice arguments
// \return View on the specified pageslice of the unary map operation.
//
// This function returns an expression representing the specified pageslice of the given unary tensor
// map operation.
*/
template< size_t... CRAs      // Compile time pageslice arguments
        , typename MT         // Tensor base type of the expression
        , typename... RRAs >  // Runtime pageslice arguments
inline decltype(auto) pageslice( const TensMapExpr<MT>& tensor, RRAs... args )
{
   BLAZE_FUNCTION_TRACE;

   return map( pageslice<CRAs...>( (~tensor).operand(), args... ), (~tensor).operation() );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific pageslice of the given binary tensor map operation.
// \ingroup pageslice
//
// \param tensor The constant binary tensor map operation.
// \param args The runtime pageslice arguments
// \return View on the specified pageslice of the binary map operation.
//
// This function returns an expression representing the specified pageslice of the given binary tensor
// map operation.
*/
template< size_t... CRAs      // Compile time pageslice arguments
        , typename MT         // Tensor base type of the expression
        , typename... RRAs >  // Runtime pageslice arguments
inline decltype(auto) pageslice( const TensTensMapExpr<MT>& tensor, RRAs... args )
{
   BLAZE_FUNCTION_TRACE;

   return map( pageslice<CRAs...>( (~tensor).leftOperand(), args... ),
               pageslice<CRAs...>( (~tensor).rightOperand(), args... ),
               (~tensor).operation() );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific pageslice of the given tensor evaluation operation.
// \ingroup pageslice
//
// \param tensor The constant tensor evaluation operation.
// \param args The runtime pageslice arguments
// \return View on the specified pageslice of the evaluation operation.
//
// This function returns an expression representing the specified pageslice of the given tensor
// evaluation operation.
*/
template< size_t... CRAs      // Compile time pageslice arguments
        , typename MT         // Tensor base type of the expression
        , typename... RRAs >  // Runtime pageslice arguments
inline decltype(auto) pageslice( const TensEvalExpr<MT>& tensor, RRAs... args )
{
   BLAZE_FUNCTION_TRACE;

   return eval( pageslice<CRAs...>( (~tensor).operand(), args... ) );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific pageslice of the given tensor serialization operation.
// \ingroup pageslice
//
// \param tensor The constant tensor serialization operation.
// \param args The runtime pageslice arguments
// \return View on the specified pageslice of the serialization operation.
//
// This function returns an expression representing the specified pageslice of the given tensor
// serialization operation.
*/
template< size_t... CRAs      // Compile time pageslice arguments
        , typename MT         // Tensor base type of the expression
        , typename... RRAs >  // Runtime pageslice arguments
inline decltype(auto) pageslice( const TensSerialExpr<MT>& tensor, RRAs... args )
{
   BLAZE_FUNCTION_TRACE;

   return serial( pageslice<CRAs...>( (~tensor).operand(), args... ) );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific pageslice of the given tensor declaration operation.
// \ingroup pageslice
//
// \param tensor The constant tensor declaration operation.
// \param args The runtime pageslice arguments
// \return View on the specified pageslice of the declaration operation.
//
// This function returns an expression representing the specified pageslice of the given tensor
// declaration operation.
*/
template< size_t... CRAs      // Compile time pageslice arguments
        , typename MT         // Tensor base type of the expression
        , typename... RRAs >  // Runtime pageslice arguments
inline decltype(auto) pageslice( const DeclExpr<MT>& tensor, RRAs... args )
{
   BLAZE_FUNCTION_TRACE;

   return pageslice<CRAs...>( (~tensor).operand(), args... );
}
/*! \endcond */
//*************************************************************************************************



//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific pageslice of the given matrix expansion operation.
// \ingroup subtensor
//
// \param tensor The constant matrix expansion operation.
// \param args Optional pageslice arguments.
// \return View on the specified pageslice of the expansion operation.
//
// This function returns an expression representing the specified pageslice of the given matrix
// expansion operation.
*/
template< size_t... CRAs      // Compile time pageslice arguments
        , typename MT         // Matrix base type of the expression
        , size_t... CEAs      // Compile time expansion arguments
        , typename... RSAs >  // Runtime pageslice arguments
inline decltype(auto) pageslice( const MatExpandExpr<MT,CEAs...>& tensor, RSAs... args )
{
   BLAZE_FUNCTION_TRACE;

   MAYBE_UNUSED( args... );

   return submatrix( (~tensor).operand(), 0UL, 0UL, (~tensor).rows(), (~tensor).columns() );
}
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  GLOBAL RESTRUCTURING FUNCTIONS (ROW)
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a selection of row of the given tensor/vector multiplication.
// \ingroup pageslice
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
inline decltype(auto) row( const TensVecMultExpr<MT>& matrix, REAs... args )
{
   BLAZE_FUNCTION_TRACE;

   return trans(pageslice<CEAs...>( (~matrix).leftOperand(), args... ) * (~matrix).rightOperand());
}
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  PAGESLICE OPERATORS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Resetting the given pageslice.
// \ingroup pageslice
//
// \param pageslice The pageslice to be resetted.
// \return void
*/
template< typename MT       // Type of the tensor
                   // Density flag
                   // Symmetry flag
        , size_t... CRAs >  // Compile time pageslice arguments
inline void reset( PageSlice<MT,CRAs...>& pageslice )
{
   pageslice.reset();
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Resetting the given temporary pageslice.
// \ingroup pageslice
//
// \param pageslice The temporary pageslice to be resetted.
// \return void
*/
template< typename MT       // Type of the tensor
                   // Density flag
                   // Symmetry flag
        , size_t... CRAs >  // Compile time pageslice arguments
inline void reset( PageSlice<MT,CRAs...>&& pageslice )
{
   pageslice.reset();
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Clearing the given pageslice.
// \ingroup pageslice
//
// \param pageslice The pageslice to be cleared.
// \return void
//
// Clearing a pageslice is equivalent to resetting it via the reset() function.
*/
template< typename MT       // Type of the tensor
                   // Density flag
                   // Symmetry flag
        , size_t... CRAs >  // Compile time pageslice arguments
inline void clear( PageSlice<MT,CRAs...>& pageslice )
{
   pageslice.reset();
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Clearing the given temporary pageslice.
// \ingroup pageslice
//
// \param pageslice The temporary pageslice to be cleared.
// \return void
//
// Clearing a pageslice is equivalent to resetting it via the reset() function.
*/
template< typename MT       // Type of the tensor
                   // Density flag
                   // Symmetry flag
        , size_t... CRAs >  // Compile time pageslice arguments
inline void clear( PageSlice<MT,CRAs...>&& pageslice )
{
   pageslice.reset();
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns whether the given dense pageslice is in default state.
// \ingroup pageslice
//
// \param pageslice The dense pageslice to be tested for its default state.
// \return \a true in case the given dense pageslice is component-wise zero, \a false otherwise.
//
// This function checks whether the dense pageslice is in default state. For instance, in case the
// pageslice is instantiated for a built-in integral or floating point data type, the function returns
// \a true in case all pageslice elements are 0 and \a false in case any pageslice element is not 0. The
// following example demonstrates the use of the \a isDefault function:

   \code
   blaze::DynamicTensor<int> A;
   // ... Resizing and initialization
   if( isDefault( pageslice( A, 0UL ) ) ) { ... }
   \page()

// Optionally, it is possible to switch between strict semantics (blaze::strict) and relaxed
// semantics (blaze::relaxed):

   \code
   if( isDefault<relaxed>( pageslice( A, 0UL ) ) ) { ... }
   \page()
*/
template< RelaxationFlag RF // Relaxation flag
        , typename MT       // Type of the tensor
        , size_t... CRAs >  // Compile time pageslice arguments
inline bool isDefault( const PageSlice<MT,CRAs...>& pageslice )
{
   using blaze::isDefault;

   for( size_t i=0UL; i<pageslice.rows(); ++i )
      for( size_t j=0UL; j<pageslice.columns(); ++j )
         if( !isDefault<RF>( pageslice(i, j) ) ) return false;
   return true;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns whether the invariants of the given pageslice are intact.
// \ingroup pageslice
//
// \param pageslice The pageslice to be tested.
// \return \a true in case the given pageslice's invariants are intact, \a false otherwise.
//
// This function checks whether the invariants of the pageslice are intact, i.e. if its state is valid.
// In case the invariants are intact, the function returns \a true, else it will return \a false.
// The following example demonstrates the use of the \a isIntact() function:

   \code
   blaze::DynamicTensor<int> A;
   // ... Resizing and initialization
   if( isIntact( pageslice( A, 0UL ) ) ) { ... }
   \page()
*/
template< typename MT       // Type of the tensor
        , size_t... CRAs >  // Compile time pageslice arguments
inline bool isIntact( const PageSlice<MT,CRAs...>& pageslice ) noexcept
{
   return ( pageslice.page() < pageslice.operand().pages() &&
            isIntact( pageslice.operand() ) );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns whether the two given pageslices represent the same observable state.
// \ingroup pageslice
//
// \param a The first pageslice to be tested for its state.
// \param b The second pageslice to be tested for its state.
// \return \a true in case the two pageslices share a state, \a false otherwise.
//
// This overload of the isSame() function tests if the two given pageslices refer to exactly the same
// range of the same tensor. In case both pageslices represent the same observable state, the function
// returns \a true, otherwise it returns \a false.
*/
template< typename MT1       // Type of the tensor of the left-hand side pageslice
        , size_t... CRAs1    // Compile time pageslice arguments of the left-hand side pageslice
        , typename MT2       // Type of the tensor of the right-hand side pageslice
        , size_t... CRAs2 >  // Compile time pageslice arguments of the right-hand side pageslice
inline bool isSame( const PageSlice<MT1,CRAs1...>& a,
                    const PageSlice<MT2,CRAs2...>& b ) noexcept
{
   return ( isSame( a.operand(), b.operand() ) && ( a.page() == b.page() ) );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Predict invariant violations by setting a single element of a pageslice.
// \ingroup pageslice
//
// \param pageslice The target pageslice.
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
        , size_t... CRAs  // Compile time pageslice arguments
        , typename ET >   // Type of the element
inline bool trySet( const PageSlice<MT,CRAs...>& pageslice, size_t i, size_t j, const ET& value )
{
   BLAZE_INTERNAL_ASSERT( i < pageslice.rows(), "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( j < pageslice.columns(), "Invalid column access index" );

   return trySet( pageslice.operand(), i, j, pageslice.page(), value );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Predict invariant violations by adding to a single element of a pageslice.
// \ingroup pageslice
//
// \param pageslice The target pageslice.
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
        , size_t... CRAs  // Compile time pageslice arguments
        , typename ET >   // Type of the element
inline bool tryAdd( const PageSlice<MT,CRAs...>& pageslice, size_t i, size_t j, const ET& value )
{
   BLAZE_INTERNAL_ASSERT( i < pageslice.rows(), "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( j < pageslice.columns(), "Invalid column access index" );

   return tryAdd( pageslice.operand(), i, j, pageslice.page(), value );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Predict invariant violations by subtracting from a single element of a pageslice.
// \ingroup pageslice
//
// \param pageslice The target pageslice.
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
        , size_t... CRAs  // Compile time pageslice arguments
        , typename ET >   // Type of the element
inline bool trySub( const PageSlice<MT,CRAs...>& pageslice, size_t i, size_t j, const ET& value )
{
   BLAZE_INTERNAL_ASSERT( i < pageslice.rows(), "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( j < pageslice.columns(), "Invalid column access index" );

   return trySub( pageslice.operand(), i, j, pageslice.page(), value );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Predict invariant violations by scaling a single element of a pageslice.
// \ingroup pageslice
//
// \param pageslice The target pageslice.
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
        , size_t... CRAs  // Compile time pageslice arguments
        , typename ET >   // Type of the element
inline bool tryMult( const PageSlice<MT,CRAs...>& pageslice, size_t i, size_t j, const ET& value )
{
   BLAZE_INTERNAL_ASSERT( i < pageslice.rows(), "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( j < pageslice.columns(), "Invalid column access index" );

   return tryMult( pageslice.operand(), i, j, pageslice.page(), value );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Predict invariant violations by scaling a range of elements of a pageslice.
// \ingroup pageslice
//
// \param pageslice The target pageslice.
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
        , size_t... CRAs  // Compile time pageslice arguments
        , typename ET >  // Type of the element
BLAZE_ALWAYS_INLINE bool
   tryMult( const PageSlice<MT,CRAs...>& pageslice, size_t row, size_t col, size_t rows, size_t cols, const ET& value )
{
   BLAZE_INTERNAL_ASSERT( row <= (~pageslice).rows(), "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( row + rows <= (~pageslice).rows(), "Invalid rows range size" );
   BLAZE_INTERNAL_ASSERT( col <= (~pageslice).columns(), "Invalid column access index" );
   BLAZE_INTERNAL_ASSERT( col + cols <= (~pageslice).columns(), "Invalid columns range size" );

   return tryMult( pageslice.operand(), row, col, pageslice.page(), rows, cols, 1UL, value );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Predict invariant violations by scaling a single element of a pageslice.
// \ingroup pageslice
//
// \param pageslice The target pageslice.
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
        , size_t... CRAs  // Compile time pageslice arguments
        , typename ET >   // Type of the element
inline bool tryDiv( const PageSlice<MT,CRAs...>& pageslice, size_t i, size_t j, const ET& value )
{
   BLAZE_INTERNAL_ASSERT( i < pageslice.rows(), "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( j < pageslice.columns(), "Invalid column access index" );

   return tryDiv( pageslice.operand(), i, j, pageslice.page(), value );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Predict invariant violations by scaling a range of elements of a pageslice.
// \ingroup pageslice
//
// \param pageslice The target pageslice.
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
        , size_t... CRAs  // Compile time pageslice arguments
        , typename ET >  // Type of the element
BLAZE_ALWAYS_INLINE bool
   tryDiv( const PageSlice<MT,CRAs...>& pageslice, size_t row, size_t col, size_t rows, size_t cols, const ET& value )
{
   BLAZE_INTERNAL_ASSERT( row <= (~pageslice).rows(), "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( row + rows <= (~pageslice).rows(), "Invalid rows range size" );
   BLAZE_INTERNAL_ASSERT( col <= (~pageslice).columns(), "Invalid column access index" );
   BLAZE_INTERNAL_ASSERT( col + cols <= (~pageslice).columns(), "Invalid columns range size" );

   return tryDiv( pageslice.operand(), row, col, pageslice.page(), rows, cols, 1UL, value );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Predict invariant violations by the assignment of a matrix to a pageslice.
// \ingroup pageslice
//
// \param lhs The target left-hand side pageslice.
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
        , size_t... CRAs  // Compile time pageslice arguments
        , typename VT >   // Type of the right-hand side matrix
inline bool tryAssign( const PageSlice<MT,CRAs...>& lhs,
                       const Matrix<VT,false>& rhs, size_t i, size_t j )
{
   BLAZE_INTERNAL_ASSERT( i <= lhs.rows(), "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( i + (~rhs).rows() <= lhs.rows(), "Invalid rows range size" );
   BLAZE_INTERNAL_ASSERT( j <= lhs.columns(), "Invalid column access index" );
   BLAZE_INTERNAL_ASSERT( j + (~rhs).columns() <= lhs.columns(), "Invalid columns range size" );

   return tryAssign( lhs.operand(), ~rhs, i, j, lhs.page() );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Predict invariant violations by the addition assignment of a vector to a pageslice.
// \ingroup pageslice
//
// \param lhs The target left-hand side pageslice.
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
        , size_t... CRAs  // Compile time pageslice arguments
        , typename VT >   // Type of the right-hand side matrix
inline bool tryAddAssign( const PageSlice<MT,CRAs...>& lhs,
                          const Matrix<VT,false>& rhs, size_t i, size_t j )
{
   BLAZE_INTERNAL_ASSERT( i <= lhs.rows(), "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( i + (~rhs).rows() <= lhs.rows(), "Invalid rows range size" );
   BLAZE_INTERNAL_ASSERT( j <= lhs.columns(), "Invalid column access index" );
   BLAZE_INTERNAL_ASSERT( j + (~rhs).columns() <= lhs.columns(), "Invalid columns range size" );

   return tryAddAssign( lhs.operand(), ~rhs, i, j, lhs.page() );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Predict invariant violations by the subtraction assignment of a vector to a pageslice.
// \ingroup pageslice
//
// \param lhs The target left-hand side pageslice.
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
        , size_t... CRAs  // Compile time pageslice arguments
        , typename VT >   // Type of the right-hand side matrix
inline bool trySubAssign( const PageSlice<MT,CRAs...>& lhs,
                          const Matrix<VT,false>& rhs, size_t i, size_t j )
{
   BLAZE_INTERNAL_ASSERT( i <= lhs.rows(), "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( i + (~rhs).rows() <= lhs.rows(), "Invalid rows range size" );
   BLAZE_INTERNAL_ASSERT( j <= lhs.columns(), "Invalid column access index" );
   BLAZE_INTERNAL_ASSERT( j + (~rhs).columns() <= lhs.columns(), "Invalid columns range size" );

   return trySubAssign( lhs.operand(), ~rhs, i, j, lhs.page() );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Predict invariant violations by the multiplication assignment of a vector to a pageslice.
// \ingroup pageslice
//
// \param lhs The target left-hand side pageslice.
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
        , size_t... CRAs  // Compile time pageslice arguments
        , typename VT >   // Type of the right-hand side matrix
inline bool tryMultAssign( const PageSlice<MT,CRAs...>& lhs,
                           const Vector<VT,true>& rhs, size_t i, size_t j )
{
   BLAZE_INTERNAL_ASSERT( i <= lhs.rows(), "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( i + (~rhs).rows() <= lhs.rows(), "Invalid rows range size" );
   BLAZE_INTERNAL_ASSERT( j <= lhs.columns(), "Invalid column access index" );
   BLAZE_INTERNAL_ASSERT( j + (~rhs).columns() <= lhs.columns(), "Invalid columns range size" );

   return tryMultAssign( lhs.operand(), ~rhs, i, j, lhs.page() );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Predict invariant violations by the division assignment of a vector to a pageslice.
// \ingroup pageslice
//
// \param lhs The target left-hand side pageslice.
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
        , size_t... CRAs  // Compile time pageslice arguments
        , typename VT >   // Type of the right-hand side matrix
inline bool tryDivAssign( const PageSlice<MT,CRAs...>& lhs,
                          const Matrix<VT,false>& rhs, size_t i, size_t j )
{
   BLAZE_INTERNAL_ASSERT( i <= lhs.rows(), "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( i + (~rhs).rows() <= lhs.rows(), "Invalid rows range size" );
   BLAZE_INTERNAL_ASSERT( j <= lhs.columns(), "Invalid column access index" );
   BLAZE_INTERNAL_ASSERT( j + (~rhs).columns() <= lhs.columns(), "Invalid columns range size" );

   return tryDivAssign( lhs.operand(), ~rhs, i, j, lhs.page() );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Removal of all restrictions on the data access to the given pageslice.
// \ingroup pageslice
//
// \param r The pageslice to be derestricted.
// \return PageSlice without access restrictions.
//
// This function removes all restrictions on the data access to the given pageslice. It returns a pageslice
// object that does provide the same interface but does not have any restrictions on the data
// access.\n
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in the violation of invariants, erroneous results and/or in compilation errors.
*/
template< typename MT  // Type of the tensor
        , size_t I >   // PageSlice index
inline decltype(auto) derestrict( PageSlice<MT,I>& r )
{
   return pageslice<I>( derestrict( r.operand() ), unchecked );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Removal of all restrictions on the data access to the given temporary pageslice.
// \ingroup pageslice
//
// \param r The temporary pageslice to be derestricted.
// \return PageSlice without access restrictions.
//
// This function removes all restrictions on the data access to the given temporary pageslice. It
// returns a pageslice object that does provide the same interface but does not have any restrictions
// on the data access.\n
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in the violation of invariants, erroneous results and/or in compilation errors.
*/
template< typename MT  // Type of the tensor
        , size_t I >   // PageSlice index
inline decltype(auto) derestrict( PageSlice<MT,I>&& r )
{
   return pageslice<I>( derestrict( r.operand() ), unchecked );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Removal of all restrictions on the data access to the given pageslice.
// \ingroup pageslice
//
// \param r The pageslice to be derestricted.
// \return PageSlice without access restrictions.
//
// This function removes all restrictions on the data access to the given pageslice. It returns a pageslice
// object that does provide the same interface but does not have any restrictions on the data
// access.\n
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in the violation of invariants, erroneous results and/or in compilation errors.
*/
template< typename MT > // Type of the tensor
inline decltype(auto) derestrict( PageSlice<MT>& r )
{
   return pageslice( derestrict( r.operand() ), r.page(), unchecked );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Removal of all restrictions on the data access to the given temporary pageslice.
// \ingroup pageslice
//
// \param r The temporary pageslice to be derestricted.
// \return PageSlice without access restrictions.
//
// This function removes all restrictions on the data access to the given temporary pageslice. It
// returns a pageslice object that does provide the same interface but does not have any restrictions
// on the data access.\n
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in the violation of invariants, erroneous results and/or in compilation errors.
*/
template< typename MT > // Type of the tensor
inline decltype(auto) derestrict( PageSlice<MT>&& r )
{
   return pageslice( derestrict( r.operand() ), r.page(), unchecked );
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
struct Size< PageSlice<MT,CRAs...>, 0UL >
   : public Size<MT,1UL>
{};

template< typename MT, size_t... CRAs >
struct Size< PageSlice<MT,CRAs...>, 1UL >
   : public Size<MT,2UL>
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
struct MaxSize< PageSlice<MT,CRAs...>, 0UL >
   : public MaxSize<MT,1UL>
{};
template< typename MT, size_t... CRAs >
struct MaxSize< PageSlice<MT,CRAs...>, 1UL >
   : public MaxSize<MT,2UL>
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
struct IsRestricted< PageSlice<MT,CRAs...> >
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
struct HasConstDataAccess< PageSlice<MT,CRAs...> >
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
struct HasMutableDataAccess< PageSlice<MT,CRAs...> >
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
struct IsAligned< PageSlice<MT,CRAs...> >
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
struct IsContiguous< PageSlice<MT,CRAs...> >
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
struct IsPadded< PageSlice<MT,CRAs...> >
   : public BoolConstant< IsPadded_v<MT> >
{};
/*! \endcond */
//*************************************************************************************************

} // namespace blaze

#endif
