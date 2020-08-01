//=================================================================================================
/*!
//  \file blaze_tensor/math/views/DilatedSubvector.h
//  \brief Header file for the implementation of the DilatedSubvector view
//
//  Copyright (C) 2012-2019 Klaus Iglberger - All Rights Reserved
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

#ifndef _BLAZE_TENSOR_MATH_VIEWS_DILATEDSUBVECTOR_H_
#define _BLAZE_TENSOR_MATH_VIEWS_DILATEDSUBVECTOR_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <algorithm>
#include <blaze/math/Aliases.h>
#include <blaze/math/Exception.h>
#include <blaze/math/expressions/CrossExpr.h>
#include <blaze/math/expressions/VecEvalExpr.h>
#include <blaze/math/expressions/VecMapExpr.h>
#include <blaze/math/expressions/VecScalarDivExpr.h>
#include <blaze/math/expressions/VecScalarMultExpr.h>
#include <blaze/math/expressions/VecSerialExpr.h>
#include <blaze/math/expressions/VecTransExpr.h>
#include <blaze/math/expressions/VecVecAddExpr.h>
#include <blaze/math/expressions/VecVecDivExpr.h>
#include <blaze/math/expressions/VecVecMapExpr.h>
#include <blaze/math/expressions/VecVecMultExpr.h>
#include <blaze/math/expressions/VecVecSubExpr.h>
#include <blaze/math/expressions/Vector.h>
#include <blaze/math/shims/IsDefault.h>
#include <blaze/math/shims/Serial.h>
#include <blaze/math/typetraits/HasConstDataAccess.h>
#include <blaze/math/typetraits/HasMutableDataAccess.h>
#include <blaze/math/typetraits/IsRestricted.h>
#include <blaze/math/typetraits/MaxSize.h>
#include <blaze/math/typetraits/Size.h>
#include <blaze/math/views/Check.h>
#include <blaze/util/Assert.h>
#include <blaze/util/FunctionTrace.h>
#include <blaze/util/IntegralConstant.h>
#include <blaze/util/SmallArray.h>
#include <blaze/util/StaticAssert.h>
#include <blaze/util/TypeList.h>
#include <blaze/util/Types.h>
#include <blaze/util/typetraits/RemoveReference.h>

#include <blaze_tensor/math/IntegerSequence.h>
#include <blaze_tensor/math/views/Forward.h>
#include <blaze_tensor/math/views/dilatedsubvector/BaseTemplate.h>
#include <blaze_tensor/math/views/dilatedsubvector/Dense.h>
// #include <blaze_tensor/math/views/dilatedsubvector/Sparse.h>

namespace blaze {

//=================================================================================================
//
//  GLOBAL FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Creating a view on a specific dilatedsubvector of the given vector.
// \ingroup dilatedsubvector
//
// \param vector The vector containing the dilatedsubvector.
// \param args Optional dilatedsubvector arguments.
// \return View on the specific dilatedsubvector of the vector.
// \exception std::invalid_argument Invalid dilatedsubvector specification.
//
// This function returns an expression representing the specified dilatedsubvector of the given vector.
// The following example demonstrates the creation of a dense and sparse dilatedsubvector:

   \code
   blaze::DynamicVector<double,blaze::columnVector> d;
   blaze::CompressedVector<int,blaze::rowVector> s;
   // ... Resizing and initialization

   // Creating a dense dilatedsubvector of size 8, starting from index 4
   auto dsv = dilatedsubvector<4UL,8UL>( d );

   // Creating a sparse dilatedsubvector of size 7, starting from index 5
   auto ssv = dilatedsubvector<5UL,7UL>( s );
   \endcode

// By default, the provided dilatedsubvector arguments are checked at runtime. In case the dilatedsubvector
// is not properly specified (i.e. if the specified first index is greater than the total size
// of the given vector or the dilatedsubvector is specified beyond the size of the vector) a
// \a std::invalid_argument exception is thrown. The checks can be skipped by providing the
// optional \a blaze::unchecked argument.

   \code
   auto dsv = dilatedsubvector<4UL,8UL>( d, unchecked );
   auto ssv = dilatedsubvector<5UL,7UL>( s, unchecked );
   \endcode

// Please note that this function creates an unaligned dense or sparse dilatedsubvector. For instance,
// the creation of the dense dilatedsubvector is equivalent to the following function call:

   \code
   auto dsv = dilatedsubvector<unaligned,4UL,8UL>( d );
   \endcode

// In contrast to unaligned dilatedsubvectors, which provide full flexibility, aligned dilatedsubvectors pose
// additional alignment restrictions. However, especially in case of dense dilatedsubvectors this may
// result in considerable performance improvements. In order to create an aligned dilatedsubvector the
// following function call has to be used:

   \code
   auto dsv = dilatedsubvector<aligned,4UL,8UL>( d );
   \endcode

// Note however that in this case the given compile time arguments \a I and \a N are subject to
// additional checks to guarantee proper alignment.
*/
template< size_t I            // Index of the first dilatedsubvector element
        , size_t N            // Size of the dilatedsubvector
        , size_t Dilation     // Step between elements of the dilatedsubvector
        , typename VT         // Type of the vector
        , bool TF             // Transpose flag
        , typename... RSAs >  // Optional dilatedsubvector arguments
inline decltype(auto) dilatedsubvector( Vector<VT,TF>& vector, RSAs... args )
{
   BLAZE_FUNCTION_TRACE;

   return dilatedsubvector<I,N,Dilation>( *vector, args... );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Creating a view on a specific dilatedsubvector of the given constant vector.
// \ingroup dilatedsubvector
//
// \param vector The constant vector containing the dilatedsubvector.
// \param args Optional dilatedsubvector arguments.
// \return View on the specific dilatedsubvector of the vector.
// \exception std::invalid_argument Invalid dilatedsubvector specification.
//
// This function returns an expression representing the specified dilatedsubvector of the given constant
// vector. The following example demonstrates the creation of a dense and sparse dilatedsubvector:

   \code
   const blaze::DynamicVector<double,blaze::columnVector> d( ... );
   const blaze::CompressedVector<int,blaze::rowVector> s( ... );
   // ... Resizing and initialization

   // Creating a dense dilatedsubvector of size 8, starting from index 4, every second element
   auto dsv = dilatedsubvector<4UL,8UL,2UL>( d );

   // Creating a sparse dilatedsubvector of size 7, starting from index 5, every third element
   auto ssv = dilatedsubvector<5UL,7UL,3UL>( s );
   \endcode

// By default, the provided dilatedsubvector arguments are checked at runtime. In case the dilatedsubvector
// is not properly specified (i.e. if the specified first index is greater than the total size
// of the given vector or the dilatedsubvector is specified beyond the size of the vector) a
// \a std::invalid_argument exception is thrown. The checks can be skipped by providing the
// optional \a blaze::unchecked argument.

   \code
   auto dsv = dilatedsubvector<4UL,8UL,2UL>( d, unchecked );
   auto ssv = dilatedsubvector<5UL,7UL,2UL>( s, unchecked );
   \endcode

// Please note that this function always creates an unaligned dense or sparse dilatedsubvector. For instance,
// the creation of the dense dilatedsubvector is equivalent to the following function call:

   \code
   auto dsv = dilatedsubvector<4UL,8UL,3UL>( d );
   \endcode
*/
template< size_t I            // Index of the first dilatedsubvector element
        , size_t N            // Size of the dilatedsubvector
        , size_t Dilation     // Step between elements of the dilatedsubvector
        , typename VT         // Type of the vector
        , bool TF             // Transpose flag
        , typename... RSAs >  // Optional dilatedsubvector arguments
inline decltype(auto) dilatedsubvector( const Vector<VT,TF>& vector, RSAs... args )
{
   BLAZE_FUNCTION_TRACE;

   return dilatedsubvector<I,N,Dilation>( *vector, args... );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Creating a view on a specific dilatedsubvector of the given temporary vector.
// \ingroup dilatedsubvector
//
// \param vector The temporary vector containing the dilatedsubvector.
// \param args Optional dilatedsubvector arguments.
// \return View on the specific dilatedsubvector of the vector.
// \exception std::invalid_argument Invalid dilatedsubvector specification.
//
// This function returns an expression representing the specified dilatedsubvector of the given
// temporary vector. In case the dilatedsubvector is not properly specified (i.e. if the specified
// first index is greater than the total size of the given vector or the dilatedsubvector is specified
// beyond the size of the vector) a \a std::invalid_argument exception is thrown.
*/
template< size_t I            // Index of the first dilatedsubvector element
        , size_t N            // Size of the dilatedsubvector
        , size_t Dilation     // Step between elements of the dilatedsubvector
        , typename VT         // Type of the vector
        , bool TF             // Transpose flag
        , typename... RSAs >  // Optional dilatedsubvector arguments
inline decltype(auto) dilatedsubvector( Vector<VT,TF>&& vector, RSAs... args )
{
   BLAZE_FUNCTION_TRACE;

   return dilatedsubvector<I,N,Dilation>( *vector, args... );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Creating a view on a specific dilatedsubvector of the given vector.
// \ingroup dilatedsubvector
//
// \param vector The vector containing the dilatedsubvector.
// \param index The index of the first element of the dilatedsubvector.
// \param size The size of the dilatedsubvector.
// \param args Optional dilatedsubvector arguments.
// \return View on the specific dilatedsubvector of the vector.
// \exception std::invalid_argument Invalid dilatedsubvector specification.
//
// This function returns an expression representing the specified dilatedsubvector of the given vector.
// The following example demonstrates the creation of a dense and sparse dilatedsubvector:

   \code
   blaze::DynamicVector<double,blaze::columnVector> d;
   blaze::CompressedVector<int,blaze::rowVector> s;
   // ... Resizing and initialization

   // Creating a dense dilatedsubvector of size 8, starting from index 4
   auto dsv = dilatedsubvector( d, 4UL, 8UL );

   // Creating a sparse dilatedsubvector of size 7, starting from index 5
   auto ssv = dilatedsubvector( s, 5UL, 7UL );
   \endcode

// By default, the provided dilatedsubvector arguments are checked at runtime. In case the dilatedsubvector
// is not properly specified (i.e. if the specified first index is greater than the total size
// of the given vector or the dilatedsubvector is specified beyond the size of the vector) a
// \a std::invalid_argument exception is thrown. The checks can be skipped by providing the
// optional \a blaze::unchecked argument.

   \code
   auto dsv = dilatedsubvector( d, 4UL, 8UL, unchecked );
   auto ssv = dilatedsubvector( s, 5UL, 7UL, unchecked );
   \endcode

// Please note that this function creates an unaligned dense or sparse dilatedsubvector. For instance,
// the creation of the dense dilatedsubvector is equivalent to the following function call:

   \code
   auto dsv = dilatedsubvector<unaligned>( d, 4UL, 8UL );
   \endcode

// In contrast to unaligned dilatedsubvectors, which provide full flexibility, aligned dilatedsubvectors pose
// additional alignment restrictions. However, especially in case of dense dilatedsubvectors this may
// result in considerable performance improvements. In order to create an aligned dilatedsubvector the
// following function call has to be used:

   \code
   auto dsv = dilatedsubvector<aligned>( d, 4UL, 8UL );
   \endcode

// Note however that in this case the given \a index and \a size are subject to additional checks
// to guarantee proper alignment.
*/
template< typename VT         // Type of the vector
        , bool TF             // Transpose flag
        , typename... RSAs >  // Optional dilatedsubvector arguments
inline decltype(auto) dilatedsubvector( Vector<VT,TF>& vector, size_t index, size_t size, size_t dilation, RSAs... args )
{
   BLAZE_FUNCTION_TRACE;

   using ReturnType = DilatedSubvector_<VT>;
   return ReturnType( *vector, index, size, dilation, args... );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Creating a view on a specific dilatedsubvector of the given constant vector.
// \ingroup dilatedsubvector
//
// \param vector The constant vector containing the dilatedsubvector.
// \param index The index of the first element of the dilatedsubvector.
// \param size The size of the dilatedsubvector.
// \param args Optional dilatedsubvector arguments.
// \return View on the specific dilatedsubvector of the vector.
// \exception std::invalid_argument Invalid dilatedsubvector specification.
//
// This function returns an expression representing the specified dilatedsubvector of the given constant
// vector. The following example demonstrates the creation of a dense and sparse dilatedsubvector:

   \code
   const blaze::DynamicVector<double,blaze::columnVector> d( ... );
   const blaze::CompressedVector<int,blaze::rowVector> s( ... );
   // ... Resizing and initialization

   // Creating a dense dilatedsubvector of size 8, starting from index 4
   auto dsv = dilatedsubvector( d, 4UL, 8UL );

   // Creating a sparse dilatedsubvector of size 7, starting from index 5
   auto ssv = dilatedsubvector( s, 5UL, 7UL );
   \endcode

// By default, the provided dilatedsubvector arguments are checked at runtime. In case the dilatedsubvector
// is not properly specified (i.e. if the specified first index is greater than the total size
// of the given vector or the dilatedsubvector is specified beyond the size of the vector) a
// \a std::invalid_argument exception is thrown. The checks can be skipped by providing the
// optional \a blaze::unchecked argument.

   \code
   auto dsv = dilatedsubvector( d, 4UL, 8UL, unchecked );
   auto ssv = dilatedsubvector( s, 5UL, 7UL, unchecked );
   \endcode

// Please note that this function creates an unaligned dense or sparse dilatedsubvector. For instance,
// the creation of the dense dilatedsubvector is equivalent to the following function call:

   \code
   auto dsv = dilatedsubvector<unaligned>( d, 4UL, 8UL );
   \endcode

// In contrast to unaligned dilatedsubvectors, which provide full flexibility, aligned dilatedsubvectors pose
// additional alignment restrictions. However, especially in case of dense dilatedsubvectors this may
// result in considerable performance improvements. In order to create an aligned dilatedsubvector the
// following function call has to be used:

   \code
   auto dsv = dilatedsubvector<aligned>( d, 4UL, 8UL );
   \endcode

// Note however that in this case the given \a index and \a size are subject to additional checks
// to guarantee proper alignment.
*/
template< typename VT         // Type of the vector
        , bool TF             // Transpose flag
        , typename... RSAs >  // Optional dilatedsubvector arguments
inline decltype(auto) dilatedsubvector( const Vector<VT,TF>& vector, size_t index, size_t size, size_t dilation, RSAs... args )
{
   BLAZE_FUNCTION_TRACE;

   using ReturnType = const DilatedSubvector_<const VT>;
   return ReturnType( *vector, index, size, dilation, args... );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Creating a view on a specific dilatedsubvector of the given temporary vector.
// \ingroup dilatedsubvector
//
// \param vector The temporary vector containing the dilatedsubvector.
// \param index The index of the first element of the dilatedsubvector.
// \param size The size of the dilatedsubvector.
// \param args Optional dilatedsubvector arguments.
// \return View on the specific dilatedsubvector of the vector.
// \exception std::invalid_argument Invalid dilatedsubvector specification.
//
// This function returns an expression representing the specified dilatedsubvector of the given
// temporary vector. In case the dilatedsubvector is not properly specified (i.e. if the specified
// first index is greater than the total size of the given vector or the dilatedsubvector is specified
// beyond the size of the vector) a \a std::invalid_argument exception is thrown.
*/
template< typename VT         // Type of the vector
        , bool TF             // Transpose flag
        , typename... RSAs >  // Optional dilatedsubvector arguments
inline decltype(auto) dilatedsubvector( Vector<VT,TF>&& vector, size_t index, size_t size, size_t dilation, RSAs... args )
{
   BLAZE_FUNCTION_TRACE;

   using ReturnType = DilatedSubvector_<VT>;
   return ReturnType( *vector, index, size, dilation, args... );
}
//*************************************************************************************************


//=================================================================================================
//
//  GLOBAL RESTRUCTURING FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific dilatedsubvector of the given vector/vector addition.
// \ingroup dilatedsubvector
//
// \param vector The constant vector/vector addition.
// \param args The runtime dilatedsubvector arguments.
// \return View on the specified dilatedsubvector of the addition.
//
// This function returns an expression representing the specified dilatedsubvector of the given
// vector/vector addition.
*/
template< size_t... CSAs      // Compile time dilatedsubvector arguments
        , typename VT         // Vector base type of the expression
        , typename... RSAs >  // Runtime dilatedsubvector arguments
inline decltype(auto) dilatedsubvector( const VecVecAddExpr<VT>& vector, RSAs... args )
{
   BLAZE_FUNCTION_TRACE;

   return dilatedsubvector<CSAs...>( (*vector).leftOperand(), args... ) +
          dilatedsubvector<CSAs...>( (*vector).rightOperand(), args... );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific dilatedsubvector of the given vector/vector subtraction.
// \ingroup dilatedsubvector
//
// \param vector The constant vector/vector subtraction.
// \param args The runtime dilatedsubvector arguments.
// \return View on the specified dilatedsubvector of the subtraction.
//
// This function returns an expression representing the specified dilatedsubvector of the given
// vector/vector subtraction.
*/
template< size_t... CSAs      // Compile time dilatedsubvector arguments
        , typename VT         // Vector base type of the expression
        , typename... RSAs >  // Runtime dilatedsubvector arguments
inline decltype(auto) dilatedsubvector( const VecVecSubExpr<VT>& vector, RSAs... args )
{
   BLAZE_FUNCTION_TRACE;

   return dilatedsubvector<CSAs...>( (*vector).leftOperand(), args... ) -
          dilatedsubvector<CSAs...>( (*vector).rightOperand(), args... );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific dilatedsubvector of the given vector/vector multiplication.
// \ingroup dilatedsubvector
//
// \param vector The constant vector/vector multiplication.
// \param args The runtime dilatedsubvector arguments.
// \return View on the specified dilatedsubvector of the multiplication.
//
// This function returns an expression representing the specified dilatedsubvector of the given
// vector/vector multiplication.
*/
template< size_t... CSAs      // Compile time dilatedsubvector arguments
        , typename VT         // Vector base type of the expression
        , typename... RSAs >  // Runtime dilatedsubvector arguments
inline decltype(auto) dilatedsubvector( const VecVecMultExpr<VT>& vector, RSAs... args )
{
   BLAZE_FUNCTION_TRACE;

   return dilatedsubvector<CSAs...>( (*vector).leftOperand(), args... ) *
          dilatedsubvector<CSAs...>( (*vector).rightOperand(), args... );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific dilatedsubvector of the given vector/vector division.
// \ingroup dilatedsubvector
//
// \param vector The constant vector/vector division.
// \param args The runtime dilatedsubvector arguments.
// \return View on the specified dilatedsubvector of the division.
//
// This function returns an expression representing the specified dilatedsubvector of the given
// vector/vector division.
*/
template< size_t... CSAs      // Compile time dilatedsubvector arguments
        , typename VT         // Vector base type of the expression
        , typename... RSAs >  // Runtime dilatedsubvector arguments
inline decltype(auto) dilatedsubvector( const VecVecDivExpr<VT>& vector, RSAs... args )
{
   BLAZE_FUNCTION_TRACE;

   return dilatedsubvector<CSAs...>( (*vector).leftOperand(), args... ) /
          dilatedsubvector<CSAs...>( (*vector).rightOperand(), args... );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific dilatedsubvector of the given vector/vector cross product.
// \ingroup dilatedsubvector
//
// \param vector The constant vector/vector cross product.
// \param args The runtime dilatedsubvector arguments.
// \return View on the specified dilatedsubvector of the cross product.
//
// This function returns an expression representing the specified dilatedsubvector of the given
// vector/vector cross product.
*/
template< size_t... CSAs      // Compile time dilatedsubvector arguments
        , typename VT         // Vector base type of the expression
        , typename... RSAs >  // Runtime dilatedsubvector arguments
inline decltype(auto) dilatedsubvector( const CrossExpr<VT>& vector, RSAs... args )
{
   BLAZE_FUNCTION_TRACE;

   using ReturnType = DilatedSubvector_< VectorType_t<VT>, CSAs... >;
   return ReturnType( *vector, args... );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific dilatedsubvector of the given vector/scalar multiplication.
// \ingroup dilatedsubvector
//
// \param vector The constant vector/scalar multiplication.
// \param args The runtime dilatedsubvector arguments.
// \return View on the specified dilatedsubvector of the multiplication.
//
// This function returns an expression representing the specified dilatedsubvector of the given
// vector/scalar multiplication.
*/
template< size_t... CSAs      // Compile time dilatedsubvector arguments
        , typename VT         // Vector base type of the expression
        , typename... RSAs >  // Runtime dilatedsubvector arguments
inline decltype(auto) dilatedsubvector( const VecScalarMultExpr<VT>& vector, RSAs... args )
{
   BLAZE_FUNCTION_TRACE;

   return dilatedsubvector<CSAs...>( (*vector).leftOperand(), args... ) * (*vector).rightOperand();
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific dilatedsubvector of the given vector/scalar division.
// \ingroup dilatedsubvector
//
// \param vector The constant vector/scalar division.
// \param args The runtime dilatedsubvector arguments.
// \return View on the specified dilatedsubvector of the division.
//
// This function returns an expression representing the specified dilatedsubvector of the given
// vector/scalar division.
*/
template< size_t... CSAs      // Compile time dilatedsubvector arguments
        , typename VT         // Vector base type of the expression
        , typename... RSAs >  // Runtime dilatedsubvector arguments
inline decltype(auto) dilatedsubvector( const VecScalarDivExpr<VT>& vector, RSAs... args )
{
   BLAZE_FUNCTION_TRACE;

   return dilatedsubvector<CSAs...>( (*vector).leftOperand(), args... ) / (*vector).rightOperand();
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific dilatedsubvector of the given unary vector map operation.
// \ingroup dilatedsubvector
//
// \param vector The constant unary vector map operation.
// \param args The runtime dilatedsubvector arguments.
// \return View on the specified dilatedsubvector of the unary map operation.
//
// This function returns an expression representing the specified dilatedsubvector of the given unary
// vector map operation.
*/
template< size_t... CSAs      // Compile time dilatedsubvector arguments
        , typename VT         // Vector base type of the expression
        , typename... RSAs >  // Runtime dilatedsubvector arguments
inline decltype(auto) dilatedsubvector( const VecMapExpr<VT>& vector, RSAs... args )
{
   BLAZE_FUNCTION_TRACE;

   return map( dilatedsubvector<CSAs...>( (*vector).operand(), args... ), (*vector).operation() );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific dilatedsubvector of the given binary vector map operation.
// \ingroup dilatedsubvector
//
// \param vector The constant binary vector map operation.
// \param args The runtime dilatedsubvector arguments.
// \return View on the specified dilatedsubvector of the binary map operation.
//
// This function returns an expression representing the specified dilatedsubvector of the given binary
// vector map operation.
*/
template< size_t... CSAs      // Compile time dilatedsubvector arguments
        , typename VT         // Vector base type of the expression
        , typename... RSAs >  // Runtime dilatedsubvector arguments
inline decltype(auto) dilatedsubvector( const VecVecMapExpr<VT>& vector, RSAs... args )
{
   BLAZE_FUNCTION_TRACE;

   return map( dilatedsubvector<CSAs...>( (*vector).leftOperand(), args... ),
               dilatedsubvector<CSAs...>( (*vector).rightOperand(), args... ),
               (*vector).operation() );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific dilatedsubvector of the given vector evaluation operation.
// \ingroup dilatedsubvector
//
// \param vector The constant vector evaluation operation.
// \param args The runtime dilatedsubvector arguments.
// \return View on the specified dilatedsubvector of the evaluation operation.
//
// This function returns an expression representing the specified dilatedsubvector of the given vector
// evaluation operation.
*/
template< size_t... CSAs      // Compile time dilatedsubvector arguments
        , typename VT         // Vector base type of the expression
        , typename... RSAs >  // Runtime dilatedsubvector arguments
inline decltype(auto) dilatedsubvector( const VecEvalExpr<VT>& vector, RSAs... args )
{
   BLAZE_FUNCTION_TRACE;

   return eval( dilatedsubvector<CSAs...>( (*vector).operand(), args... ) );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific dilatedsubvector of the given vector serialization operation.
// \ingroup dilatedsubvector
//
// \param vector The constant vector serialization operation.
// \param args The runtime dilatedsubvector arguments.
// \return View on the specified dilatedsubvector of the serialization operation.
//
// This function returns an expression representing the specified dilatedsubvector of the given vector
// serialization operation.
*/
template< size_t... CSAs      // Compile time dilatedsubvector arguments
        , typename VT         // Vector base type of the expression
        , typename... RSAs >  // Runtime dilatedsubvector arguments
inline decltype(auto) dilatedsubvector( const VecSerialExpr<VT>& vector, RSAs... args )
{
   BLAZE_FUNCTION_TRACE;

   return serial( dilatedsubvector<CSAs...>( (*vector).operand(), args... ) );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific dilatedsubvector of the given vector transpose operation.
// \ingroup dilatedsubvector
//
// \param vector The constant vector transpose operation.
// \param args The runtime dilatedsubvector arguments.
// \return View on the specified dilatedsubvector of the transpose operation.
//
// This function returns an expression representing the specified dilatedsubvector of the given vector
// transpose operation.
*/
template< size_t... CSAs      // Compile time dilatedsubvector arguments
        , typename VT         // Vector base type of the expression
        , typename... RSAs >  // Runtime dilatedsubvector arguments
inline decltype(auto) dilatedsubvector( const VecTransExpr<VT>& vector, RSAs... args )
{
   BLAZE_FUNCTION_TRACE;

   return trans( dilatedsubvector<CSAs...>( (*vector).operand(), args... ) );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific dilatedsubvector of another dilatedsubvector.
// \ingroup dilatedsubvector
//
// \param sv The given dilatedsubvector.
// \param args The optional dilatedsubvector arguments.
// \return View on the specified dilatedsubvector of the other dilatedsubvector.
//
// This function returns an expression representing the specified dilatedsubvector of the given dilatedsubvector.
*/
template< size_t I1           // Required dilatedsubvector offset
        , size_t N1           // Required size of the dilatedsubvector
        , size_t Dilation1    // Required step-size of the dilatedsubvector
        , typename VT         // Type of the vector
        , bool TF             // Transpose flag
        , bool DF             // Density flag
        , size_t I2           // Present dilatedsubvector offset
        , size_t N2           // Present size of the dilatedsubvector
        , size_t Dilation2    // Present step-size of the dilatedsubvector
        , typename... RSAs >  // Optional dilatedsubvector arguments
inline decltype(auto) dilatedsubvector( DilatedSubvector<VT,TF,DF,I2,N2,Dilation2>& sv, RSAs... args )
{
   BLAZE_FUNCTION_TRACE;

   BLAZE_STATIC_ASSERT_MSG( I1 + N1*Dilation1 <= N2*Dilation2, "Invalid dilatedsubvector specification" );

   return dilatedsubvector<I1*Dilation2+I2,N1*Dilation2,Dilation1*Dilation2>( sv.operand(), args... );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific dilatedsubvector of another constant dilatedsubvector.
// \ingroup dilatedsubvector
//
// \param sv The given constant dilatedsubvector.
// \param args The optional dilatedsubvector arguments.
// \return View on the specified dilatedsubvector of the other dilatedsubvector.
//
// This function returns an expression representing the specified dilatedsubvector of the given constant
// dilatedsubvector.
*/
template< size_t I1           // Required dilatedsubvector offset
        , size_t N1           // Required size of the dilatedsubvector
        , size_t Dilation1    // Required step-size of the dilatedsubvector
        , typename VT         // Type of the vector
        , bool TF             // Transpose flag
        , bool DF             // Density flag
        , size_t I2           // Present dilatedsubvector offset
        , size_t N2           // Present size of the dilatedsubvector
        , size_t Dilation2    // Present step-size of the dilatedsubvector
        , typename... RSAs >  // Optional dilatedsubvector arguments
inline decltype(auto) dilatedsubvector( const DilatedSubvector<VT,TF,DF,I2,N2,Dilation2>& sv, RSAs... args )
{
   BLAZE_FUNCTION_TRACE;

   BLAZE_STATIC_ASSERT_MSG( I1 + N1*Dilation1 <= N2*Dilation2, "Invalid dilatedsubvector specification" );

   return dilatedsubvector<I1*Dilation2+I2,N1*Dilation2,Dilation1*Dilation2>( sv.operand(), args... );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific dilatedsubvector of another temporary dilatedsubvector.
// \ingroup dilatedsubvector
//
// \param sv The given temporary dilatedsubvector.
// \param args The optional dilatedsubvector arguments.
// \return View on the specified dilatedsubvector of the other dilatedsubvector.
//
// This function returns an expression representing the specified dilatedsubvector of the given temporary
// dilatedsubvector.
*/
template< size_t I1           // Required dilatedsubvector offset
        , size_t N1           // Required size of the dilatedsubvector
        , size_t Dilation1    // Required step-size of the dilatedsubvector
        , typename VT         // Type of the vector
        , bool TF             // Transpose flag
        , bool DF             // Density flag
        , size_t I2           // Present dilatedsubvector offset
        , size_t N2           // Present size of the dilatedsubvector
        , size_t Dilation2    // Present step-size of the dilatedsubvector
        , typename... RSAs >  // Optional dilatedsubvector arguments
inline decltype(auto) dilatedsubvector( DilatedSubvector<VT,TF,DF,I2,N2,Dilation2>&& sv, RSAs... args )
{
   BLAZE_FUNCTION_TRACE;

   BLAZE_STATIC_ASSERT_MSG( I1 + N1*Dilation1 <= N2*Dilation2, "Invalid dilatedsubvector specification" );

   return dilatedsubvector<I1*Dilation2+I2,N1*Dilation2,Dilation1*Dilation2>( sv.operand(), args... );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific dilatedsubvector of another dilatedsubvector.
// \ingroup dilatedsubvector
//
// \param sv The given dilatedsubvector.
// \param args The optional dilatedsubvector arguments.
// \return View on the specified dilatedsubvector of the other dilatedsubvector.
// \exception std::invalid_argument Invalid dilatedsubvector specification.
//
// This function returns an expression representing the specified dilatedsubvector of the given dilatedsubvector.
*/
template< size_t I            // Index of the first dilatedsubvector element
        , size_t N            // Size of the dilatedsubvector
        , size_t Dilation     // Step-size of the dilatedsubvector
        , typename VT         // Type of the vector
        , bool TF             // Transpose flag
        , bool DF             // Density flag
        , typename... RSAs >  // Optional dilatedsubvector arguments
inline decltype(auto) dilatedsubvector( DilatedSubvector<VT,TF,DF>& sv, RSAs... args )
{
   BLAZE_FUNCTION_TRACE;

   constexpr bool isChecked( !Contains_v< TypeList<RSAs...>, Unchecked > );

   if( isChecked ) {
      if( I + N > sv.size() ) {
         BLAZE_THROW_INVALID_ARGUMENT( "Invalid dilatedsubvector specification" );
      }
   }
   else {
      BLAZE_USER_ASSERT( I + N <= sv.size(), "Invalid dilatedsubvector specification" );
   }

   return dilatedsubvector( sv.operand(), sv.offset() + I*sv.dilation(), N*sv.dilation(), Dilation*sv.dilation(), args... );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific dilatedsubvector of another constant dilatedsubvector.
// \ingroup dilatedsubvector
//
// \param sv The given constant dilatedsubvector.
// \param args The optional dilatedsubvector arguments.
// \return View on the specified dilatedsubvector of the other dilatedsubvector.
// \exception std::invalid_argument Invalid dilatedsubvector specification.
//
// This function returns an expression representing the specified dilatedsubvector of the given constant
// dilatedsubvector.
*/
template< size_t I            // Index of the first dilatedsubvector element
        , size_t N            // Size of the dilatedsubvector
        , size_t Dilation     // Step-size of the dilatedsubvector
        , typename VT         // Type of the vector
        , bool TF             // Transpose flag
        , bool DF             // Density flag
        , typename... RSAs >  // Optional dilatedsubvector arguments
inline decltype(auto) dilatedsubvector( const DilatedSubvector<VT,TF,DF>& sv, RSAs... args )
{
   BLAZE_FUNCTION_TRACE;

   constexpr bool isChecked( !Contains_v< TypeList<RSAs...>, Unchecked > );

   if( isChecked ) {
      if( I + N > sv.size() ) {
         BLAZE_THROW_INVALID_ARGUMENT( "Invalid dilatedsubvector specification" );
      }
   }
   else {
      BLAZE_USER_ASSERT( I + N <= sv.size(), "Invalid dilatedsubvector specification" );
   }

   return dilatedsubvector( sv.operand(), sv.offset() + I*sv.dilation(), N*sv.dilation(), Dilation*sv.dilation(), args... );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific dilatedsubvector of another temporary dilatedsubvector.
// \ingroup dilatedsubvector
//
// \param sv The given temporary dilatedsubvector.
// \param args The optional dilatedsubvector arguments.
// \return View on the specified dilatedsubvector of the other dilatedsubvector.
// \exception std::invalid_argument Invalid dilatedsubvector specification.
//
// This function returns an expression representing the specified dilatedsubvector of the given temporary
// dilatedsubvector.
*/
template< size_t I            // Index of the first dilatedsubvector element
        , size_t N            // Size of the dilatedsubvector
        , size_t Dilation     // Step-size of the dilatedsubvector
        , typename VT         // Type of the vector
        , bool TF             // Transpose flag
        , bool DF             // Density flag
        , typename... RSAs >  // Optional dilatedsubvector arguments
inline decltype(auto) dilatedsubvector( DilatedSubvector<VT,TF,DF>&& sv, RSAs... args )
{
   BLAZE_FUNCTION_TRACE;

   constexpr bool isChecked( !Contains_v< TypeList<RSAs...>, Unchecked > );

   if( isChecked ) {
      if( I + N > sv.size() ) {
         BLAZE_THROW_INVALID_ARGUMENT( "Invalid dilatedsubvector specification" );
      }
   }
   else {
      BLAZE_USER_ASSERT( I + N <= sv.size(), "Invalid dilatedsubvector specification" );
   }

   return dilatedsubvector( sv.operand(), sv.offset() + I*sv.dilation(), N*sv.dilation(), Dilation*sv.dilation(), args... );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific dilatedsubvector of another dilatedsubvector.
// \ingroup dilatedsubvector
//
// \param sv The given dilatedsubvector.
// \param index The index of the first element of the dilatedsubvector.
// \param size The size of the dilatedsubvector.
// \param args The optional dilatedsubvector arguments.
// \return View on the specified dilatedsubvector of the other dilatedsubvector.
// \exception std::invalid_argument Invalid dilatedsubvector specification.
//
// This function returns an expression representing the specified dilatedsubvector of the given dilatedsubvector.
*/
template< typename VT         // Type of the vector
        , bool TF             // Transpose flag
        , bool DF             // Density flag
        , size_t... CSAs      // Compile time dilatedsubvector arguments
        , typename... RSAs >  // Optional dilatedsubvector arguments
inline decltype(auto)
   dilatedsubvector( DilatedSubvector<VT,TF,DF,CSAs...>& sv, size_t index, size_t size, size_t dilation, RSAs... args )
{
   BLAZE_FUNCTION_TRACE;

   constexpr bool isChecked( !Contains_v< TypeList<RSAs...>, Unchecked > );

   if( isChecked ) {
      if( index + size * dilation > sv.size() ) {
         BLAZE_THROW_INVALID_ARGUMENT( "Invalid dilatedsubvector specification" );
      }
   }
   else {
      BLAZE_USER_ASSERT( index + size * dilation <= sv.size(), "Invalid dilatedsubvector specification" );
   }

   return dilatedsubvector( sv.operand(), sv.offset() + index*sv.dilation(), size, dilation*sv.dilation(), args... );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific dilatedsubvector of another constant dilatedsubvector.
// \ingroup dilatedsubvector
//
// \param sv The given constant dilatedsubvector.
// \param index The index of the first element of the dilatedsubvector.
// \param size The size of the dilatedsubvector.
// \param args The optional dilatedsubvector arguments.
// \return View on the specified dilatedsubvector of the other dilatedsubvector.
// \exception std::invalid_argument Invalid dilatedsubvector specification.
//
// This function returns an expression representing the specified dilatedsubvector of the given constant
// dilatedsubvector.
*/
template< typename VT         // Type of the vector
        , bool TF             // Transpose flag
        , bool DF             // Density flag
        , size_t... CSAs      // Compile time dilatedsubvector arguments
        , typename... RSAs >  // Optional dilatedsubvector arguments
inline decltype(auto)
   dilatedsubvector( const DilatedSubvector<VT,TF,DF,CSAs...>& sv, size_t index, size_t size, size_t dilation, RSAs... args )
{
   BLAZE_FUNCTION_TRACE;

   constexpr bool isChecked( !Contains_v< TypeList<RSAs...>, Unchecked > );

   if( isChecked ) {
      if( index + size * dilation > sv.size() ) {
         BLAZE_THROW_INVALID_ARGUMENT( "Invalid dilatedsubvector specification" );
      }
   }
   else {
      BLAZE_USER_ASSERT( index + size * dilation <= sv.size(), "Invalid dilatedsubvector specification" );
   }

   return dilatedsubvector( sv.operand(), sv.offset() + index*sv.dilation(), size, dilation*sv.dilation(), args... );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific dilatedsubvector of another temporary dilatedsubvector.
// \ingroup dilatedsubvector
//
// \param sv The given temporary dilatedsubvector.
// \param index The index of the first element of the dilatedsubvector.
// \param size The size of the dilatedsubvector.
// \param args The optional dilatedsubvector arguments.
// \return View on the specified dilatedsubvector of the other dilatedsubvector.
// \exception std::invalid_argument Invalid dilatedsubvector specification.
//
// This function returns an expression representing the specified dilatedsubvector of the given temporary
// dilatedsubvector.
*/
template< typename VT         // Type of the vector
        , bool TF             // Transpose flag
        , bool DF             // Density flag
        , size_t... CSAs      // Compile time dilatedsubvector arguments
        , typename... RSAs >  // Optional dilatedsubvector arguments
inline decltype(auto)
   dilatedsubvector( DilatedSubvector<VT,TF,DF,CSAs...>&& sv, size_t index, size_t size, size_t dilation, RSAs... args )
{
   BLAZE_FUNCTION_TRACE;

   constexpr bool isChecked( !Contains_v< TypeList<RSAs...>, Unchecked > );

   if( isChecked ) {
      if( index + size * dilation > sv.size() ) {
         BLAZE_THROW_INVALID_ARGUMENT( "Invalid dilatedsubvector specification" );
      }
   }
   else {
      BLAZE_USER_ASSERT( index + size * dilation <= sv.size(), "Invalid dilatedsubvector specification" );
   }

   return dilatedsubvector( sv.operand(), sv.offset() + index*sv.dilation(), size, dilation*sv.dilation(), args... );
}
/*! \endcond */
//*************************************************************************************************



//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific dilatedsubvector of another dilatedsubvector.
// \ingroup dilatedsubvector
//
// \param sv The given dilatedsubvector.
// \param index The index of the first element of the dilatedsubvector.
// \param size The size of the dilatedsubvector.
// \param args The optional dilatedsubvector arguments.
// \return View on the specified dilatedsubvector of the other dilatedsubvector.
// \exception std::invalid_argument Invalid dilatedsubvector specification.
//
// This function returns an expression representing the specified dilatedsubvector of the given dilatedsubvector.
*/
template< typename VT         // Type of the vector
        , AlignmentFlag AF    // Alignment flag
        , bool TF             // Transpose flag
        , bool DF             // Density flag
        , size_t... CSAs      // Compile time dilatedsubvector arguments
        , typename... RSAs >  // Optional dilatedsubvector arguments
inline decltype(auto)
   dilatedsubvector( Subvector<VT,AF,TF,DF,CSAs...>& sv, size_t index, size_t size, size_t dilation, RSAs... args )
{
   BLAZE_FUNCTION_TRACE;

   constexpr bool isChecked( !Contains_v< TypeList<RSAs...>, Unchecked > );

   if( isChecked ) {
      if( index + ( size - 1 ) * dilation + 1 > sv.size() ) {
         BLAZE_THROW_INVALID_ARGUMENT( "Invalid dilatedsubvector specification" );
      }
   }
   else {
      BLAZE_USER_ASSERT( index + ( size - 1 ) * dilation + 1 <= sv.size(), "Invalid dilatedsubvector specification" );
   }

   return dilatedsubvector( sv.operand(), sv.offset() + index, size, dilation, args... );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific dilatedsubvector of another constant dilatedsubvector.
// \ingroup dilatedsubvector
//
// \param sv The given constant dilatedsubvector.
// \param index The index of the first element of the dilatedsubvector.
// \param size The size of the dilatedsubvector.
// \param args The optional dilatedsubvector arguments.
// \return View on the specified dilatedsubvector of the other dilatedsubvector.
// \exception std::invalid_argument Invalid dilatedsubvector specification.
//
// This function returns an expression representing the specified dilatedsubvector of the given constant
// dilatedsubvector.
*/
template< typename VT         // Type of the vector
        , AlignmentFlag AF    // Alignment flag
        , bool TF             // Transpose flag
        , bool DF             // Density flag
        , size_t... CSAs      // Compile time dilatedsubvector arguments
        , typename... RSAs >  // Optional dilatedsubvector arguments
inline decltype(auto)
   dilatedsubvector( const Subvector<VT,AF,TF,DF,CSAs...>& sv, size_t index, size_t size, size_t dilation, RSAs... args )
{
   BLAZE_FUNCTION_TRACE;

   constexpr bool isChecked( !Contains_v< TypeList<RSAs...>, Unchecked > );

   if( isChecked ) {
      if( index + ( size - 1 ) * dilation + 1 > sv.size() ) {
         BLAZE_THROW_INVALID_ARGUMENT( "Invalid dilatedsubvector specification" );
      }
   }
   else {
      BLAZE_USER_ASSERT( index + ( size - 1 ) * dilation + 1 <= sv.size(), "Invalid dilatedsubvector specification" );
   }

   return dilatedsubvector( sv.operand(), sv.offset() + index, size, dilation, args... );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific dilatedsubvector of another temporary dilatedsubvector.
// \ingroup dilatedsubvector
//
// \param sv The given temporary dilatedsubvector.
// \param index The index of the first element of the dilatedsubvector.
// \param size The size of the dilatedsubvector.
// \param args The optional dilatedsubvector arguments.
// \return View on the specified dilatedsubvector of the other dilatedsubvector.
// \exception std::invalid_argument Invalid dilatedsubvector specification.
//
// This function returns an expression representing the specified dilatedsubvector of the given temporary
// dilatedsubvector.
*/
template< typename VT         // Type of the vector
        , AlignmentFlag AF    // Alignment flag
        , bool TF             // Transpose flag
        , bool DF             // Density flag
        , size_t... CSAs      // Compile time dilatedsubvector arguments
        , typename... RSAs >  // Optional dilatedsubvector arguments
inline decltype(auto)
   dilatedsubvector( Subvector<VT,AF,TF,DF,CSAs...>&& sv, size_t index, size_t size, size_t dilation, RSAs... args )
{
   BLAZE_FUNCTION_TRACE;

   constexpr bool isChecked( !Contains_v< TypeList<RSAs...>, Unchecked > );

   if( isChecked ) {
      if( index + ( size - 1 ) * dilation + 1 > sv.size() )
      {
         BLAZE_THROW_INVALID_ARGUMENT( "Invalid dilatedsubvector specification" );
      }
   }
   else {
      BLAZE_USER_ASSERT( index + ( size - 1 ) * dilation + 1 <= sv.size(), "Invalid dilatedsubvector specification" );
   }

   return dilatedsubvector( sv.operand(), sv.offset() + index, size, dilation, args... );
}
/*! \endcond */
//*************************************************************************************************



//=================================================================================================
//
//  GLOBAL RESTRUCTURING FUNCTIONS (ELEMENTS)
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a selection of elements on a dilatedsubvector.
// \ingroup dilatedsubvector
//
// \param sv The given dilatedsubvector.
// \param args The optional element arguments.
// \return View on the specified selection of elements on the dilatedsubvector.
//
// This function returns an expression representing the specified selection of elements on the
// given dilatedsubvector.
*/
template< size_t I1           // First element index
        , size_t... Is        // Remaining element indices
        , typename VT         // Type of the vector
        , bool TF             // Transpose flag
        , bool DF             // Density flag
        , size_t I2           // Index of the first dilatedsubvector element
        , size_t N            // Size of the dilatedsubvector
        , size_t Dilation     // Step-size of the dilatedsubvector
        , typename... REAs >  // Optional element arguments
inline decltype(auto)
   elements( DilatedSubvector<VT,TF,DF,I2,N,Dilation>& sv, REAs... args )
{
   BLAZE_FUNCTION_TRACE;

   return elements( sv.operand(), make_dilated_index_subsequence<I2,N,Dilation,I1,Is...>(), args... );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a selection of elements on a constant dilatedsubvector.
// \ingroup dilatedsubvector
//
// \param sv The given constant dilatedsubvector.
// \param args The optional element arguments.
// \return View on the specified selection of elements on the dilatedsubvector.
//
// This function returns an expression representing the specified selection of elements on the
// given constant dilatedsubvector.
*/
template< size_t I1           // First element index
        , size_t... Is        // Remaining element indices
        , typename VT         // Type of the vector
        , bool TF             // Transpose flag
        , bool DF             // Density flag
        , size_t I2           // Index of the first dilatedsubvector element
        , size_t N            // Size of the dilatedsubvector
        , size_t Dilation     // Step-size of the dilatedsubvector
        , typename... REAs >  // Optional element arguments
inline decltype(auto)
   elements( const DilatedSubvector<VT,TF,DF,I2,N,Dilation>& sv, REAs... args )
{
   BLAZE_FUNCTION_TRACE;

   return elements( sv.operand(), make_dilated_index_subsequence<I2,N,Dilation,I1,Is...>(), args... );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a selection of elements on a temporary dilatedsubvector.
// \ingroup dilatedsubvector
//
// \param sv The given temporary dilatedsubvector.
// \param args The optional element arguments.
// \return View on the specified selection of elements on the dilatedsubvector.
//
// This function returns an expression representing the specified selection of elements on the
// given temporary dilatedsubvector.
*/
template< size_t I1           // First element index
        , size_t... Is        // Remaining element indices
        , typename VT         // Type of the vector
        , bool TF             // Transpose flag
        , bool DF             // Density flag
        , size_t I2           // Index of the first dilatedsubvector element
        , size_t N            // Size of the dilatedsubvector
        , size_t Dilation     // Step-size of the dilatedsubvector
        , typename... REAs >  // Optional element arguments
inline decltype(auto)
   elements( DilatedSubvector<VT,TF,DF,I2,N,Dilation>&& sv, REAs... args )
{
   BLAZE_FUNCTION_TRACE;

   return elements( sv.operand(), make_dilated_index_subsequence<I2,N,Dilation,I1,Is...>(), args... );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a selection of elements on a dilatedsubvector.
// \ingroup dilatedsubvector
//
// \param sv The given dilatedsubvector.
// \param args The optional element arguments.
// \return View on the specified selection of elements on the dilatedsubvector.
// \exception std::invalid_argument Invalid elements specification.
//
// This function returns an expression representing the specified selection of elements on the
// given dilatedsubvector.
*/
template< size_t I            // First element index
        , size_t... Is        // Remaining element indices
        , typename VT         // Type of the vector
        , bool TF             // Transpose flag
        , bool DF             // Density flag
        , typename... REAs >  // Optional element arguments
inline decltype(auto) elements( DilatedSubvector<VT,TF,DF>& sv, REAs... args )
{
   BLAZE_FUNCTION_TRACE;

   constexpr bool isChecked( !Contains_v< TypeList<REAs...>, Unchecked > );

   if( isChecked ) {
      static constexpr size_t indices[] = { I, Is... };
      for( size_t i=0UL; i<sizeof...(Is)+1UL; ++i ) {
         if( sv.size() <= indices[i]*sv.dilation() ) {
            BLAZE_THROW_INVALID_ARGUMENT( "Invalid elements specification" );
         }
      }
   }

   return elements( sv.operand(), { I*sv.dilation()+sv.offset(), Is*sv.dilation()+sv.offset()... }, args... );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a selection of elements on a constant dilatedsubvector.
// \ingroup dilatedsubvector
//
// \param sv The given constant dilatedsubvector.
// \param args The optional element arguments.
// \return View on the specified selection of elements on the dilatedsubvector.
// \exception std::invalid_argument Invalid elements specification.
//
// This function returns an expression representing the specified selection of elements on the
// given constant dilatedsubvector.
*/
template< size_t I            // First element index
        , size_t... Is        // Remaining element indices
        , typename VT         // Type of the vector
        , bool TF             // Transpose flag
        , bool DF             // Density flag
        , typename... REAs >  // Optional element arguments
inline decltype(auto) elements( const DilatedSubvector<VT,TF,DF>& sv, REAs... args )
{
   BLAZE_FUNCTION_TRACE;

   constexpr bool isChecked( !Contains_v< TypeList<REAs...>, Unchecked > );

   if( isChecked ) {
      static constexpr size_t indices[] = { I, Is... };
      for( size_t i=0UL; i<sizeof...(Is)+1UL; ++i ) {
         if( sv.size() <= indices[i]*sv.dilation() ) {
            BLAZE_THROW_INVALID_ARGUMENT( "Invalid elements specification" );
         }
      }
   }

   return elements( sv.operand(), { I*sv.dilation()+sv.offset(), Is*sv.dilation()+sv.offset()... }, args... );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a selection of elements on a temporary dilatedsubvector.
// \ingroup dilatedsubvector
//
// \param sv The given temporary dilatedsubvector.
// \param args The optional element arguments.
// \return View on the specified selection of elements on the dilatedsubvector.
// \exception std::invalid_argument Invalid elements specification.
//
// This function returns an expression representing the specified selection of elements on the
// given temporary dilatedsubvector.
*/
template< size_t I            // First element index
        , size_t... Is        // Remaining element indices
        , typename VT         // Type of the vector
        , bool TF             // Transpose flag
        , bool DF             // Density flag
        , typename... REAs >  // Optional element arguments
inline decltype(auto) elements( DilatedSubvector<VT,TF,DF>&& sv, REAs... args )
{
   BLAZE_FUNCTION_TRACE;

   constexpr bool isChecked( !Contains_v< TypeList<REAs...>, Unchecked > );

   if( isChecked ) {
      static constexpr size_t indices[] = { I, Is... };
      for( size_t i=0UL; i<sizeof...(Is)+1UL; ++i ) {
         if( sv.size() <= indices[i]*sv.dilation() ) {
            BLAZE_THROW_INVALID_ARGUMENT( "Invalid elements specification" );
         }
      }
   }

   return elements( sv.operand(), { I*sv.dilation()+sv.offset(), Is*sv.dilation()+sv.offset()... }, args... );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a selection of elements on a dilatedsubvector.
// \ingroup dilatedsubvector
//
// \param sv The given dilatedsubvector.
// \param indices The container of element indices.
// \param n The total number of indices.
// \param args The optional element arguments.
// \return View on the specified selection of elements on the dilatedsubvector.
// \exception std::invalid_argument Invalid elements specification.
//
// This function returns an expression representing the specified selection of elements on the
// given dilatedsubvector.
*/
template< typename VT         // Type of the vector
        , bool TF             // Transpose flag
        , bool DF             // Density flag
        , size_t... CSAs      // Compile time dilatedsubvector arguments
        , typename T          // Type of the element indices
        , typename... REAs >  // Optional element arguments
inline decltype(auto)
   elements( DilatedSubvector<VT,TF,DF,CSAs...>& sv, const T* indices, size_t n, REAs... args )
{
   BLAZE_FUNCTION_TRACE;

   constexpr bool isChecked( !Contains_v< TypeList<REAs...>, Unchecked > );

   if( isChecked ) {
      for( size_t i=0UL; i<n; ++i ) {
         if( sv.size() <= indices[i] ) {
            BLAZE_THROW_INVALID_ARGUMENT( "Invalid elements specification" );
         }
      }
   }

   SmallArray<size_t,128UL> newIndices( indices, indices+n );
   std::for_each( newIndices.begin(), newIndices.end(),
      [offset = sv.offset(), dilation = sv.dilation()]( size_t& index ) {
         index = index * dilation + offset; } );

   return elements( sv.operand(), newIndices.data(), n, args... );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a selection of elements on a constant dilatedsubvector.
// \ingroup dilatedsubvector
//
// \param sv The given constant dilatedsubvector.
// \param indices The container of element indices.
// \param n The total number of indices.
// \param args The optional element arguments.
// \return View on the specified selection of elements on the dilatedsubvector.
// \exception std::invalid_argument Invalid elements specification.
//
// This function returns an expression representing the specified selection of elements on the
// given constant dilatedsubvector.
*/
template< typename VT         // Type of the vector
        , bool TF             // Transpose flag
        , bool DF             // Density flag
        , size_t... CSAs      // Compile time dilatedsubvector arguments
        , typename T          // Type of the element indices
        , typename... REAs >  // Optional element arguments
inline decltype(auto)
   elements( const DilatedSubvector<VT,TF,DF,CSAs...>& sv, const T* indices, size_t n, REAs... args )
{
   BLAZE_FUNCTION_TRACE;

   constexpr bool isChecked( !Contains_v< TypeList<REAs...>, Unchecked > );

   if( isChecked ) {
      for( size_t i=0UL; i<n; ++i ) {
         if( sv.size() <= indices[i] ) {
            BLAZE_THROW_INVALID_ARGUMENT( "Invalid elements specification" );
         }
      }
   }

   SmallArray<size_t,128UL> newIndices( indices, indices+n );
   std::for_each( newIndices.begin(), newIndices.end(),
      [offset = sv.offset(), dilation = sv.dilation()]( size_t& index ) {
         index = index * dilation + offset; } );

   return elements( sv.operand(), newIndices.data(), n, args... );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a selection of elements on a temporary dilatedsubvector.
// \ingroup dilatedsubvector
//
// \param sv The given temporary dilatedsubvector.
// \param indices The container of element indices.
// \param n The total number of indices.
// \param args The optional element arguments.
// \return View on the specified selection of elements on the dilatedsubvector.
// \exception std::invalid_argument Invalid elements specification.
//
// This function returns an expression representing the specified selection of elements on the
// given temporary dilatedsubvector.
*/
template< typename VT         // Type of the vector
        , bool TF             // Transpose flag
        , bool DF             // Density flag
        , size_t... CSAs      // Compile time dilatedsubvector arguments
        , typename T          // Type of the element indices
        , typename... REAs >  // Optional element arguments
inline decltype(auto)
   elements( DilatedSubvector<VT,TF,DF,CSAs...>&& sv, const T* indices, size_t n, REAs... args )
{
   BLAZE_FUNCTION_TRACE;

   constexpr bool isChecked( !Contains_v< TypeList<REAs...>, Unchecked > );

   if( isChecked ) {
      for( size_t i=0UL; i<n; ++i ) {
         if( sv.size() <= indices[i] ) {
            BLAZE_THROW_INVALID_ARGUMENT( "Invalid elements specification" );
         }
      }
   }

   SmallArray<size_t,128UL> newIndices( indices, indices+n );
   std::for_each( newIndices.begin(), newIndices.end(),
      [offset = sv.offset(), dilation = sv.dilation()]( size_t& index ) {
         index = index * dilation + offset; } );

   return elements( sv.operand(), newIndices.data(), n, args... );
}
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  DILATEDSUBVECTOR OPERATORS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Resetting the given dilatedsubvector.
// \ingroup dilatedsubvector
//
// \param sv The dilatedsubvector to be resetted.
// \return void
*/
template< typename VT       // Type of the vector
        , bool TF           // Transpose flag
        , bool DF           // Density flag
        , size_t... CSAs >  // Compile time dilatedsubvector arguments
inline void reset( DilatedSubvector<VT,TF,DF,CSAs...>& sv )
{
   sv.reset();
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Resetting the given temporary dilatedsubvector.
// \ingroup dilatedsubvector
//
// \param sv The temporary dilatedsubvector to be resetted.
// \return void
*/
template< typename VT       // Type of the vector
        , bool TF           // Transpose flag
        , bool DF           // Density flag
        , size_t... CSAs >  // Compile time dilatedsubvector arguments
inline void reset( DilatedSubvector<VT,TF,DF,CSAs...>&& sv )
{
   sv.reset();
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Clearing the given dilatedsubvector.
// \ingroup dilatedsubvector
//
// \param sv The dilatedsubvector to be cleared.
// \return void
*/
template< typename VT       // Type of the vector
        , bool TF           // Transpose flag
        , bool DF           // Density flag
        , size_t... CSAs >  // Compile time dilatedsubvector arguments
inline void clear( DilatedSubvector<VT,TF,DF,CSAs...>& sv )
{
   sv.reset();
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Clearing the given temporary dilatedsubvector.
// \ingroup dilatedsubvector
//
// \param sv The temporary dilatedsubvector to be cleared.
// \return void
*/
template< typename VT       // Type of the vector
        , bool TF           // Transpose flag
        , bool DF           // Density flag
        , size_t... CSAs >  // Compile time dilatedsubvector arguments
inline void clear( DilatedSubvector<VT,TF,DF,CSAs...>&& sv )
{
   sv.reset();
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns whether the given dense dilatedsubvector is in default state.
// \ingroup dilatedsubvector
//
// \param sv The dense dilatedsubvector to be tested for its default state.
// \return \a true in case the given dense dilatedsubvector is component-wise zero, \a false otherwise.
//
// This function checks whether the dense dilatedsubvector is in default state. For instance, in case
// the dilatedsubvector is instantiated for a vector of built-in integral or floating point data type,
// the function returns \a true in case all dilatedsubvector elements are 0 and \a false in case any
// dilatedsubvector element is not 0. The following example demonstrates the use of the \a isDefault
// function:

   \code
   blaze::DynamicVector<int,rowVector> v;
   // ... Resizing and initialization
   if( isDefault( dilatedsubvector( v, 10UL, 20UL ) ) ) { ... }
   \endcode

// Optionally, it is possible to switch between strict semantics (blaze::strict) and relaxed
// semantics (blaze::relaxed):

   \code
   if( isDefault<relaxed>( dilatedsubvector( v, 10UL, 20UL ) ) ) { ... }
   \endcode
*/
template< RelaxationFlag RF // Relaxation flag
        , typename VT       // Type of the dense vector
        , bool TF           // Transpose flag
        , size_t... CSAs >  // Compile time dilatedsubvector arguments
inline bool isDefault( const DilatedSubvector<VT,TF,true,CSAs...>& sv )
{
   using blaze::isDefault;

   for( size_t i=0UL; i<sv.size(); ++i )
      if( !isDefault<RF>( sv[i] ) ) return false;
   return true;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns whether the given sparse dilatedsubvector is in default state.
// \ingroup dilatedsubvector
//
// \param sv The sparse dilatedsubvector to be tested for its default state.
// \return \a true in case the given sparse dilatedsubvector is component-wise zero, \a false otherwise.
//
// This function checks whether the sparse dilatedsubvector is in default state. For instance, in case
// the dilatedsubvector is instantiated for a vector of built-in integral or floating point data type,
// the function returns \a true in case all dilatedsubvector elements are 0 and \a false in case any
// dilatedsubvector element is not 0. The following example demonstrates the use of the \a isDefault
// function:

   \code
   blaze::CompressedVector<int,rowVector> v;
   // ... Resizing and initialization
   if( isDefault( dilatedsubvector( v, 10UL, 20UL ) ) ) { ... }
   \endcode

// Optionally, it is possible to switch between strict semantics (blaze::strict) and relaxed
// semantics (blaze::relaxed):

   \code
   if( isDefault<relaxed>( dilatedsubvector( v, 10UL, 20UL ) ) ) { ... }
   \endcode
*/
template< RelaxationFlag RF // Relaxation flag
        , typename VT       // Type of the sparse vector
        , bool TF           // Transpose flag
        , size_t... CSAs >  // Compile time dilatedsubvector arguments
inline bool isDefault( const DilatedSubvector<VT,TF,false,CSAs...>& sv )
{
   using blaze::isDefault;

   for( const auto& element : *sv )
      if( !isDefault<RF>( element.value() ) ) return false;
   return true;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns whether the invariants of the given dilatedsubvector are intact.
// \ingroup dilatedsubvector
//
// \param sv The dilatedsubvector to be tested.
// \return \a true in case the given dilatedsubvector's invariants are intact, \a false otherwise.
//
// This function checks whether the invariants of the dilatedsubvector are intact, i.e. if its state
// is valid. In case the invariants are intact, the function returns \a true, else it will
// return \a false. The following example demonstrates the use of the \a isIntact() function:

   \code
   blaze::DynamicVector<int,rowVector> v;
   // ... Resizing and initialization
   if( isIntact( dilatedsubvector( v, 10UL, 20UL ) ) ) { ... }
   \endcode
*/
template< typename VT       // Type of the vector
        , bool TF           // Transpose flag
        , bool DF           // Density flag
        , size_t... CSAs >  // Compile time dilatedsubvector arguments
inline bool isIntact( const DilatedSubvector<VT,TF,DF,CSAs...>& sv ) noexcept
{
   return ( sv.offset() + sv.size()*sv.dilation() <= sv.operand().size() &&
            isIntact( sv.operand() ) );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns whether the given dilatedsubvector and vector represent the same observable state.
// \ingroup dilatedsubvector
//
// \param a The dilatedsubvector to be tested for its state.
// \param b The vector to be tested for its state.
// \return \a true in case the dilatedsubvector and vector share a state, \a false otherwise.
//
// This overload of the isSame function tests if the given dilatedsubvector refers to the entire
// range of the given vector and by that represents the same observable state. In this case,
// the function returns \a true, otherwise it returns \a false.
*/
template< typename VT       // Type of the vector
        , bool TF           // Transpose flag
        , bool DF           // Density flag
        , size_t... CSAs >  // Compile time dilatedsubvector arguments
inline bool isSame( const DilatedSubvector<VT,TF,DF,CSAs...>& a, const Vector<VT,TF>& b ) noexcept
{
   return ( isSame( a.operand(), *b ) && ( a.size() == (*b).size() ) && a.dilation() == 1 );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns whether the given vector and dilatedsubvector represent the same observable state.
// \ingroup dilatedsubvector
//
// \param a The vector to be tested for its state.
// \param b The dilatedsubvector to be tested for its state.
// \return \a true in case the vector and dilatedsubvector share a state, \a false otherwise.
//
// This overload of the isSame function tests if the given dilatedsubvector refers to the entire
// range of the given vector and by that represents the same observable state. In this case,
// the function returns \a true, otherwise it returns \a false.
*/
template< typename VT       // Type of the vector
        , bool TF           // Transpose flag
        , bool DF           // Density flag
        , size_t... CSAs >  // Compile time dilatedsubvector arguments
inline bool isSame( const Vector<VT,TF>& a, const DilatedSubvector<VT,TF,DF,CSAs...>& b ) noexcept
{
   return isSame( b, a );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns whether the two given dilatedsubvectors represent the same observable state.
// \ingroup dilatedsubvector
//
// \param a The first dilatedsubvector to be tested for its state.
// \param b The second dilatedsubvector to be tested for its state.
// \return \a true in case the two dilatedsubvectors share a state, \a false otherwise.
//
// This overload of the isSame function tests if the two given dilatedsubvectors refer to exactly the
// same range of the same vector. In case both dilatedsubvectors represent the same observable state,
// the function returns \a true, otherwise it returns \a false.
*/
template< typename VT1       // Type of the vector of the left-hand side dilatedsubvector
        , bool TF1           // Transpose flag of the left-hand side dilatedsubvector
        , bool DF1           // Density flag of the left-hand side dilatedsubvector
        , size_t... CSAs1    // Compile time dilatedsubvector arguments of the left-hand side dilatedsubvector
        , typename VT2       // Type of the vector of the right-hand side dilatedsubvector
        , bool TF2           // Transpose flag of the right-hand side dilatedsubvector
        , bool DF2           // Density flag of the right-hand side dilatedsubvector
        , size_t... CSAs2 >  // Compile time dilatedsubvector arguments of the right-hand side dilatedsubvector
inline bool isSame( const DilatedSubvector<VT1,TF1,DF1,CSAs1...>& a,
                    const DilatedSubvector<VT2,TF2,DF2,CSAs2...>& b ) noexcept
{
   return ( isSame( a.operand(), b.operand() ) &&
            ( a.offset() == b.offset() ) &&
            ( a.dilation() == b.dilation() ) &&
            ( a.size() == b.size() ) );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Predict invariant violations by setting a single element of a dilatedsubvector.
// \ingroup dilatedsubvector
//
// \param sv The target dilatedsubvector.
// \param index The index of the element to be set.
// \param value The value to be set to the element.
// \return \a true in case the operation would be successful, \a false if not.
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename VT       // Type of the vector
        , bool TF           // Transpose flag
        , bool DF           // Density flag
        , size_t... CSAs    // Compile time dilatedsubvector arguments
        , typename ET >     // Type of the element
inline bool trySet( const DilatedSubvector<VT,TF,DF,CSAs...>& sv, size_t index, const ET& value )
{
   BLAZE_INTERNAL_ASSERT( index < sv.size(), "Invalid vector access index" );

   return trySet( sv.operand(), sv.offset()+index*sv.dilation(), value );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Predict invariant violations by adding to a single element of a dilatedsubvector.
// \ingroup dilatedsubvector
//
// \param sv The target dilatedsubvector.
// \param index The index of the element to be modified.
// \param value The value to be added to the element.
// \return \a true in case the operation would be successful, \a false if not.
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename VT       // Type of the vector
        , bool TF           // Transpose flag
        , bool DF           // Density flag
        , size_t... CSAs    // Compile time dilatedsubvector arguments
        , typename ET >     // Type of the element
inline bool tryAdd( const DilatedSubvector<VT,TF,DF,CSAs...>& sv, size_t index, const ET& value )
{
   BLAZE_INTERNAL_ASSERT( index < sv.size(), "Invalid vector access index" );

   return tryAdd( sv.operand(), sv.offset()+index*sv.dilation(), value );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Predict invariant violations by subtracting from a single element of a dilatedsubvector.
// \ingroup dilatedsubvector
//
// \param sv The target dilatedsubvector.
// \param index The index of the element to be modified.
// \param value The value to be subtracting from the element.
// \return \a true in case the operation would be successful, \a false if not.
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename VT       // Type of the vector
        , bool TF           // Transpose flag
        , bool DF           // Density flag
        , size_t... CSAs    // Compile time dilatedsubvector arguments
        , typename ET >     // Type of the element
inline bool trySub( const DilatedSubvector<VT,TF,DF,CSAs...>& sv, size_t index, const ET& value )
{
   BLAZE_INTERNAL_ASSERT( index < sv.size(), "Invalid vector access index" );

   return trySub( sv.operand(), sv.offset()+index*sv.dilation(), value );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Predict invariant violations by scaling a single element of a dilatedsubvector.
// \ingroup dilatedsubvector
//
// \param sv The target dilatedsubvector.
// \param index The index of the element to be modified.
// \param value The factor for the element.
// \return \a true in case the operation would be successful, \a false if not.
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename VT       // Type of the vector
        , bool TF           // Transpose flag
        , bool DF           // Density flag
        , size_t... CSAs    // Compile time dilatedsubvector arguments
        , typename ET >     // Type of the element
inline bool tryMult( const DilatedSubvector<VT,TF,DF,CSAs...>& sv, size_t index, const ET& value )
{
   BLAZE_INTERNAL_ASSERT( index < sv.size(), "Invalid vector access index" );

   return tryMult( sv.operand(), sv.offset()+index*sv.dilation(), value );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Predict invariant violations by scaling a range of elements of a dilatedsubvector.
// \ingroup dilatedsubvector
//
// \param sv The target dilatedsubvector.
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
template< typename VT       // Type of the vector
        , bool TF           // Transpose flag
        , bool DF           // Density flag
        , size_t... CSAs    // Compile time dilatedsubvector arguments
        , typename ET >     // Type of the element
BLAZE_ALWAYS_INLINE bool
   tryMult( const DilatedSubvector<VT,TF,DF,CSAs...>& sv, size_t index, size_t size, const ET& value )
{
   BLAZE_INTERNAL_ASSERT( index <= (*sv).size(), "Invalid vector access index" );
   BLAZE_INTERNAL_ASSERT( index + size <= (*sv).size(), "Invalid range size" );

   return tryMult( sv.operand(), sv.offset()+index*sv.dilation(), size*sv.dilation(), value );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Predict invariant violations by scaling a single element of a dilatedsubvector.
// \ingroup dilatedsubvector
//
// \param sv The target dilatedsubvector.
// \param index The index of the element to be modified.
// \param value The divisor for the element.
// \return \a true in case the operation would be successful, \a false if not.
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename VT       // Type of the vector
        , bool TF           // Transpose flag
        , bool DF           // Density flag
        , size_t... CSAs    // Compile time dilatedsubvector arguments
        , typename ET >     // Type of the element
inline bool tryDiv( const DilatedSubvector<VT,TF,DF,CSAs...>& sv, size_t index, const ET& value )
{
   BLAZE_INTERNAL_ASSERT( index < sv.size(), "Invalid vector access index" );

   return tryDiv( sv.operand(), sv.offset()+index*sv.dilation(), value );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Predict invariant violations by scaling a range of elements of a dilatedsubvector.
// \ingroup dilatedsubvector
//
// \param sv The target dilatedsubvector.
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
template< typename VT       // Type of the vector
        , bool TF           // Transpose flag
        , bool DF           // Density flag
        , size_t... CSAs    // Compile time dilatedsubvector arguments
        , typename ET >     // Type of the element
BLAZE_ALWAYS_INLINE bool
   tryDiv( const DilatedSubvector<VT,TF,DF,CSAs...>& sv, size_t index, size_t size, const ET& value )
{
   BLAZE_INTERNAL_ASSERT( index <= (*sv).size(), "Invalid vector access index" );
   BLAZE_INTERNAL_ASSERT( index + size <= (*sv).size(), "Invalid range size" );

   return tryDiv( sv.operand(), sv.offset()+index*sv.dilation(), size*sv.dilation(), value );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Predict invariant violations by the assignment of a vector to a dilatedsubvector.
// \ingroup dilatedsubvector
//
// \param lhs The target left-hand side dilatedsubvector.
// \param rhs The right-hand side vector to be assigned.
// \param index The index of the first element to be modified.
// \return \a true in case the assignment would be successful, \a false if not.
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename VT1      // Type of the vector
        , bool TF           // Transpose flag
        , bool DF           // Density flag
        , size_t... CSAs    // Compile time dilatedsubvector arguments
        , typename VT2 >    // Type of the right-hand side vector
inline bool tryAssign( const DilatedSubvector<VT1,TF,DF,CSAs...>& lhs,
                       const Vector<VT2,TF>& rhs, size_t index )
{
   BLAZE_INTERNAL_ASSERT( index <= lhs.size(), "Invalid vector access index" );
   BLAZE_INTERNAL_ASSERT( index + (*rhs).size() <= lhs.size(), "Invalid vector size" );

   return tryAssign( lhs.operand(), *rhs, lhs.offset() + index * lhs.dilation() );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Predict invariant violations by the addition assignment of a vector to a dilatedsubvector.
// \ingroup dilatedsubvector
//
// \param lhs The target left-hand side dilatedsubvector.
// \param rhs The right-hand side vector to be added.
// \param index The index of the first element to be modified.
// \return \a true in case the assignment would be successful, \a false if not.
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename VT1      // Type of the vector
        , bool TF           // Transpose flag
        , bool DF           // Density flag
        , size_t... CSAs    // Compile time dilatedsubvector arguments
        , typename VT2 >    // Type of the right-hand side vector
inline bool tryAddAssign( const DilatedSubvector<VT1,TF,DF,CSAs...>& lhs,
                          const Vector<VT2,TF>& rhs, size_t index )
{
   BLAZE_INTERNAL_ASSERT( index <= lhs.size(), "Invalid vector access index" );
   BLAZE_INTERNAL_ASSERT( index + (*rhs).size() <= lhs.size(), "Invalid vector size" );

   return tryAddAssign( lhs.operand(), *rhs, lhs.offset() + index * lhs.dilation() );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Predict invariant violations by the subtraction assignment of a vector to a dilatedsubvector.
// \ingroup dilatedsubvector
//
// \param lhs The target left-hand side dilatedsubvector.
// \param rhs The right-hand side vector to be subtracted.
// \param index The index of the first element to be modified.
// \return \a true in case the assignment would be successful, \a false if not.
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename VT1      // Type of the vector
        , bool TF           // Transpose flag
        , bool DF           // Density flag
        , size_t... CSAs    // Compile time dilatedsubvector arguments
        , typename VT2 >    // Type of the right-hand side vector
inline bool trySubAssign( const DilatedSubvector<VT1,TF,DF,CSAs...>& lhs,
                          const Vector<VT2,TF>& rhs, size_t index )
{
   BLAZE_INTERNAL_ASSERT( index <= lhs.size(), "Invalid vector access index" );
   BLAZE_INTERNAL_ASSERT( index + (*rhs).size() <= lhs.size(), "Invalid vector size" );

   return trySubAssign( lhs.operand(), *rhs, lhs.offset() + index * lhs.dilation() );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Predict invariant violations by the multiplication assignment of a vector to a dilatedsubvector.
// \ingroup dilatedsubvector
//
// \param lhs The target left-hand side dilatedsubvector.
// \param rhs The right-hand side vector to be multiplied.
// \param index The index of the first element to be modified.
// \return \a true in case the assignment would be successful, \a false if not.
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename VT1      // Type of the vector
        , bool TF           // Transpose flag
        , bool DF           // Density flag
        , size_t... CSAs    // Compile time dilatedsubvector arguments
        , typename VT2 >    // Type of the right-hand side vector
inline bool tryMultAssign( const DilatedSubvector<VT1,TF,DF,CSAs...>& lhs,
                           const Vector<VT2,TF>& rhs, size_t index )
{
   BLAZE_INTERNAL_ASSERT( index <= lhs.size(), "Invalid vector access index" );
   BLAZE_INTERNAL_ASSERT( index + (*rhs).size() <= lhs.size(), "Invalid vector size" );

   return tryMultAssign(lhs.operand(), *rhs, lhs.offset() + index * lhs.dilation() );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Predict invariant violations by the division assignment of a vector to a dilatedsubvector.
// \ingroup dilatedsubvector
//
// \param lhs The target left-hand side dilatedsubvector.
// \param rhs The right-hand side vector divisor.
// \param index The index of the first element to be modified.
// \return \a true in case the assignment would be successful, \a false if not.
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename VT1      // Type of the vector
        , bool TF           // Transpose flag
        , bool DF           // Density flag
        , size_t... CSAs    // Compile time dilatedsubvector arguments
        , typename VT2 >    // Type of the right-hand side vector
inline bool tryDivAssign( const DilatedSubvector<VT1,TF,DF,CSAs...>& lhs,
                          const Vector<VT2,TF>& rhs, size_t index )
{
   BLAZE_INTERNAL_ASSERT( index <= lhs.size(), "Invalid vector access index" );
   BLAZE_INTERNAL_ASSERT( index + (*rhs).size() <= lhs.size(), "Invalid vector size" );

   return tryDivAssign(lhs.operand(), *rhs, lhs.offset() + index * lhs.dilation() );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Removal of all restrictions on the data access to the given dilatedsubvector.
// \ingroup dilatedsubvector
//
// \param sv The dilatedsubvector to be derestricted.
// \return DilatedSubvector without access restrictions.
//
// This function removes all restrictions on the data access to the given dilatedsubvector. It returns a
// dilatedsubvector that does provide the same interface but does not have any restrictions on the data
// access.\n
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in the violation of invariants, erroneous results and/or in compilation errors.
*/
template< typename VT       // Type of the vector
        , bool TF           // Transpose flag
        , bool DF           // Density flag
        , size_t I          // Index of the first element
        , size_t N          // Number of elements
        , size_t Dilation > // Steps between elements
inline decltype(auto) derestrict( DilatedSubvector<VT,TF,DF,I,N,Dilation>& sv )
{
   return dilatedsubvector<I,N,Dilation>( derestrict( sv.operand() ), unchecked );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Removal of all restrictions on the data access to the given temporary dilatedsubvector.
// \ingroup dilatedsubvector
//
// \param sv The temporary dilatedsubvector to be derestricted.
// \return DilatedSubvector without access restrictions.
//
// This function removes all restrictions on the data access to the given temporary dilatedsubvector. It
// returns a dilatedsubvector that does provide the same interface but does not have any restrictions on
// the data access.\n
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in the violation of invariants, erroneous results and/or in compilation errors.
*/
template< typename VT       // Type of the vector
        , bool TF           // Transpose flag
        , bool DF           // Density flag
        , size_t I          // Index of the first element
        , size_t N          // Number of elements
        , size_t Dilation > // Steps between elements
inline decltype(auto) derestrict( DilatedSubvector<VT,TF,DF,I,N,Dilation>&& sv )
{
   return dilatedsubvector<I,N,Dilation>( derestrict( sv.operand() ), unchecked );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Removal of all restrictions on the data access to the given dilatedsubvector.
// \ingroup dilatedsubvector
//
// \param sv The dilatedsubvector to be derestricted.
// \return DilatedSubvector without access restrictions.
//
// This function removes all restrictions on the data access to the given dilatedsubvector. It returns a
// dilatedsubvector that does provide the same interface but does not have any restrictions on the data
// access.\n
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in the violation of invariants, erroneous results and/or in compilation errors.
*/
template< typename VT       // Type of the vector
        , bool TF           // Transpose flag
        , bool DF >         // Density flag
inline decltype(auto) derestrict( DilatedSubvector<VT,TF,DF>& sv )
{
   return dilatedsubvector( derestrict( sv.operand() ), sv.offset(), sv.size(), sv.dilation(), unchecked );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Removal of all restrictions on the data access to the given temporary dilatedsubvector.
// \ingroup dilatedsubvector
//
// \param sv The temporary dilatedsubvector to be derestricted.
// \return DilatedSubvector without access restrictions.
//
// This function removes all restrictions on the data access to the given temporary dilatedsubvector. It
// returns a dilatedsubvector that does provide the same interface but does not have any restrictions on
// the data access.\n
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in the violation of invariants, erroneous results and/or in compilation errors.
*/
template< typename VT       // Type of the vector
        , bool TF           // Transpose flag
        , bool DF >         // Density flag
inline decltype(auto) derestrict( DilatedSubvector<VT,TF,DF>&& sv )
{
   return dilatedsubvector( derestrict( sv.operand() ), sv.offset(), sv.size(), sv.dilation(), unchecked );
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
template< typename VT, bool TF, bool DF, size_t I, size_t N, size_t Dilation >
struct Size< DilatedSubvector<VT,TF,DF,I,N,Dilation>, 0UL >
   : public Ptrdiff_t<N>
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
template< typename VT, bool TF, bool DF, size_t I, size_t N, size_t Dilation >
struct MaxSize< DilatedSubvector<VT,TF,DF,I,N,Dilation>, 0UL >
   : public Ptrdiff_t<N>
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
template< typename VT, bool TF, bool DF, size_t... CSAs >
struct IsRestricted< DilatedSubvector<VT,TF,DF,CSAs...> >
   : public IsRestricted<VT>
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
template< typename VT, bool TF, size_t... CSAs >
struct HasConstDataAccess< DilatedSubvector<VT,TF,true,CSAs...> >
   : public HasConstDataAccess<VT>
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
template< typename VT, bool TF, size_t... CSAs >
struct HasMutableDataAccess< DilatedSubvector<VT,TF,true,CSAs...> >
   : public HasMutableDataAccess<VT>
{};
/*! \endcond */
//*************************************************************************************************

} // namespace blaze

#endif
