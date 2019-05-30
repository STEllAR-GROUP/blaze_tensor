//=================================================================================================
/*!
//  \file blaze_array/math/views/ArraySlice.h
//  \brief Header file for the implementation of the ArraySlice view
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

#ifndef _BLAZE_TENSOR_MATH_VIEWS_ARRAYSLICE_H_
#define _BLAZE_TENSOR_MATH_VIEWS_ARRAYSLICE_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <blaze/math/expressions/Forward.h>
#include <blaze/math/expressions/SchurExpr.h>
#include <blaze/math/typetraits/HasMutableDataAccess.h>
#include <blaze/math/typetraits/IsAligned.h>
#include <blaze/math/typetraits/IsContiguous.h>
#include <blaze/math/typetraits/IsPadded.h>
#include <blaze/math/typetraits/IsResizable.h>
#include <blaze/math/typetraits/IsShrinkable.h>
#include <blaze/math/typetraits/MaxSize.h>
#include <blaze/math/typetraits/Size.h>
#include <blaze/math/views/Row.h>
#include <blaze/math/views/Submatrix.h>
#include <blaze/util/EnableIf.h>

// #include <blaze_tensor/math/expressions/ArrArrAddExpr.h>
#include <blaze_tensor/math/expressions/ArrArrMapExpr.h>
// #include <blaze_tensor/math/expressions/ArrArrMultExpr.h>
// #include <blaze_tensor/math/expressions/ArrArrSubExpr.h>
// #include <blaze_tensor/math/expressions/ArrEvalExpr.h>
#include <blaze_tensor/math/expressions/ArrMapExpr.h>
#include <blaze_tensor/math/expressions/ArrReduceExpr.h>
#include <blaze_tensor/math/expressions/ArrScalarDivExpr.h>
#include <blaze_tensor/math/expressions/ArrScalarMultExpr.h>
// #include <blaze_tensor/math/expressions/ArrSerialExpr.h>
// #include <blaze_tensor/math/expressions/ArrTransExpr.h>
// #include <blaze_tensor/math/expressions/ArrVecMultExpr.h>
#include <blaze_tensor/math/expressions/Array.h>
#include <blaze_tensor/math/expressions/Forward.h>
#include <blaze_tensor/math/expressions/MatExpandExpr.h>
#include <blaze_tensor/math/views/Forward.h>
#include <blaze_tensor/math/views/arrayslice/BaseTemplate.h>
#include <blaze_tensor/math/views/arrayslice/Dense.h>
#include <blaze_tensor/util/ArrayForEach.h>

namespace blaze {

//=================================================================================================
//
//  GLOBAL FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Creating a view on a specific arrayslice of the given array.
// \ingroup arrayslice
//
// \param array The array containing the arrayslice.
// \param args Optional arrayslice arguments.
// \return View on the specified arrayslice of the array.
// \exception std::invalid_argument Invalid arrayslice access index.
//
// This function returns an expression representing the specified arrayslice of the given array.

   \code
   blaze::DynamicArray<double> D;
   blaze::CompressedArray<double> S;
   // ... Resizing and initialization

   // Creating a view on the 3rd arrayslice of the dense array D along dimension 0
   auto arrayslice3 = arrayslice<0UL,3UL>( D );

   // Creating a view on the 4th arrayslice of the sparse array S along dimension 2
   auto arrayslice4 = arrayslice<2UL,4UL>( S );
   \page()

// By default, the provided arrayslice arguments are checked at runtime. In case the arrayslice is not properly
// specified (i.e. if the specified index is greater than or equal to the total number of the arrayslices
// in the given array) a \a std::invalid_argument exception is thrown. The checks can be skipped
// by providing the optional \a blaze::unchecked argument.

   \code
   auto arrayslice3 = arrayslice<0UL,3UL>( D, unchecked );
   auto arrayslice4 = arrayslice<0UL,4UL>( S, unchecked );
   \page()
*/
template< size_t M            // ArraySlice dimension
        , size_t I            // ArraySlice index
        , typename MT         // Type of the array
        , typename... RRAs >  // Optional arrayslice arguments
inline decltype(auto) arrayslice( Array<MT>& array, RRAs... args )
{
   BLAZE_FUNCTION_TRACE;

   using ReturnType = ArraySlice_<M,MT,I>;
   return ReturnType( ~array, args... );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Creating a view on a specific arrayslice of the given constant array.
// \ingroup arrayslice
//
// \param array The constant array containing the arrayslice.
// \param args Optional arrayslice arguments.
// \return View on the specified arrayslice of the array.
// \exception std::invalid_argument Invalid arrayslice access index.
//
// This function returns an expression representing the specified arrayslice of the given constant
// array.

   \code

   const blaze::DynamicArray<double> D( ... );
   const blaze::CompressedArray<double> S( ... );

   // Creating a view on the 3rd arrayslice of the dense array D
   auto arrayslice3 = arrayslice<0UL,3UL>( D );

   // Creating a view on the 4th arrayslice of the sparse array S
   auto arrayslice4 = arrayslice<0UL,4UL>( S );
   \page()

// By default, the provided arrayslice arguments are checked at runtime. In case the arrayslice is not properly
// specified (i.e. if the specified index is greater than or equal to the total number of the arrayslices
// in the given array) a \a std::invalid_argument exception is thrown. The checks can be skipped
// by providing the optional \a blaze::unchecked argument.

   \code
   auto arrayslice3 = arrayslice<0UL,3UL>( D, unchecked );
   auto arrayslice4 = arrayslice<0UL,4UL>( S, unchecked );
   \page()
*/
template< size_t M            // ArraySlice dimension
        , size_t I            // ArraySlice index
        , typename MT         // Type of the array
        , typename... RRAs >  // Optional arrayslice arguments
inline decltype(auto) arrayslice( const Array<MT>& array, RRAs... args )
{
   BLAZE_FUNCTION_TRACE;

   using ReturnType = const ArraySlice_<M,const MT,I>;
   return ReturnType( ~array, args... );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Creating a view on a specific arrayslice of the given temporary array.
// \ingroup arrayslice
//
// \param array The temporary array containing the arrayslice.
// \param args Optional arrayslice arguments.
// \return View on the specified arrayslice of the array.
// \exception std::invalid_argument Invalid arrayslice access index.
//
// This function returns an expression representing the specified arrayslice of the given temporary
// array. In case the arrayslice is not properly specified (i.e. if the specified index is greater
// than or equal to the total number of the arrayslices in the given array) a \a std::invalid_argument
// exception is thrown.
*/
template< size_t M            // ArraySlice dimension
        , size_t I            // ArraySlice index
        , typename MT         // Type of the array
        , typename... RRAs >  // Optional arrayslice arguments
inline decltype(auto) arrayslice( Array<MT>&& array, RRAs... args )
{
   BLAZE_FUNCTION_TRACE;

   using ReturnType = ArraySlice_<M,MT,I>;
   return ReturnType( ~array, args... );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Creating a view on a specific arrayslice of the given array.
// \ingroup arrayslice
//
// \param array The array containing the arrayslice.
// \param index The index of the arrayslice.
// \param args Optional arrayslice arguments.
// \return View on the specified arrayslice of the array.
// \exception std::invalid_argument Invalid arrayslice access index.
//
// This function returns an expression representing the specified arrayslice of the given array.

   \code
   blaze::DynamicArray<double> D;
   blaze::CompressedArray<double> S;
   // ... Resizing and initialization

   // Creating a view on the 3rd arrayslice of the dense array D along dimension 1
   auto arrayslice3 = arrayslice<1UL>( D, 3UL );

   // Creating a view on the 4th arrayslice of the sparse array S along dimension 2
   auto arrayslice4 = arrayslice<2UL>( S, 4UL );
   \page()

// By default, the provided arrayslice arguments are checked at runtime. In case the arrayslice is not properly
// specified (i.e. if the specified index is greater than or equal to the total number of the arrayslices
// in the given array) a \a std::invalid_argument exception is thrown. The checks can be skipped
// by providing the optional \a blaze::unchecked argument.

   \code
   auto arrayslice3 = arrayslice<0UL>( D, 3UL, unchecked );
   auto arrayslice4 = arrayslice<0UL>( S, 4UL, unchecked );
   \page()
*/
template< size_t M            // ArraySlice dimension
        , typename MT         // Type of the array
        , typename... RRAs >  // Optional arrayslice arguments
inline decltype(auto) arrayslice( Array<MT>& array, size_t index, RRAs... args )
{
   BLAZE_FUNCTION_TRACE;

   using ReturnType = ArraySlice_<M,MT>;
   return ReturnType( ~array, index, args... );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Creating a view on a specific arrayslice of the given constant array.
// \ingroup arrayslice
//
// \param array The constant array containing the arrayslice.
// \param index The index of the arrayslice.
// \param args Optional arrayslice arguments.
// \return View on the specified arrayslice of the array.
// \exception std::invalid_argument Invalid arrayslice access index.
//
// This function returns an expression representing the specified arrayslice of the given constant
// array.

   \code
   const blaze::DynamicArray<double> D( ... );
   const blaze::CompressedArray<double> S( ... );

   // Creating a view on the 3rd arrayslice of the dense array D
   auto arrayslice3 = arrayslice( D, 3UL );

   // Creating a view on the 4th arrayslice of the sparse array S
   auto arrayslice4 = arrayslice( S, 4UL );
   \page()

// By default, the provided arrayslice arguments are checked at runtime. In case the arrayslice is not properly
// specified (i.e. if the specified index is greater than or equal to the total number of the arrayslices
// in the given array) a \a std::invalid_argument exception is thrown. The checks can be skipped
// by providing the optional \a blaze::unchecked argument.

   \code
   auto arrayslice3 = arrayslice( D, 3UL, unchecked );
   auto arrayslice4 = arrayslice( S, 4UL, unchecked );
   \page()
*/
template< size_t M            // ArraySlice dimension
        , typename MT         // Type of the array
        , typename... RRAs >  // Optional arrayslice arguments
inline decltype(auto) arrayslice( const Array<MT>& array, size_t index, RRAs... args )
{
   BLAZE_FUNCTION_TRACE;

   using ReturnType = const ArraySlice_<M,const MT>;
   return ReturnType( ~array, index, args... );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Creating a view on a specific arrayslice of the given temporary array.
// \ingroup arrayslice
//
// \param array The temporary array containing the arrayslice.
// \param index The index of the arrayslice.
// \param args Optional arrayslice arguments.
// \return View on the specified arrayslice of the array.
// \exception std::invalid_argument Invalid arrayslice access index.
//
// This function returns an expression representing the specified arrayslice of the given temporary
// array. In case the arrayslice is not properly specified (i.e. if the specified index is greater
// than or equal to the total number of the arrayslices in the given array) a \a std::invalid_argument
// exception is thrown.
*/
template< size_t M            // ArraySlice dimension
        , typename MT         // Type of the array
        , typename... RRAs >  // Optional arrayslice arguments
inline decltype(auto) arrayslice( Array<MT>&& array, size_t index, RRAs... args )
{
   BLAZE_FUNCTION_TRACE;

   using ReturnType = ArraySlice_<M,MT>;
   return ReturnType( ~array, index, args... );
}
//*************************************************************************************************




//=================================================================================================
//
//  GLOBAL RESTRUCTURING FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific arrayslice of the given array/array addition.
// \ingroup arrayslice
//
// \param array The constant array/array addition.
// \param args The runtime arrayslice arguments.
// \return View on the specified arrayslice of the addition.
//
// This function returns an expression representing the specified arrayslice of the given array/array
// addition.
*/
// template< size_t... CRAs      // Compile time arrayslice arguments
//         , typename MT         // Array base type of the expression
//         , typename... RRAs >  // Runtime arrayslice arguments
// inline decltype(auto) arrayslice( const ArrArrAddExpr<MT>& array, RRAs... args )
// {
//    BLAZE_FUNCTION_TRACE;
//
//    return arrayslice<CRAs...>( (~array).leftOperand(), args... ) +
//           arrayslice<CRAs...>( (~array).rightOperand(), args... );
// }
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific arrayslice of the given array/array subtraction.
// \ingroup arrayslice
//
// \param array The constant array/array subtraction.
// \param args The runtime arrayslice arguments.
// \return View on the specified arrayslice of the subtraction.
//
// This function returns an expression representing the specified arrayslice of the given array/array
// subtraction.
*/
// template< size_t... CRAs      // Compile time arrayslice arguments
//         , typename MT         // Array base type of the expression
//         , typename... RRAs >  // Runtime arrayslice arguments
// inline decltype(auto) arrayslice( const ArrArrSubExpr<MT>& array, RRAs... args )
// {
//    BLAZE_FUNCTION_TRACE;
//
//    return arrayslice<CRAs...>( (~array).leftOperand(), args... ) -
//           arrayslice<CRAs...>( (~array).rightOperand(), args... );
// }
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific arrayslice of the given Schur product.
// \ingroup arrayslice
//
// \param array The constant Schur product.
// \param args The runtime arrayslice arguments.
// \return View on the specified arrayslice of the Schur product.
//
// This function returns an expression representing the specified arrayslice of the given Schur product.
*/
// template< size_t... CRAs      // Compile time arrayslice arguments
//         , typename MT         // Array base type of the expression
//         , typename... RRAs >  // Runtime arrayslice arguments
// inline decltype(auto) arrayslice( const SchurExpr<MT>& array, RRAs... args )
// {
//    BLAZE_FUNCTION_TRACE;
//
//    return arrayslice<CRAs...>( (~array).leftOperand(), args... ) *
//           arrayslice<CRAs...>( (~array).rightOperand(), args... );
// }
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific arrayslice of the given array/array multiplication.
// \ingroup arrayslice
//
// \param array The constant array/array multiplication.
// \param args The runtime arrayslice arguments
// \return View on the specified arrayslice of the multiplication.
//
// This function returns an expression representing the specified arrayslice of the given array/array
// multiplication.
*/
// template< size_t... CRAs      // Compile time arrayslice arguments
//         , typename MT         // Array base type of the expression
//         , typename... RRAs >  // Runtime arrayslice arguments
// inline decltype(auto) arrayslice( const ArrArrMultExpr<MT>& array, RRAs... args )
// {
//    BLAZE_FUNCTION_TRACE;
//
//    return arrayslice<CRAs...>( (~array).leftOperand(), args... ) * (~array).rightOperand();
// }
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific arrayslice of the given outer product.
// \ingroup arrayslice
//
// \param array The constant outer product.
// \param args Optional arrayslice arguments.
// \return View on the specified arrayslice of the outer product.
// \exception std::invalid_argument Invalid arrayslice access index.
//
// This function returns an expression representing the specified arrayslice of the given outer product.
*/
// template< size_t I            // ArraySlice index
//         , typename MT         // Array base type of the expression
//         , typename... RRAs >  // Optional arrayslice arguments
// inline decltype(auto) arrayslice( const VecTVecMultExpr<MT>& array, RRAs... args )
// {
//    BLAZE_FUNCTION_TRACE;
//
//    MAYBE_UNUSED( args... );
//
//    if( !Contains_v< TypeList<RRAs...>, Unchecked > ) {
//       if( (~array).arrayslices() <= I ) {
//          BLAZE_THARRAYSLICE_INVALID_ARGUMENT( "Invalid arrayslice access index" );
//       }
//    }
//
//    return (~array).leftOperand()[I] * (~array).rightOperand();
// }
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific arrayslice of the given outer product.
// \ingroup arrayslice
//
// \param array The constant outer product.
// \param index The index of the arrayslice.
// \param args Optional arrayslice arguments.
// \return View on the specified arrayslice of the outer product.
// \exception std::invalid_argument Invalid arrayslice access index.
//
// This function returns an expression representing the specified arrayslice of the given outer product.
*/
// template< typename MT         // Array base type of the expression
//         , typename... RRAs >  // Optional arrayslice arguments
// inline decltype(auto) arrayslice( const VecTVecMultExpr<MT>& array, size_t index, RRAs... args )
// {
//    BLAZE_FUNCTION_TRACE;
//
//    MAYBE_UNUSED( args... );
//
//    if( !Contains_v< TypeList<RRAs...>, Unchecked > ) {
//       if( (~array).arrayslices() <= index ) {
//          BLAZE_THARRAYSLICE_INVALID_ARGUMENT( "Invalid arrayslice access index" );
//       }
//    }
//
//    return (~array).leftOperand()[index] * (~array).rightOperand();
// }
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific arrayslice of the given array/scalar multiplication.
// \ingroup arrayslice
//
// \param array The constant array/scalar multiplication.
// \param args The runtime arrayslice arguments
// \return View on the specified arrayslice of the multiplication.
//
// This function returns an expression representing the specified arrayslice of the given array/scalar
// multiplication.
*/
template< size_t... CRAs      // Compile time arrayslice arguments
        , typename MT         // Array base type of the expression
        , typename... RRAs >  // Runtime arrayslice arguments
inline decltype(auto) arrayslice( const ArrScalarMultExpr<MT>& array, RRAs... args )
{
   BLAZE_FUNCTION_TRACE;

   return arrayslice<CRAs...>( (~array).leftOperand(), args... ) * (~array).rightOperand();
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific arrayslice of the given array/scalar division.
// \ingroup arrayslice
//
// \param array The constant array/scalar division.
// \param args The runtime arrayslice arguments
// \return View on the specified arrayslice of the division.
//
// This function returns an expression representing the specified arrayslice of the given array/scalar
// division.
*/
template< size_t... CRAs      // Compile time arrayslice arguments
        , typename MT         // Array base type of the expression
        , typename... RRAs >  // Runtime arrayslice arguments
inline decltype(auto) arrayslice( const ArrScalarDivExpr<MT>& array, RRAs... args )
{
   BLAZE_FUNCTION_TRACE;

   return arrayslice<CRAs...>( (~array).leftOperand(), args... ) / (~array).rightOperand();
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific arrayslice of the given unary array map operation.
// \ingroup arrayslice
//
// \param array The constant unary array map operation.
// \param args The runtime arrayslice arguments
// \return View on the specified arrayslice of the unary map operation.
//
// This function returns an expression representing the specified arrayslice of the given unary array
// map operation.
*/
template< size_t... CRAs      // Compile time arrayslice arguments
        , typename MT         // Array base type of the expression
        , typename... RRAs >  // Runtime arrayslice arguments
inline decltype(auto) arrayslice( const ArrMapExpr<MT>& array, RRAs... args )
{
   BLAZE_FUNCTION_TRACE;

   return map( arrayslice<CRAs...>( (~array).operand(), args... ), (~array).operation() );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific arrayslice of the given binary array map operation.
// \ingroup arrayslice
//
// \param array The constant binary array map operation.
// \param args The runtime arrayslice arguments
// \return View on the specified arrayslice of the binary map operation.
//
// This function returns an expression representing the specified arrayslice of the given binary array
// map operation.
*/
template< size_t... CRAs      // Compile time arrayslice arguments
        , typename MT         // Array base type of the expression
        , typename... RRAs >  // Runtime arrayslice arguments
inline decltype(auto) arrayslice( const ArrArrMapExpr<MT>& array, RRAs... args )
{
   BLAZE_FUNCTION_TRACE;

   return map( arrayslice<CRAs...>( (~array).leftOperand(), args... ),
               arrayslice<CRAs...>( (~array).rightOperand(), args... ),
               (~array).operation() );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific arrayslice of the given array evaluation operation.
// \ingroup arrayslice
//
// \param array The constant array evaluation operation.
// \param args The runtime arrayslice arguments
// \return View on the specified arrayslice of the evaluation operation.
//
// This function returns an expression representing the specified arrayslice of the given array
// evaluation operation.
*/
// template< size_t... CRAs      // Compile time arrayslice arguments
//         , typename MT         // Array base type of the expression
//         , typename... RRAs >  // Runtime arrayslice arguments
// inline decltype(auto) arrayslice( const ArrEvalExpr<MT>& array, RRAs... args )
// {
//    BLAZE_FUNCTION_TRACE;
//
//    return eval( arrayslice<CRAs...>( (~array).operand(), args... ) );
// }
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific arrayslice of the given array serialization operation.
// \ingroup arrayslice
//
// \param array The constant array serialization operation.
// \param args The runtime arrayslice arguments
// \return View on the specified arrayslice of the serialization operation.
//
// This function returns an expression representing the specified arrayslice of the given array
// serialization operation.
*/
// template< size_t... CRAs      // Compile time arrayslice arguments
//         , typename MT         // Array base type of the expression
//         , typename... RRAs >  // Runtime arrayslice arguments
// inline decltype(auto) arrayslice( const ArrSerialExpr<MT>& array, RRAs... args )
// {
//    BLAZE_FUNCTION_TRACE;
//
//    return serial( arrayslice<CRAs...>( (~array).operand(), args... ) );
// }
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific arrayslice of the given array declaration operation.
// \ingroup arrayslice
//
// \param array The constant array declaration operation.
// \param args The runtime arrayslice arguments
// \return View on the specified arrayslice of the declaration operation.
//
// This function returns an expression representing the specified arrayslice of the given array
// declaration operation.
*/
// template< size_t... CRAs      // Compile time arrayslice arguments
//         , typename MT         // Array base type of the expression
//         , typename... RRAs >  // Runtime arrayslice arguments
// inline decltype(auto) arrayslice( const DeclExpr<MT>& array, RRAs... args )
// {
//    BLAZE_FUNCTION_TRACE;
//
//    return arrayslice<CRAs...>( (~array).operand(), args... );
// }
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific arrayslice of the given array transpose operation.
// \ingroup arrayslice
//
// \param array The constant array transpose operation.
// \param args The runtime arrayslice arguments
// \return View on the specified arrayslice of the transpose operation.
//
// This function returns an expression representing the specified arrayslice of the given array
// transpose operation.
*/
// template< size_t MK           // Compile time arrayslice arguments
//         , size_t MI
//         , size_t MJ
//         , typename MT         // Array base type of the expression
//         , typename... RRAs >  // Runtime arguments
// inline decltype(auto) arrayslice( const ArrTransExpr<MT>& array, size_t index, RRAs... args )
// {
//    BLAZE_FUNCTION_TRACE;
//
//    return arrayslice<MK, MI, MJ>( evaluate( ~array ), index, args... );
// }
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific arrayslice of the given array transpose operation.
// \ingroup arrayslice
//
// \param array The constant array transpose operation.
// \param args The runtime arrayslice arguments
// \return View on the specified arrayslice of the transpose operation.
//
// This function returns an expression representing the specified arrayslice of the given array
// transpose operation.
*/
// template< typename MT         // Array base type of the expression
//         , typename... RRAs >  // Runtime arguments
// inline decltype(auto) arrayslice( const ArrTransExpr<MT>& array, size_t index, RRAs... args )
// {
//    BLAZE_FUNCTION_TRACE;
//
//    return arrayslice( evaluate( ~array ), index, args... );
// }
/*! \endcond */
//*************************************************************************************************



//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific arrayslice of the given matrix expansion operation.
// \ingroup subarray
//
// \param array The constant matrix expansion operation.
// \param args Optional arrayslice arguments.
// \return View on the specified arrayslice of the expansion operation.
//
// This function returns an expression representing the specified arrayslice of the given matrix
// expansion operation.
*/
// template< size_t... CRAs      // Compile time arrayslice arguments
//         , typename MT         // Matrix base type of the expression
//         , size_t... CEAs      // Compile time expansion arguments
//         , typename... RSAs >  // Runtime arrayslice arguments
// inline decltype(auto) arrayslice( const MatExpandExpr<MT,CEAs...>& array, RSAs... args )
// {
//    BLAZE_FUNCTION_TRACE;
//
//    MAYBE_UNUSED( args... );
//
//    return submatrix( (~array).operand(), 0UL, 0UL, (~array).rows(), (~array).columns() );
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
/*!\brief Creating a view on a selection of row of the given array/vector multiplication.
// \ingroup arrayslice
//
// \param matrix The constant array/vector multiplication.
// \param args The runtime element arguments.
// \return View on the specified row of the multiplication.
//
// This function returns an expression representing the specified elements of the given
// matrix/vector multiplication.
*/
// template< size_t... CEAs      // Compile time element arguments
//         , typename MT         // Matrix base type of the expression
//         , typename... REAs >  // Runtime element arguments
// inline decltype(auto) row( const ArrVecMultExpr<MT>& matrix, REAs... args )
// {
//    BLAZE_FUNCTION_TRACE;
//
//    return trans(arrayslice<CEAs...>( (~matrix).leftOperand(), args... ) * (~matrix).rightOperand());
// }
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  ARRAYSLICE OPERATORS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Resetting the given arrayslice.
// \ingroup arrayslice
//
// \param arrayslice The arrayslice to be resetted.
// \return void
*/
template< size_t M          // ArraySlice dimension
        , typename MT       // Type of the array
        , size_t... CRAs >  // Compile time arrayslice arguments
inline void reset( ArraySlice<M,MT,CRAs...>& arrayslice )
{
   arrayslice.reset();
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Resetting the given temporary arrayslice.
// \ingroup arrayslice
//
// \param arrayslice The temporary arrayslice to be resetted.
// \return void
*/
template< size_t M          // ArraySlice dimension
        , typename MT       // Type of the array
        , size_t... CRAs >  // Compile time arrayslice arguments
inline void reset( ArraySlice<M,MT,CRAs...>&& arrayslice )
{
   arrayslice.reset();
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Clearing the given arrayslice.
// \ingroup arrayslice
//
// \param arrayslice The arrayslice to be cleared.
// \return void
//
// Clearing a arrayslice is equivalent to resetting it via the reset() function.
*/
template< size_t M          // ArraySlice dimension
        , typename MT       // Type of the array
        , size_t... CRAs >  // Compile time arrayslice arguments
inline void clear( ArraySlice<M,MT,CRAs...>& arrayslice )
{
   arrayslice.reset();
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Clearing the given temporary arrayslice.
// \ingroup arrayslice
//
// \param arrayslice The temporary arrayslice to be cleared.
// \return void
//
// Clearing a arrayslice is equivalent to resetting it via the reset() function.
*/
template< size_t M          // ArraySlice dimension
        , typename MT       // Type of the array
        , size_t... CRAs >  // Compile time arrayslice arguments
inline void clear( ArraySlice<M,MT,CRAs...>&& arrayslice )
{
   arrayslice.reset();
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns whether the given dense arrayslice is in default state.
// \ingroup arrayslice
//
// \param arrayslice The dense arrayslice to be tested for its default state.
// \return \a true in case the given dense arrayslice is component-wise zero, \a false otherwise.
//
// This function checks whether the dense arrayslice is in default state. For instance, in case the
// arrayslice is instantiated for a built-in integral or floating point data type, the function returns
// \a true in case all arrayslice elements are 0 and \a false in case any arrayslice element is not 0. The
// following example demonstrates the use of the \a isDefault function:

   \code
   blaze::DynamicArray<int> A;
   // ... Resizing and initialization
   if( isDefault( arrayslice( A, 0UL ) ) ) { ... }
   \page()

// Optionally, it is possible to switch between strict semantics (blaze::strict) and relaxed
// semantics (blaze::relaxed):

   \code
   if( isDefault<relaxed>( arrayslice( A, 0UL ) ) ) { ... }
   \page()
*/
template< bool RF           // Relaxation flag
        , size_t M          // ArraySlice dimension
        , typename MT       // Type of the array
        , size_t... CRAs >  // Compile time arrayslice arguments
inline bool isDefault( const ArraySlice<M,MT,CRAs...>& arrayslice )
{
   using blaze::isDefault;

   constexpr size_t N = ArraySlice<M,MT,CRAs...>::num_dimensions();

   return ArrayForEachGroupedAllOf(
      arrayslice.dimensions(), [&]( std::array< size_t, N > const& indices ) {
         return isDefault( arrayslice( indices ) );
      } );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns whether the invariants of the given arrayslice are intact.
// \ingroup arrayslice
//
// \param arrayslice The arrayslice to be tested.
// \return \a true in case the given arrayslice's invariants are intact, \a false otherwise.
//
// This function checks whether the invariants of the arrayslice are intact, i.e. if its state is valid.
// In case the invariants are intact, the function returns \a true, else it will return \a false.
// The following example demonstrates the use of the \a isIntact() function:

   \code
   blaze::DynamicArray<int> A;
   // ... Resizing and initialization
   if( isIntact( arrayslice( A, 0UL ) ) ) { ... }
   \page()
*/
template< size_t M          // ArraySlice dimension
        , typename MT       // Type of the array
        , size_t... CRAs >  // Compile time arrayslice arguments
inline bool isIntact( const ArraySlice<M,MT,CRAs...>& arrayslice ) noexcept
{
   return ( arrayslice.index() < arrayslice.operand().template dimensions<M>() &&
            isIntact( arrayslice.operand() ) );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns whether the two given arrayslices represent the same observable state.
// \ingroup arrayslice
//
// \param a The first arrayslice to be tested for its state.
// \param b The second arrayslice to be tested for its state.
// \return \a true in case the two arrayslices share a state, \a false otherwise.
//
// This overload of the isSame() function tests if the two given arrayslices refer to exactly the same
// range of the same array. In case both arrayslices represent the same observable state, the function
// returns \a true, otherwise it returns \a false.
*/
template< size_t M1          // ArraySlice dimension
        , typename MT1       // Type of the array of the left-hand side arrayslice
        , size_t... CRAs1    // Compile time arrayslice arguments of the left-hand side arrayslice
        , size_t M2          // ArraySlice dimension
        , typename MT2       // Type of the array of the right-hand side arrayslice
        , size_t... CRAs2 >  // Compile time arrayslice arguments of the right-hand side arrayslice
inline bool isSame( const ArraySlice<M1,MT1,CRAs1...>& a,
                    const ArraySlice<M2,MT2,CRAs2...>& b ) noexcept
{
   return ( M1 == M2 && isSame( a.operand(), b.operand() ) && ( a.index() == b.index() ) );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Predict invariant violations by setting a single element of a arrayslice.
// \ingroup arrayslice
//
// \param arrayslice The target arrayslice.
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
template< size_t M        // ArraySlice dimension
        , typename MT     // Type of the array
        , size_t... CRAs  // Compile time arrayslice arguments
        , size_t N        // Number of dimensions
        , typename ET >   // Type of the element
inline bool trySet( const ArraySlice<M,MT,CRAs...>& arrayslice, std::array< size_t, N > const& dims, const ET& value )
{
   constexpr size_t array_N = ( ~arrayslice.operand() ).num_dimensions();
   BLAZE_STATIC_ASSERT( N + 1 == array_N );

   return trySet( arrayslice.operand(), mergeDims<M>( dims, arrayslice.index() ), value );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Predict invariant violations by adding to a single element of a arrayslice.
// \ingroup arrayslice
//
// \param arrayslice The target arrayslice.
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
template< size_t M        // Dimension of the ArraysSlice
        , typename MT     // Type of the array
        , size_t... CRAs  // Compile time arrayslice arguments
        , size_t N        // Dimensions of the ArraysSlice
        , typename ET >   // Type of the element
inline bool tryAdd( const ArraySlice<M,MT,CRAs...>& arrayslice, std::array< size_t, N > const& dims, const ET& value )
{
   constexpr size_t array_N = ( ~arrayslice.operand() ).num_dimensions();
   BLAZE_STATIC_ASSERT( N + 1 == array_N );

   return tryAdd( arrayslice.operand(), mergeDims<M>( dims, arrayslice.index() ), value );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Predict invariant violations by subtracting from a single element of a arrayslice.
// \ingroup arrayslice
//
// \param arrayslice The target arrayslice.
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
template< size_t M        // Dimension of the ArraysSlice
        , typename MT     // Type of the array
        , size_t... CRAs  // Compile time arrayslice arguments
        , size_t N        // Dimensions of the ArraysSlice
        , typename ET >   // Type of the element
inline bool trySub( const ArraySlice<M,MT,CRAs...>& arrayslice, std::array< size_t, N > const& dims, const ET& value )
{
   BLAZE_INTERNAL_ASSERT( i < arrayslice.rows(), "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( j < arrayslice.columns(), "Invalid column access index" );

   return trySub( arrayslice.operand(), mergeDims<M>( dims, arrayslice.index() ), value );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Predict invariant violations by scaling a single element of a arrayslice.
// \ingroup arrayslice
//
// \param arrayslice The target arrayslice.
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
template< size_t M        // Dimension of the ArraysSlice
        , typename MT     // Type of the array
        , size_t... CRAs  // Compile time arrayslice arguments
        , size_t N        // Dimensions of the ArraysSlice
        , typename ET >   // Type of the element
inline bool tryMult( const ArraySlice<M,MT,CRAs...>& arrayslice, std::array< size_t, N > const& dims, const ET& value )
{
   BLAZE_INTERNAL_ASSERT( i < arrayslice.rows(), "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( j < arrayslice.columns(), "Invalid column access index" );

   return tryMult( arrayslice.operand(), mergeDims<M>( dims, arrayslice.index() ), value );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Predict invariant violations by scaling a range of elements of a arrayslice.
// \ingroup arrayslice
//
// \param arrayslice The target arrayslice.
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
template< size_t M        // Dimension of the ArraysSlice
        , typename MT     // Type of the array
        , size_t... CRAs  // Compile time arrayslice arguments
        , size_t N        // Dimensions of the ArraysSlice
        , typename ET >  // Type of the element
BLAZE_ALWAYS_INLINE bool
   tryMult( const ArraySlice<M,MT,CRAs...>& arrayslice, std::array< size_t, N > const& dims, std::array< size_t, N > const& sizes, const ET& value )
{
   BLAZE_INTERNAL_ASSERT( row <= (~arrayslice).rows(), "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( row + rows <= (~arrayslice).rows(), "Invalid rows range size" );
   BLAZE_INTERNAL_ASSERT( col <= (~arrayslice).columns(), "Invalid column access index" );
   BLAZE_INTERNAL_ASSERT( col + cols <= (~arrayslice).columns(), "Invalid columns range size" );

   return tryMult( arrayslice.operand(), mergeDims<M>( dims, arrayslice.index() ), mergeDims<M>( sizes, 1UL ), value );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Predict invariant violations by scaling a single element of a arrayslice.
// \ingroup arrayslice
//
// \param arrayslice The target arrayslice.
// \param index The index of the element to be modified.
// \param value The divisor for the element.
// \return \a true in case the operation would be successful, \a false if not.
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< size_t M        // Dimension of the ArraysSlice
        , typename MT     // Type of the array
        , size_t... CRAs  // Compile time arrayslice arguments
        , size_t N        // Dimensions of the ArraysSlice
        , typename ET >   // Type of the element
inline bool tryDiv( const ArraySlice<M,MT,CRAs...>& arrayslice, std::array< size_t, N > const& dims, const ET& value )
{
   BLAZE_INTERNAL_ASSERT( i < arrayslice.rows(), "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( j < arrayslice.columns(), "Invalid column access index" );

   return tryDiv( arrayslice.operand(), mergeDims<M>( dims, arrayslice.index() ), value );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Predict invariant violations by scaling a range of elements of a arrayslice.
// \ingroup arrayslice
//
// \param arrayslice The target arrayslice.
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
template< size_t M        // Dimension of the ArraysSlice
        , typename MT     // Type of the array
        , size_t... CRAs  // Compile time arrayslice arguments
        , size_t N        // Dimensions of the ArraysSlice
        , typename ET >  // Type of the element
BLAZE_ALWAYS_INLINE bool
   tryDiv( const ArraySlice<M,MT,CRAs...>& arrayslice, std::array< size_t, N > const& dims, std::array< size_t, N > const& sizes, const ET& value )
{
   BLAZE_INTERNAL_ASSERT( row <= (~arrayslice).rows(), "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( row + rows <= (~arrayslice).rows(), "Invalid rows range size" );
   BLAZE_INTERNAL_ASSERT( col <= (~arrayslice).columns(), "Invalid column access index" );
   BLAZE_INTERNAL_ASSERT( col + cols <= (~arrayslice).columns(), "Invalid columns range size" );

   return tryDiv( arrayslice.operand(), mergeDims<M>( dims, arrayslice.index() ), mergeDims<M>( sizes, 1UL ), value );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Predict invariant violations by the assignment of a matrix to a arrayslice.
// \ingroup arrayslice
//
// \param lhs The target left-hand side arrayslice.
// \param rhs The right-hand side vector to be assigned.
// \param index The index of the first element to be modified.
// \return \a true in case the assignment would be successful, \a false if not.
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< size_t M        // Dimension of the ArraysSlice
        , typename MT     // Type of the array
        , size_t... CRAs  // Compile time arrayslice arguments
        , size_t N        // Dimensions of the ArraysSlice
        , typename VT >   // Type of the right-hand side matrix
inline bool tryAssign( const ArraySlice<M,MT,CRAs...>& lhs,
                       const Matrix<VT,false>& rhs, std::array< size_t, N > const& dims )
{
   BLAZE_INTERNAL_ASSERT( i <= lhs.rows(), "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( i + (~rhs).rows() <= lhs.rows(), "Invalid rows range size" );
   BLAZE_INTERNAL_ASSERT( j <= lhs.columns(), "Invalid column access index" );
   BLAZE_INTERNAL_ASSERT( j + (~rhs).columns() <= lhs.columns(), "Invalid columns range size" );

   return tryAssign( lhs.operand(), ~rhs, mergeDims<M>( dims, arrayslice.index() ) );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Predict invariant violations by the addition assignment of a vector to a arrayslice.
// \ingroup arrayslice
//
// \param lhs The target left-hand side arrayslice.
// \param rhs The right-hand side vector to be added.
// \param index The index of the first element to be modified.
// \return \a true in case the assignment would be successful, \a false if not.
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< size_t M        // Dimension of the ArraysSlice
        , typename MT     // Type of the array
        , size_t... CRAs  // Compile time arrayslice arguments
        , size_t N        // Dimensions of the ArraysSlice
        , typename VT >   // Type of the right-hand side matrix
inline bool tryAddAssign( const ArraySlice<M,MT,CRAs...>& lhs,
                          const Matrix<VT,false>& rhs, std::array< size_t, N > const& dims )
{
   BLAZE_INTERNAL_ASSERT( i <= lhs.rows(), "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( i + (~rhs).rows() <= lhs.rows(), "Invalid rows range size" );
   BLAZE_INTERNAL_ASSERT( j <= lhs.columns(), "Invalid column access index" );
   BLAZE_INTERNAL_ASSERT( j + (~rhs).columns() <= lhs.columns(), "Invalid columns range size" );

   return tryAddAssign( lhs.operand(), ~rhs, mergeDims<M>( dims, arrayslice.index() ) );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Predict invariant violations by the subtraction assignment of a vector to a arrayslice.
// \ingroup arrayslice
//
// \param lhs The target left-hand side arrayslice.
// \param rhs The right-hand side vector to be subtracted.
// \param index The index of the first element to be modified.
// \return \a true in case the assignment would be successful, \a false if not.
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< size_t M        // Dimension of the ArraysSlice
        , typename MT     // Type of the array
        , size_t... CRAs  // Compile time arrayslice arguments
        , size_t N        // Dimensions of the ArraysSlice
        , typename VT >   // Type of the right-hand side matrix
inline bool trySubAssign( const ArraySlice<M,MT,CRAs...>& lhs,
                          const Matrix<VT,false>& rhs, std::array< size_t, N > const& dims )
{
   BLAZE_INTERNAL_ASSERT( i <= lhs.rows(), "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( i + (~rhs).rows() <= lhs.rows(), "Invalid rows range size" );
   BLAZE_INTERNAL_ASSERT( j <= lhs.columns(), "Invalid column access index" );
   BLAZE_INTERNAL_ASSERT( j + (~rhs).columns() <= lhs.columns(), "Invalid columns range size" );

   return trySubAssign( lhs.operand(), ~rhs, mergeDims<M>( dims, arrayslice.index() ) );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Predict invariant violations by the multiplication assignment of a vector to a arrayslice.
// \ingroup arrayslice
//
// \param lhs The target left-hand side arrayslice.
// \param rhs The right-hand side vector to be multiplied.
// \param index The index of the first element to be modified.
// \return \a true in case the assignment would be successful, \a false if not.
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< size_t M        // Dimension of the ArraysSlice
        , typename MT     // Type of the array
        , size_t... CRAs  // Compile time arrayslice arguments
        , size_t N        // Dimensions of the ArraysSlice
        , typename VT >   // Type of the right-hand side matrix
inline bool tryMultAssign( const ArraySlice<M,MT,CRAs...>& lhs,
                           const Vector<VT,true>& rhs, std::array< size_t, N > const& dims )
{
   BLAZE_INTERNAL_ASSERT( i <= lhs.rows(), "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( i + (~rhs).rows() <= lhs.rows(), "Invalid rows range size" );
   BLAZE_INTERNAL_ASSERT( j <= lhs.columns(), "Invalid column access index" );
   BLAZE_INTERNAL_ASSERT( j + (~rhs).columns() <= lhs.columns(), "Invalid columns range size" );

   return tryMultAssign( lhs.operand(), ~rhs, mergeDims<M>( dims, arrayslice.index() ) );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Predict invariant violations by the division assignment of a vector to a arrayslice.
// \ingroup arrayslice
//
// \param lhs The target left-hand side arrayslice.
// \param rhs The right-hand side vector divisor.
// \param index The index of the first element to be modified.
// \return \a true in case the assignment would be successful, \a false if not.
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< size_t M        // Dimension of the ArraysSlice
        , typename MT     // Type of the array
        , size_t... CRAs  // Compile time arrayslice arguments
        , size_t N        // Dimensions of the ArraysSlice
        , typename VT >   // Type of the right-hand side matrix
inline bool tryDivAssign( const ArraySlice<M,MT,CRAs...>& lhs,
                          const Matrix<VT,false>& rhs, std::array< size_t, N > const& dims )
{
   BLAZE_INTERNAL_ASSERT( i <= lhs.rows(), "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( i + (~rhs).rows() <= lhs.rows(), "Invalid rows range size" );
   BLAZE_INTERNAL_ASSERT( j <= lhs.columns(), "Invalid column access index" );
   BLAZE_INTERNAL_ASSERT( j + (~rhs).columns() <= lhs.columns(), "Invalid columns range size" );

   return tryDivAssign( lhs.operand(), ~rhs, mergeDims<M>( dims, arrayslice.index() ) );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Removal of all restrictions on the data access to the given arrayslice.
// \ingroup arrayslice
//
// \param r The arrayslice to be derestricted.
// \return ArraySlice without access restrictions.
//
// This function removes all restrictions on the data access to the given arrayslice. It returns a arrayslice
// object that does provide the same interface but does not have any restrictions on the data
// access.\n
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in the violation of invariants, erroneous results and/or in compilation errors.
*/
template< size_t M     // ArraySlice dimension
        , typename MT  // Type of the array
        , size_t I >   // ArraySlice index
inline decltype(auto) derestrict( ArraySlice<M,MT,I>& r )
{
   return arrayslice<M,I>( derestrict( r.operand() ), unchecked );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Removal of all restrictions on the data access to the given temporary arrayslice.
// \ingroup arrayslice
//
// \param r The temporary arrayslice to be derestricted.
// \return ArraySlice without access restrictions.
//
// This function removes all restrictions on the data access to the given temporary arrayslice. It
// returns a arrayslice object that does provide the same interface but does not have any restrictions
// on the data access.\n
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in the violation of invariants, erroneous results and/or in compilation errors.
*/
template< size_t M     // ArraySlice dimension
        , typename MT  // Type of the array
        , size_t I >   // ArraySlice index
inline decltype(auto) derestrict( ArraySlice<M,MT,I>&& r )
{
   return arrayslice<M,I>( derestrict( r.operand() ), unchecked );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Removal of all restrictions on the data access to the given arrayslice.
// \ingroup arrayslice
//
// \param r The arrayslice to be derestricted.
// \return ArraySlice without access restrictions.
//
// This function removes all restrictions on the data access to the given arrayslice. It returns a arrayslice
// object that does provide the same interface but does not have any restrictions on the data
// access.\n
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in the violation of invariants, erroneous results and/or in compilation errors.
*/
template< size_t M        // ArraySlice dimension
        , typename MT >   // ArraySlice index
inline decltype(auto) derestrict( ArraySlice<M,MT>& r )
{
   return arrayslice<M>( derestrict( r.operand() ), r.index(), unchecked );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Removal of all restrictions on the data access to the given temporary arrayslice.
// \ingroup arrayslice
//
// \param r The temporary arrayslice to be derestricted.
// \return ArraySlice without access restrictions.
//
// This function removes all restrictions on the data access to the given temporary arrayslice. It
// returns a arrayslice object that does provide the same interface but does not have any restrictions
// on the data access.\n
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in the violation of invariants, erroneous results and/or in compilation errors.
*/
template< size_t M        // ArraySlice dimension
        , typename MT >   // ArraySlice index
inline decltype(auto) derestrict( ArraySlice<M,MT>&& r )
{
   return arrayslice<M>( derestrict( r.operand() ), r.index(), unchecked );
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
template< size_t M, typename MT, size_t... CRAs, size_t Index >
struct Size< ArraySlice< M, MT, CRAs... >, Index >
   : public Size< MT, ( Index < M ) ? Index : Index + 1 >
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
template< size_t M, typename MT, size_t... CRAs, size_t Index >
struct MaxSize< ArraySlice< M, MT, CRAs... >, Index >
   : public MaxSize< MT, ( Index < M ) ? Index : Index + 1 >
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
template< size_t M, typename MT, size_t... CRAs >
struct IsRestricted< ArraySlice<M,MT,CRAs...> >
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
template< size_t M, typename MT, size_t... CRAs >
struct HasConstDataAccess< ArraySlice<M,MT,CRAs...> >
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
template< size_t M, typename MT, size_t... CRAs >
struct HasMutableDataAccess< ArraySlice<M,MT,CRAs...> >
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
template< size_t M, typename MT, size_t... CRAs >
struct IsAligned< ArraySlice<M,MT,CRAs...> >
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
template< size_t M, typename MT, size_t... CRAs >
struct IsContiguous< ArraySlice<M,MT,CRAs...> >
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
template< size_t M, typename MT, size_t... CRAs >
struct IsPadded< ArraySlice<M,MT,CRAs...> >
   : public BoolConstant< IsPadded_v<MT> >
{};
/*! \endcond */
//*************************************************************************************************

} // namespace blaze

#endif
