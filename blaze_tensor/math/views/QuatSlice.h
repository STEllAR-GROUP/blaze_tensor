//=================================================================================================
/*!
//  \file blaze_quaternion/math/views/QuatSlice.h
//  \brief Header file for the implementation of the QuatSlice view
//
//  Copyright (C) 2012-2018 Klaus Iglberger - All Rights Reserved
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

#ifndef _BLAZE_TENSOR_MATH_VIEWS_QUATSLICE_H_
#define _BLAZE_TENSOR_MATH_VIEWS_QUATSLICE_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <blaze/math/expressions/Forward.h>
#include <blaze/math/expressions/SchurExpr.h>

#include <blaze_tensor/math/expressions/Forward.h>
//#include <blaze_tensor/math/expressions/TensExpandExpr.h>
//#include <blaze_tensor/math/expressions/ArrEvalExpr.h>
#include <blaze_tensor/math/expressions/ArrMapExpr.h>
#include <blaze_tensor/math/expressions/ArrReduceExpr.h>
#include <blaze_tensor/math/expressions/ArrScalarDivExpr.h>
#include <blaze_tensor/math/expressions/ArrScalarMultExpr.h>
//#include <blaze_tensor/math/expressions/ArrSerialExpr.h>
//#include <blaze_tensor/math/expressions/ArrArrAddExpr.h>
#include <blaze_tensor/math/expressions/ArrArrMapExpr.h>
//#include <blaze_tensor/math/expressions/ArrArrMultExpr.h>
//#include <blaze_tensor/math/expressions/ArrArrSubExpr.h>
//#include <blaze_tensor/math/expressions/ArrTransExpr.h>
//#include <blaze_tensor/math/expressions/ArrVecMultExpr.h>
#include <blaze_tensor/math/expressions/Array.h>
#include <blaze_tensor/math/views/Forward.h>
#include <blaze_tensor/math/views/PageSlice.h>
#include <blaze_tensor/math/views/quatslice/BaseTemplate.h>
#include <blaze_tensor/math/views/quatslice/Dense.h>
#include <blaze_tensor/math/views/Subtensor.h>
#include <blaze_tensor/util/ArrayForEach.h>

namespace blaze {

//=================================================================================================
//
//  GLOBAL FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Creating a view on a specific quatslice of the given quaternion.
// \ingroup quatslice
//
// \param quaternion The quaternion containing the quatslice.
// \param args Optional quatslice arguments.
// \return View on the specified quatslice of the quaternion.
// \exception std::invalid_argument Invalid quatslice access index.
//
// This function returns an expression representing the specified quatslice of the given quaternion.

   \code
   blaze::DynamicArray<double> D;
   blaze::CompressedArray<double> S;
   // ... Resizing and initialization

   // Creating a view on the 3rd quatslice of the dense quaternion D
   auto quatslice3 = quatslice<3UL>( D );

   // Creating a view on the 4th quatslice of the sparse quaternion S
   auto quatslice4 = quatslice<4UL>( S );
   \quat()

// By default, the provided quatslice arguments are checked at runtime. In case the quatslice is not properly
// specified (i.e. if the specified index is greater than or equal to the total number of the quatslices
// in the given quaternion) a \a std::invalid_argument exception is thrown. The checks can be skipped
// by providing the optional \a blaze::unchecked argument.

   \code
   auto quatslice3 = quatslice<3UL>( D, unchecked );
   auto quatslice4 = quatslice<4UL>( S, unchecked );
   \quat()
*/
template< size_t I            // QuatSlice index
        , typename AT         // Type of the quaternion
        , typename... RRAs >  // Optional quatslice arguments
inline decltype(auto) quatslice( Array<AT>& quaternion, RRAs... args )
{
   BLAZE_FUNCTION_TRACE;

   using ReturnType = QuatSlice_<AT,I>;
   return ReturnType( ~quaternion, args... );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Creating a view on a specific quatslice of the given constant quaternion.
// \ingroup quatslice
//
// \param quaternion The constant quaternion containing the quatslice.
// \param args Optional quatslice arguments.
// \return View on the specified quatslice of the quaternion.
// \exception std::invalid_argument Invalid quatslice access index.
//
// This function returns an expression representing the specified quatslice of the given constant
// quaternion.

   \code

   const blaze::DynamicArray<double> D( ... );
   const blaze::CompressedArray<double> S( ... );

   // Creating a view on the 3rd quatslice of the dense quaternion D
   auto quatslice3 = quatslice<3UL>( D );

   // Creating a view on the 4th quatslice of the sparse quaternion S
   auto quatslice4 = quatslice<4UL>( S );
   \quat()

// By default, the provided quatslice arguments are checked at runtime. In case the quatslice is not properly
// specified (i.e. if the specified index is greater than or equal to the total number of the quatslices
// in the given quaternion) a \a std::invalid_argument exception is thrown. The checks can be skipped
// by providing the optional \a blaze::unchecked argument.

   \code
   auto quatslice3 = quatslice<3UL>( D, unchecked );
   auto quatslice4 = quatslice<4UL>( S, unchecked );
   \quat()
*/
template< size_t I            // QuatSlice index
        , typename AT         // Type of the quaternion
        , typename... RRAs >  // Optional quatslice arguments
inline decltype(auto) quatslice( const Array<AT>& quaternion, RRAs... args )
{
   BLAZE_FUNCTION_TRACE;

   using ReturnType = const QuatSlice_<const AT,I>;
   return ReturnType( ~quaternion, args... );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Creating a view on a specific quatslice of the given temporary quaternion.
// \ingroup quatslice
//
// \param quaternion The temporary quaternion containing the quatslice.
// \param args Optional quatslice arguments.
// \return View on the specified quatslice of the quaternion.
// \exception std::invalid_argument Invalid quatslice access index.
//
// This function returns an expression representing the specified quatslice of the given temporary
// quaternion. In case the quatslice is not properly specified (i.e. if the specified index is greater
// than or equal to the total number of the quatslices in the given quaternion) a \a std::invalid_argument
// exception is thrown.
*/
template< size_t I            // QuatSlice index
        , typename AT         // Type of the quaternion
        , typename... RRAs >  // Optional quatslice arguments
inline decltype(auto) quatslice( Array<AT>&& quaternion, RRAs... args )
{
   BLAZE_FUNCTION_TRACE;

   using ReturnType = QuatSlice_<AT,I>;
   return ReturnType( ~quaternion, args... );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Creating a view on a specific quatslice of the given quaternion.
// \ingroup quatslice
//
// \param quaternion The quaternion containing the quatslice.
// \param index The index of the quatslice.
// \param args Optional quatslice arguments.
// \return View on the specified quatslice of the quaternion.
// \exception std::invalid_argument Invalid quatslice access index.
//
// This function returns an expression representing the specified quatslice of the given quaternion.

   \code
   blaze::DynamicArray<double> D;
   blaze::CompressedArray<double> S;
   // ... Resizing and initialization

   // Creating a view on the 3rd quatslice of the dense quaternion D
   auto quatslice3 = quatslice( D, 3UL );

   // Creating a view on the 4th quatslice of the sparse quaternion S
   auto quatslice4 = quatslice( S, 4UL );
   \quat()

// By default, the provided quatslice arguments are checked at runtime. In case the quatslice is not properly
// specified (i.e. if the specified index is greater than or equal to the total number of the quatslices
// in the given quaternion) a \a std::invalid_argument exception is thrown. The checks can be skipped
// by providing the optional \a blaze::unchecked argument.

   \code
   auto quatslice3 = quatslice( D, 3UL, unchecked );
   auto quatslice4 = quatslice( S, 4UL, unchecked );
   \quat()
*/
template< typename AT         // Type of the quaternion
        , typename... RRAs >  // Optional quatslice arguments
inline decltype(auto) quatslice( Array<AT>& quaternion, size_t index, RRAs... args )
{
   BLAZE_FUNCTION_TRACE;

   using ReturnType = QuatSlice_<AT>;
   return ReturnType( ~quaternion, index, args... );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Creating a view on a specific quatslice of the given constant quaternion.
// \ingroup quatslice
//
// \param quaternion The constant quaternion containing the quatslice.
// \param index The index of the quatslice.
// \param args Optional quatslice arguments.
// \return View on the specified quatslice of the quaternion.
// \exception std::invalid_argument Invalid quatslice access index.
//
// This function returns an expression representing the specified quatslice of the given constant
// quaternion.

   \code
   const blaze::DynamicArray<double> D( ... );
   const blaze::CompressedArray<double> S( ... );

   // Creating a view on the 3rd quatslice of the dense quaternion D
   auto quatslice3 = quatslice( D, 3UL );

   // Creating a view on the 4th quatslice of the sparse quaternion S
   auto quatslice4 = quatslice( S, 4UL );
   \quat()

// By default, the provided quatslice arguments are checked at runtime. In case the quatslice is not properly
// specified (i.e. if the specified index is greater than or equal to the total number of the quatslices
// in the given quaternion) a \a std::invalid_argument exception is thrown. The checks can be skipped
// by providing the optional \a blaze::unchecked argument.

   \code
   auto quatslice3 = quatslice( D, 3UL, unchecked );
   auto quatslice4 = quatslice( S, 4UL, unchecked );
   \quat()
*/
template< typename AT         // Type of the quaternion
        , typename... RRAs >  // Optional quatslice arguments
inline decltype(auto) quatslice( const Array<AT>& quaternion, size_t index, RRAs... args )
{
   BLAZE_FUNCTION_TRACE;

   using ReturnType = const QuatSlice_<const AT>;
   return ReturnType( ~quaternion, index, args... );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Creating a view on a specific quatslice of the given temporary quaternion.
// \ingroup quatslice
//
// \param quaternion The temporary quaternion containing the quatslice.
// \param index The index of the quatslice.
// \param args Optional quatslice arguments.
// \return View on the specified quatslice of the quaternion.
// \exception std::invalid_argument Invalid quatslice access index.
//
// This function returns an expression representing the specified quatslice of the given temporary
// quaternion. In case the quatslice is not properly specified (i.e. if the specified index is greater
// than or equal to the total number of the quatslices in the given quaternion) a \a std::invalid_argument
// exception is thrown.
*/
template< typename AT         // Type of the quaternion
        , typename... RRAs >  // Optional quatslice arguments
inline decltype(auto) quatslice( Array<AT>&& quaternion, size_t index, RRAs... args )
{
   BLAZE_FUNCTION_TRACE;

   using ReturnType = QuatSlice_<AT>;
   return ReturnType( ~quaternion, index, args... );
}
//*************************************************************************************************




//=================================================================================================
//
//  GLOBAL RESTRUCTURING FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific quatslice of the given quaternion/quaternion addition.
// \ingroup quatslice
//
// \param quaternion The constant quaternion/quaternion addition.
// \param args The runtime quatslice arguments.
// \return View on the specified quatslice of the addition.
//
// This function returns an expression representing the specified quatslice of the given quaternion/quaternion
// addition.
*/
//template< size_t... CRAs      // Compile time quatslice arguments
//        , typename AT         // Array base type of the expression
//        , typename... RRAs >  // Runtime quatslice arguments
//inline decltype(auto) quatslice( const ArrArrAddExpr<AT>& quaternion, RRAs... args )
//{
//   BLAZE_FUNCTION_TRACE;
//
//   return quatslice<CRAs...>( (~quaternion).leftOperand(), args... ) +
//          quatslice<CRAs...>( (~quaternion).rightOperand(), args... );
//}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific quatslice of the given quaternion/quaternion subtraction.
// \ingroup quatslice
//
// \param quaternion The constant quaternion/quaternion subtraction.
// \param args The runtime quatslice arguments.
// \return View on the specified quatslice of the subtraction.
//
// This function returns an expression representing the specified quatslice of the given quaternion/quaternion
// subtraction.
*/
//template< size_t... CRAs      // Compile time quatslice arguments
//        , typename AT         // Array base type of the expression
//        , typename... RRAs >  // Runtime quatslice arguments
//inline decltype(auto) quatslice( const ArrArrSubExpr<AT>& quaternion, RRAs... args )
//{
//   BLAZE_FUNCTION_TRACE;
//
//   return quatslice<CRAs...>( (~quaternion).leftOperand(), args... ) -
//          quatslice<CRAs...>( (~quaternion).rightOperand(), args... );
//}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific quatslice of the given Schur product.
// \ingroup quatslice
//
// \param quaternion The constant Schur product.
// \param args The runtime quatslice arguments.
// \return View on the specified quatslice of the Schur product.
//
// This function returns an expression representing the specified quatslice of the given Schur product.
*/
//template< size_t... CRAs      // Compile time quatslice arguments
//        , typename AT         // Array base type of the expression
//        , typename... RRAs >  // Runtime quatslice arguments
//inline decltype(auto) quatslice( const SchurExpr<AT>& quaternion, RRAs... args )
//{
//   BLAZE_FUNCTION_TRACE;
//
//   return quatslice<CRAs...>( (~quaternion).leftOperand(), args... ) *
//          quatslice<CRAs...>( (~quaternion).rightOperand(), args... );
//}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific quatslice of the given quaternion/quaternion multiplication.
// \ingroup quatslice
//
// \param quaternion The constant quaternion/quaternion multiplication.
// \param args The runtime quatslice arguments
// \return View on the specified quatslice of the multiplication.
//
// This function returns an expression representing the specified quatslice of the given quaternion/quaternion
// multiplication.
*/
//template< size_t... CRAs      // Compile time quatslice arguments
//        , typename AT         // Array base type of the expression
//        , typename... RRAs >  // Runtime quatslice arguments
//inline decltype(auto) quatslice( const ArrArrMultExpr<AT>& quaternion, RRAs... args )
//{
//   BLAZE_FUNCTION_TRACE;
//
//   return quatslice<CRAs...>( (~quaternion).leftOperand(), args... ) * (~quaternion).rightOperand();
//}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific quatslice of the given outer product.
// \ingroup quatslice
//
// \param quaternion The constant outer product.
// \param args Optional quatslice arguments.
// \return View on the specified quatslice of the outer product.
// \exception std::invalid_argument Invalid quatslice access index.
//
// This function returns an expression representing the specified quatslice of the given outer product.
*/
// template< size_t I            // QuatSlice index
//         , typename AT         // Array base type of the expression
//         , typename... RRAs >  // Optional quatslice arguments
// inline decltype(auto) quatslice( const VecTVecMultExpr<AT>& quaternion, RRAs... args )
// {
//    BLAZE_FUNCTION_TRACE;
//
//    MAYBE_UNUSED( args... );
//
//    if( !Contains_v< TypeList<RRAs...>, Unchecked > ) {
//       if( (~quaternion).quatslices() <= I ) {
//          BLAZE_THQUATSLICE_INVALID_ARGUMENT( "Invalid quatslice access index" );
//       }
//    }
//
//    return (~quaternion).leftOperand()[I] * (~quaternion).rightOperand();
// }
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific quatslice of the given outer product.
// \ingroup quatslice
//
// \param quaternion The constant outer product.
// \param index The index of the quatslice.
// \param args Optional quatslice arguments.
// \return View on the specified quatslice of the outer product.
// \exception std::invalid_argument Invalid quatslice access index.
//
// This function returns an expression representing the specified quatslice of the given outer product.
*/
// template< typename AT         // Array base type of the expression
//         , typename... RRAs >  // Optional quatslice arguments
// inline decltype(auto) quatslice( const VecTVecMultExpr<AT>& quaternion, size_t index, RRAs... args )
// {
//    BLAZE_FUNCTION_TRACE;
//
//    MAYBE_UNUSED( args... );
//
//    if( !Contains_v< TypeList<RRAs...>, Unchecked > ) {
//       if( (~quaternion).quatslices() <= index ) {
//          BLAZE_THQUATSLICE_INVALID_ARGUMENT( "Invalid quatslice access index" );
//       }
//    }
//
//    return (~quaternion).leftOperand()[index] * (~quaternion).rightOperand();
// }
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific quatslice of the given quaternion/scalar multiplication.
// \ingroup quatslice
//
// \param quaternion The constant quaternion/scalar multiplication.
// \param args The runtime quatslice arguments
// \return View on the specified quatslice of the multiplication.
//
// This function returns an expression representing the specified quatslice of the given quaternion/scalar
// multiplication.
*/
template< size_t... CRAs      // Compile time quatslice arguments
        , typename AT         // Array base type of the expression
        , typename... RRAs >  // Runtime quatslice arguments
inline decltype(auto) quatslice( const ArrScalarMultExpr<AT>& quaternion, RRAs... args )
{
   BLAZE_FUNCTION_TRACE;

   return quatslice<CRAs...>( (~quaternion).leftOperand(), args... ) * (~quaternion).rightOperand();
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific quatslice of the given quaternion/scalar division.
// \ingroup quatslice
//
// \param quaternion The constant quaternion/scalar division.
// \param args The runtime quatslice arguments
// \return View on the specified quatslice of the division.
//
// This function returns an expression representing the specified quatslice of the given quaternion/scalar
// division.
*/
template< size_t... CRAs      // Compile time quatslice arguments
        , typename AT         // Array base type of the expression
        , typename... RRAs >  // Runtime quatslice arguments
inline decltype(auto) quatslice( const ArrScalarDivExpr<AT>& quaternion, RRAs... args )
{
   BLAZE_FUNCTION_TRACE;

   return quatslice<CRAs...>( (~quaternion).leftOperand(), args... ) / (~quaternion).rightOperand();
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific quatslice of the given unary quaternion map operation.
// \ingroup quatslice
//
// \param quaternion The constant unary quaternion map operation.
// \param args The runtime quatslice arguments
// \return View on the specified quatslice of the unary map operation.
//
// This function returns an expression representing the specified quatslice of the given unary quaternion
// map operation.
*/
template< size_t... CRAs      // Compile time quatslice arguments
        , typename AT         // Array base type of the expression
        , typename... RRAs >  // Runtime quatslice arguments
inline decltype(auto) quatslice( const ArrMapExpr<AT>& quaternion, RRAs... args )
{
   BLAZE_FUNCTION_TRACE;

   return map( quatslice<CRAs...>( (~quaternion).operand(), args... ), (~quaternion).operation() );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific quatslice of the given binary quaternion map operation.
// \ingroup quatslice
//
// \param quaternion The constant binary quaternion map operation.
// \param args The runtime quatslice arguments
// \return View on the specified quatslice of the binary map operation.
//
// This function returns an expression representing the specified quatslice of the given binary quaternion
// map operation.
*/
template< size_t... CRAs      // Compile time quatslice arguments
        , typename AT         // Array base type of the expression
        , typename... RRAs >  // Runtime quatslice arguments
inline decltype(auto) quatslice( const ArrArrMapExpr<AT>& quaternion, RRAs... args )
{
   BLAZE_FUNCTION_TRACE;

   return map( quatslice<CRAs...>( (~quaternion).leftOperand(), args... ),
               quatslice<CRAs...>( (~quaternion).rightOperand(), args... ),
               (~quaternion).operation() );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific quatslice of the given quaternion evaluation operation.
// \ingroup quatslice
//
// \param quaternion The constant quaternion evaluation operation.
// \param args The runtime quatslice arguments
// \return View on the specified quatslice of the evaluation operation.
//
// This function returns an expression representing the specified quatslice of the given quaternion
// evaluation operation.
*/
//template< size_t... CRAs      // Compile time quatslice arguments
//        , typename AT         // Array base type of the expression
//        , typename... RRAs >  // Runtime quatslice arguments
//inline decltype(auto) quatslice( const ArrEvalExpr<AT>& quaternion, RRAs... args )
//{
//   BLAZE_FUNCTION_TRACE;
//
//   return eval( quatslice<CRAs...>( (~quaternion).operand(), args... ) );
//}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific quatslice of the given quaternion serialization operation.
// \ingroup quatslice
//
// \param quaternion The constant quaternion serialization operation.
// \param args The runtime quatslice arguments
// \return View on the specified quatslice of the serialization operation.
//
// This function returns an expression representing the specified quatslice of the given quaternion
// serialization operation.
*/
//template< size_t... CRAs      // Compile time quatslice arguments
//        , typename AT         // Array base type of the expression
//        , typename... RRAs >  // Runtime quatslice arguments
//inline decltype(auto) quatslice( const ArrSerialExpr<AT>& quaternion, RRAs... args )
//{
//   BLAZE_FUNCTION_TRACE;
//
//   return serial( quatslice<CRAs...>( (~quaternion).operand(), args... ) );
//}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific quatslice of the given quaternion declaration operation.
// \ingroup quatslice
//
// \param quaternion The constant quaternion declaration operation.
// \param args The runtime quatslice arguments
// \return View on the specified quatslice of the declaration operation.
//
// This function returns an expression representing the specified quatslice of the given quaternion
// declaration operation.
*/
//template< size_t... CRAs      // Compile time quatslice arguments
//        , typename AT         // Array base type of the expression
//        , typename... RRAs >  // Runtime quatslice arguments
//inline decltype(auto) quatslice( const DeclExpr<AT>& quaternion, RRAs... args )
//{
//   BLAZE_FUNCTION_TRACE;
//
//   return quatslice<CRAs...>( (~quaternion).operand(), args... );
//}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific quatslice of the given quaternion transpose operation.
// \ingroup quatslice
//
// \param quaternion The constant quaternion transpose operation.
// \param args The runtime quatslice arguments
// \return View on the specified quatslice of the transpose operation.
//
// This function returns an expression representing the specified quatslice of the given quaternion
// transpose operation.
*/
//template< size_t MK           // Compile time quatslice arguments
//        , size_t MI
//        , size_t MJ
//        , typename AT         // Array base type of the expression
//        , typename... RRAs >  // Runtime arguments
//inline decltype(auto) quatslice( const ArrTransExpr<AT>& quaternion, size_t index, RRAs... args )
//{
//   BLAZE_FUNCTION_TRACE;
//
//   return quatslice<MK, MI, MJ>( evaluate( ~quaternion ), index, args... );
//}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific quatslice of the given quaternion transpose operation.
// \ingroup quatslice
//
// \param quaternion The constant quaternion transpose operation.
// \param args The runtime quatslice arguments
// \return View on the specified quatslice of the transpose operation.
//
// This function returns an expression representing the specified quatslice of the given quaternion
// transpose operation.
*/
//template< typename AT         // Array base type of the expression
//        , typename... RRAs >  // Runtime arguments
//inline decltype(auto) quatslice( const ArrTransExpr<AT>& quaternion, size_t index, RRAs... args )
//{
//   BLAZE_FUNCTION_TRACE;
//
//   return quatslice( evaluate( ~quaternion ), index, args... );
//}
/*! \endcond */
//*************************************************************************************************



//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific quatslice of the given matrix expansion operation.
// \ingroup subquaternion
//
// \param quaternion The constant matrix expansion operation.
// \param args Optional quatslice arguments.
// \return View on the specified quatslice of the expansion operation.
//
// This function returns an expression representing the specified quatslice of the given matrix
// expansion operation.
*/
//template< size_t... CRAs      // Compile time quatslice arguments
//        , typename AT         // Matrix base type of the expression
//        , size_t... CEAs      // Compile time expansion arguments
//        , typename... RSAs >  // Runtime quatslice arguments
//inline decltype(auto) quatslice( const TensExpandExpr<AT,CEAs...>& quaternion, RSAs... args )
//{
//   BLAZE_FUNCTION_TRACE;
//
//   MAYBE_UNUSED( args... );
//
//   return submatrix( (~quaternion).operand(), 0UL, 0UL, (~quaternion).rows(), (~quaternion).columns() );
//}
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  GLOBAL RESTRUCTURING FUNCTIONS (ROW)
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a selection of row of the given quaternion/vector multiplication.
// \ingroup quatslice
//
// \param matrix The constant quaternion/vector multiplication.
// \param args The runtime element arguments.
// \return View on the specified row of the multiplication.
//
// This function returns an expression representing the specified elements of the given
// matrix/vector multiplication.
*/
//template< size_t... CEAs      // Compile time element arguments
//        , typename AT         // Matrix base type of the expression
//        , typename... REAs >  // Runtime element arguments
//inline decltype(auto) row( const ArrVecMultExpr<AT>& matrix, REAs... args )
//{
//   BLAZE_FUNCTION_TRACE;
//
//   return trans(quatslice<CEAs...>( (~matrix).leftOperand(), args... ) * (~matrix).rightOperand());
//}
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  QUATSLICE OPERATORS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Resetting the given quatslice.
// \ingroup quatslice
//
// \param quatslice The quatslice to be resetted.
// \return void
*/
template< typename AT       // Type of the quaternion
                   // Density flag
                   // Symmetry flag
        , size_t... CRAs >  // Compile time quatslice arguments
inline void reset( QuatSlice<AT,CRAs...>& quatslice )
{
   quatslice.reset();
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Resetting the given temporary quatslice.
// \ingroup quatslice
//
// \param quatslice The temporary quatslice to be resetted.
// \return void
*/
template< typename AT       // Type of the quaternion
                   // Density flag
                   // Symmetry flag
        , size_t... CRAs >  // Compile time quatslice arguments
inline void reset( QuatSlice<AT,CRAs...>&& quatslice )
{
   quatslice.reset();
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Clearing the given quatslice.
// \ingroup quatslice
//
// \param quatslice The quatslice to be cleared.
// \return void
//
// Clearing a quatslice is equivalent to resetting it via the reset() function.
*/
template< typename AT       // Type of the quaternion
                   // Density flag
                   // Symmetry flag
        , size_t... CRAs >  // Compile time quatslice arguments
inline void clear( QuatSlice<AT,CRAs...>& quatslice )
{
   quatslice.reset();
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Clearing the given temporary quatslice.
// \ingroup quatslice
//
// \param quatslice The temporary quatslice to be cleared.
// \return void
//
// Clearing a quatslice is equivalent to resetting it via the reset() function.
*/
template< typename AT       // Type of the quaternion
                   // Density flag
                   // Symmetry flag
        , size_t... CRAs >  // Compile time quatslice arguments
inline void clear( QuatSlice<AT,CRAs...>&& quatslice )
{
   quatslice.reset();
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns whether the given dense quatslice is in default state.
// \ingroup quatslice
//
// \param quatslice The dense quatslice to be tested for its default state.
// \return \a true in case the given dense quatslice is component-wise zero, \a false otherwise.
//
// This function checks whether the dense quatslice is in default state. For instance, in case the
// quatslice is instantiated for a built-in integral or floating point data type, the function returns
// \a true in case all quatslice elements are 0 and \a false in case any quatslice element is not 0. The
// following example demonstrates the use of the \a isDefault function:

   \code
   blaze::DynamicArray<int> A;
   // ... Resizing and initialization
   if( isDefault( quatslice( A, 0UL ) ) ) { ... }
   \quat()

// Optionally, it is possible to switch between strict semantics (blaze::strict) and relaxed
// semantics (blaze::relaxed):

   \code
   if( isDefault<relaxed>( quatslice( A, 0UL ) ) ) { ... }
   \quat()
*/
template< bool RF           // Relaxation flag
        , typename AT       // Type of the quaternion
        , size_t... CRAs >  // Compile time quatslice arguments
inline bool isDefault( const QuatSlice<AT,CRAs...>& quatslice )
{
   using blaze::isDefault;

   for( size_t k=0UL; k<(~sm).pages(); ++k )
      for( size_t i=0UL; i<quatslice.rows(); ++i )
         for( size_t j=0UL; j<quatslice.columns(); ++j )
            if( !isDefault<RF>( quatslice(k, i, j) ) )
               return false;
   return true;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns whether the invariants of the given quatslice are intact.
// \ingroup quatslice
//
// \param quatslice The quatslice to be tested.
// \return \a true in case the given quatslice's invariants are intact, \a false otherwise.
//
// This function checks whether the invariants of the quatslice are intact, i.e. if its state is valid.
// In case the invariants are intact, the function returns \a true, else it will return \a false.
// The following example demonstrates the use of the \a isIntact() function:

   \code
   blaze::DynamicArray<int> A;
   // ... Resizing and initialization
   if( isIntact( quatslice( A, 0UL ) ) ) { ... }
   \quat()
*/
template< typename AT       // Type of the quaternion
        , size_t... CRAs >  // Compile time quatslice arguments
inline bool isIntact( const QuatSlice<AT,CRAs...>& quatslice ) noexcept
{
   return ( quatslice.quat() < quatslice.operand().quats() &&
            isIntact( quatslice.operand() ) );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns whether the two given quatslices represent the same observable state.
// \ingroup quatslice
//
// \param a The first quatslice to be tested for its state.
// \param b The second quatslice to be tested for its state.
// \return \a true in case the two quatslices share a state, \a false otherwise.
//
// This overload of the isSame() function tests if the two given quatslices refer to exactly the same
// range of the same quaternion. In case both quatslices represent the same observable state, the function
// returns \a true, otherwise it returns \a false.
*/
template< typename AT1       // Type of the quaternion of the left-hand side quatslice
        , size_t... CRAs1    // Compile time quatslice arguments of the left-hand side quatslice
        , typename AT2       // Type of the quaternion of the right-hand side quatslice
        , size_t... CRAs2 >  // Compile time quatslice arguments of the right-hand side quatslice
inline bool isSame( const QuatSlice<AT1,CRAs1...>& a,
                    const QuatSlice<AT2,CRAs2...>& b ) noexcept
{
   return ( isSame( a.operand(), b.operand() ) && ( a.quat() == b.quat() ) );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Predict invariant violations by setting a single element of a quatslice.
// \ingroup quatslice
//
// \param quatslice The target quatslice.
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
template< typename AT     // Type of the quaternion
        , size_t... CRAs  // Compile time quatslice arguments
        , typename ET >   // Type of the element
inline bool trySet( const QuatSlice<AT,CRAs...>& quatslice, size_t k, size_t i, size_t j, const ET& value )
{
   BLAZE_INTERNAL_ASSERT( k < quatslice.pages(),   "Invalid page access index" );
   BLAZE_INTERNAL_ASSERT( i < quatslice.rows(),    "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( j < quatslice.columns(), "Invalid column access index" );

   return trySet( quatslice.operand(), quatslice.quat(), k, i, j, value );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Predict invariant violations by adding to a single element of a quatslice.
// \ingroup quatslice
//
// \param quatslice The target quatslice.
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
template< typename AT     // Type of the quaternion
        , size_t... CRAs  // Compile time quatslice arguments
        , typename ET >   // Type of the element
inline bool tryAdd( const QuatSlice<AT,CRAs...>& quatslice, size_t k, size_t i, size_t j, const ET& value )
{
   BLAZE_INTERNAL_ASSERT( k < quatslice.pages(),   "Invalid page access index" );
   BLAZE_INTERNAL_ASSERT( i < quatslice.rows(), "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( j < quatslice.columns(), "Invalid column access index" );

   return tryAdd( quatslice.operand(), quatslice.quat(), k, i, j, value );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Predict invariant violations by subtracting from a single element of a quatslice.
// \ingroup quatslice
//
// \param quatslice The target quatslice.
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
template< typename AT     // Type of the quaternion
        , size_t... CRAs  // Compile time quatslice arguments
        , typename ET >   // Type of the element
inline bool trySub( const QuatSlice<AT,CRAs...>& quatslice, size_t k, size_t i, size_t j, const ET& value )
{
   BLAZE_INTERNAL_ASSERT( k < quatslice.pages(),   "Invalid page access index" );
   BLAZE_INTERNAL_ASSERT( i < quatslice.rows(), "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( j < quatslice.columns(), "Invalid column access index" );

   return trySub( quatslice.operand(), quatslice.quat(), k, i, j, value );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Predict invariant violations by scaling a single element of a quatslice.
// \ingroup quatslice
//
// \param quatslice The target quatslice.
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
template< typename AT     // Type of the quaternion
        , size_t... CRAs  // Compile time quatslice arguments
        , typename ET >   // Type of the element
inline bool tryMult( const QuatSlice<AT,CRAs...>& quatslice, size_t k, size_t i, size_t j, const ET& value )
{
   BLAZE_INTERNAL_ASSERT( k < quatslice.pages(),   "Invalid page access index" );
   BLAZE_INTERNAL_ASSERT( i < quatslice.rows(), "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( j < quatslice.columns(), "Invalid column access index" );

   return tryMult( quatslice.operand(), quatslice.quat(), k, i, j, value );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Predict invariant violations by scaling a range of elements of a quatslice.
// \ingroup quatslice
//
// \param quatslice The target quatslice.
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
template< typename AT     // Type of the quaternion
        , size_t... CRAs  // Compile time quatslice arguments
        , typename ET >  // Type of the element
BLAZE_ALWAYS_INLINE bool
   tryMult( const QuatSlice<AT,CRAs...>& quatslice, size_t page, size_t row, size_t col, size_t pages, size_t rows, size_t cols, const ET& value )
{
   BLAZE_INTERNAL_ASSERT( page <= (~quatslice).pages(), "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( page + pages <= (~quatslice).pages(), "Invalid rows range size" );
   BLAZE_INTERNAL_ASSERT( row <= (~quatslice).rows(), "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( row + rows <= (~quatslice).rows(), "Invalid rows range size" );
   BLAZE_INTERNAL_ASSERT( col <= (~quatslice).columns(), "Invalid column access index" );
   BLAZE_INTERNAL_ASSERT( col + cols <= (~quatslice).columns(), "Invalid columns range size" );

   return tryMult( quatslice.operand(), quatslice.quat(), page, row, col, 1UL, pages, rows, cols, value );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Predict invariant violations by scaling a single element of a quatslice.
// \ingroup quatslice
//
// \param quatslice The target quatslice.
// \param index The index of the element to be modified.
// \param value The divisor for the element.
// \return \a true in case the operation would be successful, \a false if not.
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename AT     // Type of the quaternion
        , size_t... CRAs  // Compile time quatslice arguments
        , typename ET >   // Type of the element
inline bool tryDiv( const QuatSlice<AT,CRAs...>& quatslice, size_t k, size_t i, size_t j, const ET& value )
{
   BLAZE_INTERNAL_ASSERT( k < quatslice.pages(), "Invalid page access index" );
   BLAZE_INTERNAL_ASSERT( i < quatslice.rows(), "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( j < quatslice.columns(), "Invalid column access index" );

   return tryDiv( quatslice.operand(), quatslice.quat(), k, i, j, value );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Predict invariant violations by scaling a range of elements of a quatslice.
// \ingroup quatslice
//
// \param quatslice The target quatslice.
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
template< typename AT     // Type of the quaternion
        , size_t... CRAs  // Compile time quatslice arguments
        , typename ET >  // Type of the element
BLAZE_ALWAYS_INLINE bool
   tryDiv( const QuatSlice<AT,CRAs...>& quatslice, size_t page, size_t row, size_t col, size_t pages, size_t rows, size_t cols, const ET& value )
{
   BLAZE_INTERNAL_ASSERT( page <= (~quatslice).pages(), "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( page + pages <= (~quatslice).pages(), "Invalid rows range size" );
   BLAZE_INTERNAL_ASSERT( row <= (~quatslice).rows(), "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( row + rows <= (~quatslice).rows(), "Invalid rows range size" );
   BLAZE_INTERNAL_ASSERT( col <= (~quatslice).columns(), "Invalid column access index" );
   BLAZE_INTERNAL_ASSERT( col + cols <= (~quatslice).columns(), "Invalid columns range size" );

   return tryDiv( quatslice.operand(), quatslice.quat(), page, row, col, 1UL, pages, rows, cols, value );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Predict invariant violations by the assignment of a matrix to a quatslice.
// \ingroup quatslice
//
// \param lhs The target left-hand side quatslice.
// \param rhs The right-hand side vector to be assigned.
// \param index The index of the first element to be modified.
// \return \a true in case the assignment would be successful, \a false if not.
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename AT     // Type of the quaternion
        , size_t... CRAs  // Compile time quatslice arguments
        , typename TT >   // Type of the right-hand side tensor
inline bool tryAssign( const QuatSlice<AT,CRAs...>& lhs,
                       const Tensor<TT>& rhs, size_t k, size_t i, size_t j )
{
   BLAZE_INTERNAL_ASSERT( k <= lhs.pages(), "Invalid page access index" );
   BLAZE_INTERNAL_ASSERT( k + (~rhs).pages() <= lhs.pages(), "Invalid page range size" );
   BLAZE_INTERNAL_ASSERT( i <= lhs.rows(), "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( i + (~rhs).rows() <= lhs.rows(), "Invalid rows range size" );
   BLAZE_INTERNAL_ASSERT( j <= lhs.columns(), "Invalid column access index" );
   BLAZE_INTERNAL_ASSERT( j + (~rhs).columns() <= lhs.columns(), "Invalid columns range size" );

   return tryAssign( lhs.operand(), ~rhs, lhs.quat(), k, i, j  );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Predict invariant violations by the addition assignment of a vector to a quatslice.
// \ingroup quatslice
//
// \param lhs The target left-hand side quatslice.
// \param rhs The right-hand side vector to be added.
// \param index The index of the first element to be modified.
// \return \a true in case the assignment would be successful, \a false if not.
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename AT     // Type of the quaternion
        , size_t... CRAs  // Compile time quatslice arguments
        , typename TT >   // Type of the right-hand side matrix
inline bool tryAddAssign( const QuatSlice<AT,CRAs...>& lhs,
                          const Tensor<TT>& rhs, size_t k, size_t i, size_t j )
{
   BLAZE_INTERNAL_ASSERT( k <= lhs.pages(), "Invalid page access index" );
   BLAZE_INTERNAL_ASSERT( k + (~rhs).pages() <= lhs.pages(), "Invalid page range size" );
   BLAZE_INTERNAL_ASSERT( i <= lhs.rows(), "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( i + (~rhs).rows() <= lhs.rows(), "Invalid rows range size" );
   BLAZE_INTERNAL_ASSERT( j <= lhs.columns(), "Invalid column access index" );
   BLAZE_INTERNAL_ASSERT( j + (~rhs).columns() <= lhs.columns(), "Invalid columns range size" );

   return tryAddAssign( lhs.operand(), ~rhs, lhs.quat(), k, i, j );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Predict invariant violations by the subtraction assignment of a vector to a quatslice.
// \ingroup quatslice
//
// \param lhs The target left-hand side quatslice.
// \param rhs The right-hand side vector to be subtracted.
// \param index The index of the first element to be modified.
// \return \a true in case the assignment would be successful, \a false if not.
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename AT     // Type of the quaternion
        , size_t... CRAs  // Compile time quatslice arguments
        , typename TT >   // Type of the right-hand side matrix
inline bool trySubAssign( const QuatSlice<AT,CRAs...>& lhs,
                          const Tensor<TT>& rhs, size_t k, size_t i, size_t j )
{
   BLAZE_INTERNAL_ASSERT( k <= lhs.pages(), "Invalid page access index" );
   BLAZE_INTERNAL_ASSERT( k + (~rhs).pages() <= lhs.pages(), "Invalid page range size" );
   BLAZE_INTERNAL_ASSERT( i <= lhs.rows(), "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( i + (~rhs).rows() <= lhs.rows(), "Invalid rows range size" );
   BLAZE_INTERNAL_ASSERT( j <= lhs.columns(), "Invalid column access index" );
   BLAZE_INTERNAL_ASSERT( j + (~rhs).columns() <= lhs.columns(), "Invalid columns range size" );

   return trySubAssign( lhs.operand(), ~rhs, lhs.quat(), k, i, j );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Predict invariant violations by the multiplication assignment of a vector to a quatslice.
// \ingroup quatslice
//
// \param lhs The target left-hand side quatslice.
// \param rhs The right-hand side vector to be multiplied.
// \param index The index of the first element to be modified.
// \return \a true in case the assignment would be successful, \a false if not.
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename AT     // Type of the quaternion
        , size_t... CRAs  // Compile time quatslice arguments
        , typename VT >   // Type of the right-hand side matrix
inline bool tryMultAssign( const QuatSlice<AT,CRAs...>& lhs,
                           const Vector<VT,true>& rhs, size_t k, size_t i, size_t j )
{
   BLAZE_INTERNAL_ASSERT( k <= lhs.pages(), "Invalid page access index" );
   BLAZE_INTERNAL_ASSERT( k + (~rhs).pages() <= lhs.pages(), "Invalid page range size" );
   BLAZE_INTERNAL_ASSERT( i <= lhs.rows(), "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( i + (~rhs).rows() <= lhs.rows(), "Invalid rows range size" );
   BLAZE_INTERNAL_ASSERT( j <= lhs.columns(), "Invalid column access index" );
   BLAZE_INTERNAL_ASSERT( j + (~rhs).columns() <= lhs.columns(), "Invalid columns range size" );

   return tryMultAssign( lhs.operand(), ~rhs, lhs.quat(), k, i, j );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Predict invariant violations by the division assignment of a vector to a quatslice.
// \ingroup quatslice
//
// \param lhs The target left-hand side quatslice.
// \param rhs The right-hand side vector divisor.
// \param index The index of the first element to be modified.
// \return \a true in case the assignment would be successful, \a false if not.
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename AT     // Type of the quaternion
        , size_t... CRAs  // Compile time quatslice arguments
        , typename TT >   // Type of the right-hand side matrix
inline bool tryDivAssign( const QuatSlice<AT,CRAs...>& lhs,
                          const Tensor<TT>& rhs, size_t k, size_t i, size_t j )
{
   BLAZE_INTERNAL_ASSERT( k <= lhs.pages(), "Invalid page access index" );
   BLAZE_INTERNAL_ASSERT( k + (~rhs).pages() <= lhs.pages(), "Invalid page range size" );
   BLAZE_INTERNAL_ASSERT( i <= lhs.rows(), "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( i + (~rhs).rows() <= lhs.rows(), "Invalid rows range size" );
   BLAZE_INTERNAL_ASSERT( j <= lhs.columns(), "Invalid column access index" );
   BLAZE_INTERNAL_ASSERT( j + (~rhs).columns() <= lhs.columns(), "Invalid columns range size" );

   return tryDivAssign( lhs.operand(), ~rhs, lhs.quat(), k, i, j  );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Removal of all restrictions on the data access to the given quatslice.
// \ingroup quatslice
//
// \param r The quatslice to be derestricted.
// \return QuatSlice without access restrictions.
//
// This function removes all restrictions on the data access to the given quatslice. It returns a quatslice
// object that does provide the same interface but does not have any restrictions on the data
// access.\n
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in the violation of invariants, erroneous results and/or in compilation errors.
*/
template< typename AT  // Type of the quaternion
        , size_t I >   // QuatSlice index
inline decltype(auto) derestrict( QuatSlice<AT,I>& r )
{
   return quatslice<I>( derestrict( r.operand() ), unchecked );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Removal of all restrictions on the data access to the given temporary quatslice.
// \ingroup quatslice
//
// \param r The temporary quatslice to be derestricted.
// \return QuatSlice without access restrictions.
//
// This function removes all restrictions on the data access to the given temporary quatslice. It
// returns a quatslice object that does provide the same interface but does not have any restrictions
// on the data access.\n
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in the violation of invariants, erroneous results and/or in compilation errors.
*/
template< typename AT  // Type of the quaternion
        , size_t I >   // QuatSlice index
inline decltype(auto) derestrict( QuatSlice<AT,I>&& r )
{
   return quatslice<I>( derestrict( r.operand() ), unchecked );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Removal of all restrictions on the data access to the given quatslice.
// \ingroup quatslice
//
// \param r The quatslice to be derestricted.
// \return QuatSlice without access restrictions.
//
// This function removes all restrictions on the data access to the given quatslice. It returns a quatslice
// object that does provide the same interface but does not have any restrictions on the data
// access.\n
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in the violation of invariants, erroneous results and/or in compilation errors.
*/
template< typename AT > // Type of the quaternion
inline decltype(auto) derestrict( QuatSlice<AT>& r )
{
   return quatslice( derestrict( r.operand() ), r.quat(), unchecked );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Removal of all restrictions on the data access to the given temporary quatslice.
// \ingroup quatslice
//
// \param r The temporary quatslice to be derestricted.
// \return QuatSlice without access restrictions.
//
// This function removes all restrictions on the data access to the given temporary quatslice. It
// returns a quatslice object that does provide the same interface but does not have any restrictions
// on the data access.\n
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in the violation of invariants, erroneous results and/or in compilation errors.
*/
template< typename AT > // Type of the quaternion
inline decltype(auto) derestrict( QuatSlice<AT>&& r )
{
   return quatslice( derestrict( r.operand() ), r.quat(), unchecked );
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
template< typename AT, size_t... CRAs >
struct Size< QuatSlice<AT,CRAs...>, 0UL >
   : public Size<AT,1UL>
{};

template< typename AT, size_t... CRAs >
struct Size< QuatSlice<AT,CRAs...>, 1UL >
   : public Size<AT,2UL>
{};

template< typename AT, size_t... CRAs >
struct Size< QuatSlice<AT,CRAs...>, 2UL >
   : public Size<AT,3UL>
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
template< typename AT, size_t... CRAs >
struct MaxSize< QuatSlice<AT,CRAs...>, 0UL >
   : public MaxSize<AT,1UL>
{};

template< typename AT, size_t... CRAs >
struct MaxSize< QuatSlice<AT,CRAs...>, 1UL >
   : public MaxSize<AT,2UL>
{};

template< typename AT, size_t... CRAs >
struct MaxSize< QuatSlice<AT,CRAs...>, 2UL >
   : public MaxSize<AT,3UL>
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
template< typename AT, size_t... CRAs >
struct IsRestricted< QuatSlice<AT,CRAs...> >
   : public IsRestricted<AT>
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
template< typename AT, size_t... CRAs >
struct HasConstDataAccess< QuatSlice<AT,CRAs...> >
   : public HasConstDataAccess<AT>
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
template< typename AT, size_t... CRAs >
struct HasMutableDataAccess< QuatSlice<AT,CRAs...> >
   : public HasMutableDataAccess<AT>
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
template< typename AT, size_t... CRAs >
struct IsAligned< QuatSlice<AT,CRAs...> >
   : public BoolConstant< IsAligned_v<AT> >
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
template< typename AT, size_t... CRAs >
struct IsContiguous< QuatSlice<AT,CRAs...> >
   : public IsContiguous<AT>
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
template< typename AT, size_t... CRAs >
struct IsPadded< QuatSlice<AT,CRAs...> >
   : public BoolConstant< IsPadded_v<AT> >
{};
/*! \endcond */
//*************************************************************************************************

} // namespace blaze

#endif
