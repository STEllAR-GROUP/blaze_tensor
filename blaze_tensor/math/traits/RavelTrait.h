//=================================================================================================
/*!
//  \file blaze_tensor/math/traits/RavelTrait.h
//  \brief Header file for the ravel trait
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

#ifndef _BLAZE_TENSOR_MATH_TRAITS_RAVELTRAIT_H_
#define _BLAZE_TENSOR_MATH_TRAITS_RAVELTRAIT_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <utility>
#include <blaze/math/Infinity.h>
#include <blaze/util/InvalidType.h>
#include <blaze/util/Types.h>


namespace blaze {

//=================================================================================================
//
//  CLASS DEFINITION
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename > struct RavelTrait;
template< typename, typename = void > struct RavelTraitEval1;
template< typename, typename = void > struct RavelTraitEval2;
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename T >
auto evalRavelTrait( T& )
   -> typename RavelTraitEval1<T>::Type;

template< typename T >
auto evalRavelTrait( const T& )
   -> typename RavelTrait<T>::Type;

template< typename T >
auto evalRavelTrait( const volatile T& )
   -> typename RavelTrait<T>::Type;
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Base template for the RavelTrait class.
// \ingroup math_traits
//
// \section raveltrait_general General
//
// The RavelTrait class template offers the possibility to select the resulting data type when
// raveling a dense or sparse vector or matrix. RavelTrait defines the nested type \a Type,
// which represents the resulting data type of the ravel operation. In case the given data type
// is not a dense or sparse vector or matrix type, the resulting data type \a Type is set to
// \a INVALID_TYPE. Note that \a const and \a volatile qualifiers and reference modifiers are
// generally ignored.
//
//
// \section raveltrait_specializations Creating custom specializations
//
// Per default, RavelTrait supports all vector and matrix types of the Blaze library (including
// views and adaptors). For all other data types it is possible to specialize the RavelTrait
// template. The following example shows the according specialization for the DynamicVector class
// template:

   \code
   template< typename Type, bool SO, size_t... CEAs >
   struct RavelTrait< DynamicMatrix<Type,SO>, CEAs... >
   {
      using Type = DynamicVector<Type,( SO == rowMajor ? rowVector : columnVector )>;
   };
   \endcode

// \n \section raveltrait_examples Examples
//
// The following example demonstrates the use of the RavelTrait template, where depending on
// the given vector or matrix type the resulting type is selected:

   \code
   using blaze::columnMajor;
   using blaze::rowMajor;

   // Definition of the resulting type of a dynamic columnMajor matrix
   using MatrixType1 = blaze::DynamicMatrix<int,columnMajor>;
   using ResultType1 = typename blaze::RavelTrait<MatrixType1>::Type;

   // Definition of the resulting type of a static rowMajor matrix
   using MatrixType2 = blaze::StaticMatrix<int,5UL,rowMajor>;
   using ResultType2 = typename blaze::RavelTrait<MatrixType2>::Type;
   \endcode
*/
template< typename T >      // Type of the operand
struct RavelTrait
{
 public:
   //**********************************************************************************************
   /*! \cond BLAZE_INTERNAL */
   using Type = decltype( evalRavelTrait( std::declval<T&>() ) );
   /*! \endcond */
   //**********************************************************************************************
};
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Auxiliary alias declaration for the RavelTrait type trait.
// \ingroup math_traits
//
// The RavelTrait_t alias declaration provides a convenient shortcut to access the nested
// \a Type of the RavelTrait class template. For instance, given the matrix type \a MT the
// following two type definitions are identical:

   \code
   using Type1 = typename blaze::RavelTrait<MT>::Type;
   using Type2 = blaze::RavelTrait_t<MT>;
   \endcode
*/
template< typename T >      // Type of the operand
using RavelTrait_t = typename RavelTrait<T>::Type;
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief First auxiliary helper struct for the RavelTrait type trait.
// \ingroup math_traits
*/
template< typename T   // Type of the operand
        , typename >   // Restricting condition
struct RavelTraitEval1
{
 public:
   //**********************************************************************************************
   using Type = typename RavelTraitEval2<T>::Type;
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Second auxiliary helper struct for the RavelTrait type trait.
// \ingroup math_traits
*/
template< typename MT  // Type of the operand
        , typename >   // Restricting condition
struct RavelTraitEval2
{
 public:
   //**********************************************************************************************
   using Type = INVALID_TYPE;
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************

} // namespace blaze

#endif
