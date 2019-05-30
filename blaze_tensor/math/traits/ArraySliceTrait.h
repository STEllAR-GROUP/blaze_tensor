//=================================================================================================
/*!
//  \file blaze_array/math/traits/ArraySliceTrait.h
//  \brief Header file for the arrayslice trait
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

#ifndef _BLAZE_TENSOR_MATH_TRAITS_ARRAYSLICETRAIT_H_
#define _BLAZE_TENSOR_MATH_TRAITS_ARRAYSLICETRAIT_H_


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
template< size_t, typename, size_t... > struct ArraySliceTrait;
template< size_t, typename, size_t, typename = void > struct ArraySliceTraitEval1;
template< size_t, typename, size_t, typename = void > struct ArraySliceTraitEval2;
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< size_t M, size_t I, typename T >
auto evalArraySliceTrait( T& )
   -> typename ArraySliceTraitEval1<M,T,I>::Type;

template< size_t M, typename T >
auto evalArraySliceTrait( T& )
   -> typename ArraySliceTraitEval2<M,T,inf>::Type;

template< size_t M, size_t I, typename T >
auto evalArraySliceTrait( const T& )
   -> typename ArraySliceTrait<M,T,I>::Type;

template< size_t M, typename T >
auto evalArraySliceTrait( const T& )
   -> typename ArraySliceTrait<M,T>::Type;

template< size_t M, size_t I, typename T >
auto evalArraySliceTrait( const volatile T& )
   -> typename ArraySliceTrait<M,T,I>::Type;

template< size_t M, typename T >
auto evalArraySliceTrait( const volatile T& )
   -> typename ArraySliceTrait<M,T>::Type;
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Base template for the ArraySliceTrait class.
// \ingroup math_traits
//
// \section pagetrait_general General
//
// The ArraySliceTrait class template offers the possibility to select the resulting data type when
// creating a view on a specific page of a dense or sparse array. ArraySliceTrait defines the nested
// type \a Type, which represents the resulting data type of the page operation. In case the
// given data type is not a dense or sparse array type, the resulting data type \a Type is
// set to \a INVALID_TYPE. Note that \a const and \a volatile qualifiers and reference modifiers
// are generally ignored.
//
//
// \section pagetrait_specializations Creating custom specializations
//
// Per default, ArraySliceTrait supports all array types of the Blaze library (including views and
// adaptors). For all other data types it is possible to specialize the ArraySliceTrait template. The
// following example shows the according specialization for the DynamicArray class template:

   \code
   template< size_t M, typename T1, size_t... CRAs >
   struct ArraySliceTrait< M, DynamicArray<M,T1>, CRAs... >
   {
      using Type = DynamicArray<M-1,T1>;
   };
   \endcode

// \n \section arrayslicetrait_examples Examples
//
// The following example demonstrates the use of the ArraySliceTrait template, where depending on
// the given array type the resulting page type is selected:

   \code
   // Definition of the page type of a dynamic array
   using ArrayType1 = blaze::DynamicArray<3,int>;
   using ResultType1 = typename blaze::ArraySliceTrait<3,ArrayType1>::Type;
   \endcode
*/
template< size_t M          // ArraySlice dimension
        , typename MT       // Type of the array
        , size_t... CRAs >  // Compile time page arguments
struct ArraySliceTrait
{
 public:
   //**********************************************************************************************
   /*! \cond BLAZE_INTERNAL */
   using Type = decltype( evalArraySliceTrait<M,CRAs...>( std::declval<MT&>() ) );
   /*! \endcond */
   //**********************************************************************************************
};
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Auxiliary alias declaration for the ArraySliceTrait type trait.
// \ingroup math_traits
//
// The ArraySliceTrait_t alias declaration provides a convenient shortcut to access the nested
// \a Type of the ArraySliceTrait class template. For instance, given the array type \a MT the
// following two type definitions are identical:

   \code
   using Type1 = typename blaze::ArraySliceTrait<MT>::Type;
   using Type2 = blaze::ArraySliceTrait_t<MT>;
   \endcode
*/
template< size_t M          // ArraySlice dimension
        , typename MT       // Type of the array
        , size_t... CRAs >  // Compile time page arguments
using ArraySliceTrait_t = typename ArraySliceTrait<M,MT,CRAs...>::Type;
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief First auxiliary helper struct for the ArraySliceTrait type trait.
// \ingroup math_traits
*/
template< size_t M          // ArraySlice dimension
        , typename MT  // Type of the array
        , size_t I     // Compile time dimensional index
        , typename >   // Restricting condition
struct ArraySliceTraitEval1
{
 public:
   //**********************************************************************************************
   using Type = typename ArraySliceTraitEval2<M,MT,I>::Type;
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Second auxiliary helper struct for the ArraySliceTrait type trait.
// \ingroup math_traits
*/
template< size_t M     // ArraySlice dimension
        , typename MT  // Type of the array
        , size_t I     // Compile time dimensional index
        , typename >   // Restricting condition
struct ArraySliceTraitEval2
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
