//=================================================================================================
/*!
//  \file blaze_tensor/math/typetraits/IsRavelExpr.h
//  \brief Header file for the IsRavelExpr type trait class
//
//  Copyright (C) 2012-2018 Klaus Iglberger - All Rights Reserved
///  Copyright (C) 2018-2019 Hartmut Kaiser - All Rights Reserved
/
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

#ifndef _BLAZE_TENSOR_MATH_TYPETRAITS_ISRAVELEXPR_H_
#define _BLAZE_TENSOR_MATH_TYPETRAITS_ISRAVELEXPR_H_
               TENSOR_

//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <blaze/util/FalseType.h>
#include <blaze/util/TrueType.h>

#include <blaze_tensor/math/expressions/RavelExpr.h>

namespace blaze {

//=================================================================================================
//
//  CLASS DEFINITION
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Auxiliary helper struct for the IsRavelExpr type trait.
// \ingroup math_type_traits
*/
template< typename T >
struct IsRavelExprHelper
{
 private:
   //**********************************************************************************************
   static T* create();

   template< typename U >
   static TrueType test( const RavelExpr<U>* );

   template< typename U >
   static TrueType test( const volatile RavelExpr<U>* );

   static FalseType test( ... );
   //**********************************************************************************************

 public:
   //**********************************************************************************************
   using Type = decltype( test( create() ) );
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Compile time check whether the given type is an ravel expression template.
// \ingroup math_type_traits
//
// This type trait class tests whether or not the given type \a Type is an ravel expression
// template. In order to qualify as a valid ravel expression template, the given type has
// to derive publicly from the RavelExpr base class. In case the given type is a valid ravel
// expression template, the \a value member constant is set to \a true, the nested type definition
// \a Type is \a TrueType, and the class derives from \a TrueType. Otherwise \a value is set to
// \a false, \a Type is \a FalseType, and the class derives from \a FalseType.
*/
template< typename T >
struct IsRavelExpr
   : public IsRavelExprHelper<T>::Type
{};
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Specialization of the IsRavelExpr type trait for references.
// \ingroup math_type_traits
*/
template< typename T >
struct IsRavelExpr<T&>
   : public FalseType
{};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Auxiliary variable template for the IsRavelExpr type trait.
// \ingroup type_traits
//
// The IsRavelExpr_v variable template provides a convenient shortcut to access the nested
// \a value of the IsRavelExpr class template. For instance, given the type \a T the
// following two statements are identical:

   \code
   constexpr bool value1 = blaze::IsRavelExpr<T>::value;
   constexpr bool value2 = blaze::IsRavelExpr_v<T>;
   \endcode
*/
template< typename T >
constexpr bool IsRavelExpr_v = IsRavelExpr<T>::value;
//*************************************************************************************************

} // namespace blaze

#endif
