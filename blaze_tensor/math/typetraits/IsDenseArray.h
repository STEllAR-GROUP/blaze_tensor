//=================================================================================================
/*!
//  \file blaze_tensor/math/typetraits/IsDenseArray.h
//  \brief Header file for the IsDenseArray type trait
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

#ifndef _BLAZE_TENSOR_MATH_TYPETRAITS_ISDENSEARRAY_H_
#define _BLAZE_TENSOR_MATH_TYPETRAITS_ISDENSEARRAY_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <utility>
#include <blaze/util/IntegralConstant.h>

#include <blaze_tensor/math/expressions/DenseArray.h>

namespace blaze {

//=================================================================================================
//
//  CLASS DEFINITION
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Auxiliary helper struct for the IsDenseArray type trait.
// \ingroup math_type_traits
*/
template< typename T >
struct IsDenseArrayHelper
{
 private:
   //**********************************************************************************************
   template< typename MT >
   static TrueType test( const DenseArray<MT>& );

   template< typename MT >
   static TrueType test( const volatile DenseArray<MT>& );

   static FalseType test( ... );
   //**********************************************************************************************

 public:
   //**********************************************************************************************
   using Type = decltype( test( std::declval<T&>() ) );
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Compile time check for dense array types.
// \ingroup math_type_traits
//
// This type trait tests whether or not the given template parameter is a dense, N-dimensional
// array type. In case the type is a dense array type, the \a value member constant is set
// to \a true, the nested type definition \a Type is \a TrueType, and the class derives from
// \a TrueType. Otherwise \a yes is set to \a false, \a Type is \a FalseType, and the class
// derives from \a FalseType.

   \code
   blaze::IsDenseArray< DynamicArray<double> >::value        // Evaluates to 1
   blaze::IsDenseArray< const DynamicArray<float> >::Type    // Results in TrueType
   blaze::IsDenseArray< volatile DynamicArray<int> >         // Is derived from TrueType
   blaze::IsDenseArray< CompressedMatrix<double,false>::value    // Evaluates to 0
   blaze::IsDenseArray< CompressedVector<double,true> >::Type    // Results in FalseType
   blaze::IsDenseArray< DynamicVector<double,true> >             // Is derived from FalseType
   \endcode
*/
template< typename T >
struct IsDenseArray
   : public IsDenseArrayHelper<T>::Type
{};
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Auxiliary variable template for the IsDenseArray type trait.
// \ingroup type_traits
//
// The IsDenseArray_v variable template provides a convenient shortcut to access the nested
// \a value of the IsDenseArray class template. For instance, given the type \a T the
// following two statements are identical:

   \code
   constexpr bool value1 = blaze::IsDenseArray<T>::value;
   constexpr bool value2 = blaze::IsDenseArray_v<T>;
   \endcode
*/
template< typename T >
constexpr bool IsDenseArray_v = IsDenseArray<T>::value;
//*************************************************************************************************

} // namespace blaze

#endif
