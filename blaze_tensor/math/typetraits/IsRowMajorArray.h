//=================================================================================================
/*!
//  \file blaze/math/typetraits/IsRowMajorArray.h
//  \brief Header file for the IsRowMajorArray type trait
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

#ifndef _BLAZE_TENSOR_MATH_TYPETRAITS_ISROWMAJORARRAY_H_
#define _BLAZE_TENSOR_MATH_TYPETRAITS_ISROWMAJORARRAY_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <utility>
#include <blaze/util/IntegralConstant.h>

#include <blaze_tensor/math/expressions/Array.h>
#include <blaze_tensor/math/StorageOrder.h>

namespace blaze {

//=================================================================================================
//
//  CLASS DEFINITION
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Auxiliary helper struct for the IsRowMajorArray type trait.
// \ingroup math_type_traits
*/
template< typename T >
struct IsRowMajorArrayHelper
{
 private:
   //**********************************************************************************************
   template< typename MT >
   static TrueType test( const Array<MT>& );

   template< typename MT >
   static TrueType test( const volatile Array<MT>& );

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
/*!\brief Compile time check for row-major matrix types.
// \ingroup math_type_traits
//
// This type trait tests whether or not the given template argument is a row-major dense or
// sparse matrix type (i.e., a matrix whose storage order is set to \a true). In case the type
// is a row-major matrix type, the \a value member constant is set to \a true, the nested type
// definition \a Type is \a TrueType, and the class derives from \a TrueType. Otherwise \a value
// is set to \a false, \a Type is \a FalseType, and the class derives from \a FalseType.

   \code
   using blaze::StaticArray;
   using blaze::DynamicArray;
   using blaze::CompressedArray;
   using blaze::rowMajor;
   using blaze::columnMajor;

   blaze::IsRowMajorArray< StaticArray<float,3U,3U,rowMajor> >::value      // Evaluates to 1
   blaze::IsRowMajorArray< const DynamicArray<double,rowMajor> >::Type     // Results in TrueType
   blaze::IsRowMajorArray< volatile CompressedArray<int,rowMajor> >        // Is derived from TrueType
   blaze::IsRowMajorArray< StaticArray<float,3U,3U,columnMajor> >::value   // Evaluates to 0
   blaze::IsRowMajorArray< const DynamicArray<double,columnMajor> >::Type  // Results in FalseType
   blaze::IsRowMajorArray< volatile CompressedArray<int,columnMajor> >     // Is derived from FalseType
   \endcode
*/
template< typename T >
struct IsRowMajorArray
   : public IsRowMajorArrayHelper<T>::Type
{};
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Auxiliary variable template for the IsRowMajorArray type trait.
// \ingroup type_traits
//
// The IsRowMajorArray_v variable template provides a convenient shortcut to access the nested
// \a value of the IsRowMajorArray class template. For instance, given the type \a T the
// following two statements are identical:

   \code
   constexpr bool value1 = blaze::IsRowMajorArray<T>::value;
   constexpr bool value2 = blaze::IsRowMajorArray_v<T>;
   \endcode
*/
template< typename T >
constexpr bool IsRowMajorArray_v = IsRowMajorArray<T>::value;
//*************************************************************************************************

} // namespace blaze

#endif
