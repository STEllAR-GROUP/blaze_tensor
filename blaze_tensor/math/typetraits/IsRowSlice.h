//=================================================================================================
/*!
//  \file blaze_tensor/math/typetraits/IsRowSlice.h
//  \brief Header file for the IsRowSlice type trait
//
//  Copyright (C) 2012-2018 Klaus Iglberger - All Rights Reserved
//  Copyright (C) 2018 Hartmut Kaiser - All Rights Reserved
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

#ifndef _BLAZE_TENSOR_MATH_TYPETRAITS_ISROWSLICE_H_
#define _BLAZE_TENSOR_MATH_TYPETRAITS_ISROWSLICE_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <blaze/util/FalseType.h>
#include <blaze/util/TrueType.h>

#include <blaze_tensor/math/views/Forward.h>

namespace blaze {

//=================================================================================================
//
//  CLASS DEFINITION
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Compile time check for rowslices.
// \ingroup math_type_traits
//
// This type trait tests whether or not the given template parameter is a rowslice (i.e. a view on a
// rowslice of a dense or sparse matrix). In case the type is a rowslice, the \a value member constant is
// set to \a true, the nested type definition \a Type is \a TrueType, and the class derives from
// \a TrueType. Otherwise \a value is set to \a false, \a Type is \a FalseType, and the class
// derives from \a FalseType.

   \code
   using blaze::aligned;

   using MatrixType1 = blaze::StaticMatrix<int,10UL,16UL>;
   using MatrixType2 = blaze::DynamicMatrix<double>;
   using MatrixType3 = blaze::CompressedMatrix<float>;

   MatrixType1 A;
   MatrixType2 B( 100UL, 200UL );
   MatrixType3 C( 200UL, 250UL );

   using RowSliceType1 = decltype( blaze::rowslice<4UL>( A ) );
   using RowSliceType2 = decltype( blaze::rowslice( B, 16UL ) );
   using RowSliceType3 = decltype( blaze::rowslice( C, 17UL ) );

   blaze::IsRowSlice< RowSliceType1 >::value          // Evaluates to 1
   blaze::IsRowSlice< const RowSliceType2 >::Type     // Results in TrueType
   blaze::IsRowSlice< volatile RowSliceType3 >        // Is derived from TrueType
   blaze::IsRowSlice< MatrixType1 >::value       // Evaluates to 0
   blaze::IsRowSlice< const MatrixType2 >::Type  // Results in FalseType
   blaze::IsRowSlice< volatile MatrixType3 >     // Is derived from FalseType
   \endcode
*/
template< typename T >
struct IsRowSlice
   : public FalseType
{};
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Specialization of the IsRowSlice type trait for 'RowSlice'.
// \ingroup math_type_traits
*/
template< typename MT, size_t... CRAs >
struct IsRowSlice< RowSlice<MT,CRAs...> >
   : public TrueType
{};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Specialization of the IsRowSlice type trait for 'const RowSlice'.
// \ingroup math_type_traits
*/
template< typename MT, size_t... CRAs >
struct IsRowSlice< const RowSlice<MT,CRAs...> >
   : public TrueType
{};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Specialization of the IsRowSlice type trait for 'volatile RowSlice'.
// \ingroup math_type_traits
*/
template< typename MT, size_t... CRAs >
struct IsRowSlice< volatile RowSlice<MT,CRAs...> >
   : public TrueType
{};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Specialization of the IsRowSlice type trait for 'const volatile RowSlice'.
// \ingroup math_type_traits
*/
template< typename MT, size_t... CRAs >
struct IsRowSlice< const volatile RowSlice<MT,CRAs...> >
   : public TrueType
{};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Auxiliary variable template for the IsRowSlice type trait.
// \ingroup type_traits
//
// The IsRowSlice_v variable template provides a convenient shortcut to access the nested \a value
// of the IsRowSlice class template. For instance, given the type \a T the following two statements
// are identical:

   \code
   constexpr bool value1 = blaze::IsRowSlice<T>::value;
   constexpr bool value2 = blaze::IsRowSlice_v<T>;
   \endcode
*/
template< typename T >
constexpr bool IsRowSlice_v = IsRowSlice<T>::value;
//*************************************************************************************************

} // namespace blaze

#endif
