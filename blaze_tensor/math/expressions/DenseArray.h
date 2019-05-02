//=================================================================================================
/*!
//  \file blaze_array/math/expressions/DenseArray.h
//  \brief Header file for the DenseArray base class
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

#ifndef _BLAZE_TENSOR_MATH_EXPRESSIONS_DENSEARRAY_H_
#define _BLAZE_TENSOR_MATH_EXPRESSIONS_DENSEARRAY_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <blaze/math/typetraits/HasConstDataAccess.h>
#include <blaze/math/typetraits/HasMutableDataAccess.h>
#include <blaze/system/Inline.h>
#include <blaze/util/DisableIf.h>
#include <blaze/util/EnableIf.h>
#include <blaze/util/MaybeUnused.h>

#include <blaze_tensor/math/expressions/Array.h>


namespace blaze {

//=================================================================================================
//
//  CLASS DEFINITION
//
//=================================================================================================

//*************************************************************************************************
/*!\defgroup dense_array Dense Arrays
// \ingroup array
*/
/*!\defgroup dense_array_expression Expressions
// \ingroup dense_array
*/
/*!\brief Base class for dense arrays.
// \ingroup dense_array
//
// The DenseArray class is a base class for all dense array classes. It provides an
// abstraction from the actual type of the dense array, but enables a conversion back
// to this type via the Array base class.
*/
template< typename TT > // Type of the dense array
struct DenseArray
   : public Array<TT>
{};
//*************************************************************************************************




//=================================================================================================
//
//  GLOBAL FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*!\name DenseArray global functions */
//@{
template< typename TT >
BLAZE_ALWAYS_INLINE typename TT::ElementType* data( DenseArray<TT>& dm ) noexcept;

template< typename TT >
BLAZE_ALWAYS_INLINE typename TT::ElementType* data( const DenseArray<TT>& dm ) noexcept;

template< typename TT >
BLAZE_ALWAYS_INLINE size_t spacing( const DenseArray<TT>& dm ) noexcept;
//@}
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Backend implementation of the \c data() function for arrays without mutable data access.
// \ingroup dense_array
//
// \param dm The given dense array.
// \return Pointer to the internal element storage.
//
// This function returns the internal storage of a dense array without mutable data access,
// which is represented by \c nullptr.
*/
template< typename TT > // Type of the array
BLAZE_ALWAYS_INLINE DisableIf_t< HasMutableDataAccess_v<TT>, typename TT::ElementType* >
   data_backend( DenseArray<TT>& dm ) noexcept
{
   MAYBE_UNUSED( dm );

   return nullptr;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Backend implementation of the \c data() function for arrays with mutable data access.
// \ingroup dense_array
//
// \param dm The given dense array.
// \return Pointer to the internal element storage.
//
// This function returns the internal storage of a dense array with mutable data access.
*/
template< typename TT > // Type of the array
BLAZE_ALWAYS_INLINE EnableIf_t< HasMutableDataAccess_v<TT>, typename TT::ElementType* >
   data_backend( DenseArray<TT>& dm ) noexcept
{
   return (~dm).data();
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Low-level data access to the dense array elements.
// \ingroup dense_array
//
// \param dm The given dense array.
// \return Pointer to the internal element storage.
//
// This function provides a unified interface to access the given dense array's internal
// element storage. In contrast to the \c data() member function, which is only available
// in case the array has some internal storage, this function can be used on all kinds of
// dense arrays. In case the given dense array does not provide low-level data access,
// the function returns \c nullptr.
*/
template< typename TT > // Type of the array
BLAZE_ALWAYS_INLINE typename TT::ElementType* data( DenseArray<TT>& dm ) noexcept
{
   return data_backend( ~dm );
}
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Backend implementation of the \c data() function for arrays without constant data access.
// \ingroup dense_array
//
// \param dm The given dense array.
// \return Pointer to the internal element storage.
//
// This function returns the internal storage of a dense array without constant data access,
// which is represented by \c nullptr.
*/
template< typename TT > // Type of the array
BLAZE_ALWAYS_INLINE DisableIf_t< HasConstDataAccess_v<TT>, typename TT::ElementType* >
   data_backend( const DenseArray<TT>& dm ) noexcept
{
   MAYBE_UNUSED( dm );

   return nullptr;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Backend implementation of the \c data() function for arrays with constant data access.
// \ingroup dense_array
//
// \param dm The given dense array.
// \return Pointer to the internal element storage.
//
// This function returns the internal storage of a dense array with constant data access.
*/
template< typename TT > // Type of the array
BLAZE_ALWAYS_INLINE EnableIf_t< HasConstDataAccess_v<TT>, typename TT::ElementType* >
   data_backend( const DenseArray<TT>& dm ) noexcept
{
   return (~dm).data();
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Low-level data access to the dense array elements.
// \ingroup dense_array
//
// \param dm The given dense array.
// \return Pointer to the internal element storage.
//
// This function provides a unified interface to access the given dense array's internal
// element storage. In contrast to the \c data() member function, which is only available
// in case the array has some internal storage, this function can be used on all kinds of
// dense arrays. In case the given dense array does not provide low-level data access,
// the function returns \c nullptr.
*/
template< typename TT > // Type of the array
BLAZE_ALWAYS_INLINE typename TT::ElementType* data( const DenseArray<TT>& dm ) noexcept
{
   return data_backend( ~dm );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns the spacing between the beginning of two rows/columns.
// \ingroup dense_array
//
// \param dm The given array.
// \return The spacing between the beginning of two rows/columns.
*/
template< typename TT > // Type of the array
BLAZE_ALWAYS_INLINE size_t spacing( const DenseArray<TT>& dm ) noexcept
{
   return (~dm).spacing();
}
//*************************************************************************************************

} // namespace blaze

#endif
