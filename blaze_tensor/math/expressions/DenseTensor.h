//=================================================================================================
/*!
//  \file blaze_tensor/math/expressions/DenseTensor.h
//  \brief Header file for the DenseTensor base class
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

#ifndef _BLAZE_TENSOR_MATH_EXPRESSIONS_DENSETENSOR_H_
#define _BLAZE_TENSOR_MATH_EXPRESSIONS_DENSETENSOR_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <blaze/math/typetraits/HasConstDataAccess.h>
#include <blaze/math/typetraits/HasMutableDataAccess.h>
#include <blaze/system/Inline.h>
#include <blaze/util/EnableIf.h>
#include <blaze/util/MaybeUnused.h>

#include <blaze_tensor/math/expressions/Tensor.h>


namespace blaze {

//=================================================================================================
//
//  CLASS DEFINITION
//
//=================================================================================================

//*************************************************************************************************
/*!\defgroup dense_tensor Dense Tensors
// \ingroup tensor
*/
/*!\defgroup dense_tensor_expression Expressions
// \ingroup dense_tensor
*/
/*!\brief Base class for dense matrices.
// \ingroup dense_tensor
//
// The DenseTensor class is a base class for all dense tensor classes. It provides an
// abstraction from the actual type of the dense tensor, but enables a conversion back
// to this type via the Tensor base class.
*/
template< typename TT > // Type of the dense tensor
struct DenseTensor
   : public Tensor<TT>
{};
//*************************************************************************************************




//=================================================================================================
//
//  GLOBAL FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*!\name DenseTensor global functions */
//@{
template< typename TT >
BLAZE_ALWAYS_INLINE typename TT::ElementType* data( DenseTensor<TT>& dm ) noexcept;

template< typename TT >
BLAZE_ALWAYS_INLINE typename TT::ElementType* data( const DenseTensor<TT>& dm ) noexcept;

template< typename TT >
BLAZE_ALWAYS_INLINE size_t spacing( const DenseTensor<TT>& dm ) noexcept;
//@}
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Backend implementation of the \c data() function for matrices without mutable data access.
// \ingroup dense_tensor
//
// \param dm The given dense tensor.
// \return Pointer to the internal element storage.
//
// This function returns the internal storage of a dense tensor without mutable data access,
// which is represented by \c nullptr.
*/
template< typename TT > // Type of the tensor
BLAZE_ALWAYS_INLINE EnableIf_t< !HasMutableDataAccess_v<TT>, typename TT::ElementType* >
   data_backend( DenseTensor<TT>& dm ) noexcept
{
   MAYBE_UNUSED( dm );

   return nullptr;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Backend implementation of the \c data() function for matrices with mutable data access.
// \ingroup dense_tensor
//
// \param dm The given dense tensor.
// \return Pointer to the internal element storage.
//
// This function returns the internal storage of a dense tensor with mutable data access.
*/
template< typename TT > // Type of the tensor
BLAZE_ALWAYS_INLINE EnableIf_t< HasMutableDataAccess_v<TT>, typename TT::ElementType* >
   data_backend( DenseTensor<TT>& dm ) noexcept
{
   return (~dm).data();
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Low-level data access to the dense tensor elements.
// \ingroup dense_tensor
//
// \param dm The given dense tensor.
// \return Pointer to the internal element storage.
//
// This function provides a unified interface to access the given dense tensor's internal
// element storage. In contrast to the \c data() member function, which is only available
// in case the tensor has some internal storage, this function can be used on all kinds of
// dense matrices. In case the given dense tensor does not provide low-level data access,
// the function returns \c nullptr.
*/
template< typename TT > // Type of the tensor
BLAZE_ALWAYS_INLINE typename TT::ElementType* data( DenseTensor<TT>& dm ) noexcept
{
   return data_backend( ~dm );
}
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Backend implementation of the \c data() function for matrices without constant data access.
// \ingroup dense_tensor
//
// \param dm The given dense tensor.
// \return Pointer to the internal element storage.
//
// This function returns the internal storage of a dense tensor without constant data access,
// which is represented by \c nullptr.
*/
template< typename TT > // Type of the tensor
BLAZE_ALWAYS_INLINE EnableIf_t< !HasConstDataAccess_v<TT>, typename TT::ElementType* >
   data_backend( const DenseTensor<TT>& dm ) noexcept
{
   MAYBE_UNUSED( dm );

   return nullptr;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Backend implementation of the \c data() function for matrices with constant data access.
// \ingroup dense_tensor
//
// \param dm The given dense tensor.
// \return Pointer to the internal element storage.
//
// This function returns the internal storage of a dense tensor with constant data access.
*/
template< typename TT > // Type of the tensor
BLAZE_ALWAYS_INLINE EnableIf_t< HasConstDataAccess_v<TT>, typename TT::ElementType* >
   data_backend( const DenseTensor<TT>& dm ) noexcept
{
   return (~dm).data();
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Low-level data access to the dense tensor elements.
// \ingroup dense_tensor
//
// \param dm The given dense tensor.
// \return Pointer to the internal element storage.
//
// This function provides a unified interface to access the given dense tensor's internal
// element storage. In contrast to the \c data() member function, which is only available
// in case the tensor has some internal storage, this function can be used on all kinds of
// dense matrices. In case the given dense tensor does not provide low-level data access,
// the function returns \c nullptr.
*/
template< typename TT > // Type of the tensor
BLAZE_ALWAYS_INLINE typename TT::ElementType* data( const DenseTensor<TT>& dm ) noexcept
{
   return data_backend( ~dm );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns the spacing between the beginning of two rows/columns.
// \ingroup dense_tensor
//
// \param dm The given tensor.
// \return The spacing between the beginning of two rows/columns.
*/
template< typename TT > // Type of the tensor
BLAZE_ALWAYS_INLINE size_t spacing( const DenseTensor<TT>& dm ) noexcept
{
   return (~dm).spacing();
}
//*************************************************************************************************

} // namespace blaze

#endif
