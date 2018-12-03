//=================================================================================================
/*!
//  \file blaze_tensor/math/typetraits/StorageOrder.h
//  \brief Header file for the StorageOrder type trait
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

#ifndef _BLAZE_TENSOR_MATH_TYPETRAITS_STORAGEORDER_H_
#define _BLAZE_TENSOR_MATH_TYPETRAITS_STORAGEORDER_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <utility>
#include <blaze/math/typetraits/StorageOrder.h>
#include <blaze/util/IntegralConstant.h>
#include <blaze/util/typetraits/RemoveCV.h>

#include <blaze_tensor/math/expressions/Tensor.h>

namespace blaze {

//=================================================================================================
//
//  CLASS DEFINITION
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Auxiliary helper struct for the StorageOrder type trait.
// \ingroup math_type_traits
*/
template< typename T >
struct TensorStorageOrderHelper
{
 private:
   //**********************************************************************************************
   template< typename MT >
   static BoolConstant<rowMajor> test( const Tensor<MT>& );
   //**********************************************************************************************

 public:
   //**********************************************************************************************
   using Type = decltype( test( std::declval< RemoveCV_t<T> >() ) );
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Evaluation of the storage order of a given tensor type.
// \ingroup math_type_traits
//
// Via this type trait it is possible to evaluate the storage order of a given tensor type.
// In case the given type is a row-major tensor type the nested boolean \a value is set to
// \a rowMajor, in case it is a column-major tensor type it is set to \a columnMajor. If the
// given type is not a tensor type a compilation error is created.

   \code
   using RowMajorTensor    = blaze::DynamicTensor<int,blaze::rowMajor>;
   using ColumnMajorTensor = blaze::DynamicTensor<int,blaze::columnMajor>;

   blaze::TensorStorageOrder<RowMajorTensor>::value     // Evaluates to blaze::rowMajor
   blaze::TensorStorageOrder<ColumnMajorTensor>::value  // Evaluates to blaze::columnMajor
   blaze::TensorStorageOrder<int>::value                // Compilation error!
   \endcode
*/
template< typename T >
struct TensorStorageOrder
   : public TensorStorageOrderHelper< T >::Type
{};
//*************************************************************************************************

//*************************************************************************************************
/*!\brief Auxiliary variable template for the TensorStorageOrder type trait.
// \ingroup type_traits
//
// The StorageOrder_v variable template provides a convenient shortcut to access the nested
// \a value of the StorageOrder class template. For instance, given the matrix type \a T the
// following two statements are identical:

   \code
   constexpr bool value1 = blaze::TensorStorageOrder<T>::value;
   constexpr bool value2 = blaze::TensorStorageOrder_v<T>;
   \endcode
*/
template< typename T >
constexpr bool TensorStorageOrder_v = TensorStorageOrder<T>::value;
//*************************************************************************************************

} // namespace blaze

#endif
