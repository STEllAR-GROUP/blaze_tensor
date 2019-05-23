//=================================================================================================
/*!
//  \file blaze_tensor/math/dense/DynamicMatrix.h
//  \brief Header file for the implementation of a dynamic MxN matrix
//
//  Copyright (C) 2012-2019 Klaus Iglberger - All Rights Reserved
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

#ifndef _BLAZE_TENSOR_MATH_DENSE_DYNAMICMATRIX_H_
#define _BLAZE_TENSOR_MATH_DENSE_DYNAMICMATRIX_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <blaze/math/dense/DynamicMatrix.h>
#include <blaze/math/dense/DynamicVector.h>
#include <blaze/math/traits/ExpandTrait.h>
#include <blaze/math/typetraits/IsColumnVector.h>
#include <blaze/math/typetraits/IsDenseMatrix.h>

#include <blaze_tensor/math/dense/Forward.h>
#include <blaze_tensor/math/traits/DilatedSubmatrixTrait.h>
#include <blaze_tensor/math/traits/MultTrait.h>
#include <blaze_tensor/math/traits/RavelTrait.h>
#include <blaze_tensor/math/typetraits/IsDenseTensor.h>
#include <blaze_tensor/math/typetraits/IsTensor.h>

namespace blaze {

//=================================================================================================
//
//  EXPANDTRAIT SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename T  // Type to be expanded
        , size_t E >  // Compile time expansion
struct ExpandTraitEval2< T, E
                       , EnableIf_t< IsDenseMatrix_v<T> &&
                                     ( ( E == inf ) ||
                                       ( ( Size_v<T,0UL> == DefaultSize_v ) &&
                                         ( MaxSize_v<T,0UL> == DefaultMaxSize_v ) &&
                                         ( Size_v<T,1UL> == DefaultSize_v ) &&
                                         ( MaxSize_v<T,1UL> == DefaultMaxSize_v ) ) ) > >
{
   using Type = DynamicTensor< ElementType_t<T> >;
};
/*! \endcond */
//*************************************************************************************************




//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename T > // Type to be expanded
struct RavelTraitEval2< T
                       , EnableIf_t< IsDenseMatrix_v<T> &&
                                     ( ( ( Size_v<T,0UL> == DefaultSize_v ) &&
                                         ( MaxSize_v<T,0UL> == DefaultMaxSize_v ) &&
                                         ( Size_v<T,1UL> == DefaultSize_v ) &&
                                         ( MaxSize_v<T,1UL> == DefaultMaxSize_v ) ) ) > >
{
   using Type = DynamicVector< ElementType_t<T>, rowVector >;
};
/*! \endcond */
//*************************************************************************************************


//=================================================================================================
//
//  DILATEDSUBMATRIXTRAIT SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename MT >
struct DilatedSubmatrixTraitEval2< MT, inf, inf, inf, inf, inf, inf
                          , EnableIf_t< IsDenseMatrix_v<MT> &&
                                        ( Size_v<MT,0UL> == DefaultSize_v ||
                                          Size_v<MT,1UL> == DefaultSize_v ) &&
                                        ( MaxSize_v<MT,0UL> == DefaultMaxSize_v ||
                                          MaxSize_v<MT,1UL> == DefaultMaxSize_v ) > >
{
   using Type = DynamicMatrix< RemoveConst_t< ElementType_t<MT> >, StorageOrder_v<MT> >;
};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  MULTTRAIT SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename T1, typename T2 >
struct MultTraitEval2<  T1, T2
                     , EnableIf_t< IsTensor_v<T1> &&
                                   IsColumnVector_v<T2> &&
                                   ( IsDenseTensor_v<T1> || IsDenseVector_v<T2> ) &&
                                   ( (Size_v<T1,0UL> == DefaultSize_v && Size_v<T1,1UL> == DefaultSize_v
                                      && Size_v<T2,1UL> == DefaultSize_v) ||
                                        Size_v<T2,0UL> == DefaultSize_v ) &&
                                   ( (MaxSize_v<T1,0UL> == DefaultMaxSize_v && MaxSize_v<T1,1UL> == DefaultMaxSize_v
                                      && MaxSize_v<T1,2UL> == DefaultMaxSize_v) ||
                                       MaxSize_v<T2,0UL> == DefaultMaxSize_v )> >
{
   using ET1 = ElementType_t<T1>;
   using ET2 = ElementType_t<T2>;

   using Type = DynamicMatrix< MultTrait_t<ET1,ET2>, false >;
};

//template< typename ET1, typename ET2 >
//struct MultTraitEval2<DynamicTensor<ET1>, DynamicVector<ET2, true> >
//{
//      using Type = DynamicMatrix< MultTrait_t<ET1,ET2>, true >;
//};

/*! \endcond */
//*************************************************************************************************


} // namespace blaze

#endif
