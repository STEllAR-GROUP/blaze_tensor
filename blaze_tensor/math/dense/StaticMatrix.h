//=================================================================================================
/*!
//  \file blaze_tensor/math/dense/StaticMatrix.h
//  \brief Header file for the implementation of a uniform matrix
//
//  Copyright (C) 2012-2019 Klaus Iglberger - All Rights Reserved
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

#ifndef _BLAZE_TENSOR_MATH_DENSE_STATICMATRIX_H_
#define _BLAZE_TENSOR_MATH_DENSE_STATICMATRIX_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <blaze/math/dense/StaticMatrix.h>
#include <blaze/math/dense/StaticVector.h>
#include <blaze/math/traits/ExpandTrait.h>

#include <blaze_tensor/math/dense/Forward.h>
#include <blaze_tensor/math/traits/RavelTrait.h>


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
                                     ( E != inf ) &&
                                     ( Size_v<T,0UL> != DefaultSize_v ) &&
                                     ( MaxSize_v<T,0UL> != DefaultMaxSize_v ) &&
                                     ( Size_v<T,1UL> != DefaultSize_v ) &&
                                     ( MaxSize_v<T,1UL> != DefaultMaxSize_v ) > >
{
   static constexpr size_t O = ( E );
   static constexpr size_t M = ( Size_v<T,0UL> );
   static constexpr size_t N = ( Size_v<T,1UL> );

   using Type = StaticTensor< ElementType_t<T>, O, M, N >;
};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  RAVELTRAIT SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename T > // Type to be expanded
struct RavelTraitEval2< T
                       , EnableIf_t< IsDenseMatrix_v<T> &&
                                     ( Size_v<T,0UL> != DefaultSize_v ) &&
                                     ( MaxSize_v<T,0UL> != DefaultMaxSize_v ) &&
                                     ( Size_v<T,1UL> != DefaultSize_v ) &&
                                     ( MaxSize_v<T,1UL> != DefaultMaxSize_v ) > >
{
   using Type = StaticVector< ElementType_t<T>, Size_v<T,0UL> * Size_v<T,1UL>, rowVector >;
};
/*! \endcond */
//*************************************************************************************************




} // namespace blaze

#endif
