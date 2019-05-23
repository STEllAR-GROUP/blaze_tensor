//=================================================================================================
/*!
//  \file blaze_tensor/math/dense/UniformMatrix.h
//  \brief Header file for the implementation of a uniform matrix
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

#ifndef _BLAZE_TENSOR_MATH_DENSE_UNIFORMMATRIX_H_
#define _BLAZE_TENSOR_MATH_DENSE_UNIFORMMATRIX_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <blaze/math/dense/UniformMatrix.h>
#include <blaze/math/dense/UniformVector.h>
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
struct ExpandTraitEval1< T, E
                       , EnableIf_t< IsMatrix_v<T> && IsRowMajorMatrix_v<T> &&
                                     IsUniform_v<T> && !IsZero_v<T> > >
{
   using Type = UniformTensor< ElementType_t<T> >;
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
template< typename T >
struct RavelTraitEval1< T
                    , EnableIf_t< IsDenseMatrix_v<T> && IsUniform_v<T> && !IsZero_v<T> > >
{
   using Type = UniformVector< ElementType_t<T>, rowVector >;
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
template< typename MT, size_t I, size_t J, size_t M, size_t N, size_t RowDilation, size_t ColumnDilation >
struct DilatedSubmatrixTraitEval1< MT, I, J, M, N, RowDilation, ColumnDilation
                          , EnableIf_t< IsUniform_v<MT> && !IsZero_v<MT> > >
{
   using Type = UniformMatrix< RemoveConst_t< ElementType_t<MT> >, StorageOrder_v<MT> >;
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
struct MultTraitEval2< T1, T2
                     , EnableIf_t< IsTensor_v<T1> &&
                                   IsColumnVector_v<T2> &&
                                   IsUniform_v<T1> &&
                                   !( IsZero_v<T1> || IsZero_v<T2> ) > >
{
   using ET1 = ElementType_t<T1>;
   using ET2 = ElementType_t<T2>;

   using Type = UniformMatrix< MultTrait_t<ET1,ET2>, false >;
};

/*! \endcond */
//*************************************************************************************************

} // namespace blaze

#endif
