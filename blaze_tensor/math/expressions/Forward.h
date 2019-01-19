//=================================================================================================
/*!
//  \file blaze_tensor/math/expressions/Forward.h
//  \brief Header file for all forward declarations for expression class templates
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

#ifndef _BLAZE_TENSOR_MATH_EXPRESSIONS_FORWARD_H_
#define _BLAZE_TENSOR_MATH_EXPRESSIONS_FORWARD_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <blaze/math/expressions/Forward.h>

namespace blaze {

//=================================================================================================
//
//  ::blaze NAMESPACE FORWARD DECLARATIONS
//
//=================================================================================================

template< typename > struct DenseTensor;
template< typename > class DTensSerialExpr;
template< typename, typename > class DTensDTensAddExpr;
template< typename, typename > class DTensDTensMultExpr;
template< typename, typename > class DTensDTensSchurExpr;
template< typename, typename > class DTensDTensSubExpr;
template< typename, typename > class DTensMapExpr;
template< typename, typename > class DTensScalarMultExpr;
template< typename, typename > class DTensScalarDivExpr;
template< typename, typename, typename > class DTensDTensMapExpr;
template< typename, size_t... > class DTensTransExpr;
template< typename, size_t... > class DMatExpandExpr;



template< typename TT1, typename TT2 >
decltype(auto) operator+( const DenseTensor<TT1>&, const DenseTensor<TT2>& );

template< typename TT1, typename TT2 >
decltype(auto) operator-( const DenseTensor<TT1>&, const DenseTensor<TT2>& );

template< typename TT1, typename TT2 >
decltype(auto) operator*( const DenseTensor<TT1>&, const DenseTensor<TT2>& );

template< typename TT1, typename TT2 >
decltype(auto) operator%( const DenseTensor<TT1>&, const DenseTensor<TT2>& );


template< size_t O, size_t M, size_t N, typename MT, typename ... RTAs>
decltype(auto) trans( const DenseTensor<MT>& dm, RTAs... args );

template< typename MT, typename ... RTAs>
decltype(auto) trans( const DenseTensor<MT>& dm, RTAs... args );


template< typename TT >
decltype(auto) eval( const DenseTensor<TT>& );

template< typename TT >
decltype(auto) serial( const DenseTensor<TT>& );

template< typename TT >
inline decltype(auto) inv( const DenseTensor<TT>& );

template< typename TT, typename OP >
decltype(auto) map( const DenseTensor<TT>&, OP );

template< typename TT1, typename TT2, typename OP >
decltype(auto) map( const DenseTensor<TT1>&, const DenseTensor<TT2>&, OP );

template< typename TT, typename OP >
decltype(auto) reduce( const DenseTensor<TT>&, OP );

template< typename TT, bool SO >
decltype(auto) expand( const DenseMatrix<TT, SO>&, size_t );

template< size_t E, typename TT, bool SO >
decltype(auto) expand( const DenseMatrix<TT, SO>& );

template< size_t E, typename TT, bool SO >
decltype(auto) expand( const DenseMatrix<TT, SO>&, size_t );

} // namespace blaze

#endif
