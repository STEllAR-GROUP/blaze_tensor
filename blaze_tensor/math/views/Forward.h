//=================================================================================================
/*!
//  \file blaze_tensor/math/views/Forward.h
//  \brief Header file for all forward declarations for views
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

#ifndef _BLAZE_TENSOR_MATH_VIEWS_FORWARD_H_
#define _BLAZE_TENSOR_MATH_VIEWS_FORWARD_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <blaze/math/expressions/Forward.h>

#include <blaze_tensor/math/expressions/Forward.h>
// #include <blaze_tensor/math/views/column/BaseTemplate.h>
// #include <blaze_tensor/math/views/columns/BaseTemplate.h>
// #include <blaze_tensor/math/views/elements/BaseTemplate.h>
#include <blaze_tensor/math/views/columnslice/BaseTemplate.h>
#include <blaze_tensor/math/views/pageslice/BaseTemplate.h>
#include <blaze_tensor/math/views/rowslice/BaseTemplate.h>
// #include <blaze_tensor/math/views/rows/BaseTemplate.h>
// #include <blaze_tensor/math/views/page/BaseTemplate.h>
// #include <blaze_tensor/math/views/pages/BaseTemplate.h>
#include <blaze_tensor/math/views/subtensor/BaseTemplate.h>

namespace blaze {

//=================================================================================================
//
//  ::blaze NAMESPACE FORWARD DECLARATIONS
//
//=================================================================================================

// template< AlignmentFlag AF, size_t K, size_t I, size_t J, size_t O, size_t M, size_t N, typename TT, typename... RSAs >
// decltype(auto) submatrix( Tensor<TT>&, RSAs... );
//
// template< AlignmentFlag AF, size_t K, size_t I, size_t J, size_t O, size_t M, size_t N, typename TT, typename... RSAs >
// decltype(auto) submatrix( const Tensor<TT>&, RSAs... );
//
// template< AlignmentFlag AF, size_t K, size_t I, size_t J, size_t O, size_t M, size_t N, typename TT, typename... RSAs >
// decltype(auto) submatrix( Tensor<TT>&&, RSAs... );
//
// template< AlignmentFlag AF, size_t K, size_t I, size_t J, size_t O, size_t M, size_t N, typename TT, typename... RSAs >
// decltype(auto) submatrix( Tensor<TT>&, size_t, size_t, size_t, size_t, RSAs... );
//
// template< AlignmentFlag AF, size_t K, size_t I, size_t J, size_t O, size_t M, size_t N, typename TT, typename... RSAs >
// decltype(auto) submatrix( const Tensor<TT>&, size_t, size_t, size_t, size_t, RSAs... );
//
// template< AlignmentFlag AF, size_t K, size_t I, size_t J, size_t O, size_t M, size_t N, typename TT, typename... RSAs >
// decltype(auto) submatrix( Tensor<TT>&&, size_t, size_t, size_t, size_t, RSAs... );
//
// template< size_t I, size_t... Is, typename TT, typename... RRAs >
// decltype(auto) rows( Tensor<TT>&, RRAs... );
//
// template< size_t I, size_t... Is, typename TT, typename... RRAs >
// decltype(auto) rows( const Tensor<TT>&, RRAs... );
//
// template< size_t I, size_t... Is, typename TT, typename... RRAs >
// decltype(auto) rows( Tensor<TT>&&, RRAs... );
//
// template< typename TT, typename T, typename... RRAs >
// decltype(auto) rows( Tensor<TT>&, const T*, size_t, RRAs... );
//
// template< typename TT, typename T, typename... RRAs >
// decltype(auto) rows( const Tensor<TT>&, const T*, size_t, RRAs... );
//
// template< typename TT, typename T, typename... RRAs >
// decltype(auto) rows( Tensor<TT>&&, const T*, size_t, RRAs... );

// template< size_t I, typename TT, typename... RCAs >
// decltype(auto) column( Tensor<TT>&, RCAs... );
//
// template< size_t I, typename TT, typename... RCAs >
// decltype(auto) column( const Tensor<TT>&, RCAs... );
//
// template< size_t I, typename TT, typename... RCAs >
// decltype(auto) column( Tensor<TT>&&, RCAs... );
//
// template< typename TT, typename... RCAs >
// decltype(auto) column( Tensor<TT>&, size_t, RCAs... );
//
// template< typename TT, typename... RCAs >
// decltype(auto) column( const Tensor<TT>&, size_t, RCAs... );
//
// template< typename TT, typename... RCAs >
// decltype(auto) column( Tensor<TT>&&, size_t, RCAs... );
//
// template< size_t I, size_t... Is, typename TT, typename... RCAs >
// decltype(auto) columns( Tensor<TT>&, RCAs... );
//
// template< size_t I, size_t... Is, typename TT, typename... RCAs >
// decltype(auto) columns( const Tensor<TT>&, RCAs... );
//
// template< size_t I, size_t... Is, typename TT, typename... RCAs >
// decltype(auto) columns( Tensor<TT>&&, RCAs... );
//
// template< typename TT, typename T, typename... RCAs >
// decltype(auto) columns( Tensor<TT>&, const T*, size_t, RCAs... );
//
// template< typename TT, typename T, typename... RCAs >
// decltype(auto) columns( const Tensor<TT>&, const T*, size_t, RCAs... );
//
// template< typename TT, typename T, typename... RCAs >
// decltype(auto) columns( Tensor<TT>&&, const T*, size_t, RCAs... );

template< size_t I, typename TT, typename... RRAs >
decltype(auto) pageslice( Tensor<TT>&, RRAs... );

template< size_t I, typename TT, typename... RRAs >
decltype(auto) pageslice( const Tensor<TT>&, RRAs... );

template< size_t I, typename TT, typename... RRAs >
decltype(auto) pageslice( Tensor<TT>&&, RRAs... );

template< typename TT, typename... RRAs >
decltype(auto) pageslice( Tensor<TT>&, size_t, RRAs... );

template< typename TT, typename... RRAs >
decltype(auto) pageslice( const Tensor<TT>&, size_t, RRAs... );

template< typename TT, typename... RRAs >
decltype(auto) pageslice( Tensor<TT>&&, size_t, RRAs... );

// template< size_t I, size_t... Is, typename TT, typename... RCAs >
// decltype(auto) pages( Tensor<TT>&, RCAs... );
//
// template< size_t I, size_t... Is, typename TT, typename... RCAs >
// decltype(auto) pages( const Tensor<TT>&, RCAs... );
//
// template< size_t I, size_t... Is, typename TT, typename... RCAs >
// decltype(auto) pages( Tensor<TT>&&, RCAs... );
//
// template< typename TT, typename T, typename... RCAs >
// decltype(auto) pages( Tensor<TT>&, const T*, size_t, RCAs... );
//
// template< typename TT, typename T, typename... RCAs >
// decltype(auto) pages( const Tensor<TT>&, const T*, size_t, RCAs... );
//
// template< typename TT, typename T, typename... RCAs >
// decltype(auto) pages( Tensor<TT>&&, const T*, size_t, RCAs... );

} // namespace blaze

#endif
