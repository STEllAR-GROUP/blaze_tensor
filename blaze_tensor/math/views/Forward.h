//=================================================================================================
/*!
//  \file blaze_tensor/math/views/Forward.h
//  \brief Header file for all forward declarations for views
//
//  Copyright (C) 2012-2018 Klaus Iglberger - All Rights Reserved
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

#ifndef _BLAZE_TENSOR_MATH_VIEWS_FORWARD_H_
#define _BLAZE_TENSOR_MATH_VIEWS_FORWARD_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <blaze/math/expressions/Forward.h>
#include <blaze/math/views/subvector/BaseTemplate.h>
#include <blaze/math/views/submatrix/BaseTemplate.h>

#include <blaze_tensor/math/expressions/Forward.h>
#include <blaze_tensor/math/views/columnslice/BaseTemplate.h>
#include <blaze_tensor/math/views/dilatedsubmatrix/BaseTemplate.h>
#include <blaze_tensor/math/views/dilatedsubvector/BaseTemplate.h>
#include <blaze_tensor/math/views/dilatedsubtensor/BaseTemplate.h>
#include <blaze_tensor/math/views/pageslice/BaseTemplate.h>
#include <blaze_tensor/math/views/rowslice/BaseTemplate.h>
#include <blaze_tensor/math/views/subtensor/BaseTemplate.h>

namespace blaze {

//=================================================================================================
//
//  ::blaze NAMESPACE FORWARD DECLARATIONS
//
//=================================================================================================

template< size_t I, size_t M, size_t Dilation, typename VT, bool TF, typename... RSAs >
decltype(auto) dilatedsubvector( Vector<VT,TF>&, RSAs... );

template< size_t I, size_t M, size_t Dilation, typename VT, bool TF, typename... RSAs >
decltype(auto) dilatedsubvector( const Vector<VT,TF>&, RSAs... );

template< size_t I, size_t M, size_t Dilation, typename VT, bool TF, typename... RSAs >
decltype(auto) dilatedsubvector( Vector<VT,TF>&&, RSAs... );

template< typename VT, bool TF, typename... RSAs >
decltype(auto) dilatedsubvector( Vector<VT,TF>&, size_t, size_t, size_t, RSAs... );

template< typename VT, bool TF, typename... RSAs >
decltype(auto) dilatedsubvector( const Vector<VT,TF>&, size_t, size_t, size_t, RSAs... );

template< typename VT, bool TF, typename... RSAs >
decltype(auto) dilatedsubvector( Vector<VT,TF>&&, size_t, size_t, size_t, RSAs... );

template< typename VT, bool TF, bool DF, size_t... CSAs, typename... RSAs >
inline decltype(auto) dilatedsubvector( DilatedSubvector<VT,TF,DF,CSAs...>& sv,
   size_t index, size_t size, size_t dilation, RSAs... args);

template< typename VT, bool TF, bool DF, size_t... CSAs, typename... RSAs >
inline decltype(auto) dilatedsubvector( const DilatedSubvector<VT,TF,DF,CSAs...>& sv,
   size_t index, size_t size, size_t dilation, RSAs... args);

template< typename VT, bool TF, bool DF, size_t... CSAs, typename... RSAs >
inline decltype(auto) dilatedsubvector( DilatedSubvector<VT,TF,DF,CSAs...>&& sv,
   size_t index, size_t size, size_t dilation, RSAs... args);

template< typename VT, AlignmentFlag AF, bool TF, bool DF, size_t... CSAs, typename... RSAs >
inline decltype(auto) dilatedsubvector( Subvector<VT,AF,TF,DF,CSAs...>& sv,
   size_t index, size_t size, size_t dilation, RSAs... args);

template< typename VT, AlignmentFlag AF, bool TF, bool DF, size_t... CSAs, typename... RSAs >
inline decltype(auto) dilatedsubvector( const Subvector<VT,AF,TF,DF,CSAs...>& sv,
   size_t index, size_t size, size_t dilation, RSAs... args);

template< typename VT, AlignmentFlag AF, bool TF, bool DF, size_t... CSAs, typename... RSAs >
inline decltype(auto) dilatedsubvector( Subvector<VT,AF,TF,DF,CSAs...>&& sv,
   size_t index, size_t size, size_t dilation, RSAs... args);



template< size_t I, size_t J, size_t M, size_t N, size_t RowDilation, size_t ColumnDilation, typename MT, bool SO, typename... RSAs >
decltype(auto) dilatedsubmatrix( Matrix<MT,SO>&, RSAs... );

template< size_t I, size_t J, size_t M, size_t N, size_t RowDilation, size_t ColumnDilation, typename MT, bool SO, typename... RSAs >
decltype(auto) dilatedsubmatrix( const Matrix<MT,SO>&, RSAs... );

template< size_t I, size_t J, size_t M, size_t N, size_t RowDilation, size_t ColumnDilation, typename MT, bool SO, typename... RSAs >
decltype(auto) dilatedsubmatrix( Matrix<MT,SO>&&, RSAs... );

template< typename MT, bool SO, typename... RSAs >
decltype(auto) dilatedsubmatrix( Matrix<MT,SO>&, size_t, size_t, size_t, size_t, size_t, size_t, RSAs... );

template< typename MT, bool SO, typename... RSAs >
decltype(auto) dilatedsubmatrix( const Matrix<MT,SO>&, size_t, size_t, size_t, size_t, size_t, size_t, RSAs... );

template< typename MT, bool SO, typename... RSAs >
decltype(auto) dilatedsubmatrix( Matrix<MT,SO>&&, size_t, size_t, size_t, size_t, size_t, size_t, RSAs... );

template< typename MT, bool SO, bool DF, size_t... CSAs, typename... RSAs >
inline decltype(auto) dilatedsubmatrix( DilatedSubmatrix<MT,SO,DF,CSAs...>& sm,
   size_t row, size_t column, size_t m, size_t n, size_t rowdilation, size_t columndilation, RSAs... args);

template< typename MT,bool SO, bool DF, size_t... CSAs, typename... RSAs >
inline decltype(auto) dilatedsubmatrix( const DilatedSubmatrix<MT,SO,DF,CSAs...>& sm,
   size_t row, size_t column, size_t m, size_t n, size_t rowdilation, size_t columndilation, RSAs... args);

template< typename MT, bool SO, bool DF, size_t... CSAs, typename... RSAs >
inline decltype(auto) dilatedsubmatrix( DilatedSubmatrix<MT, SO, DF, CSAs...>&& sm,
   size_t row, size_t column, size_t m, size_t n, size_t rowdilation, size_t columndilation, RSAs... args);

template< typename MT, AlignmentFlag AF, bool SO, bool DF, size_t... CSAs, typename... RSAs >
inline decltype(auto) dilatedsubmatrix( Submatrix<MT,AF,SO,DF,CSAs...>& sm,
   size_t row, size_t column, size_t m, size_t n, size_t rowdilation, size_t columndilation, RSAs... args);

template< typename MT, AlignmentFlag AF, bool SO, bool DF, size_t... CSAs, typename... RSAs >
inline decltype(auto) dilatedsubmatrix( const Submatrix<MT,AF,SO,DF,CSAs...>& sm,
   size_t row, size_t column, size_t m, size_t n, size_t rowdilation, size_t columndilation, RSAs... args);

template< typename MT, AlignmentFlag AF, bool SO, bool DF, size_t... CSAs, typename... RSAs >
inline decltype(auto) dilatedsubmatrix( Submatrix<MT, AF, SO, DF, CSAs...>&& sm,
   size_t row, size_t column, size_t m, size_t n, size_t rowdilation, size_t columndilation, RSAs... args);




template< size_t K, size_t I, size_t J, size_t O, size_t M, size_t N, size_t PageDilation, size_t RowDilation, size_t ColumnDilation, typename TT, typename... RSAs >
decltype(auto) dilatedsubtensor( Tensor<TT>&, RSAs... );

template< size_t K, size_t I, size_t J, size_t O, size_t M, size_t N, size_t PageDilation, size_t RowDilation, size_t ColumnDilation, typename TT, typename... RSAs >
decltype(auto) dilatedsubtensor( const Tensor<TT>&, RSAs... );

template< size_t K, size_t I, size_t J, size_t O, size_t M, size_t N, size_t PageDilation, size_t RowDilation, size_t ColumnDilation, typename TT, typename... RSAs >
decltype(auto) dilatedsubtensor( Tensor<TT>&&, RSAs... );

template< typename TT, typename... RSAs >
decltype(auto) dilatedsubtensor( Tensor<TT>&, size_t, size_t, size_t, size_t, size_t, size_t, size_t, size_t, size_t, RSAs... );

template< typename TT, typename... RSAs >
decltype(auto) dilatedsubtensor( const Tensor<TT>&, size_t, size_t, size_t, size_t, size_t, size_t, size_t, size_t, size_t, RSAs... );

template< typename TT, typename... RSAs >
decltype(auto) dilatedsubtensor( Tensor<TT>&&, size_t, size_t, size_t, size_t, size_t, size_t, size_t, size_t, size_t, RSAs... );

template< typename TT, bool DF, size_t... CSAs, typename... RSAs >
inline decltype(auto) dilatedsubtensor( DilatedSubtensor<TT,DF,CSAs...>& st,
   size_t page, size_t row, size_t column, size_t o, size_t m, size_t n, size_t pagedilation, size_t rowdilation, size_t columndilation, RSAs... args);

template< typename TT, bool DF, size_t... CSAs, typename... RSAs >
inline decltype(auto) dilatedsubtensor( const DilatedSubtensor<TT,DF,CSAs...>& st,
   size_t page, size_t row, size_t column, size_t o, size_t m, size_t n, size_t pagedilation, size_t rowdilation, size_t columndilation, RSAs... args);

template< typename TT, bool DF, size_t... CSAs, typename... RSAs >
inline decltype(auto) dilatedsubtensor( DilatedSubtensor<TT , DF, CSAs...>&& st,
   size_t page, size_t row, size_t column, size_t o, size_t m, size_t n, size_t pagedilation, size_t rowdilation, size_t columndilation, RSAs... args);

template< typename TT, AlignmentFlag AF, bool DF, size_t... CSAs, typename... RSAs >
inline decltype(auto) dilatedsubtensor( Subtensor<TT,AF,DF,CSAs...>& st,
   size_t page, size_t row, size_t column, size_t o, size_t m, size_t n, size_t pagedilation, size_t rowdilation, size_t columndilation, RSAs... args);

template< typename TT, AlignmentFlag AF, bool DF, size_t... CSAs, typename... RSAs >
inline decltype(auto) dilatedsubtensor( const Subtensor<TT,AF,DF,CSAs...>& st,
   size_t page, size_t row, size_t column, size_t o, size_t m, size_t n, size_t pagedilation, size_t rowdilation, size_t columndilation, RSAs... args);

template< typename TT, AlignmentFlag AF, bool DF, size_t... CSAs, typename... RSAs >
inline decltype(auto) dilatedsubtensor( Subtensor<TT, AF, DF, CSAs...>&& st,
   size_t page, size_t row, size_t column, size_t o, size_t m, size_t n, size_t pagedilation, size_t rowdilation, size_t columndilation, RSAs... args);

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
