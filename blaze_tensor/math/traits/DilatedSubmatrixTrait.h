//=================================================================================================
/*!
//  \file blaze_tensor/math/traits/DilatedSubmatrixTrait.h
//  \brief Header file for the dilatedsubmatrix trait
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

#ifndef _BLAZE_TENSOR_MATH_TRAITS_DILATEDSUBMATRIXTRAIT_H_
#define _BLAZE_TENSOR_MATH_TRAITS_DILATEDSUBMATRIXTRAIT_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <utility>
#include <blaze/math/Infinity.h>
#include <blaze/util/InvalidType.h>
#include <blaze/util/Types.h>


namespace blaze {

//=================================================================================================
//
//  CLASS DEFINITION
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename, size_t... > struct DilatedSubmatrixTrait;
template< typename, size_t, size_t, size_t, size_t, size_t, size_t, typename = void > struct DilatedSubmatrixTraitEval1;
template< typename, size_t, size_t, size_t, size_t, size_t, size_t, typename = void > struct DilatedSubmatrixTraitEval2;
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< size_t I, size_t J, size_t M, size_t N, size_t RowDilation,
   size_t ColumnDilation, typename T >
auto evalDilatedSubmatrixTrait( T& ) -> typename DilatedSubmatrixTraitEval1< T,
   I, J, M, N, RowDilation, ColumnDilation >::Type;

template< typename T >
auto evalDilatedSubmatrixTrait( T& )
   -> typename DilatedSubmatrixTraitEval2<T,inf,inf,inf,inf,inf,inf>::Type;

template< size_t I, size_t J, size_t M, size_t N, size_t RowDilation,
   size_t ColumnDilation, typename T >
auto evalDilatedSubmatrixTrait( const T& ) -> typename DilatedSubmatrixTrait< T,
   I, J, M, N, RowDilation, ColumnDilation >::Type;

template< typename T >
auto evalDilatedSubmatrixTrait( const T& )
   -> typename DilatedSubmatrixTrait<T>::Type;

template< size_t I, size_t J, size_t M, size_t N, size_t RowDilation,
   size_t ColumnDilation, typename T >
auto evalDilatedSubmatrixTrait( const volatile T& ) ->
   typename DilatedSubmatrixTrait< T, I, J, M, N, RowDilation,
      ColumnDilation >::Type;

template< typename T >
auto evalDilatedSubmatrixTrait( const volatile T& )
   -> typename DilatedSubmatrixTrait<T>::Type;
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Base template for the SubmatrixTrait class.
// \ingroup math_traits
//
// \section submatrixtrait_general General
//
// The SubmatrixTrait class template offers the possibility to select the resulting data type
// when creating a submatrix of a dense or sparse matrix. SubmatrixTrait defines the nested
// type \a Type, which represents the resulting data type of the submatrix operation. In case
// the given data type is not a dense or sparse matrix type, the resulting data type \a Type
// is set to \a INVALID_TYPE. Note that \a const and \a volatile qualifiers and reference
// modifiers are generally ignored.
//
//
// \section submatrixtrait_specializations Creating custom specializations
//
// Per default, SubmatrixTrait supports all matrix types of the Blaze library (including views
// and adaptors). For all other data types it is possible to specialize the SubmatrixTrait
// template. The following example shows the according specialization for the DynamicMatrix
// class template:

   \code
   template< typename T1, bool SO >
   struct SubmatrixTrait< DynamicMatrix<T1,SO> >
   {
      using Type = DynamicMatrix<T1,SO>;
   };
   \endcode

// \n \section submatrixtrait_examples Examples
//
// The following example demonstrates the use of the SubmatrixTrait template, where depending
// on the given matrix type the according result type is selected:

   \code
   using blaze::rowMajor;
   using blaze::columnMajor;

   // Definition of the result type of a row-major dynamic matrix
   using MatrixType1 = blaze::DynamicMatrix<int,rowMajor>;
   using ResultType1 = typename blaze::SubmatrixTrait<MatrixType1>::Type;

   // Definition of the result type for the inner four elements of a 4x4 column-major static matrix
   using MatrixType2 = blaze::StaticMatrix<int,4UL,4UL,columnMajor>;
   using ResultType2 = typename blaze::SubmatrixTrait<MatrixType2,1UL,1UL,2UL,2UL>::Type;
   \endcode
*/
template< typename MT       // Type of the matrix
        , size_t... CSAs >  // Compile time dilatedsubmatrix arguments
struct DilatedSubmatrixTrait
{
 public:
   //**********************************************************************************************
   /*! \cond BLAZE_INTERNAL */
   using Type = decltype( evalDilatedSubmatrixTrait<CSAs...>( std::declval<MT&>() ) );
   /*! \endcond */
   //**********************************************************************************************
};
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Auxiliary alias declaration for the SubmatrixTrait type trait.
// \ingroup math_traits
//
// The SubmatrixTrait_t alias declaration provides a convenient shortcut to access the nested
// \a Type of the SubmatrixTrait class template. For instance, given the matrix type \a MT the
// following two type definitions are identical:

   \code
   using Type1 = typename blaze::SubmatrixTrait<MT>::Type;
   using Type2 = blaze::SubmatrixTrait_t<MT>;
   \endcode
*/
template< typename MT       // Type of the matrix
        , size_t... CSAs >  // Compile time submatrix arguments
using DilatedSubmatrixTrait_t = typename DilatedSubmatrixTrait<MT,CSAs...>::Type;
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief First auxiliary helper struct for the SubmatrixTrait type trait.
// \ingroup math_traits
*/
template< typename MT            // Type of the matrix
        , size_t I               // Index of the first row
        , size_t J               // Index of the first column
        , size_t M               // Number of rows
        , size_t N               // Number of columns
        , size_t RowDilation     // Step between rows
        , size_t ColumnDilation  // Step between columns
        , typename >             // Restricting condition
struct DilatedSubmatrixTraitEval1
{
 public:
   //**********************************************************************************************
   using Type = typename DilatedSubmatrixTraitEval2<MT,I,J,M,N,RowDilation,ColumnDilation>::Type;
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Second auxiliary helper struct for the DilatedSubmatrixTrait type trait.
// \ingroup math_traits
*/
template< typename MT            // Type of the matrix
        , size_t I               // Index of the first row
        , size_t J               // Index of the first column
        , size_t M               // Number of rows
        , size_t N               // Number of columns
        , size_t RowDilation     // Step between rows
        , size_t ColumnDilation  // Step between columns
        , typename >             // Restricting condition
struct DilatedSubmatrixTraitEval2
{
 public:
   //**********************************************************************************************
   using Type = INVALID_TYPE;
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************

} // namespace blaze

#endif
