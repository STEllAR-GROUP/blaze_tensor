//=================================================================================================
/*!
//  \file blaze_tensor/math/expressions/DQuatTransExprData.h
//  \brief Header file for the implementation of the DQuatTransExprData class template
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

#ifndef _BLAZE_TENSOR_MATH_EXPROESSIONS_DQUATTRANSEXPRDATA_H_
#define _BLAZE_TENSOR_MATH_EXPROESSIONS_DQUATTRANSEXPRDATA_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <blaze/util/Assert.h>
#include <blaze/util/SmallArray.h>
#include <blaze/util/Types.h>
#include <blaze/util/MaybeUnused.h>

#include <array>
#include <algorithm>
#include <initializer_list>

namespace blaze {

//=================================================================================================
//
//  CLASS DEFINITION
//
//=================================================================================================

template< size_t... CTAs >  // Compile time expression arguments
struct DQuatTransExprData;

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Auxiliary class template for the data members of the DQuatTransExpr class.
// \ingroup elements
//
// The auxiliary DQuatTransExprData class template represents an abstraction of the data members of the
// QuatTransExpr class template. The necessary set of data members is selected depending on the number
// of compile time element arguments. The basic implementation of DQuatTransExprData adapts the class
// template to the requirements of multiple compile time element arguments.
*/
template<>
struct DQuatTransExprData<0UL, 1UL, 2UL, 3UL>
{
 public:
   //**Type definitions****************************************************************************
   using Indices = std::array<size_t, 4UL>;  //!< Type of the container for transpose indices.
   //**********************************************************************************************

   //**Constructors********************************************************************************
   /*!\name Constructors */
   //@{
   template< typename... RTAs >
   explicit inline DQuatTransExprData( RTAs... args ) noexcept
   {
      MAYBE_UNUSED( args... );
   }

   DQuatTransExprData( const DQuatTransExprData& ) = default;
   //@}
   //**********************************************************************************************

   //**Destructor**********************************************************************************
   /*!\name Destructor */
   //@{
   ~DQuatTransExprData() = default;
   //@}
   //**********************************************************************************************

   //**Assignment operators************************************************************************
   /*!\name Assignment operators */
   //@{
   DQuatTransExprData& operator=( const DQuatTransExprData& ) = delete;
   //@}
   //**********************************************************************************************

   //**Utility functions***************************************************************************
   /*!\name Utility functions */
   //@{

   /*! \cond BLAZE_INTERNAL */
   /*!\brief Returns the mapped page indices
   //
   // \return The mapped page indices.
   */
   static inline constexpr Indices idces( ) noexcept
   {
      return Indices{0UL, 1UL, 2UL, 3UL};
   }

   /*! \cond BLAZE_INTERNAL */
   /*!\brief Returns the mapped quat index of the specified element in the underlying tensor.
   //
   // \param l Quat access index.
   // \param k Page access index.
   // \param i Row access index.
   // \param j Column access index.
   // \return The mapped page index of the specified element.
   */
   static inline constexpr size_t quat  ( size_t l, size_t k, size_t i, size_t j ) noexcept
   {
      return l;
   }

   /*! \cond BLAZE_INTERNAL */
   /*!\brief Returns the mapped page index of the specified element in the underlying tensor.
   //
   // \param l Quat access index.
   // \param k Page access index.
   // \param i Row access index.
   // \param j Column access index.
   // \return The mapped page index of the specified element.
   */
   static inline constexpr size_t page  ( size_t l, size_t k, size_t i, size_t j ) noexcept
   {
      return k;
   }

   /*! \cond BLAZE_INTERNAL */
   /*!\brief Returns the mapped row index of the specified element in the underlying tensor.
   //
   // \param l Quat access index.
   // \param k Page access index.
   // \param i Row access index.
   // \param j Column access index.
   // \return The mapped row index of the specified element.
   */
   static inline constexpr size_t row   ( size_t l, size_t k, size_t i, size_t j ) noexcept
   {
      return i;
   }

   /*! \cond BLAZE_INTERNAL */
   /*!\brief Returns the mapped column index of the specified element in the underlying tensor.
   //
   // \param l Quat access index.
   // \param k Page access index.
   // \param i Row access index.
   // \param j Column access index.
   // \return The mapped column index of the specified element.
   */
   static inline constexpr size_t column( size_t l, size_t k, size_t i, size_t j ) noexcept
   {
      return j;
   }
   //@}
   //**********************************************************************************************

   /*! \cond BLAZE_INTERNAL */
   /*!\brief Returns the reversely mapped quat index of the specified element in the underlying tensor.
   //
   // \param l Quat access index.
   // \param k Page access index.
   // \param i Row access index.
   // \param j Column access index.
   // \return The mapped page index of the specified element.
   */
   static inline constexpr size_t reverse_quat  ( size_t l, size_t k, size_t i, size_t j ) noexcept
   {
      return l;
   }

   /*! \cond BLAZE_INTERNAL */
   /*!\brief Returns the reversely mapped page index of the specified element in the underlying tensor.
   //
   // \param l Quat access index.
   // \param k Page access index.
   // \param i Row access index.
   // \param j Column access index.
   // \return The mapped page index of the specified element.
   */
   static inline constexpr size_t reverse_page  ( size_t l, size_t k, size_t i, size_t j ) noexcept
   {
      return k;
   }

   /*! \cond BLAZE_INTERNAL */
   /*!\brief Returns the reversely mapped row index of the specified element in the underlying tensor.
   //
   // \param l Quat access index.
   // \param k Page access index.
   // \param i Row access index.
   // \param j Column access index.
   // \return The mapped row index of the specified element.
   */
   static inline constexpr size_t reverse_row   ( size_t l, size_t k, size_t i, size_t j ) noexcept
   {
      return i;
   }

   /*! \cond BLAZE_INTERNAL */
   /*!\brief Returns the reversely mapped column index of the specified element in the underlying tensor.
   //
   // \param l Quat access index.
   // \param k Page access index.
   // \param i Row access index.
   // \param j Column access index.
   // \return The mapped column index of the specified element.
   */
   static inline constexpr size_t reverse_column( size_t l, size_t k, size_t i, size_t j ) noexcept
   {
      return j;
   }
   //@}
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************



template<>
struct DQuatTransExprData<0UL, 1UL, 3UL, 2UL>
{
 public:
   //**Type definitions****************************************************************************
   using Indices = std::array<size_t, 4UL>;  //!< Type of the container for transpose indices.
   //**********************************************************************************************

   //**Constructors********************************************************************************
   /*!\name Constructors */
   //@{
   template< typename... RTAs >
   explicit inline DQuatTransExprData( RTAs... args ) noexcept
   {
      MAYBE_UNUSED( args... );
   }

   DQuatTransExprData( const DQuatTransExprData& ) = default;
   //@}
   //**********************************************************************************************

   //**Destructor**********************************************************************************
   /*!\name Destructor */
   //@{
   ~DQuatTransExprData() = default;
   //@}
   //**********************************************************************************************

   //**Assignment operators************************************************************************
   /*!\name Assignment operators */
   //@{
   DQuatTransExprData& operator=( const DQuatTransExprData& ) = delete;
   //@}
   //**********************************************************************************************

   //**Utility functions***************************************************************************
   /*!\name Utility functions */
   //@{

   /*! \cond BLAZE_INTERNAL */
   /*!\brief Returns the mapped page indices
   //
   // \return The mapped page indices.
   */
   static inline constexpr Indices idces( ) noexcept
   {
      return Indices{0UL, 1UL, 3UL, 2UL};
   }

   /*! \cond BLAZE_INTERNAL */
   /*!\brief Returns the mapped page index of the specified element in the underlying tensor.
   //
   // \param l Quat access index.
   // \param k Page access index.
   // \param i Row access index.
   // \param j Column access index.
   // \return The mapped page index of the specified element.
   */
   static inline constexpr size_t quat  ( size_t l, size_t k, size_t i, size_t j ) noexcept
   {
      return l;
   }

   /*! \cond BLAZE_INTERNAL */
   /*!\brief Returns the mapped page index of the specified element in the underlying tensor.
   //
   // \param l Quat access index.
   // \param k Page access index.
   // \param i Row access index.
   // \param j Column access index.
   // \return The mapped page index of the specified element.
   */
   static inline constexpr size_t page  ( size_t l, size_t k, size_t i, size_t j ) noexcept
   {
      return k;
   }

   /*! \cond BLAZE_INTERNAL */
   /*!\brief Returns the mapped row index of the specified element in the underlying tensor.
   //
   // \param l Quat access index.
   // \param k Page access index.
   // \param i Row access index.
   // \param j Column access index.
   // \return The mapped row index of the specified element.
   */
   static inline constexpr size_t row   ( size_t l, size_t k, size_t i, size_t j ) noexcept
   {
      return j;
   }

   /*! \cond BLAZE_INTERNAL */
   /*!\brief Returns the mapped column index of the specified element in the underlying tensor.
   //
   // \param l Quat access index.
   // \param k Page access index.
   // \param i Row access index.
   // \param j Column access index.
   // \return The mapped column index of the specified element.
   */
   static inline constexpr size_t column( size_t l, size_t k, size_t i, size_t j ) noexcept
   {
      return i;
   }
   //@}
   //**********************************************************************************************

   /*! \cond BLAZE_INTERNAL */
   /*!\brief Returns the reversely mapped quat index of the specified element in the underlying tensor.
   //
   // \param l Quat access index.
   // \param k Page access index.
   // \param i Row access index.
   // \param j Column access index.
   // \return The mapped page index of the specified element.
   */
   static inline constexpr size_t reverse_quat  ( size_t l, size_t k, size_t i, size_t j ) noexcept
   {
      return l;
   }

   /*! \cond BLAZE_INTERNAL */
   /*!\brief Returns the reversely mapped page index of the specified element in the underlying tensor.
   //
   // \param l Quat access index.
   // \param k Page access index.
   // \param i Row access index.
   // \param j Column access index.
   // \return The mapped page index of the specified element.
   */
   static inline constexpr size_t reverse_page  ( size_t l, size_t k, size_t i, size_t j ) noexcept
   {
      return k;
   }

   /*! \cond BLAZE_INTERNAL */
   /*!\brief Returns the reversely mapped row index of the specified element in the underlying tensor.
   //
   // \param l Quat access index.
   // \param k Page access index.
   // \param i Row access index.
   // \param j Column access index.
   // \return The mapped row index of the specified element.
   */
   static inline constexpr size_t reverse_row   ( size_t l, size_t k, size_t i, size_t j ) noexcept
   {
      return j;
   }

   /*! \cond BLAZE_INTERNAL */
   /*!\brief Returns the reversely mapped column index of the specified element in the underlying tensor.
   //
   // \param l Quat access index.
   // \param k Page access index.
   // \param i Row access index.
   // \param j Column access index.
   // \return The mapped column index of the specified element.
   */
   static inline constexpr size_t reverse_column( size_t l, size_t k, size_t i, size_t j ) noexcept
   {
      return i;
   }
   //@}
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************


template<>
struct DQuatTransExprData<0UL, 2UL, 1UL, 3UL>
{
 public:
   //**Type definitions****************************************************************************
   using Indices = std::array<size_t, 4UL>;  //!< Type of the container for transpose indices.
   //**********************************************************************************************

   //**Constructors********************************************************************************
   /*!\name Constructors */
   //@{
   template< typename... RTAs >
   explicit inline DQuatTransExprData( RTAs... args ) noexcept
   {
      MAYBE_UNUSED( args... );
   }

   DQuatTransExprData( const DQuatTransExprData& ) = default;
   //@}
   //**********************************************************************************************

   //**Destructor**********************************************************************************
   /*!\name Destructor */
   //@{
   ~DQuatTransExprData() = default;
   //@}
   //**********************************************************************************************

   //**Assignment operators************************************************************************
   /*!\name Assignment operators */
   //@{
   DQuatTransExprData& operator=( const DQuatTransExprData& ) = delete;
   //@}
   //**********************************************************************************************

   //**Utility functions***************************************************************************
   /*!\name Utility functions */
   //@{

   /*! \cond BLAZE_INTERNAL */
   /*!\brief Returns the mapped page indices
   //
   // \return The mapped page indices.
   */
   static inline constexpr Indices idces( ) noexcept
   {
      return Indices{0UL, 2UL, 1UL, 3UL};
   }

   /*! \cond BLAZE_INTERNAL */
   /*!\brief Returns the mapped page index of the specified element in the underlying tensor.
   //
   // \param l Quat access index.
   // \param k Page access index.
   // \param i Row access index.
   // \param j Column access index.
   // \return The mapped page index of the specified element.
   */
   static inline constexpr size_t quat  ( size_t l, size_t k, size_t i, size_t j ) noexcept
   {
      return l;
   }

   /*! \cond BLAZE_INTERNAL */
   /*!\brief Returns the mapped page index of the specified element in the underlying tensor.
   //
   // \param l Quat access index.
   // \param k Page access index.
   // \param i Row access index.
   // \param j Column access index.
   // \return The mapped page index of the specified element.
   */
   static inline constexpr size_t page  ( size_t l, size_t k, size_t i, size_t j ) noexcept
   {
      return i;
   }

   /*! \cond BLAZE_INTERNAL */
   /*!\brief Returns the mapped row index of the specified element in the underlying tensor.
   //
   // \param l Quat access index.
   // \param k Page access index.
   // \param i Row access index.
   // \param j Column access index.
   // \return The mapped row index of the specified element.
   */
   static inline constexpr size_t row   ( size_t l, size_t k, size_t i, size_t j ) noexcept
   {
      return k;
   }

   /*! \cond BLAZE_INTERNAL */
   /*!\brief Returns the mapped column index of the specified element in the underlying tensor.
   //
   // \param l Quat access index.
   // \param k Page access index.
   // \param i Row access index.
   // \param j Column access index.
   // \return The mapped column index of the specified element.
   */
   static inline constexpr size_t column( size_t l, size_t k, size_t i, size_t j ) noexcept
   {
      return j;
   }
   //@}
   //**********************************************************************************************

   /*! \cond BLAZE_INTERNAL */
   /*!\brief Returns the reversely mapped quat index of the specified element in the underlying tensor.
   //
   // \param l Quat access index.
   // \param k Page access index.
   // \param i Row access index.
   // \param j Column access index.
   // \return The mapped page index of the specified element.
   */
   static inline constexpr size_t reverse_quat  ( size_t l, size_t k, size_t i, size_t j ) noexcept
   {
      return l;
   }

   /*! \cond BLAZE_INTERNAL */
   /*!\brief Returns the reversely mapped page index of the specified element in the underlying tensor.
   //
   // \param l Quat access index.
   // \param k Page access index.
   // \param i Row access index.
   // \param j Column access index.
   // \return The mapped page index of the specified element.
   */
   static inline constexpr size_t reverse_page  ( size_t l, size_t k, size_t i, size_t j ) noexcept
   {
      return i;
   }

   /*! \cond BLAZE_INTERNAL */
   /*!\brief Returns the reversely mapped row index of the specified element in the underlying tensor.
   //
   // \param l Quat access index.
   // \param k Page access index.
   // \param i Row access index.
   // \param j Column access index.
   // \return The mapped row index of the specified element.
   */
   static inline constexpr size_t reverse_row   ( size_t l, size_t k, size_t i, size_t j ) noexcept
   {
      return k;
   }

   /*! \cond BLAZE_INTERNAL */
   /*!\brief Returns the reversely mapped column index of the specified element in the underlying tensor.
   //
   // \param l Quat access index.
   // \param k Page access index.
   // \param i Row access index.
   // \param j Column access index.
   // \return The mapped column index of the specified element.
   */
   static inline constexpr size_t reverse_column( size_t l, size_t k, size_t i, size_t j ) noexcept
   {
      return j;
   }
   //@}
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************

template<>
struct DQuatTransExprData<0UL, 2UL, 3UL, 1UL>
{
 public:
   //**Type definitions****************************************************************************
   using Indices = std::array<size_t, 4UL>;  //!< Type of the container for transpose indices.
   //**********************************************************************************************

   //**Constructors********************************************************************************
   /*!\name Constructors */
   //@{
   template< typename... RTAs >
   explicit inline DQuatTransExprData( RTAs... args ) noexcept
   {
      MAYBE_UNUSED( args... );
   }

   DQuatTransExprData( const DQuatTransExprData& ) = default;
   //@}
   //**********************************************************************************************

   //**Destructor**********************************************************************************
   /*!\name Destructor */
   //@{
   ~DQuatTransExprData() = default;
   //@}
   //**********************************************************************************************

   //**Assignment operators************************************************************************
   /*!\name Assignment operators */
   //@{
   DQuatTransExprData& operator=( const DQuatTransExprData& ) = delete;
   //@}
   //**********************************************************************************************

   //**Utility functions***************************************************************************
   /*!\name Utility functions */
   //@{

   /*! \cond BLAZE_INTERNAL */
   /*!\brief Returns the mapped page indices
   //
   // \return The mapped page indices.
   */
   static inline constexpr Indices idces( ) noexcept
   {
      return Indices{0UL, 2UL, 3UL, 1UL};
   }

   /*! \cond BLAZE_INTERNAL */
   /*!\brief Returns the mapped page index of the specified element in the underlying tensor.
   //
   // \param l Quat access index.
   // \param k Page access index.
   // \param i Row access index.
   // \param j Column access index.
   // \return The mapped page index of the specified element.
   */
   static inline constexpr size_t quat  ( size_t l, size_t k, size_t i, size_t j ) noexcept
   {
      return l;
   }

   /*! \cond BLAZE_INTERNAL */
   /*!\brief Returns the mapped page index of the specified element in the underlying tensor.
   //
   // \param l Quat access index.
   // \param k Page access index.
   // \param i Row access index.
   // \param j Column access index.
   // \return The mapped page index of the specified element.
   */
   static inline constexpr size_t page  ( size_t l, size_t k, size_t i, size_t j ) noexcept
   {
      return i;
   }

   /*! \cond BLAZE_INTERNAL */
   /*!\brief Returns the mapped row index of the specified element in the underlying tensor.
   //
   // \param l Quat access index.
   // \param k Page access index.
   // \param i Row access index.
   // \param j Column access index.
   // \return The mapped row index of the specified element.
   */
   static inline constexpr size_t row   ( size_t l, size_t k, size_t i, size_t j ) noexcept
   {
      return j;
   }

   /*! \cond BLAZE_INTERNAL */
   /*!\brief Returns the mapped column index of the specified element in the underlying tensor.
   //
   // \param l Quat access index.
   // \param k Page access index.
   // \param i Row access index.
   // \param j Column access index.
   // \return The mapped column index of the specified element.
   */
   static inline constexpr size_t column( size_t l, size_t k, size_t i, size_t j ) noexcept
   {
      return k;
   }
   //@}
   //**********************************************************************************************

   /*! \cond BLAZE_INTERNAL */
   /*!\brief Returns the reversely mapped quat index of the specified element in the underlying tensor.
   //
   // \param l Quat access index.
   // \param k Page access index.
   // \param i Row access index.
   // \param j Column access index.
   // \return The mapped page index of the specified element.
   */
   static inline constexpr size_t reverse_quat  ( size_t l, size_t k, size_t i, size_t j ) noexcept
   {
      return l;
   }

   /*! \cond BLAZE_INTERNAL */
   /*!\brief Returns the reversely mapped page index of the specified element in the underlying tensor.
   //
   // \param l Quat access index.
   // \param k Page access index.
   // \param i Row access index.
   // \param j Column access index.
   // \return The mapped page index of the specified element.
   */
   static inline constexpr size_t reverse_page  ( size_t l, size_t k, size_t i, size_t j ) noexcept
   {
      return j;
   }

   /*! \cond BLAZE_INTERNAL */
   /*!\brief Returns the reversely mapped row index of the specified element in the underlying tensor.
   //
   // \param l Quat access index.
   // \param k Page access index.
   // \param i Row access index.
   // \param j Column access index.
   // \return The mapped row index of the specified element.
   */
   static inline constexpr size_t reverse_row   ( size_t l, size_t k, size_t i, size_t j ) noexcept
   {
      return k;
   }

   /*! \cond BLAZE_INTERNAL */
   /*!\brief Returns the reversely mapped column index of the specified element in the underlying tensor.
   //
   // \param l Quat access index.
   // \param k Page access index.
   // \param i Row access index.
   // \param j Column access index.
   // \return The mapped column index of the specified element.
   */
   static inline constexpr size_t reverse_column( size_t l, size_t k, size_t i, size_t j ) noexcept
   {
      return i;
   }
   //@}
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************


template<>
struct DQuatTransExprData<0UL, 3UL, 1UL, 2UL>
{
 public:
   //**Type definitions****************************************************************************
   using Indices = std::array<size_t, 4UL>;  //!< Type of the container for transpose indices.
   //**********************************************************************************************

   //**Constructors********************************************************************************
   /*!\name Constructors */
   //@{
   template< typename... RTAs >
   explicit inline DQuatTransExprData( RTAs... args ) noexcept
   {
      MAYBE_UNUSED( args... );
   }

   DQuatTransExprData( const DQuatTransExprData& ) = default;
   //@}
   //**********************************************************************************************

   //**Destructor**********************************************************************************
   /*!\name Destructor */
   //@{
   ~DQuatTransExprData() = default;
   //@}
   //**********************************************************************************************

   //**Assignment operators************************************************************************
   /*!\name Assignment operators */
   //@{
   DQuatTransExprData& operator=( const DQuatTransExprData& ) = delete;
   //@}
   //**********************************************************************************************

   //**Utility functions***************************************************************************
   /*!\name Utility functions */
   //@{

   /*! \cond BLAZE_INTERNAL */
   /*!\brief Returns the mapped page indices
   //
   // \return The mapped page indices.
   */
   static inline constexpr Indices idces( ) noexcept
   {
      return Indices{0UL, 3UL, 1UL, 2UL};
   }

   /*! \cond BLAZE_INTERNAL */
   /*!\brief Returns the mapped page index of the specified element in the underlying tensor.
   //
   // \param l Quat access index.
   // \param k Page access index.
   // \param i Row access index.
   // \param j Column access index.
   // \return The mapped page index of the specified element.
   */
   static inline constexpr size_t quat  ( size_t l, size_t k, size_t i, size_t j ) noexcept
   {
      return l;
   }

   /*! \cond BLAZE_INTERNAL */
   /*!\brief Returns the mapped page index of the specified element in the underlying tensor.
   //
   // \param l Quat access index.
   // \param k Page access index.
   // \param i Row access index.
   // \param j Column access index.
   // \return The mapped page index of the specified element.
   */
   static inline constexpr size_t page  ( size_t l, size_t k, size_t i, size_t j ) noexcept
   {
      return j;
   }

   /*! \cond BLAZE_INTERNAL */
   /*!\brief Returns the mapped row index of the specified element in the underlying tensor.
   //
   // \param l Quat access index.
   // \param k Page access index.
   // \param i Row access index.
   // \param j Column access index.
   // \return The mapped row index of the specified element.
   */
   static inline constexpr size_t row   ( size_t l, size_t k, size_t i, size_t j ) noexcept
   {
      return k;
   }

   /*! \cond BLAZE_INTERNAL */
   /*!\brief Returns the mapped column index of the specified element in the underlying tensor.
   //
   // \param l Quat access index.
   // \param k Page access index.
   // \param i Row access index.
   // \param j Column access index.
   // \return The mapped column index of the specified element.
   */
   static inline constexpr size_t column( size_t l, size_t k, size_t i, size_t j ) noexcept
   {
      return i;
   }
   //@}
   //**********************************************************************************************

   /*! \cond BLAZE_INTERNAL */
   /*!\brief Returns the reversely mapped quat index of the specified element in the underlying tensor.
   //
   // \param l Quat access index.
   // \param k Page access index.
   // \param i Row access index.
   // \param j Column access index.
   // \return The mapped page index of the specified element.
   */
   static inline constexpr size_t reverse_quat  ( size_t l, size_t k, size_t i, size_t j ) noexcept
   {
      return l;
   }

   /*! \cond BLAZE_INTERNAL */
   /*!\brief Returns the reversely mapped page index of the specified element in the underlying tensor.
   //
   // \param l Quat access index.
   // \param k Page access index.
   // \param i Row access index.
   // \param j Column access index.
   // \return The mapped page index of the specified element.
   */
   static inline constexpr size_t reverse_page  ( size_t l, size_t k, size_t i, size_t j ) noexcept
   {
      return i;
   }

   /*! \cond BLAZE_INTERNAL */
   /*!\brief Returns the reversely mapped row index of the specified element in the underlying tensor.
   //
   // \param l Quat access index.
   // \param k Page access index.
   // \param i Row access index.
   // \param j Column access index.
   // \return The mapped row index of the specified element.
   */
   static inline constexpr size_t reverse_row   ( size_t l, size_t k, size_t i, size_t j ) noexcept
   {
      return j;
   }

   /*! \cond BLAZE_INTERNAL */
   /*!\brief Returns the reversely mapped column index of the specified element in the underlying tensor.
   //
   // \param l Quat access index.
   // \param k Page access index.
   // \param i Row access index.
   // \param j Column access index.
   // \return The mapped column index of the specified element.
   */
   static inline constexpr size_t reverse_column( size_t l, size_t k, size_t i, size_t j ) noexcept
   {
      return k;
   }
   //@}
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************

template<>
struct DQuatTransExprData<0UL, 3UL, 2UL, 1UL>
{
 public:
   //**Type definitions****************************************************************************
   using Indices = std::array<size_t, 4UL>;  //!< Type of the container for transpose indices.
   //**********************************************************************************************

   //**Constructors********************************************************************************
   /*!\name Constructors */
   //@{
   template< typename... RTAs >
   explicit inline DQuatTransExprData( RTAs... args ) noexcept
   {
      MAYBE_UNUSED( args... );
   }

   DQuatTransExprData( const DQuatTransExprData& ) = default;
   //@}
   //**********************************************************************************************

   //**Destructor**********************************************************************************
   /*!\name Destructor */
   //@{
   ~DQuatTransExprData() = default;
   //@}
   //**********************************************************************************************

   //**Assignment operators************************************************************************
   /*!\name Assignment operators */
   //@{
   DQuatTransExprData& operator=( const DQuatTransExprData& ) = delete;
   //@}
   //**********************************************************************************************

   //**Utility functions***************************************************************************
   /*!\name Utility functions */
   //@{

   /*! \cond BLAZE_INTERNAL */
   /*!\brief Returns the mapped page indices
   //
   // \return The mapped page indices.
   */
   static inline constexpr Indices idces( ) noexcept
   {
      return Indices{0UL, 3UL, 2UL, 1UL};
   }

   /*! \cond BLAZE_INTERNAL */
   /*!\brief Returns the mapped page index of the specified element in the underlying tensor.
   //
   // \param l Quat access index.
   // \param k Page access index.
   // \param i Row access index.
   // \param j Column access index.
   // \return The mapped page index of the specified element.
   */
   static inline constexpr size_t quat  ( size_t l, size_t k, size_t i, size_t j ) noexcept
   {
      return l;
   }

   /*! \cond BLAZE_INTERNAL */
   /*!\brief Returns the mapped page index of the specified element in the underlying tensor.
   //
   // \param l Quat access index.
   // \param k Page access index.
   // \param i Row access index.
   // \param j Column access index.
   // \return The mapped page index of the specified element.
   */
   static inline constexpr size_t page  ( size_t l, size_t k, size_t i, size_t j ) noexcept
   {
      return j;
   }

   /*! \cond BLAZE_INTERNAL */
   /*!\brief Returns the mapped row index of the specified element in the underlying tensor.
   //
   // \param l Quat access index.
   // \param k Page access index.
   // \param i Row access index.
   // \param j Column access index.
   // \return The mapped row index of the specified element.
   */
   static inline constexpr size_t row   ( size_t l, size_t k, size_t i, size_t j ) noexcept
   {
      return i;
   }

   /*! \cond BLAZE_INTERNAL */
   /*!\brief Returns the mapped column index of the specified element in the underlying tensor.
   //
   // \param l Quat access index.
   // \param k Page access index.
   // \param i Row access index.
   // \param j Column access index.
   // \return The mapped column index of the specified element.
   */
   static inline constexpr size_t column( size_t l, size_t k, size_t i, size_t j ) noexcept
   {
      return k;
   }
   //@}
   //**********************************************************************************************

   /*! \cond BLAZE_INTERNAL */
   /*!\brief Returns the reversely mapped quat index of the specified element in the underlying tensor.
   //
   // \param l Quat access index.
   // \param k Page access index.
   // \param i Row access index.
   // \param j Column access index.
   // \return The mapped page index of the specified element.
   */
   static inline constexpr size_t reverse_quat  ( size_t l, size_t k, size_t i, size_t j ) noexcept
   {
      return l;
   }

   /*! \cond BLAZE_INTERNAL */
   /*!\brief Returns the reversely mapped page index of the specified element in the underlying tensor.
   //
   // \param l Quat access index.
   // \param k Page access index.
   // \param i Row access index.
   // \param j Column access index.
   // \return The mapped page index of the specified element.
   */
   static inline constexpr size_t reverse_page  ( size_t l, size_t k, size_t i, size_t j ) noexcept
   {
      return j;
   }

   /*! \cond BLAZE_INTERNAL */
   /*!\brief Returns the reversely mapped row index of the specified element in the underlying tensor.
   //
   // \param l Quat access index.
   // \param k Page access index.
   // \param i Row access index.
   // \param j Column access index.
   // \return The mapped row index of the specified element.
   */
   static inline constexpr size_t reverse_row   ( size_t l, size_t k, size_t i, size_t j ) noexcept
   {
      return i;
   }

   /*! \cond BLAZE_INTERNAL */
   /*!\brief Returns the reversely mapped column index of the specified element in the underlying tensor.
   //
   // \param l Quat access index.
   // \param k Page access index.
   // \param i Row access index.
   // \param j Column access index.
   // \return The mapped column index of the specified element.
   */
   static inline constexpr size_t reverse_column( size_t l, size_t k, size_t i, size_t j ) noexcept
   {
      return k;
   }
   //@}
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************


template<>
struct DQuatTransExprData<1UL, 0UL, 2UL, 3UL>
{
 public:
   //**Type definitions****************************************************************************
   using Indices = std::array<size_t, 4UL>;  //!< Type of the container for transpose indices.
   //**********************************************************************************************

   //**Constructors********************************************************************************
   /*!\name Constructors */
   //@{
   template< typename... RTAs >
   explicit inline DQuatTransExprData( RTAs... args ) noexcept
   {
      MAYBE_UNUSED( args... );
   }

   DQuatTransExprData( const DQuatTransExprData& ) = default;
   //@}
   //**********************************************************************************************

   //**Destructor**********************************************************************************
   /*!\name Destructor */
   //@{
   ~DQuatTransExprData() = default;
   //@}
   //**********************************************************************************************

   //**Assignment operators************************************************************************
   /*!\name Assignment operators */
   //@{
   DQuatTransExprData& operator=( const DQuatTransExprData& ) = delete;
   //@}
   //**********************************************************************************************

   //**Utility functions***************************************************************************
   /*!\name Utility functions */
   //@{

   /*! \cond BLAZE_INTERNAL */
   /*!\brief Returns the mapped page indices
   //
   // \return The mapped page indices.
   */
   static inline constexpr Indices idces( ) noexcept
   {
      return Indices{1UL, 0UL, 2UL, 3UL};
   }

   /*! \cond BLAZE_INTERNAL */
   /*!\brief Returns the mapped page index of the specified element in the underlying tensor.
   //
   // \param l Quat access index.
   // \param k Page access index.
   // \param i Row access index.
   // \param j Column access index.
   // \return The mapped page index of the specified element.
   */
   static inline constexpr size_t quat  ( size_t l, size_t k, size_t i, size_t j ) noexcept
   {
      return k;
   }

   /*! \cond BLAZE_INTERNAL */
   /*!\brief Returns the mapped page index of the specified element in the underlying tensor.
   //
   // \param l Quat access index.
   // \param k Page access index.
   // \param i Row access index.
   // \param j Column access index.
   // \return The mapped page index of the specified element.
   */
   static inline constexpr size_t page  ( size_t l, size_t k, size_t i, size_t j ) noexcept
   {
      return l;
   }

   /*! \cond BLAZE_INTERNAL */
   /*!\brief Returns the mapped row index of the specified element in the underlying tensor.
   //
   // \param l Quat access index.
   // \param k Page access index.
   // \param i Row access index.
   // \param j Column access index.
   // \return The mapped row index of the specified element.
   */
   static inline constexpr size_t row   ( size_t l, size_t k, size_t i, size_t j ) noexcept
   {
      return i;
   }

   /*! \cond BLAZE_INTERNAL */
   /*!\brief Returns the mapped column index of the specified element in the underlying tensor.
   //
   // \param l Quat access index.
   // \param k Page access index.
   // \param i Row access index.
   // \param j Column access index.
   // \return The mapped column index of the specified element.
   */
   static inline constexpr size_t column( size_t l, size_t k, size_t i, size_t j ) noexcept
   {
      return j;
   }
   //@}
   //**********************************************************************************************

   /*! \cond BLAZE_INTERNAL */
   /*!\brief Returns the reversely mapped quat index of the specified element in the underlying tensor.
   //
   // \param l Quat access index.
   // \param k Page access index.
   // \param i Row access index.
   // \param j Column access index.
   // \return The mapped page index of the specified element.
   */
   static inline constexpr size_t reverse_quat  ( size_t l, size_t k, size_t i, size_t j ) noexcept
   {
      return k;
   }

   /*! \cond BLAZE_INTERNAL */
   /*!\brief Returns the reversely mapped page index of the specified element in the underlying tensor.
   //
   // \param l Quat access index.
   // \param k Page access index.
   // \param i Row access index.
   // \param j Column access index.
   // \return The mapped page index of the specified element.
   */
   static inline constexpr size_t reverse_page  ( size_t l, size_t k, size_t i, size_t j ) noexcept
   {
      return l;
   }

   /*! \cond BLAZE_INTERNAL */
   /*!\brief Returns the reversely mapped row index of the specified element in the underlying tensor.
   //
   // \param l Quat access index.
   // \param k Page access index.
   // \param i Row access index.
   // \param j Column access index.
   // \return The mapped row index of the specified element.
   */
   static inline constexpr size_t reverse_row   ( size_t l, size_t k, size_t i, size_t j ) noexcept
   {
      return i;
   }

   /*! \cond BLAZE_INTERNAL */
   /*!\brief Returns the reversely mapped column index of the specified element in the underlying tensor.
   //
   // \param l Quat access index.
   // \param k Page access index.
   // \param i Row access index.
   // \param j Column access index.
   // \return The mapped column index of the specified element.
   */
   static inline constexpr size_t reverse_column( size_t l, size_t k, size_t i, size_t j ) noexcept
   {
      return j;
   }
   //@}
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************



template<>
struct DQuatTransExprData<1UL, 0UL, 3UL, 2UL>
{
 public:
   //**Type definitions****************************************************************************
   using Indices = std::array<size_t, 4UL>;  //!< Type of the container for transpose indices.
   //**********************************************************************************************

   //**Constructors********************************************************************************
   /*!\name Constructors */
   //@{
   template< typename... RTAs >
   explicit inline DQuatTransExprData( RTAs... args ) noexcept
   {
      MAYBE_UNUSED( args... );
   }

   DQuatTransExprData( const DQuatTransExprData& ) = default;
   //@}
   //**********************************************************************************************

   //**Destructor**********************************************************************************
   /*!\name Destructor */
   //@{
   ~DQuatTransExprData() = default;
   //@}
   //**********************************************************************************************

   //**Assignment operators************************************************************************
   /*!\name Assignment operators */
   //@{
   DQuatTransExprData& operator=( const DQuatTransExprData& ) = delete;
   //@}
   //**********************************************************************************************

   //**Utility functions***************************************************************************
   /*!\name Utility functions */
   //@{

   /*! \cond BLAZE_INTERNAL */
   /*!\brief Returns the mapped page indices
   //
   // \return The mapped page indices.
   */
   static inline constexpr Indices idces( ) noexcept
   {
      return Indices{1UL, 0UL, 3UL, 2UL};
   }

   /*! \cond BLAZE_INTERNAL */
   /*!\brief Returns the mapped page index of the specified element in the underlying tensor.
   //
   // \param l Quat access index.
   // \param k Page access index.
   // \param i Row access index.
   // \param j Column access index.
   // \return The mapped page index of the specified element.
   */
   static inline constexpr size_t quat  ( size_t l, size_t k, size_t i, size_t j ) noexcept
   {
      return k;
   }

   /*! \cond BLAZE_INTERNAL */
   /*!\brief Returns the mapped page index of the specified element in the underlying tensor.
   //
   // \param l Quat access index.
   // \param k Page access index.
   // \param i Row access index.
   // \param j Column access index.
   // \return The mapped page index of the specified element.
   */
   static inline constexpr size_t page  ( size_t l, size_t k, size_t i, size_t j ) noexcept
   {
      return l;
   }

   /*! \cond BLAZE_INTERNAL */
   /*!\brief Returns the mapped row index of the specified element in the underlying tensor.
   //
   // \param l Quat access index.
   // \param k Page access index.
   // \param i Row access index.
   // \param j Column access index.
   // \return The mapped row index of the specified element.
   */
   static inline constexpr size_t row   ( size_t l, size_t k, size_t i, size_t j ) noexcept
   {
      return j;
   }

   /*! \cond BLAZE_INTERNAL */
   /*!\brief Returns the mapped column index of the specified element in the underlying tensor.
   //
   // \param l Quat access index.
   // \param k Page access index.
   // \param i Row access index.
   // \param j Column access index.
   // \return The mapped column index of the specified element.
   */
   static inline constexpr size_t column( size_t l, size_t k, size_t i, size_t j ) noexcept
   {
      return i;
   }
   //@}
   //**********************************************************************************************

   /*! \cond BLAZE_INTERNAL */
   /*!\brief Returns the reversely mapped quat index of the specified element in the underlying tensor.
   //
   // \param l Quat access index.
   // \param k Page access index.
   // \param i Row access index.
   // \param j Column access index.
   // \return The mapped page index of the specified element.
   */
   static inline constexpr size_t reverse_quat  ( size_t l, size_t k, size_t i, size_t j ) noexcept
   {
      return k;
   }

   /*! \cond BLAZE_INTERNAL */
   /*!\brief Returns the reversely mapped page index of the specified element in the underlying tensor.
   //
   // \param l Quat access index.
   // \param k Page access index.
   // \param i Row access index.
   // \param j Column access index.
   // \return The mapped page index of the specified element.
   */
   static inline constexpr size_t reverse_page  ( size_t l, size_t k, size_t i, size_t j ) noexcept
   {
      return l;
   }

   /*! \cond BLAZE_INTERNAL */
   /*!\brief Returns the reversely mapped row index of the specified element in the underlying tensor.
   //
   // \param l Quat access index.
   // \param k Page access index.
   // \param i Row access index.
   // \param j Column access index.
   // \return The mapped row index of the specified element.
   */
   static inline constexpr size_t reverse_row   ( size_t l, size_t k, size_t i, size_t j ) noexcept
   {
      return j;
   }

   /*! \cond BLAZE_INTERNAL */
   /*!\brief Returns the reversely mapped column index of the specified element in the underlying tensor.
   //
   // \param l Quat access index.
   // \param k Page access index.
   // \param i Row access index.
   // \param j Column access index.
   // \return The mapped column index of the specified element.
   */
   static inline constexpr size_t reverse_column( size_t l, size_t k, size_t i, size_t j ) noexcept
   {
      return i;
   }
   //@}
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************


template<>
struct DQuatTransExprData<1UL, 2UL, 0UL, 3UL>
{
 public:
   //**Type definitions****************************************************************************
   using Indices = std::array<size_t, 4UL>;  //!< Type of the container for transpose indices.
   //**********************************************************************************************

   //**Constructors********************************************************************************
   /*!\name Constructors */
   //@{
   template< typename... RTAs >
   explicit inline DQuatTransExprData( RTAs... args ) noexcept
   {
      MAYBE_UNUSED( args... );
   }

   DQuatTransExprData( const DQuatTransExprData& ) = default;
   //@}
   //**********************************************************************************************

   //**Destructor**********************************************************************************
   /*!\name Destructor */
   //@{
   ~DQuatTransExprData() = default;
   //@}
   //**********************************************************************************************

   //**Assignment operators************************************************************************
   /*!\name Assignment operators */
   //@{
   DQuatTransExprData& operator=( const DQuatTransExprData& ) = delete;
   //@}
   //**********************************************************************************************

   //**Utility functions***************************************************************************
   /*!\name Utility functions */
   //@{

   /*! \cond BLAZE_INTERNAL */
   /*!\brief Returns the mapped page indices
   //
   // \return The mapped page indices.
   */
   static inline constexpr Indices idces( ) noexcept
   {
      return Indices{1UL, 2UL, 0UL, 3UL};
   }

   /*! \cond BLAZE_INTERNAL */
   /*!\brief Returns the mapped page index of the specified element in the underlying tensor.
   //
   // \param l Quat access index.
   // \param k Page access index.
   // \param i Row access index.
   // \param j Column access index.
   // \return The mapped page index of the specified element.
   */
   static inline constexpr size_t quat  ( size_t l, size_t k, size_t i, size_t j ) noexcept
   {
      return k;
   }

   /*! \cond BLAZE_INTERNAL */
   /*!\brief Returns the mapped page index of the specified element in the underlying tensor.
   //
   // \param l Quat access index.
   // \param k Page access index.
   // \param i Row access index.
   // \param j Column access index.
   // \return The mapped page index of the specified element.
   */
   static inline constexpr size_t page  ( size_t l, size_t k, size_t i, size_t j ) noexcept
   {
      return i;
   }

   /*! \cond BLAZE_INTERNAL */
   /*!\brief Returns the mapped row index of the specified element in the underlying tensor.
   //
   // \param l Quat access index.
   // \param k Page access index.
   // \param i Row access index.
   // \param j Column access index.
   // \return The mapped row index of the specified element.
   */
   static inline constexpr size_t row   ( size_t l, size_t k, size_t i, size_t j ) noexcept
   {
      return l;
   }

   /*! \cond BLAZE_INTERNAL */
   /*!\brief Returns the mapped column index of the specified element in the underlying tensor.
   //
   // \param l Quat access index.
   // \param k Page access index.
   // \param i Row access index.
   // \param j Column access index.
   // \return The mapped column index of the specified element.
   */
   static inline constexpr size_t column( size_t l, size_t k, size_t i, size_t j ) noexcept
   {
      return j;
   }
   //@}
   //**********************************************************************************************

   /*! \cond BLAZE_INTERNAL */
   /*!\brief Returns the reversely mapped quat index of the specified element in the underlying tensor.
   //
   // \param l Quat access index.
   // \param k Page access index.
   // \param i Row access index.
   // \param j Column access index.
   // \return The mapped page index of the specified element.
   */
   static inline constexpr size_t reverse_quat  ( size_t l, size_t k, size_t i, size_t j ) noexcept
   {
      return i;
   }

   /*! \cond BLAZE_INTERNAL */
   /*!\brief Returns the reversely mapped page index of the specified element in the underlying tensor.
   //
   // \param l Quat access index.
   // \param k Page access index.
   // \param i Row access index.
   // \param j Column access index.
   // \return The mapped page index of the specified element.
   */
   static inline constexpr size_t reverse_page  ( size_t l, size_t k, size_t i, size_t j ) noexcept
   {
      return l;
   }

   /*! \cond BLAZE_INTERNAL */
   /*!\brief Returns the reversely mapped row index of the specified element in the underlying tensor.
   //
   // \param l Quat access index.
   // \param k Page access index.
   // \param i Row access index.
   // \param j Column access index.
   // \return The mapped row index of the specified element.
   */
   static inline constexpr size_t reverse_row   ( size_t l, size_t k, size_t i, size_t j ) noexcept
   {
      return k;
   }

   /*! \cond BLAZE_INTERNAL */
   /*!\brief Returns the reversely mapped column index of the specified element in the underlying tensor.
   //
   // \param l Quat access index.
   // \param k Page access index.
   // \param i Row access index.
   // \param j Column access index.
   // \return The mapped column index of the specified element.
   */
   static inline constexpr size_t reverse_column( size_t l, size_t k, size_t i, size_t j ) noexcept
   {
      return j;
   }
   //@}
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************


template<>
struct DQuatTransExprData<1UL, 2UL, 3UL, 0UL>
{
 public:
   //**Type definitions****************************************************************************
   using Indices = std::array<size_t, 4UL>;  //!< Type of the container for transpose indices.
   //**********************************************************************************************

   //**Constructors********************************************************************************
   /*!\name Constructors */
   //@{
   template< typename... RTAs >
   explicit inline DQuatTransExprData( RTAs... args ) noexcept
   {
      MAYBE_UNUSED( args... );
   }

   DQuatTransExprData( const DQuatTransExprData& ) = default;
   //@}
   //**********************************************************************************************

   //**Destructor**********************************************************************************
   /*!\name Destructor */
   //@{
   ~DQuatTransExprData() = default;
   //@}
   //**********************************************************************************************

   //**Assignment operators************************************************************************
   /*!\name Assignment operators */
   //@{
   DQuatTransExprData& operator=( const DQuatTransExprData& ) = delete;
   //@}
   //**********************************************************************************************

   //**Utility functions***************************************************************************
   /*!\name Utility functions */
   //@{

   /*! \cond BLAZE_INTERNAL */
   /*!\brief Returns the mapped page indices
   //
   // \return The mapped page indices.
   */
   static inline constexpr Indices idces( ) noexcept
   {
      return Indices{1UL, 2UL, 3UL, 0UL};
   }

   /*! \cond BLAZE_INTERNAL */
   /*!\brief Returns the mapped page index of the specified element in the underlying tensor.
   //
   // \param l Quat access index.
   // \param k Page access index.
   // \param i Row access index.
   // \param j Column access index.
   // \return The mapped page index of the specified element.
   */
   static inline constexpr size_t quat  ( size_t l, size_t k, size_t i, size_t j ) noexcept
   {
      return k;
   }

   /*! \cond BLAZE_INTERNAL */
   /*!\brief Returns the mapped page index of the specified element in the underlying tensor.
   //
   // \param l Quat access index.
   // \param k Page access index.
   // \param i Row access index.
   // \param j Column access index.
   // \return The mapped page index of the specified element.
   */
   static inline constexpr size_t page  ( size_t l, size_t k, size_t i, size_t j ) noexcept
   {
      return i;
   }

   /*! \cond BLAZE_INTERNAL */
   /*!\brief Returns the mapped row index of the specified element in the underlying tensor.
   //
   // \param l Quat access index.
   // \param k Page access index.
   // \param i Row access index.
   // \param j Column access index.
   // \return The mapped row index of the specified element.
   */
   static inline constexpr size_t row   ( size_t l, size_t k, size_t i, size_t j ) noexcept
   {
      return j;
   }

   /*! \cond BLAZE_INTERNAL */
   /*!\brief Returns the mapped column index of the specified element in the underlying tensor.
   //
   // \param l Quat access index.
   // \param k Page access index.
   // \param i Row access index.
   // \param j Column access index.
   // \return The mapped column index of the specified element.
   */
   static inline constexpr size_t column( size_t l, size_t k, size_t i, size_t j ) noexcept
   {
      return l;
   }
   //@}
   //**********************************************************************************************

   /*! \cond BLAZE_INTERNAL */
   /*!\brief Returns the reversely mapped quat index of the specified element in the underlying tensor.
   //
   // \param l Quat access index.
   // \param k Page access index.
   // \param i Row access index.
   // \param j Column access index.
   // \return The mapped page index of the specified element.
   */
   static inline constexpr size_t reverse_quat  ( size_t l, size_t k, size_t i, size_t j ) noexcept
   {
      return j;
   }

   /*! \cond BLAZE_INTERNAL */
   /*!\brief Returns the reversely mapped page index of the specified element in the underlying tensor.
   //
   // \param l Quat access index.
   // \param k Page access index.
   // \param i Row access index.
   // \param j Column access index.
   // \return The mapped page index of the specified element.
   */
   static inline constexpr size_t reverse_page  ( size_t l, size_t k, size_t i, size_t j ) noexcept
   {
      return l;
   }

   /*! \cond BLAZE_INTERNAL */
   /*!\brief Returns the reversely mapped row index of the specified element in the underlying tensor.
   //
   // \param l Quat access index.
   // \param k Page access index.
   // \param i Row access index.
   // \param j Column access index.
   // \return The mapped row index of the specified element.
   */
   static inline constexpr size_t reverse_row   ( size_t l, size_t k, size_t i, size_t j ) noexcept
   {
      return k;
   }

   /*! \cond BLAZE_INTERNAL */
   /*!\brief Returns the reversely mapped column index of the specified element in the underlying tensor.
   //
   // \param l Quat access index.
   // \param k Page access index.
   // \param i Row access index.
   // \param j Column access index.
   // \return The mapped column index of the specified element.
   */
   static inline constexpr size_t reverse_column( size_t l, size_t k, size_t i, size_t j ) noexcept
   {
      return i;
   }
   //@}
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************



template<>
struct DQuatTransExprData<1UL, 3UL, 0UL, 2UL>
{
 public:
   //**Type definitions****************************************************************************
   using Indices = std::array<size_t, 4UL>;  //!< Type of the container for transpose indices.
   //**********************************************************************************************

   //**Constructors********************************************************************************
   /*!\name Constructors */
   //@{
   template< typename... RTAs >
   explicit inline DQuatTransExprData( RTAs... args ) noexcept
   {
      MAYBE_UNUSED( args... );
   }

   DQuatTransExprData( const DQuatTransExprData& ) = default;
   //@}
   //**********************************************************************************************

   //**Destructor**********************************************************************************
   /*!\name Destructor */
   //@{
   ~DQuatTransExprData() = default;
   //@}
   //**********************************************************************************************

   //**Assignment operators************************************************************************
   /*!\name Assignment operators */
   //@{
   DQuatTransExprData& operator=( const DQuatTransExprData& ) = delete;
   //@}
   //**********************************************************************************************

   //**Utility functions***************************************************************************
   /*!\name Utility functions */
   //@{

   /*! \cond BLAZE_INTERNAL */
   /*!\brief Returns the mapped page indices
   //
   // \return The mapped page indices.
   */
   static inline constexpr Indices idces( ) noexcept
   {
      return Indices{1UL, 3UL, 0UL, 2UL};
   }

   /*! \cond BLAZE_INTERNAL */
   /*!\brief Returns the mapped page index of the specified element in the underlying tensor.
   //
   // \param l Quat access index.
   // \param k Page access index.
   // \param i Row access index.
   // \param j Column access index.
   // \return The mapped page index of the specified element.
   */
   static inline constexpr size_t quat  ( size_t l, size_t k, size_t i, size_t j ) noexcept
   {
      return k;
   }

   /*! \cond BLAZE_INTERNAL */
   /*!\brief Returns the mapped page index of the specified element in the underlying tensor.
   //
   // \param l Quat access index.
   // \param k Page access index.
   // \param i Row access index.
   // \param j Column access index.
   // \return The mapped page index of the specified element.
   */
   static inline constexpr size_t page  ( size_t l, size_t k, size_t i, size_t j ) noexcept
   {
      return j;
   }

   /*! \cond BLAZE_INTERNAL */
   /*!\brief Returns the mapped row index of the specified element in the underlying tensor.
   //
   // \param l Quat access index.
   // \param k Page access index.
   // \param i Row access index.
   // \param j Column access index.
   // \return The mapped row index of the specified element.
   */
   static inline constexpr size_t row   ( size_t l, size_t k, size_t i, size_t j ) noexcept
   {
      return l;
   }

   /*! \cond BLAZE_INTERNAL */
   /*!\brief Returns the mapped column index of the specified element in the underlying tensor.
   //
   // \param l Quat access index.
   // \param k Page access index.
   // \param i Row access index.
   // \param j Column access index.
   // \return The mapped column index of the specified element.
   */
   static inline constexpr size_t column( size_t l, size_t k, size_t i, size_t j ) noexcept
   {
      return i;
   }
   //@}
   //**********************************************************************************************

   /*! \cond BLAZE_INTERNAL */
   /*!\brief Returns the reversely mapped quat index of the specified element in the underlying tensor.
   //
   // \param l Quat access index.
   // \param k Page access index.
   // \param i Row access index.
   // \param j Column access index.
   // \return The mapped page index of the specified element.
   */
   static inline constexpr size_t reverse_quat  ( size_t l, size_t k, size_t i, size_t j ) noexcept
   {
      return i;
   }

   /*! \cond BLAZE_INTERNAL */
   /*!\brief Returns the reversely mapped page index of the specified element in the underlying tensor.
   //
   // \param l Quat access index.
   // \param k Page access index.
   // \param i Row access index.
   // \param j Column access index.
   // \return The mapped page index of the specified element.
   */
   static inline constexpr size_t reverse_page  ( size_t l, size_t k, size_t i, size_t j ) noexcept
   {
      return l;
   }

   /*! \cond BLAZE_INTERNAL */
   /*!\brief Returns the reversely mapped row index of the specified element in the underlying tensor.
   //
   // \param l Quat access index.
   // \param k Page access index.
   // \param i Row access index.
   // \param j Column access index.
   // \return The mapped row index of the specified element.
   */
   static inline constexpr size_t reverse_row   ( size_t l, size_t k, size_t i, size_t j ) noexcept
   {
      return j;
   }

   /*! \cond BLAZE_INTERNAL */
   /*!\brief Returns the reversely mapped column index of the specified element in the underlying tensor.
   //
   // \param l Quat access index.
   // \param k Page access index.
   // \param i Row access index.
   // \param j Column access index.
   // \return The mapped column index of the specified element.
   */
   static inline constexpr size_t reverse_column( size_t l, size_t k, size_t i, size_t j ) noexcept
   {
      return k;
   }
   //@}
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************


template<>
struct DQuatTransExprData<2UL, 0UL, 1UL, 3UL>
{
 public:
   //**Type definitions****************************************************************************
   using Indices = std::array<size_t, 4UL>;  //!< Type of the container for transpose indices.
   //**********************************************************************************************

   //**Constructors********************************************************************************
   /*!\name Constructors */
   //@{
   template< typename... RTAs >
   explicit inline DQuatTransExprData( RTAs... args ) noexcept
   {
      MAYBE_UNUSED( args... );
   }

   DQuatTransExprData( const DQuatTransExprData& ) = default;
   //@}
   //**********************************************************************************************

   //**Destructor**********************************************************************************
   /*!\name Destructor */
   //@{
   ~DQuatTransExprData() = default;
   //@}
   //**********************************************************************************************

   //**Assignment operators************************************************************************
   /*!\name Assignment operators */
   //@{
   DQuatTransExprData& operator=( const DQuatTransExprData& ) = delete;
   //@}
   //**********************************************************************************************

   //**Utility functions***************************************************************************
   /*!\name Utility functions */
   //@{

   /*! \cond BLAZE_INTERNAL */
   /*!\brief Returns the mapped page indices
   //
   // \return The mapped page indices.
   */
   static inline constexpr Indices idces( ) noexcept
   {
      return Indices{2UL, 0UL, 1UL, 3UL};
   }

   /*! \cond BLAZE_INTERNAL */
   /*!\brief Returns the mapped page index of the specified element in the underlying tensor.
   //
   // \param l Quat access index.
   // \param k Page access index.
   // \param i Row access index.
   // \param j Column access index.
   // \return The mapped page index of the specified element.
   */
   static inline constexpr size_t quat  ( size_t l, size_t k, size_t i, size_t j ) noexcept
   {
      return i;
   }

   /*! \cond BLAZE_INTERNAL */
   /*!\brief Returns the mapped page index of the specified element in the underlying tensor.
   //
   // \param l Quat access index.
   // \param k Page access index.
   // \param i Row access index.
   // \param j Column access index.
   // \return The mapped page index of the specified element.
   */
   static inline constexpr size_t page  ( size_t l, size_t k, size_t i, size_t j ) noexcept
   {
      return l;
   }

   /*! \cond BLAZE_INTERNAL */
   /*!\brief Returns the mapped row index of the specified element in the underlying tensor.
   //
   // \param l Quat access index.
   // \param k Page access index.
   // \param i Row access index.
   // \param j Column access index.
   // \return The mapped row index of the specified element.
   */
   static inline constexpr size_t row   ( size_t l, size_t k, size_t i, size_t j ) noexcept
   {
      return k;
   }

   /*! \cond BLAZE_INTERNAL */
   /*!\brief Returns the mapped column index of the specified element in the underlying tensor.
   //
   // \param l Quat access index.
   // \param k Page access index.
   // \param i Row access index.
   // \param j Column access index.
   // \return The mapped column index of the specified element.
   */
   static inline constexpr size_t column( size_t l, size_t k, size_t i, size_t j ) noexcept
   {
      return j;
   }
   //@}
   //**********************************************************************************************

   /*! \cond BLAZE_INTERNAL */
   /*!\brief Returns the reversely mapped quat index of the specified element in the underlying tensor.
   //
   // \param l Quat access index.
   // \param k Page access index.
   // \param i Row access index.
   // \param j Column access index.
   // \return The mapped page index of the specified element.
   */
   static inline constexpr size_t reverse_quat  ( size_t l, size_t k, size_t i, size_t j ) noexcept
   {
      return k;
   }

   /*! \cond BLAZE_INTERNAL */
   /*!\brief Returns the reversely mapped page index of the specified element in the underlying tensor.
   //
   // \param l Quat access index.
   // \param k Page access index.
   // \param i Row access index.
   // \param j Column access index.
   // \return The mapped page index of the specified element.
   */
   static inline constexpr size_t reverse_page  ( size_t l, size_t k, size_t i, size_t j ) noexcept
   {
      return i;
   }

   /*! \cond BLAZE_INTERNAL */
   /*!\brief Returns the reversely mapped row index of the specified element in the underlying tensor.
   //
   // \param l Quat access index.
   // \param k Page access index.
   // \param i Row access index.
   // \param j Column access index.
   // \return The mapped row index of the specified element.
   */
   static inline constexpr size_t reverse_row   ( size_t l, size_t k, size_t i, size_t j ) noexcept
   {
      return l;
   }

   /*! \cond BLAZE_INTERNAL */
   /*!\brief Returns the reversely mapped column index of the specified element in the underlying tensor.
   //
   // \param l Quat access index.
   // \param k Page access index.
   // \param i Row access index.
   // \param j Column access index.
   // \return The mapped column index of the specified element.
   */
   static inline constexpr size_t reverse_column( size_t l, size_t k, size_t i, size_t j ) noexcept
   {
      return j;
   }
   //@}
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************

template<>
struct DQuatTransExprData<2UL, 0UL, 3UL, 1UL>
{
 public:
   //**Type definitions****************************************************************************
   using Indices = std::array<size_t, 4UL>;  //!< Type of the container for transpose indices.
   //**********************************************************************************************

   //**Constructors********************************************************************************
   /*!\name Constructors */
   //@{
   template< typename... RTAs >
   explicit inline DQuatTransExprData( RTAs... args ) noexcept
   {
      MAYBE_UNUSED( args... );
   }

   DQuatTransExprData( const DQuatTransExprData& ) = default;
   //@}
   //**********************************************************************************************

   //**Destructor**********************************************************************************
   /*!\name Destructor */
   //@{
   ~DQuatTransExprData() = default;
   //@}
   //**********************************************************************************************

   //**Assignment operators************************************************************************
   /*!\name Assignment operators */
   //@{
   DQuatTransExprData& operator=( const DQuatTransExprData& ) = delete;
   //@}
   //**********************************************************************************************

   //**Utility functions***************************************************************************
   /*!\name Utility functions */
   //@{

   /*! \cond BLAZE_INTERNAL */
   /*!\brief Returns the mapped page indices
   //
   // \return The mapped page indices.
   */
   static inline constexpr Indices idces( ) noexcept
   {
      return Indices{2UL, 0UL, 3UL, 1UL};
   }

   /*! \cond BLAZE_INTERNAL */
   /*!\brief Returns the mapped page index of the specified element in the underlying tensor.
   //
   // \param l Quat access index.
   // \param k Page access index.
   // \param i Row access index.
   // \param j Column access index.
   // \return The mapped page index of the specified element.
   */
   static inline constexpr size_t quat  ( size_t l, size_t k, size_t i, size_t j ) noexcept
   {
      return i;
   }

   /*! \cond BLAZE_INTERNAL */
   /*!\brief Returns the mapped page index of the specified element in the underlying tensor.
   //
   // \param l Quat access index.
   // \param k Page access index.
   // \param i Row access index.
   // \param j Column access index.
   // \return The mapped page index of the specified element.
   */
   static inline constexpr size_t page  ( size_t l, size_t k, size_t i, size_t j ) noexcept
   {
      return l;
   }

   /*! \cond BLAZE_INTERNAL */
   /*!\brief Returns the mapped row index of the specified element in the underlying tensor.
   //
   // \param l Quat access index.
   // \param k Page access index.
   // \param i Row access index.
   // \param j Column access index.
   // \return The mapped row index of the specified element.
   */
   static inline constexpr size_t row   ( size_t l, size_t k, size_t i, size_t j ) noexcept
   {
      return j;
   }

   /*! \cond BLAZE_INTERNAL */
   /*!\brief Returns the mapped column index of the specified element in the underlying tensor.
   //
   // \param l Quat access index.
   // \param k Page access index.
   // \param i Row access index.
   // \param j Column access index.
   // \return The mapped column index of the specified element.
   */
   static inline constexpr size_t column( size_t l, size_t k, size_t i, size_t j ) noexcept
   {
      return k;
   }
   //@}
   //**********************************************************************************************

   /*! \cond BLAZE_INTERNAL */
   /*!\brief Returns the reversely mapped quat index of the specified element in the underlying tensor.
   //
   // \param l Quat access index.
   // \param k Page access index.
   // \param i Row access index.
   // \param j Column access index.
   // \return The mapped page index of the specified element.
   */
   static inline constexpr size_t reverse_quat  ( size_t l, size_t k, size_t i, size_t j ) noexcept
   {
      return k;
   }

   /*! \cond BLAZE_INTERNAL */
   /*!\brief Returns the reversely mapped page index of the specified element in the underlying tensor.
   //
   // \param l Quat access index.
   // \param k Page access index.
   // \param i Row access index.
   // \param j Column access index.
   // \return The mapped page index of the specified element.
   */
   static inline constexpr size_t reverse_page  ( size_t l, size_t k, size_t i, size_t j ) noexcept
   {
      return j;
   }

   /*! \cond BLAZE_INTERNAL */
   /*!\brief Returns the reversely mapped row index of the specified element in the underlying tensor.
   //
   // \param l Quat access index.
   // \param k Page access index.
   // \param i Row access index.
   // \param j Column access index.
   // \return The mapped row index of the specified element.
   */
   static inline constexpr size_t reverse_row   ( size_t l, size_t k, size_t i, size_t j ) noexcept
   {
      return l;
   }

   /*! \cond BLAZE_INTERNAL */
   /*!\brief Returns the reversely mapped column index of the specified element in the underlying tensor.
   //
   // \param l Quat access index.
   // \param k Page access index.
   // \param i Row access index.
   // \param j Column access index.
   // \return The mapped column index of the specified element.
   */
   static inline constexpr size_t reverse_column( size_t l, size_t k, size_t i, size_t j ) noexcept
   {
      return i;
   }
   //@}
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************


template<>
struct DQuatTransExprData<2UL, 1UL, 0UL, 3UL>
{
 public:
   //**Type definitions****************************************************************************
   using Indices = std::array<size_t, 4UL>;  //!< Type of the container for transpose indices.
   //**********************************************************************************************

   //**Constructors********************************************************************************
   /*!\name Constructors */
   //@{
   template< typename... RTAs >
   explicit inline DQuatTransExprData( RTAs... args ) noexcept
   {
      MAYBE_UNUSED( args... );
   }

   DQuatTransExprData( const DQuatTransExprData& ) = default;
   //@}
   //**********************************************************************************************

   //**Destructor**********************************************************************************
   /*!\name Destructor */
   //@{
   ~DQuatTransExprData() = default;
   //@}
   //**********************************************************************************************

   //**Assignment operators************************************************************************
   /*!\name Assignment operators */
   //@{
   DQuatTransExprData& operator=( const DQuatTransExprData& ) = delete;
   //@}
   //**********************************************************************************************

   //**Utility functions***************************************************************************
   /*!\name Utility functions */
   //@{

   /*! \cond BLAZE_INTERNAL */
   /*!\brief Returns the mapped page indices
   //
   // \return The mapped page indices.
   */
   static inline constexpr Indices idces( ) noexcept
   {
      return Indices{2UL, 1UL, 0UL, 3UL};
   }

   /*! \cond BLAZE_INTERNAL */
   /*!\brief Returns the mapped page index of the specified element in the underlying tensor.
   //
   // \param l Quat access index.
   // \param k Page access index.
   // \param i Row access index.
   // \param j Column access index.
   // \return The mapped page index of the specified element.
   */
   static inline constexpr size_t quat  ( size_t l, size_t k, size_t i, size_t j ) noexcept
   {
      return i;
   }

   /*! \cond BLAZE_INTERNAL */
   /*!\brief Returns the mapped page index of the specified element in the underlying tensor.
   //
   // \param l Quat access index.
   // \param k Page access index.
   // \param i Row access index.
   // \param j Column access index.
   // \return The mapped page index of the specified element.
   */
   static inline constexpr size_t page  ( size_t l, size_t k, size_t i, size_t j ) noexcept
   {
      return k;
   }

   /*! \cond BLAZE_INTERNAL */
   /*!\brief Returns the mapped row index of the specified element in the underlying tensor.
   //
   // \param l Quat access index.
   // \param k Page access index.
   // \param i Row access index.
   // \param j Column access index.
   // \return The mapped row index of the specified element.
   */
   static inline constexpr size_t row   ( size_t l, size_t k, size_t i, size_t j ) noexcept
   {
      return l;
   }

   /*! \cond BLAZE_INTERNAL */
   /*!\brief Returns the mapped column index of the specified element in the underlying tensor.
   //
   // \param l Quat access index.
   // \param k Page access index.
   // \param i Row access index.
   // \param j Column access index.
   // \return The mapped column index of the specified element.
   */
   static inline constexpr size_t column( size_t l, size_t k, size_t i, size_t j ) noexcept
   {
      return j;
   }
   //@}
   //**********************************************************************************************

   /*! \cond BLAZE_INTERNAL */
   /*!\brief Returns the reversely mapped quat index of the specified element in the underlying tensor.
   //
   // \param l Quat access index.
   // \param k Page access index.
   // \param i Row access index.
   // \param j Column access index.
   // \return The mapped page index of the specified element.
   */
   static inline constexpr size_t reverse_quat  ( size_t l, size_t k, size_t i, size_t j ) noexcept
   {
      return i;
   }

   /*! \cond BLAZE_INTERNAL */
   /*!\brief Returns the reversely mapped page index of the specified element in the underlying tensor.
   //
   // \param l Quat access index.
   // \param k Page access index.
   // \param i Row access index.
   // \param j Column access index.
   // \return The mapped page index of the specified element.
   */
   static inline constexpr size_t reverse_page  ( size_t l, size_t k, size_t i, size_t j ) noexcept
   {
      return k;
   }

   /*! \cond BLAZE_INTERNAL */
   /*!\brief Returns the reversely mapped row index of the specified element in the underlying tensor.
   //
   // \param l Quat access index.
   // \param k Page access index.
   // \param i Row access index.
   // \param j Column access index.
   // \return The mapped row index of the specified element.
   */
   static inline constexpr size_t reverse_row   ( size_t l, size_t k, size_t i, size_t j ) noexcept
   {
      return l;
   }

   /*! \cond BLAZE_INTERNAL */
   /*!\brief Returns the reversely mapped column index of the specified element in the underlying tensor.
   //
   // \param l Quat access index.
   // \param k Page access index.
   // \param i Row access index.
   // \param j Column access index.
   // \return The mapped column index of the specified element.
   */
   static inline constexpr size_t reverse_column( size_t l, size_t k, size_t i, size_t j ) noexcept
   {
      return j;
   }
   //@}
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************


template<>
struct DQuatTransExprData<2UL, 1UL, 3UL, 0UL>
{
 public:
   //**Type definitions****************************************************************************
   using Indices = std::array<size_t, 4UL>;  //!< Type of the container for transpose indices.
   //**********************************************************************************************

   //**Constructors********************************************************************************
   /*!\name Constructors */
   //@{
   template< typename... RTAs >
   explicit inline DQuatTransExprData( RTAs... args ) noexcept
   {
      MAYBE_UNUSED( args... );
   }

   DQuatTransExprData( const DQuatTransExprData& ) = default;
   //@}
   //**********************************************************************************************

   //**Destructor**********************************************************************************
   /*!\name Destructor */
   //@{
   ~DQuatTransExprData() = default;
   //@}
   //**********************************************************************************************

   //**Assignment operators************************************************************************
   /*!\name Assignment operators */
   //@{
   DQuatTransExprData& operator=( const DQuatTransExprData& ) = delete;
   //@}
   //**********************************************************************************************

   //**Utility functions***************************************************************************
   /*!\name Utility functions */
   //@{

   /*! \cond BLAZE_INTERNAL */
   /*!\brief Returns the mapped page indices
   //
   // \return The mapped page indices.
   */
   static inline constexpr Indices idces( ) noexcept
   {
      return Indices{2UL, 1UL, 3UL, 0UL};
   }

   /*! \cond BLAZE_INTERNAL */
   /*!\brief Returns the mapped page index of the specified element in the underlying tensor.
   //
   // \param l Quat access index.
   // \param k Page access index.
   // \param i Row access index.
   // \param j Column access index.
   // \return The mapped page index of the specified element.
   */
   static inline constexpr size_t quat  ( size_t l, size_t k, size_t i, size_t j ) noexcept
   {
      return i;
   }

   /*! \cond BLAZE_INTERNAL */
   /*!\brief Returns the mapped page index of the specified element in the underlying tensor.
   //
   // \param l Quat access index.
   // \param k Page access index.
   // \param i Row access index.
   // \param j Column access index.
   // \return The mapped page index of the specified element.
   */
   static inline constexpr size_t page  ( size_t l, size_t k, size_t i, size_t j ) noexcept
   {
      return k;
   }

   /*! \cond BLAZE_INTERNAL */
   /*!\brief Returns the mapped row index of the specified element in the underlying tensor.
   //
   // \param l Quat access index.
   // \param k Page access index.
   // \param i Row access index.
   // \param j Column access index.
   // \return The mapped row index of the specified element.
   */
   static inline constexpr size_t row   ( size_t l, size_t k, size_t i, size_t j ) noexcept
   {
      return j;
   }

   /*! \cond BLAZE_INTERNAL */
   /*!\brief Returns the mapped column index of the specified element in the underlying tensor.
   //
   // \param l Quat access index.
   // \param k Page access index.
   // \param i Row access index.
   // \param j Column access index.
   // \return The mapped column index of the specified element.
   */
   static inline constexpr size_t column( size_t l, size_t k, size_t i, size_t j ) noexcept
   {
      return l;
   }
   //@}
   //**********************************************************************************************

   /*! \cond BLAZE_INTERNAL */
   /*!\brief Returns the reversely mapped quat index of the specified element in the underlying tensor.
   //
   // \param l Quat access index.
   // \param k Page access index.
   // \param i Row access index.
   // \param j Column access index.
   // \return The mapped page index of the specified element.
   */
   static inline constexpr size_t reverse_quat  ( size_t l, size_t k, size_t i, size_t j ) noexcept
   {
      return j;
   }

   /*! \cond BLAZE_INTERNAL */
   /*!\brief Returns the reversely mapped page index of the specified element in the underlying tensor.
   //
   // \param l Quat access index.
   // \param k Page access index.
   // \param i Row access index.
   // \param j Column access index.
   // \return The mapped page index of the specified element.
   */
   static inline constexpr size_t reverse_page  ( size_t l, size_t k, size_t i, size_t j ) noexcept
   {
      return k;
   }

   /*! \cond BLAZE_INTERNAL */
   /*!\brief Returns the reversely mapped row index of the specified element in the underlying tensor.
   //
   // \param l Quat access index.
   // \param k Page access index.
   // \param i Row access index.
   // \param j Column access index.
   // \return The mapped row index of the specified element.
   */
   static inline constexpr size_t reverse_row   ( size_t l, size_t k, size_t i, size_t j ) noexcept
   {
      return l;
   }

   /*! \cond BLAZE_INTERNAL */
   /*!\brief Returns the reversely mapped column index of the specified element in the underlying tensor.
   //
   // \param l Quat access index.
   // \param k Page access index.
   // \param i Row access index.
   // \param j Column access index.
   // \return The mapped column index of the specified element.
   */
   static inline constexpr size_t reverse_column( size_t l, size_t k, size_t i, size_t j ) noexcept
   {
      return i;
   }
   //@}
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************


template<>
struct DQuatTransExprData<2UL, 3UL, 0UL, 1UL>
{
 public:
   //**Type definitions****************************************************************************
   using Indices = std::array<size_t, 4UL>;  //!< Type of the container for transpose indices.
   //**********************************************************************************************

   //**Constructors********************************************************************************
   /*!\name Constructors */
   //@{
   template< typename... RTAs >
   explicit inline DQuatTransExprData( RTAs... args ) noexcept
   {
      MAYBE_UNUSED( args... );
   }

   DQuatTransExprData( const DQuatTransExprData& ) = default;
   //@}
   //**********************************************************************************************

   //**Destructor**********************************************************************************
   /*!\name Destructor */
   //@{
   ~DQuatTransExprData() = default;
   //@}
   //**********************************************************************************************

   //**Assignment operators************************************************************************
   /*!\name Assignment operators */
   //@{
   DQuatTransExprData& operator=( const DQuatTransExprData& ) = delete;
   //@}
   //**********************************************************************************************

   //**Utility functions***************************************************************************
   /*!\name Utility functions */
   //@{

   /*! \cond BLAZE_INTERNAL */
   /*!\brief Returns the mapped page indices
   //
   // \return The mapped page indices.
   */
   static inline constexpr Indices idces( ) noexcept
   {
      return Indices{2UL, 3UL, 0UL, 1UL};
   }

   /*! \cond BLAZE_INTERNAL */
   /*!\brief Returns the mapped page index of the specified element in the underlying tensor.
   //
   // \param l Quat access index.
   // \param k Page access index.
   // \param i Row access index.
   // \param j Column access index.
   // \return The mapped page index of the specified element.
   */
   static inline constexpr size_t quat  ( size_t l, size_t k, size_t i, size_t j ) noexcept
   {
      return i;
   }

   /*! \cond BLAZE_INTERNAL */
   /*!\brief Returns the mapped page index of the specified element in the underlying tensor.
   //
   // \param l Quat access index.
   // \param k Page access index.
   // \param i Row access index.
   // \param j Column access index.
   // \return The mapped page index of the specified element.
   */
   static inline constexpr size_t page  ( size_t l, size_t k, size_t i, size_t j ) noexcept
   {
      return j;
   }

   /*! \cond BLAZE_INTERNAL */
   /*!\brief Returns the mapped row index of the specified element in the underlying tensor.
   //
   // \param l Quat access index.
   // \param k Page access index.
   // \param i Row access index.
   // \param j Column access index.
   // \return The mapped row index of the specified element.
   */
   static inline constexpr size_t row   ( size_t l, size_t k, size_t i, size_t j ) noexcept
   {
      return l;
   }

   /*! \cond BLAZE_INTERNAL */
   /*!\brief Returns the mapped column index of the specified element in the underlying tensor.
   //
   // \param l Quat access index.
   // \param k Page access index.
   // \param i Row access index.
   // \param j Column access index.
   // \return The mapped column index of the specified element.
   */
   static inline constexpr size_t column( size_t l, size_t k, size_t i, size_t j ) noexcept
   {
      return k;
   }
   //@}
   //**********************************************************************************************

   /*! \cond BLAZE_INTERNAL */
   /*!\brief Returns the reversely mapped quat index of the specified element in the underlying tensor.
   //
   // \param l Quat access index.
   // \param k Page access index.
   // \param i Row access index.
   // \param j Column access index.
   // \return The mapped page index of the specified element.
   */
   static inline constexpr size_t reverse_quat  ( size_t l, size_t k, size_t i, size_t j ) noexcept
   {
      return i;
   }

   /*! \cond BLAZE_INTERNAL */
   /*!\brief Returns the reversely mapped page index of the specified element in the underlying tensor.
   //
   // \param l Quat access index.
   // \param k Page access index.
   // \param i Row access index.
   // \param j Column access index.
   // \return The mapped page index of the specified element.
   */
   static inline constexpr size_t reverse_page  ( size_t l, size_t k, size_t i, size_t j ) noexcept
   {
      return j;
   }

   /*! \cond BLAZE_INTERNAL */
   /*!\brief Returns the reversely mapped row index of the specified element in the underlying tensor.
   //
   // \param l Quat access index.
   // \param k Page access index.
   // \param i Row access index.
   // \param j Column access index.
   // \return The mapped row index of the specified element.
   */
   static inline constexpr size_t reverse_row   ( size_t l, size_t k, size_t i, size_t j ) noexcept
   {
      return l;
   }

   /*! \cond BLAZE_INTERNAL */
   /*!\brief Returns the reversely mapped column index of the specified element in the underlying tensor.
   //
   // \param l Quat access index.
   // \param k Page access index.
   // \param i Row access index.
   // \param j Column access index.
   // \return The mapped column index of the specified element.
   */
   static inline constexpr size_t reverse_column( size_t l, size_t k, size_t i, size_t j ) noexcept
   {
      return k;
   }
   //@}
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************


template<>
struct DQuatTransExprData<2UL, 3UL, 1UL, 0UL>
{
 public:
   //**Type definitions****************************************************************************
   using Indices = std::array<size_t, 4UL>;  //!< Type of the container for transpose indices.
   //**********************************************************************************************

   //**Constructors********************************************************************************
   /*!\name Constructors */
   //@{
   template< typename... RTAs >
   explicit inline DQuatTransExprData( RTAs... args ) noexcept
   {
      MAYBE_UNUSED( args... );
   }

   DQuatTransExprData( const DQuatTransExprData& ) = default;
   //@}
   //**********************************************************************************************

   //**Destructor**********************************************************************************
   /*!\name Destructor */
   //@{
   ~DQuatTransExprData() = default;
   //@}
   //**********************************************************************************************

   //**Assignment operators************************************************************************
   /*!\name Assignment operators */
   //@{
   DQuatTransExprData& operator=( const DQuatTransExprData& ) = delete;
   //@}
   //**********************************************************************************************

   //**Utility functions***************************************************************************
   /*!\name Utility functions */
   //@{

   /*! \cond BLAZE_INTERNAL */
   /*!\brief Returns the mapped page indices
   //
   // \return The mapped page indices.
   */
   static inline constexpr Indices idces( ) noexcept
   {
      return Indices{2UL, 3UL, 1UL, 0UL};
   }

   /*! \cond BLAZE_INTERNAL */
   /*!\brief Returns the mapped page index of the specified element in the underlying tensor.
   //
   // \param l Quat access index.
   // \param k Page access index.
   // \param i Row access index.
   // \param j Column access index.
   // \return The mapped page index of the specified element.
   */
   static inline constexpr size_t quat  ( size_t l, size_t k, size_t i, size_t j ) noexcept
   {
      return i;
   }

   /*! \cond BLAZE_INTERNAL */
   /*!\brief Returns the mapped page index of the specified element in the underlying tensor.
   //
   // \param l Quat access index.
   // \param k Page access index.
   // \param i Row access index.
   // \param j Column access index.
   // \return The mapped page index of the specified element.
   */
   static inline constexpr size_t page  ( size_t l, size_t k, size_t i, size_t j ) noexcept
   {
      return j;
   }

   /*! \cond BLAZE_INTERNAL */
   /*!\brief Returns the mapped row index of the specified element in the underlying tensor.
   //
   // \param l Quat access index.
   // \param k Page access index.
   // \param i Row access index.
   // \param j Column access index.
   // \return The mapped row index of the specified element.
   */
   static inline constexpr size_t row   ( size_t l, size_t k, size_t i, size_t j ) noexcept
   {
      return k;
   }

   /*! \cond BLAZE_INTERNAL */
   /*!\brief Returns the mapped column index of the specified element in the underlying tensor.
   //
   // \param l Quat access index.
   // \param k Page access index.
   // \param i Row access index.
   // \param j Column access index.
   // \return The mapped column index of the specified element.
   */
   static inline constexpr size_t column( size_t l, size_t k, size_t i, size_t j ) noexcept
   {
      return l;
   }
   //@}
   //**********************************************************************************************

   /*! \cond BLAZE_INTERNAL */
   /*!\brief Returns the reversely mapped quat index of the specified element in the underlying tensor.
   //
   // \param l Quat access index.
   // \param k Page access index.
   // \param i Row access index.
   // \param j Column access index.
   // \return The mapped page index of the specified element.
   */
   static inline constexpr size_t reverse_quat  ( size_t l, size_t k, size_t i, size_t j ) noexcept
   {
      return j;
   }

   /*! \cond BLAZE_INTERNAL */
   /*!\brief Returns the reversely mapped page index of the specified element in the underlying tensor.
   //
   // \param l Quat access index.
   // \param k Page access index.
   // \param i Row access index.
   // \param j Column access index.
   // \return The mapped page index of the specified element.
   */
   static inline constexpr size_t reverse_page  ( size_t l, size_t k, size_t i, size_t j ) noexcept
   {
      return i;
   }

   /*! \cond BLAZE_INTERNAL */
   /*!\brief Returns the reversely mapped row index of the specified element in the underlying tensor.
   //
   // \param l Quat access index.
   // \param k Page access index.
   // \param i Row access index.
   // \param j Column access index.
   // \return The mapped row index of the specified element.
   */
   static inline constexpr size_t reverse_row   ( size_t l, size_t k, size_t i, size_t j ) noexcept
   {
      return l;
   }

   /*! \cond BLAZE_INTERNAL */
   /*!\brief Returns the reversely mapped column index of the specified element in the underlying tensor.
   //
   // \param l Quat access index.
   // \param k Page access index.
   // \param i Row access index.
   // \param j Column access index.
   // \return The mapped column index of the specified element.
   */
   static inline constexpr size_t reverse_column( size_t l, size_t k, size_t i, size_t j ) noexcept
   {
      return k;
   }
   //@}
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************


template<>
struct DQuatTransExprData<3UL, 0UL, 1UL, 2UL>
{
 public:
   //**Type definitions****************************************************************************
   using Indices = std::array<size_t, 4UL>;  //!< Type of the container for transpose indices.
   //**********************************************************************************************

   //**Constructors********************************************************************************
   /*!\name Constructors */
   //@{
   template< typename... RTAs >
   explicit inline DQuatTransExprData( RTAs... args ) noexcept
   {
      MAYBE_UNUSED( args... );
   }

   DQuatTransExprData( const DQuatTransExprData& ) = default;
   //@}
   //**********************************************************************************************

   //**Destructor**********************************************************************************
   /*!\name Destructor */
   //@{
   ~DQuatTransExprData() = default;
   //@}
   //**********************************************************************************************

   //**Assignment operators************************************************************************
   /*!\name Assignment operators */
   //@{
   DQuatTransExprData& operator=( const DQuatTransExprData& ) = delete;
   //@}
   //**********************************************************************************************

   //**Utility functions***************************************************************************
   /*!\name Utility functions */
   //@{

   /*! \cond BLAZE_INTERNAL */
   /*!\brief Returns the mapped page indices
   //
   // \return The mapped page indices.
   */
   static inline constexpr Indices idces( ) noexcept
   {
      return Indices{3UL, 0UL, 1UL, 2UL};
   }

   /*! \cond BLAZE_INTERNAL */
   /*!\brief Returns the mapped page index of the specified element in the underlying tensor.
   //
   // \param l Quat access index.
   // \param k Page access index.
   // \param i Row access index.
   // \param j Column access index.
   // \return The mapped page index of the specified element.
   */
   static inline constexpr size_t quat  ( size_t l, size_t k, size_t i, size_t j ) noexcept
   {
      return j;
   }

   /*! \cond BLAZE_INTERNAL */
   /*!\brief Returns the mapped page index of the specified element in the underlying tensor.
   //
   // \param l Quat access index.
   // \param k Page access index.
   // \param i Row access index.
   // \param j Column access index.
   // \return The mapped page index of the specified element.
   */
   static inline constexpr size_t page  ( size_t l, size_t k, size_t i, size_t j ) noexcept
   {
      return l;
   }

   /*! \cond BLAZE_INTERNAL */
   /*!\brief Returns the mapped row index of the specified element in the underlying tensor.
   //
   // \param l Quat access index.
   // \param k Page access index.
   // \param i Row access index.
   // \param j Column access index.
   // \return The mapped row index of the specified element.
   */
   static inline constexpr size_t row   ( size_t l, size_t k, size_t i, size_t j ) noexcept
   {
      return k;
   }

   /*! \cond BLAZE_INTERNAL */
   /*!\brief Returns the mapped column index of the specified element in the underlying tensor.
   //
   // \param l Quat access index.
   // \param k Page access index.
   // \param i Row access index.
   // \param j Column access index.
   // \return The mapped column index of the specified element.
   */
   static inline constexpr size_t column( size_t l, size_t k, size_t i, size_t j ) noexcept
   {
      return i;
   }
   //@}
   //**********************************************************************************************

   /*! \cond BLAZE_INTERNAL */
   /*!\brief Returns the reversely mapped quat index of the specified element in the underlying tensor.
   //
   // \param l Quat access index.
   // \param k Page access index.
   // \param i Row access index.
   // \param j Column access index.
   // \return The mapped page index of the specified element.
   */
   static inline constexpr size_t reverse_quat  ( size_t l, size_t k, size_t i, size_t j ) noexcept
   {
      return k;
   }

   /*! \cond BLAZE_INTERNAL */
   /*!\brief Returns the reversely mapped page index of the specified element in the underlying tensor.
   //
   // \param l Quat access index.
   // \param k Page access index.
   // \param i Row access index.
   // \param j Column access index.
   // \return The mapped page index of the specified element.
   */
   static inline constexpr size_t reverse_page  ( size_t l, size_t k, size_t i, size_t j ) noexcept
   {
      return i;
   }

   /*! \cond BLAZE_INTERNAL */
   /*!\brief Returns the reversely mapped row index of the specified element in the underlying tensor.
   //
   // \param l Quat access index.
   // \param k Page access index.
   // \param i Row access index.
   // \param j Column access index.
   // \return The mapped row index of the specified element.
   */
   static inline constexpr size_t reverse_row   ( size_t l, size_t k, size_t i, size_t j ) noexcept
   {
      return j;
   }

   /*! \cond BLAZE_INTERNAL */
   /*!\brief Returns the reversely mapped column index of the specified element in the underlying tensor.
   //
   // \param l Quat access index.
   // \param k Page access index.
   // \param i Row access index.
   // \param j Column access index.
   // \return The mapped column index of the specified element.
   */
   static inline constexpr size_t reverse_column( size_t l, size_t k, size_t i, size_t j ) noexcept
   {
      return l;
   }
   //@}
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************


template<>
struct DQuatTransExprData<3UL, 0UL, 2UL, 1UL>
{
 public:
   //**Type definitions****************************************************************************
   using Indices = std::array<size_t, 4UL>;  //!< Type of the container for transpose indices.
   //**********************************************************************************************

   //**Constructors********************************************************************************
   /*!\name Constructors */
   //@{
   template< typename... RTAs >
   explicit inline DQuatTransExprData( RTAs... args ) noexcept
   {
      MAYBE_UNUSED( args... );
   }

   DQuatTransExprData( const DQuatTransExprData& ) = default;
   //@}
   //**********************************************************************************************

   //**Destructor**********************************************************************************
   /*!\name Destructor */
   //@{
   ~DQuatTransExprData() = default;
   //@}
   //**********************************************************************************************

   //**Assignment operators************************************************************************
   /*!\name Assignment operators */
   //@{
   DQuatTransExprData& operator=( const DQuatTransExprData& ) = delete;
   //@}
   //**********************************************************************************************

   //**Utility functions***************************************************************************
   /*!\name Utility functions */
   //@{

   /*! \cond BLAZE_INTERNAL */
   /*!\brief Returns the mapped page indices
   //
   // \return The mapped page indices.
   */
   static inline constexpr Indices idces( ) noexcept
   {
      return Indices{3UL, 0UL, 2UL, 1UL};
   }

   /*! \cond BLAZE_INTERNAL */
   /*!\brief Returns the mapped page index of the specified element in the underlying tensor.
   //
   // \param l Quat access index.
   // \param k Page access index.
   // \param i Row access index.
   // \param j Column access index.
   // \return The mapped page index of the specified element.
   */
   static inline constexpr size_t quat  ( size_t l, size_t k, size_t i, size_t j ) noexcept
   {
      return j;
   }

   /*! \cond BLAZE_INTERNAL */
   /*!\brief Returns the mapped page index of the specified element in the underlying tensor.
   //
   // \param l Quat access index.
   // \param k Page access index.
   // \param i Row access index.
   // \param j Column access index.
   // \return The mapped page index of the specified element.
   */
   static inline constexpr size_t page  ( size_t l, size_t k, size_t i, size_t j ) noexcept
   {
      return l;
   }

   /*! \cond BLAZE_INTERNAL */
   /*!\brief Returns the mapped row index of the specified element in the underlying tensor.
   //
   // \param l Quat access index.
   // \param k Page access index.
   // \param i Row access index.
   // \param j Column access index.
   // \return The mapped row index of the specified element.
   */
   static inline constexpr size_t row   ( size_t l, size_t k, size_t i, size_t j ) noexcept
   {
      return i;
   }

   /*! \cond BLAZE_INTERNAL */
   /*!\brief Returns the mapped column index of the specified element in the underlying tensor.
   //
   // \param l Quat access index.
   // \param k Page access index.
   // \param i Row access index.
   // \param j Column access index.
   // \return The mapped column index of the specified element.
   */
   static inline constexpr size_t column( size_t l, size_t k, size_t i, size_t j ) noexcept
   {
      return k;
   }
   //@}
   //**********************************************************************************************

   /*! \cond BLAZE_INTERNAL */
   /*!\brief Returns the reversely mapped quat index of the specified element in the underlying tensor.
   //
   // \param l Quat access index.
   // \param k Page access index.
   // \param i Row access index.
   // \param j Column access index.
   // \return The mapped page index of the specified element.
   */
   static inline constexpr size_t reverse_quat  ( size_t l, size_t k, size_t i, size_t j ) noexcept
   {
      return k;
   }

   /*! \cond BLAZE_INTERNAL */
   /*!\brief Returns the reversely mapped page index of the specified element in the underlying tensor.
   //
   // \param l Quat access index.
   // \param k Page access index.
   // \param i Row access index.
   // \param j Column access index.
   // \return The mapped page index of the specified element.
   */
   static inline constexpr size_t reverse_page  ( size_t l, size_t k, size_t i, size_t j ) noexcept
   {
      return j;
   }

   /*! \cond BLAZE_INTERNAL */
   /*!\brief Returns the reversely mapped row index of the specified element in the underlying tensor.
   //
   // \param l Quat access index.
   // \param k Page access index.
   // \param i Row access index.
   // \param j Column access index.
   // \return The mapped row index of the specified element.
   */
   static inline constexpr size_t reverse_row   ( size_t l, size_t k, size_t i, size_t j ) noexcept
   {
      return i;
   }

   /*! \cond BLAZE_INTERNAL */
   /*!\brief Returns the reversely mapped column index of the specified element in the underlying tensor.
   //
   // \param l Quat access index.
   // \param k Page access index.
   // \param i Row access index.
   // \param j Column access index.
   // \return The mapped column index of the specified element.
   */
   static inline constexpr size_t reverse_column( size_t l, size_t k, size_t i, size_t j ) noexcept
   {
      return l;
   }
   //@}
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************


template<>
struct DQuatTransExprData<3UL, 1UL, 0UL, 2UL>
{
 public:
   //**Type definitions****************************************************************************
   using Indices = std::array<size_t, 4UL>;  //!< Type of the container for transpose indices.
   //**********************************************************************************************

   //**Constructors********************************************************************************
   /*!\name Constructors */
   //@{
   template< typename... RTAs >
   explicit inline DQuatTransExprData( RTAs... args ) noexcept
   {
      MAYBE_UNUSED( args... );
   }

   DQuatTransExprData( const DQuatTransExprData& ) = default;
   //@}
   //**********************************************************************************************

   //**Destructor**********************************************************************************
   /*!\name Destructor */
   //@{
   ~DQuatTransExprData() = default;
   //@}
   //**********************************************************************************************

   //**Assignment operators************************************************************************
   /*!\name Assignment operators */
   //@{
   DQuatTransExprData& operator=( const DQuatTransExprData& ) = delete;
   //@}
   //**********************************************************************************************

   //**Utility functions***************************************************************************
   /*!\name Utility functions */
   //@{

   /*! \cond BLAZE_INTERNAL */
   /*!\brief Returns the mapped page indices
   //
   // \return The mapped page indices.
   */
   static inline constexpr Indices idces( ) noexcept
   {
      return Indices{3UL, 1UL, 0UL, 2UL};
   }

   /*! \cond BLAZE_INTERNAL */
   /*!\brief Returns the mapped page index of the specified element in the underlying tensor.
   //
   // \param l Quat access index.
   // \param k Page access index.
   // \param i Row access index.
   // \param j Column access index.
   // \return The mapped page index of the specified element.
   */
   static inline constexpr size_t quat  ( size_t l, size_t k, size_t i, size_t j ) noexcept
   {
      return j;
   }

   /*! \cond BLAZE_INTERNAL */
   /*!\brief Returns the mapped page index of the specified element in the underlying tensor.
   //
   // \param l Quat access index.
   // \param k Page access index.
   // \param i Row access index.
   // \param j Column access index.
   // \return The mapped page index of the specified element.
   */
   static inline constexpr size_t page  ( size_t l, size_t k, size_t i, size_t j ) noexcept
   {
      return k;
   }

   /*! \cond BLAZE_INTERNAL */
   /*!\brief Returns the mapped row index of the specified element in the underlying tensor.
   //
   // \param l Quat access index.
   // \param k Page access index.
   // \param i Row access index.
   // \param j Column access index.
   // \return The mapped row index of the specified element.
   */
   static inline constexpr size_t row   ( size_t l, size_t k, size_t i, size_t j ) noexcept
   {
      return l;
   }

   /*! \cond BLAZE_INTERNAL */
   /*!\brief Returns the mapped column index of the specified element in the underlying tensor.
   //
   // \param l Quat access index.
   // \param k Page access index.
   // \param i Row access index.
   // \param j Column access index.
   // \return The mapped column index of the specified element.
   */
   static inline constexpr size_t column( size_t l, size_t k, size_t i, size_t j ) noexcept
   {
      return i;
   }
   //@}
   //**********************************************************************************************

   /*! \cond BLAZE_INTERNAL */
   /*!\brief Returns the reversely mapped quat index of the specified element in the underlying tensor.
   //
   // \param l Quat access index.
   // \param k Page access index.
   // \param i Row access index.
   // \param j Column access index.
   // \return The mapped page index of the specified element.
   */
   static inline constexpr size_t reverse_quat  ( size_t l, size_t k, size_t i, size_t j ) noexcept
   {
      return i;
   }

   /*! \cond BLAZE_INTERNAL */
   /*!\brief Returns the reversely mapped page index of the specified element in the underlying tensor.
   //
   // \param l Quat access index.
   // \param k Page access index.
   // \param i Row access index.
   // \param j Column access index.
   // \return The mapped page index of the specified element.
   */
   static inline constexpr size_t reverse_page  ( size_t l, size_t k, size_t i, size_t j ) noexcept
   {
      return k;
   }

   /*! \cond BLAZE_INTERNAL */
   /*!\brief Returns the reversely mapped row index of the specified element in the underlying tensor.
   //
   // \param l Quat access index.
   // \param k Page access index.
   // \param i Row access index.
   // \param j Column access index.
   // \return The mapped row index of the specified element.
   */
   static inline constexpr size_t reverse_row   ( size_t l, size_t k, size_t i, size_t j ) noexcept
   {
      return j;
   }

   /*! \cond BLAZE_INTERNAL */
   /*!\brief Returns the reversely mapped column index of the specified element in the underlying tensor.
   //
   // \param l Quat access index.
   // \param k Page access index.
   // \param i Row access index.
   // \param j Column access index.
   // \return The mapped column index of the specified element.
   */
   static inline constexpr size_t reverse_column( size_t l, size_t k, size_t i, size_t j ) noexcept
   {
      return l;
   }
   //@}
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************


template<>
struct DQuatTransExprData<3UL, 1UL, 2UL, 0UL>
{
 public:
   //**Type definitions****************************************************************************
   using Indices = std::array<size_t, 4UL>;  //!< Type of the container for transpose indices.
   //**********************************************************************************************

   //**Constructors********************************************************************************
   /*!\name Constructors */
   //@{
   template< typename... RTAs >
   explicit inline DQuatTransExprData( RTAs... args ) noexcept
   {
      MAYBE_UNUSED( args... );
   }

   DQuatTransExprData( const DQuatTransExprData& ) = default;
   //@}
   //**********************************************************************************************

   //**Destructor**********************************************************************************
   /*!\name Destructor */
   //@{
   ~DQuatTransExprData() = default;
   //@}
   //**********************************************************************************************

   //**Assignment operators************************************************************************
   /*!\name Assignment operators */
   //@{
   DQuatTransExprData& operator=( const DQuatTransExprData& ) = delete;
   //@}
   //**********************************************************************************************

   //**Utility functions***************************************************************************
   /*!\name Utility functions */
   //@{

   /*! \cond BLAZE_INTERNAL */
   /*!\brief Returns the mapped page indices
   //
   // \return The mapped page indices.
   */
   static inline constexpr Indices idces( ) noexcept
   {
      return Indices{3UL, 1UL, 2UL, 0UL};
   }

   /*! \cond BLAZE_INTERNAL */
   /*!\brief Returns the mapped page index of the specified element in the underlying tensor.
   //
   // \param l Quat access index.
   // \param k Page access index.
   // \param i Row access index.
   // \param j Column access index.
   // \return The mapped page index of the specified element.
   */
   static inline constexpr size_t quat  ( size_t l, size_t k, size_t i, size_t j ) noexcept
   {
      return j;
   }

   /*! \cond BLAZE_INTERNAL */
   /*!\brief Returns the mapped page index of the specified element in the underlying tensor.
   //
   // \param l Quat access index.
   // \param k Page access index.
   // \param i Row access index.
   // \param j Column access index.
   // \return The mapped page index of the specified element.
   */
   static inline constexpr size_t page  ( size_t l, size_t k, size_t i, size_t j ) noexcept
   {
      return k;
   }

   /*! \cond BLAZE_INTERNAL */
   /*!\brief Returns the mapped row index of the specified element in the underlying tensor.
   //
   // \param l Quat access index.
   // \param k Page access index.
   // \param i Row access index.
   // \param j Column access index.
   // \return The mapped row index of the specified element.
   */
   static inline constexpr size_t row   ( size_t l, size_t k, size_t i, size_t j ) noexcept
   {
      return i;
   }

   /*! \cond BLAZE_INTERNAL */
   /*!\brief Returns the mapped column index of the specified element in the underlying tensor.
   //
   // \param l Quat access index.
   // \param k Page access index.
   // \param i Row access index.
   // \param j Column access index.
   // \return The mapped column index of the specified element.
   */
   static inline constexpr size_t column( size_t l, size_t k, size_t i, size_t j ) noexcept
   {
      return l;
   }
   //@}
   //**********************************************************************************************

   /*! \cond BLAZE_INTERNAL */
   /*!\brief Returns the reversely mapped quat index of the specified element in the underlying tensor.
   //
   // \param l Quat access index.
   // \param k Page access index.
   // \param i Row access index.
   // \param j Column access index.
   // \return The mapped page index of the specified element.
   */
   static inline constexpr size_t reverse_quat  ( size_t l, size_t k, size_t i, size_t j ) noexcept
   {
      return j;
   }

   /*! \cond BLAZE_INTERNAL */
   /*!\brief Returns the reversely mapped page index of the specified element in the underlying tensor.
   //
   // \param l Quat access index.
   // \param k Page access index.
   // \param i Row access index.
   // \param j Column access index.
   // \return The mapped page index of the specified element.
   */
   static inline constexpr size_t reverse_page  ( size_t l, size_t k, size_t i, size_t j ) noexcept
   {
      return k;
   }

   /*! \cond BLAZE_INTERNAL */
   /*!\brief Returns the reversely mapped row index of the specified element in the underlying tensor.
   //
   // \param l Quat access index.
   // \param k Page access index.
   // \param i Row access index.
   // \param j Column access index.
   // \return The mapped row index of the specified element.
   */
   static inline constexpr size_t reverse_row   ( size_t l, size_t k, size_t i, size_t j ) noexcept
   {
      return i;
   }

   /*! \cond BLAZE_INTERNAL */
   /*!\brief Returns the reversely mapped column index of the specified element in the underlying tensor.
   //
   // \param l Quat access index.
   // \param k Page access index.
   // \param i Row access index.
   // \param j Column access index.
   // \return The mapped column index of the specified element.
   */
   static inline constexpr size_t reverse_column( size_t l, size_t k, size_t i, size_t j ) noexcept
   {
      return l;
   }
   //@}
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************


template<>
struct DQuatTransExprData<3UL, 2UL, 0UL, 1UL>
{
 public:
   //**Type definitions****************************************************************************
   using Indices = std::array<size_t, 4UL>;  //!< Type of the container for transpose indices.
   //**********************************************************************************************

   //**Constructors********************************************************************************
   /*!\name Constructors */
   //@{
   template< typename... RTAs >
   explicit inline DQuatTransExprData( RTAs... args ) noexcept
   {
      MAYBE_UNUSED( args... );
   }

   DQuatTransExprData( const DQuatTransExprData& ) = default;
   //@}
   //**********************************************************************************************

   //**Destructor**********************************************************************************
   /*!\name Destructor */
   //@{
   ~DQuatTransExprData() = default;
   //@}
   //**********************************************************************************************

   //**Assignment operators************************************************************************
   /*!\name Assignment operators */
   //@{
   DQuatTransExprData& operator=( const DQuatTransExprData& ) = delete;
   //@}
   //**********************************************************************************************

   //**Utility functions***************************************************************************
   /*!\name Utility functions */
   //@{

   /*! \cond BLAZE_INTERNAL */
   /*!\brief Returns the mapped page indices
   //
   // \return The mapped page indices.
   */
   static inline constexpr Indices idces( ) noexcept
   {
      return Indices{3UL, 2UL, 0UL, 1UL};
   }

   /*! \cond BLAZE_INTERNAL */
   /*!\brief Returns the mapped page index of the specified element in the underlying tensor.
   //
   // \param l Quat access index.
   // \param k Page access index.
   // \param i Row access index.
   // \param j Column access index.
   // \return The mapped page index of the specified element.
   */
   static inline constexpr size_t quat  ( size_t l, size_t k, size_t i, size_t j ) noexcept
   {
      return j;
   }

   /*! \cond BLAZE_INTERNAL */
   /*!\brief Returns the mapped page index of the specified element in the underlying tensor.
   //
   // \param l Quat access index.
   // \param k Page access index.
   // \param i Row access index.
   // \param j Column access index.
   // \return The mapped page index of the specified element.
   */
   static inline constexpr size_t page  ( size_t l, size_t k, size_t i, size_t j ) noexcept
   {
      return i;
   }

   /*! \cond BLAZE_INTERNAL */
   /*!\brief Returns the mapped row index of the specified element in the underlying tensor.
   //
   // \param l Quat access index.
   // \param k Page access index.
   // \param i Row access index.
   // \param j Column access index.
   // \return The mapped row index of the specified element.
   */
   static inline constexpr size_t row   ( size_t l, size_t k, size_t i, size_t j ) noexcept
   {
      return l;
   }

   /*! \cond BLAZE_INTERNAL */
   /*!\brief Returns the mapped column index of the specified element in the underlying tensor.
   //
   // \param l Quat access index.
   // \param k Page access index.
   // \param i Row access index.
   // \param j Column access index.
   // \return The mapped column index of the specified element.
   */
   static inline constexpr size_t column( size_t l, size_t k, size_t i, size_t j ) noexcept
   {
      return k;
   }
   //@}
   //**********************************************************************************************

   /*! \cond BLAZE_INTERNAL */
   /*!\brief Returns the reversely mapped quat index of the specified element in the underlying tensor.
   //
   // \param l Quat access index.
   // \param k Page access index.
   // \param i Row access index.
   // \param j Column access index.
   // \return The mapped page index of the specified element.
   */
   static inline constexpr size_t reverse_quat  ( size_t l, size_t k, size_t i, size_t j ) noexcept
   {
      return i;
   }

   /*! \cond BLAZE_INTERNAL */
   /*!\brief Returns the reversely mapped page index of the specified element in the underlying tensor.
   //
   // \param l Quat access index.
   // \param k Page access index.
   // \param i Row access index.
   // \param j Column access index.
   // \return The mapped page index of the specified element.
   */
   static inline constexpr size_t reverse_page  ( size_t l, size_t k, size_t i, size_t j ) noexcept
   {
      return j;
   }

   /*! \cond BLAZE_INTERNAL */
   /*!\brief Returns the reversely mapped row index of the specified element in the underlying tensor.
   //
   // \param l Quat access index.
   // \param k Page access index.
   // \param i Row access index.
   // \param j Column access index.
   // \return The mapped row index of the specified element.
   */
   static inline constexpr size_t reverse_row   ( size_t l, size_t k, size_t i, size_t j ) noexcept
   {
      return l;
   }

   /*! \cond BLAZE_INTERNAL */
   /*!\brief Returns the reversely mapped column index of the specified element in the underlying tensor.
   //
   // \param l Quat access index.
   // \param k Page access index.
   // \param i Row access index.
   // \param j Column access index.
   // \return The mapped column index of the specified element.
   */
   static inline constexpr size_t reverse_column( size_t l, size_t k, size_t i, size_t j ) noexcept
   {
      return k;
   }
   //@}
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************



template<>
struct DQuatTransExprData<3UL, 2UL, 1UL, 0UL>
{
 public:
   //**Type definitions****************************************************************************
   using Indices = std::array<size_t, 4UL>;  //!< Type of the container for transpose indices.
   //**********************************************************************************************

   //**Constructors********************************************************************************
   /*!\name Constructors */
   //@{
   template< typename... RTAs >
   explicit inline DQuatTransExprData( RTAs... args ) noexcept
   {
      MAYBE_UNUSED( args... );
   }

   DQuatTransExprData( const DQuatTransExprData& ) = default;
   //@}
   //**********************************************************************************************

   //**Destructor**********************************************************************************
   /*!\name Destructor */
   //@{
   ~DQuatTransExprData() = default;
   //@}
   //**********************************************************************************************

   //**Assignment operators************************************************************************
   /*!\name Assignment operators */
   //@{
   DQuatTransExprData& operator=( const DQuatTransExprData& ) = delete;
   //@}
   //**********************************************************************************************

   //**Utility functions***************************************************************************
   /*!\name Utility functions */
   //@{

   /*! \cond BLAZE_INTERNAL */
   /*!\brief Returns the mapped page indices
   //
   // \return The mapped page indices.
   */
   static inline constexpr Indices idces( ) noexcept
   {
      return Indices{3UL, 2UL, 1UL, 0UL};
   }

   /*! \cond BLAZE_INTERNAL */
   /*!\brief Returns the mapped page index of the specified element in the underlying tensor.
   //
   // \param l Quat access index.
   // \param k Page access index.
   // \param i Row access index.
   // \param j Column access index.
   // \return The mapped page index of the specified element.
   */
   static inline constexpr size_t quat  ( size_t l, size_t k, size_t i, size_t j ) noexcept
   {
      return j;
   }

   /*! \cond BLAZE_INTERNAL */
   /*!\brief Returns the mapped page index of the specified element in the underlying tensor.
   //
   // \param l Quat access index.
   // \param k Page access index.
   // \param i Row access index.
   // \param j Column access index.
   // \return The mapped page index of the specified element.
   */
   static inline constexpr size_t page  ( size_t l, size_t k, size_t i, size_t j ) noexcept
   {
      return i;
   }

   /*! \cond BLAZE_INTERNAL */
   /*!\brief Returns the mapped row index of the specified element in the underlying tensor.
   //
   // \param l Quat access index.
   // \param k Page access index.
   // \param i Row access index.
   // \param j Column access index.
   // \return The mapped row index of the specified element.
   */
   static inline constexpr size_t row   ( size_t l, size_t k, size_t i, size_t j ) noexcept
   {
      return k;
   }

   /*! \cond BLAZE_INTERNAL */
   /*!\brief Returns the mapped column index of the specified element in the underlying tensor.
   //
   // \param l Quat access index.
   // \param k Page access index.
   // \param i Row access index.
   // \param j Column access index.
   // \return The mapped column index of the specified element.
   */
   static inline constexpr size_t column( size_t l, size_t k, size_t i, size_t j ) noexcept
   {
      return l;
   }
   //@}
   //**********************************************************************************************

   /*! \cond BLAZE_INTERNAL */
   /*!\brief Returns the reversely mapped quat index of the specified element in the underlying tensor.
   //
   // \param l Quat access index.
   // \param k Page access index.
   // \param i Row access index.
   // \param j Column access index.
   // \return The mapped page index of the specified element.
   */
   static inline constexpr size_t reverse_quat  ( size_t l, size_t k, size_t i, size_t j ) noexcept
   {
      return j;
   }

   /*! \cond BLAZE_INTERNAL */
   /*!\brief Returns the reversely mapped page index of the specified element in the underlying tensor.
   //
   // \param l Quat access index.
   // \param k Page access index.
   // \param i Row access index.
   // \param j Column access index.
   // \return The mapped page index of the specified element.
   */
   static inline constexpr size_t reverse_page  ( size_t l, size_t k, size_t i, size_t j ) noexcept
   {
      return i;
   }

   /*! \cond BLAZE_INTERNAL */
   /*!\brief Returns the reversely mapped row index of the specified element in the underlying tensor.
   //
   // \param l Quat access index.
   // \param k Page access index.
   // \param i Row access index.
   // \param j Column access index.
   // \return The mapped row index of the specified element.
   */
   static inline constexpr size_t reverse_row   ( size_t l, size_t k, size_t i, size_t j ) noexcept
   {
      return k;
   }

   /*! \cond BLAZE_INTERNAL */
   /*!\brief Returns the reversely mapped column index of the specified element in the underlying tensor.
   //
   // \param l Quat access index.
   // \param k Page access index.
   // \param i Row access index.
   // \param j Column access index.
   // \return The mapped column index of the specified element.
   */
   static inline constexpr size_t reverse_column( size_t l, size_t k, size_t i, size_t j ) noexcept
   {
      return l;
   }
   //@}
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************



//=================================================================================================
//
//  CLASS TEMPLATE SPECIALIZATION FOR ZERO COMPILE TIME ELEMENT ARGUMENTS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Specialization of the DQuatTransExprData class template for zero compile time element arguments.
// \ingroup elements
//
// This specialization of DQuatTransExprData adapts the class template to the requirements of zero
// compile time element arguments.
*/
template<>
struct DQuatTransExprData<>
{
 public:
   //**Type definitions****************************************************************************
   using Indices = SmallArray<size_t, 4UL>;  //!< Type of the container for transpose indices.
   //**********************************************************************************************

   //**Constructors********************************************************************************
   /*!\name Constructors */
   //@{
   template< typename... RTAs >
   explicit inline DQuatTransExprData( RTAs... args );

   template< typename T, typename... RTAs >
   explicit inline DQuatTransExprData( T* indices, size_t n, RTAs... args );

   template< typename T, typename... RTAs >
   explicit inline DQuatTransExprData( const T* indices, size_t n, RTAs... args );

   DQuatTransExprData( const DQuatTransExprData& ) = default;
   DQuatTransExprData( DQuatTransExprData&& ) = default;
   //@}
   //**********************************************************************************************

   //**Destructor**********************************************************************************
   /*!\name Destructor */
   //@{
   ~DQuatTransExprData() = default;
   //@}
   //**********************************************************************************************

   //**Assignment operators************************************************************************
   /*!\name Assignment operators */
   //@{
   DQuatTransExprData& operator=( const DQuatTransExprData& ) = delete;
   //@}
   //**********************************************************************************************

   //**Utility functions***************************************************************************
   /*!\name Utility functions */
   //@{
   inline Indices const& idces() const noexcept;
   inline size_t quat  ( size_t l, size_t k, size_t i, size_t j ) const noexcept;
   inline size_t page  ( size_t l, size_t k, size_t i, size_t j ) const noexcept;
   inline size_t row   ( size_t l, size_t k, size_t i, size_t j ) const noexcept;
   inline size_t column( size_t l, size_t k, size_t i, size_t j ) const noexcept;
   inline size_t reverse_quat  ( size_t l, size_t k, size_t i, size_t j ) const noexcept;
   inline size_t reverse_page  ( size_t l, size_t k, size_t i, size_t j ) const noexcept;
   inline size_t reverse_row   ( size_t l, size_t k, size_t i, size_t j ) const noexcept;
   inline size_t reverse_column( size_t l, size_t k, size_t i, size_t j ) const noexcept;
   //@}
   //**********************************************************************************************

 private:
   //**Member variables****************************************************************************
   /*!\name Member variables */
   //@{
   Indices indices_;          //!< The indices of the elements in the tensor.
   Indices reverse_indices_;  //!< The reversely mapped indices of the elements in the tensor.
   //@}
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief The constructor for DQuatTransExprData.
//
// \param indices Pointer to the first index of the selected elements.
// \param n The total number of indices.
// \param args The optional element arguments.
*/
template< typename... RTAs >  // Optional element arguments
inline DQuatTransExprData<>::DQuatTransExprData( RTAs... args )
   : indices_( {3, 2, 1, 0} )  // The indices of the elements in the tensor
   , reverse_indices_( {3, 2, 1, 0} )
{
   MAYBE_UNUSED( args... );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief The constructor for DQuatTransExprData.
//
// \param indices Pointer to the first index of the selected elements.
// \param n The total number of indices.
// \param args The optional element arguments.
*/
template< typename T          // Type of the element indices
        , typename... RTAs >  // Optional element arguments
inline DQuatTransExprData<>::DQuatTransExprData( T* indices, size_t n, RTAs... args )
   : indices_( indices, indices+n )  // The indices of the elements in the quaternion
   , reverse_indices_(n)
{
   BLAZE_USER_ASSERT( n == 4, "Invalid number of transpose mapping indices (should be four)" );
   MAYBE_UNUSED( args... );

   for( size_t i = 0; i != 4; ++i )
   {
      reverse_indices_[indices_[i]] = i;
   }
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief The constructor for DQuatTransExprData.
//
// \param indices Pointer to the first index of the selected elements.
// \param n The total number of indices.
// \param args The optional element arguments.
*/
template< typename T          // Type of the element indices
        , typename... RTAs >  // Optional element arguments
inline DQuatTransExprData<>::DQuatTransExprData( const T* indices, size_t n, RTAs... args )
   : indices_( indices, indices+n )  // The indices of the elements in the quaternion
   , reverse_indices_(n)
{
   BLAZE_USER_ASSERT( n == 4, "Invalid number of transpose mapping indices (should be four)" );
   MAYBE_UNUSED( args... );

   for( size_t i = 0; i != 4; ++i )
   {
      reverse_indices_[indices_[i]] = i;
   }
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns the mapped page index of the specified element in the underlying tensor.
//
// \param k Page access index.
// \param i Row access index.
// \param j column access index.
// \return The mapped page index of the specified element.
*/
inline DQuatTransExprData<>::Indices const& DQuatTransExprData<>::idces( ) const noexcept
{
   return indices_;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns the mapped quat index of the specified element in the underlying tensor.
//
// \param l Quat access index.
// \param k Page access index.
// \param i Row access index.
// \param j column access index.
// \return The mapped page index of the specified element.
*/
inline size_t DQuatTransExprData<>::quat( size_t l, size_t k, size_t i, size_t j ) const noexcept
{
   switch (indices_[0])
   {
   case 0: return l;
   case 1: return k;
   case 2: return i;
   case 3: return j;
   }
   BLAZE_USER_ASSERT( true, "Invalid quat transposition mapping access index" );
   return -1;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns the mapped page index of the specified element in the underlying tensor.
//
// \param l Quat access index.
// \param k Page access index.
// \param i Row access index.
// \param j column access index.
// \return The mapped page index of the specified element.
*/
inline size_t DQuatTransExprData<>::page( size_t l, size_t k, size_t i, size_t j ) const noexcept
{
   switch (indices_[1])
   {
   case 0: return l;
   case 1: return k;
   case 2: return i;
   case 3: return j;
   }
   BLAZE_USER_ASSERT( true, "Invalid page transposition mapping access index" );
   return -1;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns the mapped row index of the specified element in the underlying tensor.
//
// \param l Quat access index.
// \param k Page access index.
// \param i Row access index.
// \param j column access index.
// \return The mapped row index of the specified element.
*/
inline size_t DQuatTransExprData<>::row( size_t l, size_t k, size_t i, size_t j ) const noexcept
{
   switch (indices_[2])
   {
   case 0: return l;
   case 1: return k;
   case 2: return i;
   case 3: return j;
   }
   BLAZE_USER_ASSERT( true, "Invalid row transposition mapping access index" );
   return -1;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns the mapped column index of the specified element in the underlying tensor.
//
// \param l Quat access index.
// \param k Page access index.
// \param i Row access index.
// \param j column access index.
// \return The mapped column index of the specified element.
*/
inline size_t DQuatTransExprData<>::column( size_t l, size_t k, size_t i, size_t j ) const noexcept
{
   switch (indices_[3])
   {
   case 0: return l;
   case 1: return k;
   case 2: return i;
   case 3: return j;
   }
   BLAZE_USER_ASSERT( true, "Invalid column transposition mapping access index" );
   return -1;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns the reversely mapped quat index of the specified element in the underlying tensor.
//
// \param l Quat access index.
// \param k Page access index.
// \param i Row access index.
// \param j column access index.
// \return The mapped page index of the specified element.
*/
inline size_t DQuatTransExprData<>::reverse_quat( size_t l, size_t k, size_t i, size_t j ) const noexcept
{
   switch (reverse_indices_[0])
   {
   case 0: return l;
   case 1: return k;
   case 2: return i;
   case 3: return j;
   }
   BLAZE_USER_ASSERT( true, "Invalid quat transposition mapping access index" );
   return -1;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns the reversely mapped page index of the specified element in the underlying tensor.
//
// \param l Quat access index.
// \param k Page access index.
// \param i Row access index.
// \param j column access index.
// \return The mapped page index of the specified element.
*/
inline size_t DQuatTransExprData<>::reverse_page( size_t l, size_t k, size_t i, size_t j ) const noexcept
{
   switch (reverse_indices_[1])
   {
   case 0: return l;
   case 1: return k;
   case 2: return i;
   case 3: return j;
   }
   BLAZE_USER_ASSERT( true, "Invalid page transposition mapping access index" );
   return -1;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns the reversely mapped page index of the specified element in the underlying tensor.
//
// \param k Page access index.
// \param i Row access index.
// \param j column access index.
// \return The mapped row index of the specified element.
*/
inline size_t DQuatTransExprData<>::reverse_row( size_t l, size_t k, size_t i, size_t j ) const noexcept
{
   switch (reverse_indices_[2])
   {
   case 0: return l;
   case 1: return k;
   case 2: return i;
   case 3: return j;
   }
   BLAZE_USER_ASSERT( true, "Invalid row transposition mapping access index" );
   return -1;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns the reversely mapped column index of the specified element in the underlying tensor.
//
// \param k Page access index.
// \param i Row access index.
// \param j column access index.
// \return The mapped column index of the specified element.
*/
inline size_t DQuatTransExprData<>::reverse_column( size_t l, size_t k, size_t i, size_t j ) const noexcept
{
   switch (reverse_indices_[3])
   {
   case 0: return l;
   case 1: return k;
   case 2: return i;
   case 3: return j;
   }
   BLAZE_USER_ASSERT( true, "Invalid column transposition mapping access index" );
   return -1;
}
/*! \endcond */
//*************************************************************************************************

} // namespace blaze

#endif
