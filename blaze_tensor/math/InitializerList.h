//=================================================================================================
/*!
//  \file blaze_tensor/math/InitializerList.h
//  \brief Header file for the extended initializer_list functionality
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

#ifndef _BLAZE_TENSOR_TENSOR_MATH_INITIALIZERLIST_H_
#define _BLAZE_TENSOR_TENSOR_MATH_INITIALIZERLIST_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <blaze/math/InitializerList.h>
#include <array>

namespace blaze {

//=================================================================================================
//
//  UTILITY FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Determines the number of non-zero elements contained in the given initializer list.
// \ingroup math
//
// \param list The given initializer list
// \return The number of non-zeros elements.
*/
template< typename Type >
inline constexpr size_t nonZeros(
   initializer_list< initializer_list< initializer_list< Type > > >
      list ) noexcept
{
   size_t nonzeros( 0UL );

   for (const auto& colList : list) {
      for (const auto& rowList : colList) {
         nonzeros += nonZeros(rowList);
      }
   }

   return nonzeros;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Determines the maximum number of pages specified by the given initializer list.
// \ingroup math
//
// \param list The given initializer list
// \return The maximum number of pages.
*/
template< typename Type >
inline constexpr size_t determineQuats( initializer_list< initializer_list<
      initializer_list< initializer_list< initializer_list< Type > > > > >
      list ) noexcept
{
   size_t cubes( 0UL );
   for( const auto& cube : list ) {
      cubes = max( cubes, cube.size() );
   }
   return cubes;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Determines the maximum number of pages specified by the given initializer list.
// \ingroup math
//
// \param list The given initializer list
// \return The maximum number of pages.
*/
template< typename Type >
inline constexpr size_t determinePages( initializer_list< initializer_list<
      initializer_list< initializer_list< initializer_list< Type > > > > >
      list ) noexcept
{
   size_t pages( 0UL );

   for( const auto& cube : list ) {
      for( const auto& page_list : cube ) {
         pages = max( pages, page_list.size() );
      }
   }
   return pages;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Determines the maximum number of rows specified by the given initializer list.
// \ingroup math
//
// \param list The given initializer list
// \return The maximum number of rows.
*/
template< typename Type >
inline constexpr size_t determineRows( initializer_list< initializer_list<
      initializer_list< initializer_list< initializer_list< Type > > > > >
      list ) noexcept
{
   size_t rows( 0UL );

   for( const auto& cube : list ) {
      for( const auto& page_list : cube ) {
         for( const auto& row_list : page_list ) {
            rows = max( rows, row_list.size() );
         }
      }
   }
   return rows;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Determines the maximum number of columns specified by the given initializer list.
// \ingroup math
//
// \param list The given initializer list
// \return The maximum number of columns.
*/
template< typename Type >
inline constexpr size_t determineColumns( initializer_list< initializer_list<
      initializer_list< initializer_list< initializer_list< Type > > > > >
      list ) noexcept
{
   size_t cols( 0UL );

   for( const auto& cube : list ) {
      for( const auto& page_list : cube ) {
         for( const auto& row_list : page_list ) {
            for(const auto& col_list : row_list) {
               cols = max( cols, col_list.size() );
            }
         }
      }
   }
   return cols;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Determines the maximum number of pages specified by the given initializer list.
// \ingroup math
//
// \param list The given initializer list
// \return The maximum number of pages.
*/
template< typename Type >
inline constexpr size_t determinePages( initializer_list<
   initializer_list< initializer_list< initializer_list< Type > > > >
      list ) noexcept
{
   size_t pages( 0UL );

   for( const auto& cube : list ) {
      pages = max( pages, cube.size() );
   }
   return pages;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Determines the maximum number of columns specified by the given initializer list.
// \ingroup math
//
// \param list The given initializer list
// \return The maximum number of columns.
*/
template< typename Type >
inline constexpr size_t determineColumns( initializer_list<
   initializer_list< initializer_list< initializer_list< Type > > > >
      list ) noexcept
{
   size_t cols( 0UL );

   for( const auto& cube : list ) {
      for( const auto& page_list : cube ) {
         for( const auto& row_list : page_list ) {
            cols = max( cols, row_list.size() );
         }
      }
   }
   return cols;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Determines the maximum number of rows specified by the given initializer list.
// \ingroup math
//
// \param list The given initializer list
// \return The maximum number of rows.
*/
template< typename Type >
inline constexpr size_t determineRows( initializer_list<
   initializer_list< initializer_list< initializer_list< Type > > > >
      list ) noexcept
{
   size_t rows( 0UL );

   for( const auto& cube : list ) {
      for( const auto& page_list : cube ) {
         rows = max( rows, page_list.size() );
      }
   }
   return rows;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Determines the maximum number of columns specified by the given initializer list.
// \ingroup math
//
// \param list The given initializer list
// \return The maximum number of columns.
*/
template< typename Type >
inline constexpr size_t determineColumns(
   initializer_list< initializer_list< initializer_list< Type > > >
      list ) noexcept
{
   size_t cols( 0UL );

   for (const auto& page : list) {
      for (const auto& rowList : page) {
         cols = max(cols, rowList.size());
      }
   }
   return cols;
}
//*************************************************************************************************

//*************************************************************************************************
/*!\brief Determines the maximum number of rows specified by the given initializer list.
// \ingroup math
//
// \param list The given initializer list
// \return The maximum number of rows.
*/
template< typename Type >
inline constexpr size_t determineRows(
   initializer_list< initializer_list< initializer_list< Type > > >
      list ) noexcept
{
   size_t rows( 0UL );

   for (const auto& page : list) {
      rows = max(rows, page.size());
   }
   return rows;
}
//*************************************************************************************************

//*************************************************************************************************
/*!\brief Define a nested initializer list type of given dimensionality.
// \ingroup math
*/
template< size_t N, typename Type >
struct nested_initializer_list;

template< typename Type >
struct nested_initializer_list< 1, Type > : initializer_list< Type >
{
   using type = initializer_list< Type >;

   constexpr nested_initializer_list(type rhs) : type(std::move(rhs)) {}

   constexpr std::array< size_t, 1 > dimensions() const
   {
      std::array< size_t, 1 > dims {
         this->type::size()
      };
      return dims;
   }

   template< typename C >
   void transfer_data( C& rhs )
   {
      std::fill(
         std::copy( this->type::begin(), this->type::end(), ( ~rhs ).begin() ),
         ( ~rhs ).end(),
         Type() );
   }
};

template< typename Type >
struct nested_initializer_list< 2, Type >
   : initializer_list< initializer_list< Type > >
{
   using type = initializer_list< initializer_list< Type > >;

   constexpr nested_initializer_list(type rhs) : type(std::move(rhs)) {}

   constexpr std::array< size_t, 2 > dimensions() const
   {
      std::array< size_t, 2 > dims {
         determineColumns( *this ),
         this->type::size()
      };
      return dims;
   }


   template< typename C >
   void transfer_data( C& rhs )
   {
      size_t i( 0UL );
      for( const auto& rowList : *this ) {
         std::fill(
            std::copy( rowList.begin(), rowList.end(), ( ~rhs ).begin( i ) ),
            ( ~rhs ).end( i ),
            Type() );
         ++i;
      }
   }
};

template< typename Type >
struct nested_initializer_list< 3, Type >
   : initializer_list< initializer_list< initializer_list< Type > > >
{
   using type =
      initializer_list< initializer_list< initializer_list< Type > > >;

   constexpr nested_initializer_list(type rhs) : type(std::move(rhs)) {}

   constexpr std::array< size_t, 3 > dimensions() const
   {
      std::array< size_t, 3 > dims = {
         determineColumns( *this ),
         determineRows( *this ),
         this->type::size()
      };
      return dims;
   }

   template< typename C >
   void transfer_data( C& rhs )
   {
      size_t k( 0UL );
      for (const auto& page : *this) {
         size_t i( 0UL );
         for (const auto& rowList : page) {
            std::fill(
               std::copy(
                  rowList.begin(), rowList.end(), ( ~rhs ).begin( i, k ) ),
               ( ~rhs ).end( i, k ),
               Type() );
            ++i;
         }
         ++k;
      }
   }
};

template< typename Type >
struct nested_initializer_list< 4, Type >
   : initializer_list<
         initializer_list< initializer_list< initializer_list< Type > > > >
{
   using type = initializer_list<
      initializer_list< initializer_list< initializer_list< Type > > > >;

   constexpr nested_initializer_list(type rhs) : type(std::move(rhs)) {}

   constexpr std::array< size_t, 4 > dimensions() const
   {
      std::array< size_t, 4 > dims = {
         determineColumns( *this ),
         determineRows( *this ),
         determinePages( *this ),
         this->type::size()
      };
      return dims;
   }

   template< typename C >
   void transfer_data( C& rhs )
   {
      size_t l( 0UL );
      for( const auto& cube : *this ) {
         size_t k( 0UL );
         for( const auto& page_list : cube ) {
            size_t i( 0UL );
            for( const auto& row_list : page_list ) {
               std::fill( std::copy( row_list.begin(),
                             row_list.end(),
                             ( ~rhs ).begin( i, l, k ) ),
                  ( ~rhs ).end( i, l, k ),
                  Type() );
               ++i;
            }
            ++k;
         }
         ++l;
      }
   }
};

template< typename Type >
struct nested_initializer_list< 5, Type >
   : initializer_list< initializer_list<
         initializer_list< initializer_list< initializer_list< Type > > > > >
{
   using type = initializer_list< initializer_list<
      initializer_list< initializer_list< initializer_list< Type > > > > >;

   constexpr nested_initializer_list(type rhs) : type(std::move(rhs)) {}

   constexpr std::array< size_t, 5 > dimensions() const
   {
      std::array< size_t, 5 > dims = {
         determineColumns( *this ),
         determineRows( *this ),
         determinePages( *this ),
         determineQuats( *this ),
         this->type::size()
      };
      return dims;
   }
};
//*************************************************************************************************

} // namespace blaze

#endif
