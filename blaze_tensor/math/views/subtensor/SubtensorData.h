//=================================================================================================
/*!
//  \file blaze_tensor/math/views/subtensor/SubtensorData.h
//  \brief Header file for the implementation of the SubtensorData class template
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

#ifndef _BLAZE_TENSOR_MATH_VIEWS_SUBTENSOR_SUBTENSORDATA_H_
#define _BLAZE_TENSOR_MATH_VIEWS_SUBTENSOR_SUBTENSORDATA_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <blaze/util/Types.h>
#include <blaze/util/Unused.h>


namespace blaze {

//=================================================================================================
//
//  CLASS DEFINITION
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Auxiliary class template for the data members of the Subtensor class.
// \ingroup subtensor
//
// The auxiliary SubtensorData class template represents an abstraction of the data members of
// the Subtensor class template. The necessary set of data members is selected depending on the
// number of compile time subtensor arguments.
*/
template< size_t... CSAs >  // Compile time subtensor arguments
struct SubtensorData
{};
//*************************************************************************************************




//=================================================================================================
//
//  CLASS TEMPLATE SPECIALIZATION FOR ZERO COMPILE TIME ARGUMENTS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Specialization of the SubtensorData class template for zero compile time subtensor
//        arguments.
// \ingroup subtensor
//
// This specialization of SubtensorData adapts the class template to the requirements of zero
// compile time subtensor arguments.
*/
template<>
struct SubtensorData<>
{
 public:
   //**Constructors********************************************************************************
   /*!\name Constructors */
   //@{
   template< typename... RSAs >
   explicit inline SubtensorData( size_t pindex, size_t rindex, size_t cindex, size_t o, size_t m, size_t n, RSAs... args );

   SubtensorData( const SubtensorData& ) = default;
   //@}
   //**********************************************************************************************

   //**Destructor**********************************************************************************
   /*!\name Destructor */
   //@{
   ~SubtensorData() = default;
   //@}
   //**********************************************************************************************

   //**Assignment operators************************************************************************
   /*!\name Assignment operators */
   //@{
   SubtensorData& operator=( const SubtensorData& ) = delete;
   //@}
   //**********************************************************************************************

   //**Utility functions***************************************************************************
   /*!\name Utility functions */
   //@{
   inline size_t row    () const noexcept;
   inline size_t column () const noexcept;
   inline size_t page   () const noexcept;
   inline size_t rows   () const noexcept;
   inline size_t columns() const noexcept;
   inline size_t pages  () const noexcept;
   //@}
   //**********************************************************************************************

 private:
   //**Member variables****************************************************************************
   /*!\name Member variables */
   //@{
   const size_t page_;    //!< The first page of the subtensor.
   const size_t row_;     //!< The first row of the subtensor.
   const size_t column_;  //!< The first column of the subtensor.
   const size_t o_;       //!< The number of pages of the subtensor.
   const size_t m_;       //!< The number of rows of the subtensor.
   const size_t n_;       //!< The number of columns of the subtensor.
   //@}
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief The constructor for SubtensorData.
//
// \param rindex The index of the first row of the subtensor in the given tensor.
// \param cindex The index of the first column of the subtensor in the given tensor.
// \param m The number of rows of the subtensor.
// \param n The number of columns of the subtensor.
// \param args The optional subtensor arguments.
*/
template< typename... RSAs >  // Optional subtensor arguments
inline SubtensorData<>::SubtensorData( size_t pindex, size_t rindex, size_t cindex, size_t o, size_t m, size_t n, RSAs... args )
   : page_  ( pindex )  // The first page of the subtensor
   , row_   ( rindex )  // The first row of the subtensor
   , column_( cindex )  // The first column of the subtensor
   , o_     ( o      )  // The number of columns of the subtensor
   , m_     ( m      )  // The number of rows of the subtensor
   , n_     ( n      )  // The number of columns of the subtensor
{
   UNUSED_PARAMETER( args... );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns the index of the first row of the subtensor in the underlying tensor.
//
// \return The index of the first row.
*/
inline size_t SubtensorData<>::row() const noexcept
{
   return row_;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns the index of the first column of the subtensor in the underlying tensor.
//
// \return The index of the first column.
*/
inline size_t SubtensorData<>::column() const noexcept
{
   return column_;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns the index of the first page of the subtensor in the underlying tensor.
//
// \return The index of the first page.
*/
inline size_t SubtensorData<>::page() const noexcept
{
   return page_;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns the number of rows of the subtensor.
//
// \return The number of rows of the subtensor.
*/
inline size_t SubtensorData<>::rows() const noexcept
{
   return m_;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns the number of columns of the subtensor.
//
// \return The number of columns of the subtensor.
*/
inline size_t SubtensorData<>::columns() const noexcept
{
   return n_;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns the number of pages of the subtensor.
//
// \return The number of pages of the subtensor.
*/
inline size_t SubtensorData<>::pages() const noexcept
{
   return o_;
}
/*! \endcond */
//*************************************************************************************************



//=================================================================================================
//
//  CLASS TEMPLATE SPECIALIZATION FOR FOUR COMPILE TIME ARGUMENTS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Specialization of the SubtensorData class template for four compile time subtensor
//        arguments.
// \ingroup subtensor
//
// This specialization of SubtensorData adapts the class template to the requirements of two
// compile time arguments.
*/
template< size_t K    // Index of the first page
        , size_t I    // Index of the first row
        , size_t J    // Index of the first column
        , size_t O    // Number of pages
        , size_t M    // Number of rows
        , size_t N >  // Number of columns
struct SubtensorData<K,I,J,O,M,N>
{
 public:
   //**Constructors********************************************************************************
   /*!\name Constructors */
   //@{
   template< typename... RSAs >
   explicit inline SubtensorData( RSAs... args );

   SubtensorData( const SubtensorData& ) = default;
   //@}
   //**********************************************************************************************

   //**Destructor**********************************************************************************
   /*!\name Destructor */
   //@{
   ~SubtensorData() = default;
   //@}
   //**********************************************************************************************

   //**Assignment operators************************************************************************
   /*!\name Assignment operators */
   //@{
   SubtensorData& operator=( const SubtensorData& ) = delete;
   //@}
   //**********************************************************************************************

   //**Utility functions***************************************************************************
   /*!\name Utility functions */
   //@{
   static inline constexpr size_t row    () noexcept;
   static inline constexpr size_t column () noexcept;
   static inline constexpr size_t page   () noexcept;
   static inline constexpr size_t rows   () noexcept;
   static inline constexpr size_t columns() noexcept;
   static inline constexpr size_t pages  () noexcept;
   //@}
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief The constructor for SubtensorData.
//
// \param args The optional subtensor arguments.
*/
template< size_t K    // Index of the first page
        , size_t I    // Index of the first row
        , size_t J    // Index of the first column
        , size_t O    // Number of pages
        , size_t M    // Number of rows
        , size_t N >  // Number of columns
template< typename... RSAs >  // Optional subtensor arguments
inline SubtensorData<K,I,J,O,M,N>::SubtensorData( RSAs... args )
{
   UNUSED_PARAMETER( args... );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns the index of the first row of the subtensor in the underlying tensor.
//
// \return The index of the first row.
*/
template< size_t K    // Index of the first page
        , size_t I    // Index of the first row
        , size_t J    // Index of the first column
        , size_t O    // Number of pages
        , size_t M    // Number of rows
        , size_t N >  // Number of columns
inline constexpr size_t SubtensorData<K,I,J,O,M,N>::row() noexcept
{
   return I;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns the index of the first column of the subtensor in the underlying tensor.
//
// \return The index of the first column.
*/
template< size_t K    // Index of the first page
        , size_t I    // Index of the first row
        , size_t J    // Index of the first column
        , size_t O    // Number of pages
        , size_t M    // Number of rows
        , size_t N >  // Number of columns
inline constexpr size_t SubtensorData<K,I,J,O,M,N>::column() noexcept
{
   return J;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns the index of the first page of the subtensor in the underlying tensor.
//
// \return The index of the first page.
*/
template< size_t K    // Index of the first page
        , size_t I    // Index of the first row
        , size_t J    // Index of the first column
        , size_t O    // Number of pages
        , size_t M    // Number of rows
        , size_t N >  // Number of columns
inline constexpr size_t SubtensorData<K,I,J,O,M,N>::page() noexcept
{
   return K;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns the number of rows of the subtensor.
//
// \return The number of rows of the subtensor.
*/
template< size_t K    // Index of the first page
        , size_t I    // Index of the first row
        , size_t J    // Index of the first column
        , size_t O    // Number of pages
        , size_t M    // Number of rows
        , size_t N >  // Number of columns
inline constexpr size_t SubtensorData<K,I,J,O,M,N>::rows() noexcept
{
   return M;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns the number of columns of the subtensor.
//
// \return The number of columns of the subtensor.
*/
template< size_t K    // Index of the first page
        , size_t I    // Index of the first row
        , size_t J    // Index of the first column
        , size_t O    // Number of pages
        , size_t M    // Number of rows
        , size_t N >  // Number of columns
inline constexpr size_t SubtensorData<K,I,J,O,M,N>::columns() noexcept
{
   return N;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns the number of pages of the subtensor.
//
// \return The number of columns of the subtensor.
*/
template< size_t K    // Index of the first page
        , size_t I    // Index of the first row
        , size_t J    // Index of the first column
        , size_t O    // Number of pages
        , size_t M    // Number of rows
        , size_t N >  // Number of columns
inline constexpr size_t SubtensorData<K,I,J,O,M,N>::pages() noexcept
{
   return O;
}
/*! \endcond */
//*************************************************************************************************

} // namespace blaze

#endif
