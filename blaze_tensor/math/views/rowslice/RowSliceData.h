//=================================================================================================
/*!
//  \file blaze_tensor/math/views/rowslice/RowSliceData.h
//  \brief Header file for the implementation of the RowSliceData class template
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

#ifndef _BLAZE_TENSOR_MATH_VIEWS_ROWSLICE_ROWSLICEDATA_H_
#define _BLAZE_TENSOR_MATH_VIEWS_ROWSLICE_ROWSLICEDATA_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <blaze/util/Types.h>
#include <blaze/util/MaybeUnused.h>


namespace blaze {

//=================================================================================================
//
//  CLASS DEFINITION
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Auxiliary class template for the data members of the RowSlice class.
// \ingroup rowslice
//
// The auxiliary RowSliceData class template represents an abstraction of the data members of the
// RowSlice class template. The necessary set of data members is selected depending on the number
// of compile time rowslice arguments.
*/
template< size_t... CRAs >  // Compile time rowslice arguments
struct RowSliceData
{};
//*************************************************************************************************




//=================================================================================================
//
//  CLASS TEMPLATE SPECIALIZATION FOR ZERO COMPILE TIME ROWSLICE INDICES
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Specialization of the RowSliceData class template for zero compile time rowslice arguments.
// \ingroup rowslice
//
// This specialization of RowSliceData adapts the class template to the requirements of zero compile
// time rowslice arguments.
*/
template<>
struct RowSliceData<>
{
 public:
   //**Constructors********************************************************************************
   /*!\name Constructors */
   //@{
   template< typename... RRAs >
   explicit inline RowSliceData( size_t index, RRAs... args );

   RowSliceData( const RowSliceData& ) = default;
   //@}
   //**********************************************************************************************

   //**Destructor**********************************************************************************
   /*!\name Destructor */
   //@{
   ~RowSliceData() = default;
   //@}
   //**********************************************************************************************

   //**Assignment operators************************************************************************
   /*!\name Assignment operators */
   //@{
   RowSliceData& operator=( const RowSliceData& ) = delete;
   //@}
   //**********************************************************************************************

   //**Utility functions***************************************************************************
   /*!\name Utility functions */
   //@{
   inline size_t row() const noexcept;
   //@}
   //**********************************************************************************************

 private:
   //**Member variables****************************************************************************
   /*!\name Member variables */
   //@{
   const size_t rowslice_;  //!< The index of the rowslice in the tensor.
   //@}
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief The constructor for RowSliceData.
//
// \param index The index of the rowslice.
// \param args The optional rowslice arguments.
*/
template< typename... RRAs >  // Optional rowslice arguments
inline RowSliceData<>::RowSliceData( size_t index, RRAs... args )
   : rowslice_( index )  // The index of the rowslice in the tensor
{
   MAYBE_UNUSED( args... );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns the index of the rowslice of the underlying dense tensor.
//
// \return The index of the rowslice.
*/
inline size_t RowSliceData<>::row() const noexcept
{
   return rowslice_;
}
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  CLASS TEMPLATE SPECIALIZATION FOR ONE COMPILE TIME ROWSLICE INDEX
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Specialization of the RowSliceData class template for a single compile time rowslice argument.
// \ingroup rowslice
//
// This specialization of RowSliceData adapts the class template to the requirements of a single
// compile time rowslice argument.
*/
template< size_t Index >  // Compile time rowslice index
struct RowSliceData<Index>
{
 public:
   //**Constructors********************************************************************************
   /*!\name Constructors */
   //@{
   template< typename... RRAs >
   explicit inline RowSliceData( RRAs... args );

   RowSliceData( const RowSliceData& ) = default;
   //@}
   //**********************************************************************************************

   //**Destructor**********************************************************************************
   /*!\name Destructor */
   //@{
   ~RowSliceData() = default;
   //@}
   //**********************************************************************************************

   //**Assignment operators************************************************************************
   /*!\name Assignment operators */
   //@{
   RowSliceData& operator=( const RowSliceData& ) = delete;
   //@}
   //**********************************************************************************************

   //**Utility functions***************************************************************************
   /*!\name Utility functions */
   //@{
   static inline constexpr size_t row() noexcept;
   //@}
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief The constructor for RowSliceData.
//
// \param args The optional rowslice arguments.
*/
template< size_t Index >      // Compile time rowslice index
template< typename... RRAs >  // Optional rowslice arguments
inline RowSliceData<Index>::RowSliceData( RRAs... args )
{
   MAYBE_UNUSED( args... );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns the index of the rowslice of the underlying dense tensor.
//
// \return The index of the rowslice.
*/
template< size_t Index >  // Compile time rowslice index
inline constexpr size_t RowSliceData<Index>::row() noexcept
{
   return Index;
}
/*! \endcond */
//*************************************************************************************************

} // namespace blaze

#endif
