//=================================================================================================
/*!
//  \file blaze_tensor/math/views/columnslice/ColumnSliceData.h
//  \brief Header file for the implementation of the ColumnSliceData class template
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

#ifndef _BLAZE_TENSOR_MATH_VIEWS_COLUMNSLICE_COLUMNSLICEDATA_H_
#define _BLAZE_TENSOR_MATH_VIEWS_COLUMNSLICE_COLUMNSLICEDATA_H_


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
/*!\brief Auxiliary class template for the data members of the ColumnSlice class.
// \ingroup columnslice
//
// The auxiliary ColumnSliceData class template represents an abstraction of the data members of the
// ColumnSlice class template. The necessary set of data members is selected depending on the number
// of compile time columnslice arguments.
*/
template< size_t... CRAs >  // Compile time columnslice arguments
struct ColumnSliceData
{};
//*************************************************************************************************




//=================================================================================================
//
//  CLASS TEMPLATE SPECIALIZATION FOR ZERO COMPILE TIME COLUMNSLICE INDICES
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Specialization of the ColumnSliceData class template for zero compile time columnslice arguments.
// \ingroup columnslice
//
// This specialization of ColumnSliceData adapts the class template to the requirements of zero compile
// time columnslice arguments.
*/
template<>
struct ColumnSliceData<>
{
 public:
   //**Constructors********************************************************************************
   /*!\name Constructors */
   //@{
   template< typename... RRAs >
   explicit inline ColumnSliceData( size_t index, RRAs... args );

   ColumnSliceData( const ColumnSliceData& ) = default;
   //@}
   //**********************************************************************************************

   //**Destructor**********************************************************************************
   /*!\name Destructor */
   //@{
   ~ColumnSliceData() = default;
   //@}
   //**********************************************************************************************

   //**Assignment operators************************************************************************
   /*!\name Assignment operators */
   //@{
   ColumnSliceData& operator=( const ColumnSliceData& ) = delete;
   //@}
   //**********************************************************************************************

   //**Utility functions***************************************************************************
   /*!\name Utility functions */
   //@{
   inline size_t column() const noexcept;
   //@}
   //**********************************************************************************************

 private:
   //**Member variables****************************************************************************
   /*!\name Member variables */
   //@{
   const size_t columnslice_;  //!< The index of the columnslice in the tensor.
   //@}
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief The constructor for ColumnSliceData.
//
// \param index The index of the columnslice.
// \param args The optional columnslice arguments.
*/
template< typename... RRAs >  // Optional columnslice arguments
inline ColumnSliceData<>::ColumnSliceData( size_t index, RRAs... args )
   : columnslice_( index )  // The index of the columnslice in the tensor
{
   MAYBE_UNUSED( args... );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns the index of the columnslice of the underlying dense tensor.
//
// \return The index of the columnslice.
*/
inline size_t ColumnSliceData<>::column() const noexcept
{
   return columnslice_;
}
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  CLASS TEMPLATE SPECIALIZATION FOR ONE COMPILE TIME COLUMNSLICE INDEX
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Specialization of the ColumnSliceData class template for a single compile time columnslice argument.
// \ingroup columnslice
//
// This specialization of ColumnSliceData adapts the class template to the requirements of a single
// compile time columnslice argument.
*/
template< size_t Index >  // Compile time columnslice index
struct ColumnSliceData<Index>
{
 public:
   //**Constructors********************************************************************************
   /*!\name Constructors */
   //@{
   template< typename... RRAs >
   explicit inline ColumnSliceData( RRAs... args );

   ColumnSliceData( const ColumnSliceData& ) = default;
   //@}
   //**********************************************************************************************

   //**Destructor**********************************************************************************
   /*!\name Destructor */
   //@{
   ~ColumnSliceData() = default;
   //@}
   //**********************************************************************************************

   //**Assignment operators************************************************************************
   /*!\name Assignment operators */
   //@{
   ColumnSliceData& operator=( const ColumnSliceData& ) = delete;
   //@}
   //**********************************************************************************************

   //**Utility functions***************************************************************************
   /*!\name Utility functions */
   //@{
   static inline constexpr size_t column() noexcept;
   //@}
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief The constructor for ColumnSliceData.
//
// \param args The optional columnslice arguments.
*/
template< size_t Index >      // Compile time columnslice index
template< typename... RRAs >  // Optional columnslice arguments
inline ColumnSliceData<Index>::ColumnSliceData( RRAs... args )
{
   MAYBE_UNUSED( args... );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns the index of the columnslice of the underlying dense tensor.
//
// \return The index of the columnslice.
*/
template< size_t Index >  // Compile time columnslice index
inline constexpr size_t ColumnSliceData<Index>::column() noexcept
{
   return Index;
}
/*! \endcond */
//*************************************************************************************************

} // namespace blaze

#endif
