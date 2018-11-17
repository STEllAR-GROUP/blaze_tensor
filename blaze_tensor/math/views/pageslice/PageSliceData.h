//=================================================================================================
/*!
//  \file blaze_tensor/math/views/pageslice/PageSliceData.h
//  \brief Header file for the implementation of the PageSliceData class template
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

#ifndef _BLAZE_TENSOR_MATH_VIEWS_PAGESLICE_PAGESLICEDATA_H_
#define _BLAZE_TENSOR_MATH_VIEWS_PAGESLICE_PAGESLICEDATA_H_


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
/*!\brief Auxiliary class template for the data members of the PageSlice class.
// \ingroup pageslice
//
// The auxiliary PageSliceData class template represents an abstraction of the data members of the
// PageSlice class template. The necessary set of data members is selected depending on the number
// of compile time pageslice arguments.
*/
template< size_t... CRAs >  // Compile time pageslice arguments
struct PageSliceData
{};
//*************************************************************************************************




//=================================================================================================
//
//  CLASS TEMPLATE SPECIALIZATION FOR ZERO COMPILE TIME PAGESLICE INDICES
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Specialization of the PageSliceData class template for zero compile time pageslice arguments.
// \ingroup pageslice
//
// This specialization of PageSliceData adapts the class template to the requirements of zero compile
// time pageslice arguments.
*/
template<>
struct PageSliceData<>
{
 public:
   //**Constructors********************************************************************************
   /*!\name Constructors */
   //@{
   template< typename... RRAs >
   explicit inline PageSliceData( size_t index, RRAs... args );

   PageSliceData( const PageSliceData& ) = default;
   //@}
   //**********************************************************************************************

   //**Destructor**********************************************************************************
   /*!\name Destructor */
   //@{
   ~PageSliceData() = default;
   //@}
   //**********************************************************************************************

   //**Assignment operators************************************************************************
   /*!\name Assignment operators */
   //@{
   PageSliceData& operator=( const PageSliceData& ) = delete;
   //@}
   //**********************************************************************************************

   //**Utility functions***************************************************************************
   /*!\name Utility functions */
   //@{
   inline size_t page() const noexcept;
   //@}
   //**********************************************************************************************

 private:
   //**Member variables****************************************************************************
   /*!\name Member variables */
   //@{
   const size_t pageslice_;  //!< The index of the pageslice in the tensor.
   //@}
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief The constructor for PageSliceData.
//
// \param index The index of the pageslice.
// \param args The optional pageslice arguments.
*/
template< typename... RRAs >  // Optional pageslice arguments
inline PageSliceData<>::PageSliceData( size_t index, RRAs... args )
   : pageslice_( index )  // The index of the pageslice in the tensor
{
   UNUSED_PARAMETER( args... );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns the index of the pageslice of the underlying dense tensor.
//
// \return The index of the pageslice.
*/
inline size_t PageSliceData<>::page() const noexcept
{
   return pageslice_;
}
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  CLASS TEMPLATE SPECIALIZATION FOR ONE COMPILE TIME PAGESLICE INDEX
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Specialization of the PageSliceData class template for a single compile time pageslice argument.
// \ingroup pageslice
//
// This specialization of PageSliceData adapts the class template to the requirements of a single
// compile time pageslice argument.
*/
template< size_t Index >  // Compile time pageslice index
struct PageSliceData<Index>
{
 public:
   //**Constructors********************************************************************************
   /*!\name Constructors */
   //@{
   template< typename... RRAs >
   explicit inline PageSliceData( RRAs... args );

   PageSliceData( const PageSliceData& ) = default;
   //@}
   //**********************************************************************************************

   //**Destructor**********************************************************************************
   /*!\name Destructor */
   //@{
   ~PageSliceData() = default;
   //@}
   //**********************************************************************************************

   //**Assignment operators************************************************************************
   /*!\name Assignment operators */
   //@{
   PageSliceData& operator=( const PageSliceData& ) = delete;
   //@}
   //**********************************************************************************************

   //**Utility functions***************************************************************************
   /*!\name Utility functions */
   //@{
   static inline constexpr size_t page() noexcept;
   //@}
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief The constructor for PageSliceData.
//
// \param args The optional pageslice arguments.
*/
template< size_t Index >      // Compile time pageslice index
template< typename... RRAs >  // Optional pageslice arguments
inline PageSliceData<Index>::PageSliceData( RRAs... args )
{
   UNUSED_PARAMETER( args... );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns the index of the pageslice of the underlying dense tensor.
//
// \return The index of the pageslice.
*/
template< size_t Index >  // Compile time pageslice index
inline constexpr size_t PageSliceData<Index>::page() noexcept
{
   return Index;
}
/*! \endcond */
//*************************************************************************************************

} // namespace blaze

#endif
