//=================================================================================================
/*!
//  \file blaze_tensor/math/views/arrayslice/ArraySliceData.h
//  \brief Header file for the implementation of the ArraySliceData class template
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

#ifndef _BLAZE_TENSOR_MATH_VIEWS_ARRAYSLICE_ARRAYSLICEDATA_H_
#define _BLAZE_TENSOR_MATH_VIEWS_ARRAYSLICE_ARRAYSLICEDATA_H_


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
/*!\brief Auxiliary class template for the data members of the ArraySlice class.
// \ingroup arrayslice
//
// The auxiliary ArraySliceData class template represents an abstraction of the data members of the
// ArraySlice class template. The necessary set of data members is selected depending on the number
// of compile time arrayslice arguments.
*/
template< size_t... CRAs >  // Compile time arrayslice arguments
struct ArraySliceData
{};
//*************************************************************************************************




//=================================================================================================
//
//  CLASS TEMPLATE SPECIALIZATION FOR ZERO COMPILE TIME PAGESLICE INDICES
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Specialization of the ArraySliceData class template for zero compile time arrayslice arguments.
// \ingroup arrayslice
//
// This specialization of ArraySliceData adapts the class template to the requirements of zero compile
// time arrayslice arguments.
*/
template<>
struct ArraySliceData<>
{
 public:
   //**Constructors********************************************************************************
   /*!\name Constructors */
   //@{
   template< typename... RRAs >
   explicit inline ArraySliceData( size_t index, RRAs... args );

   ArraySliceData( const ArraySliceData& ) = default;
   //@}
   //**********************************************************************************************

   //**Destructor**********************************************************************************
   /*!\name Destructor */
   //@{
   ~ArraySliceData() = default;
   //@}
   //**********************************************************************************************

   //**Assignment operators************************************************************************
   /*!\name Assignment operators */
   //@{
   ArraySliceData& operator=( const ArraySliceData& ) = delete;
   //@}
   //**********************************************************************************************

   //**Utility functions***************************************************************************
   /*!\name Utility functions */
   //@{
   inline size_t index() const noexcept;
   //@}
   //**********************************************************************************************

 private:
   //**Member variables****************************************************************************
   /*!\name Member variables */
   //@{
   const size_t arrayslice_;  //!< The index of the arrayslice in the tensor.
   //@}
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief The constructor for ArraySliceData.
//
// \param index The index of the arrayslice.
// \param args The optional arrayslice arguments.
*/
template< typename... RRAs >  // Optional arrayslice arguments
inline ArraySliceData<>::ArraySliceData( size_t index, RRAs... args )
   : arrayslice_( index )  // The index of the arrayslice in the tensor
{
   MAYBE_UNUSED( args... );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns the index of the arrayslice of the underlying dense tensor.
//
// \return The index of the arrayslice.
*/
inline size_t ArraySliceData<>::index() const noexcept
{
   return arrayslice_;
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
/*!\brief Specialization of the ArraySliceData class template for a single compile time arrayslice argument.
// \ingroup arrayslice
//
// This specialization of ArraySliceData adapts the class template to the requirements of a single
// compile time arrayslice argument.
*/
template< size_t Index >  // Compile time arrayslice index
struct ArraySliceData<Index>
{
 public:
   //**Constructors********************************************************************************
   /*!\name Constructors */
   //@{
   template< typename... RRAs >
   explicit inline ArraySliceData( RRAs... args );

   ArraySliceData( const ArraySliceData& ) = default;
   //@}
   //**********************************************************************************************

   //**Destructor**********************************************************************************
   /*!\name Destructor */
   //@{
   ~ArraySliceData() = default;
   //@}
   //**********************************************************************************************

   //**Assignment operators************************************************************************
   /*!\name Assignment operators */
   //@{
   ArraySliceData& operator=( const ArraySliceData& ) = delete;
   //@}
   //**********************************************************************************************

   //**Utility functions***************************************************************************
   /*!\name Utility functions */
   //@{
   static inline constexpr size_t index() noexcept;
   //@}
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief The constructor for ArraySliceData.
//
// \param args The optional arrayslice arguments.
*/
template< size_t Index >      // Compile time arrayslice index
template< typename... RRAs >  // Optional arrayslice arguments
inline ArraySliceData<Index>::ArraySliceData( RRAs... args )
{
   MAYBE_UNUSED( args... );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns the index of the arrayslice of the underlying dense tensor.
//
// \return The index of the arrayslice.
*/
template< size_t Index >  // Compile time arrayslice index
inline constexpr size_t ArraySliceData<Index>::index() noexcept
{
   return Index;
}
/*! \endcond */
//*************************************************************************************************

} // namespace blaze

#endif
