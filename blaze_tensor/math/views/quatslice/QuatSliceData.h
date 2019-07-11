//=================================================================================================
/*!
//  \file blaze_tensor/math/views/quatslice/QuatSliceData.h
//  \brief Header file for the implementation of the QuatSliceData class template
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

#ifndef _BLAZE_TENSOR_MATH_VIEWS_QUATSLICE_QUATSLICEDATA_H_
#define _BLAZE_TENSOR_MATH_VIEWS_QUATSLICE_QUATSLICEDATA_H_


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
/*!\brief Auxiliary class template for the data members of the QuatSlice class.
// \ingroup quatslice
//
// The auxiliary QuatSliceData class template represents an abstraction of the data members of the
// QuatSlice class template. The necessary set of data members is selected depending on the number
// of compile time quatslice arguments.
*/
template< size_t... CRAs >  // Compile time quatslice arguments
struct QuatSliceData
{};
//*************************************************************************************************




//=================================================================================================
//
//  CLASS TEMPLATE SPECIALIZATION FOR ZERO COMPILE TIME QUATSLICE INDICES
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Specialization of the QuatSliceData class template for zero compile time quatslice arguments.
// \ingroup quatslice
//
// This specialization of QuatSliceData adapts the class template to the requirements of zero compile
// time quatslice arguments.
*/
template<>
struct QuatSliceData<>
{
 public:
   //**Constructors********************************************************************************
   /*!\name Constructors */
   //@{
   template< typename... RRAs >
   explicit inline QuatSliceData( size_t index, RRAs... args );

   QuatSliceData( const QuatSliceData& ) = default;
   //@}
   //**********************************************************************************************

   //**Destructor**********************************************************************************
   /*!\name Destructor */
   //@{
   ~QuatSliceData() = default;
   //@}
   //**********************************************************************************************

   //**Assignment operators************************************************************************
   /*!\name Assignment operators */
   //@{
   QuatSliceData& operator=( const QuatSliceData& ) = delete;
   //@}
   //**********************************************************************************************

   //**Utility functions***************************************************************************
   /*!\name Utility functions */
   //@{
   inline size_t quat() const noexcept;
   //@}
   //**********************************************************************************************

 private:
   //**Member variables****************************************************************************
   /*!\name Member variables */
   //@{
   const size_t quatslice_;  //!< The index of the quatslice in the quaternion.
   //@}
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief The constructor for QuatSliceData.
//
// \param index The index of the quatslice.
// \param args The optional quatslice arguments.
*/
template< typename... RRAs >  // Optional quatslice arguments
inline QuatSliceData<>::QuatSliceData( size_t index, RRAs... args )
   : quatslice_( index )  // The index of the quatslice in the tensor
{
   MAYBE_UNUSED( args... );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns the index of the quatslice of the underlying dense tensor.
//
// \return The index of the quatslice.
*/
inline size_t QuatSliceData<>::quat() const noexcept
{
   return quatslice_;
}
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  CLASS TEMPLATE SPECIALIZATION FOR ONE COMPILE TIME QUATSLICE INDEX
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Specialization of the QuatSliceData class template for a single compile time quatslice argument.
// \ingroup quatslice
//
// This specialization of QuatSliceData adapts the class template to the requirements of a single
// compile time quatslice argument.
*/
template< size_t Index >  // Compile time quatslice index
struct QuatSliceData<Index>
{
 public:
   //**Constructors********************************************************************************
   /*!\name Constructors */
   //@{
   template< typename... RRAs >
   explicit inline QuatSliceData( RRAs... args );

   QuatSliceData( const QuatSliceData& ) = default;
   //@}
   //**********************************************************************************************

   //**Destructor**********************************************************************************
   /*!\name Destructor */
   //@{
   ~QuatSliceData() = default;
   //@}
   //**********************************************************************************************

   //**Assignment operators************************************************************************
   /*!\name Assignment operators */
   //@{
   QuatSliceData& operator=( const QuatSliceData& ) = delete;
   //@}
   //**********************************************************************************************

   //**Utility functions***************************************************************************
   /*!\name Utility functions */
   //@{
   static inline constexpr size_t quat() noexcept;
   //@}
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief The constructor for QuatSliceData.
//
// \param args The optional quatslice arguments.
*/
template< size_t Index >      // Compile time quatslice index
template< typename... RRAs >  // Optional quatslice arguments
inline QuatSliceData<Index>::QuatSliceData( RRAs... args )
{
   MAYBE_UNUSED( args... );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns the index of the quatslice of the underlying dense tensor.
//
// \return The index of the quatslice.
*/
template< size_t Index >  // Compile time quatslice index
inline constexpr size_t QuatSliceData<Index>::quat() noexcept
{
   return Index;
}
/*! \endcond */
//*************************************************************************************************

} // namespace blaze

#endif
