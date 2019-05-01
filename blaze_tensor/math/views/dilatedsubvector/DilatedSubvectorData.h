//=================================================================================================
/*!
//  \file blaze_tensor/math/views/dilatedsubvector/DilatedSubvectorData.h
//  \brief Header file for the implementation of the DilatedSubvectorData class template
//
//  Copyright (C) 2012-2019 Klaus Iglberger - All Rights Reserved
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

#ifndef _BLAZE_TENSOR_MATH_VIEWS_DILATEDSUBVECTOR_DILATEDSUBVECTORDATA_H_
#define _BLAZE_TENSOR_MATH_VIEWS_DILATEDSUBVECTOR_DILATEDSUBVECTORDATA_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <blaze/util/MaybeUnused.h>
#include <blaze/util/Types.h>


namespace blaze {

//=================================================================================================
//
//  CLASS DEFINITION
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Auxiliary class template for the data members of the Subvector class.
// \ingroup dilatedsubvector
//
// The auxiliary DilatedSubvectorData class template represents an abstraction of the data members of
// the Subvector class template. The necessary set of data members is selected depending on the
// number of compile time dilatedsubvector arguments.
*/
template< size_t... CSAs >  // Compile time dilatedsubvector arguments
struct DilatedSubvectorData
{};
//*************************************************************************************************




//=================================================================================================
//
//  CLASS TEMPLATE SPECIALIZATION FOR ZERO COMPILE TIME ARGUMENTS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Specialization of the DilatedSubvectorData class template for zero compile time dilatedsubvector
//        arguments.
// \ingroup dilatedsubvector
//
// This specialization of DilatedSubvectorData adapts the class template to the requirements of zero
// compile time dilatedsubvector arguments.
*/
template<>
struct DilatedSubvectorData<>
{
 public:
   //**Constructors********************************************************************************
   /*!\name Constructors */
   //@{
   template< typename... RSAs >
   explicit inline DilatedSubvectorData( size_t index, size_t n, size_t dilation, RSAs... args );

   DilatedSubvectorData( const DilatedSubvectorData& ) = default;
   //@}
   //**********************************************************************************************

   //**Destructor**********************************************************************************
   /*!\name Destructor */
   //@{
   ~DilatedSubvectorData() = default;
   //@}
   //**********************************************************************************************

   //**Assignment operators************************************************************************
   /*!\name Assignment operators */
   //@{
   DilatedSubvectorData& operator=( const DilatedSubvectorData& ) = delete;
   //@}
   //**********************************************************************************************

   //**Utility functions***************************************************************************
   /*!\name Utility functions */
   //@{
   inline size_t offset  () const noexcept;
   inline size_t size    () const noexcept;
   inline size_t dilation() const noexcept;
   //@}
   //**********************************************************************************************

 private:
   //**Member variables****************************************************************************
   /*!\name Member variables */
   //@{
   const size_t offset_;   //!< The offset of the dilatedsubvector within the vector.
   const size_t size_;     //!< The size of the dilatedsubvector.
   const size_t dilation_; //!< The step-size of the dilatedsubvector.
   //@}
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief The constructor for DilatedSubvectorData.
//
// \param index The offset of the dilatedsubvector within the given vector.
// \param n The size of the dilatedsubvector.
// \param args The optional dilatedsubvector arguments.
*/
template< typename... RSAs >  // Optional dilatedsubvector arguments
inline DilatedSubvectorData<>::DilatedSubvectorData( size_t index, size_t n, size_t dilation, RSAs... args )
   : offset_  ( index    ) // The offset of the dilatedsubvector within the vector
   , size_    ( n        ) // The size of the dilatedsubvector
   , dilation_( dilation ) // The steps-size of the dilatedsubvector
{
   MAYBE_UNUSED( args... );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns the offset of the dilatedsubvector within the underlying vector.
//
// \return The offset of the dilatedsubvector.
*/
inline size_t DilatedSubvectorData<>::offset() const noexcept
{
   return offset_;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns the current size/dimension of the dilatedsubvector.
//
// \return The size of the dilatedsubvector.
*/
inline size_t DilatedSubvectorData<>::size() const noexcept
{
   return size_;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns the current step-size of the dilatedsubvector.
//
// \return The step-size of the dilatedsubvector.
*/
inline size_t DilatedSubvectorData<>::dilation() const noexcept
{
   return dilation_;
}
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  CLASS TEMPLATE SPECIALIZATION FOR THREE COMPILE TIME ARGUMENTS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Specialization of the DilatedSubvectorData class template for two compile time dilatedsubvector
//        arguments.
// \ingroup dilatedsubvector
//
// This specialization of DilatedSubvectorData adapts the class template to the requirements of two
// compile time arguments.
*/
template< size_t I            // Index of the first element
        , size_t N            // Number of elements
        , size_t Dilation >   // Step-size in between of elements
struct DilatedSubvectorData<I,N,Dilation>
{
 public:
   //**Constructors********************************************************************************
   /*!\name Constructors */
   //@{
   template< typename... RSAs >
   explicit inline DilatedSubvectorData( RSAs... args );

   DilatedSubvectorData( const DilatedSubvectorData& ) = default;
   //@}
   //**********************************************************************************************

   //**Destructor**********************************************************************************
   /*!\name Destructor */
   //@{
   ~DilatedSubvectorData() = default;
   //@}
   //**********************************************************************************************

   //**Assignment operators************************************************************************
   /*!\name Assignment operators */
   //@{
   DilatedSubvectorData& operator=( const DilatedSubvectorData& ) = delete;
   //@}
   //**********************************************************************************************

   //**Utility functions***************************************************************************
   /*!\name Utility functions */
   //@{
   static inline constexpr size_t offset  () noexcept;
   static inline constexpr size_t size    () noexcept;
   static inline constexpr size_t dilation() noexcept;
   //@}
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief The constructor for DilatedSubvectorData.
//
// \param args The optional dilatedsubvector arguments.
*/
template< size_t I            // Index of the first element
        , size_t N            // Number of elements
        , size_t Dilation >   // Step-size in between of elements
template< typename... RSAs >  // Optional dilatedsubvector arguments
inline DilatedSubvectorData<I,N,Dilation>::DilatedSubvectorData( RSAs... args )
{
   MAYBE_UNUSED( args... );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns the offset of the dilatedsubvector within the underlying vector.
//
// \return The offset of the dilatedsubvector.
*/
template< size_t I            // Index of the first element
        , size_t N            // Number of elements
        , size_t Dilation >   // Step-size in between of elements
inline constexpr size_t DilatedSubvectorData<I,N,Dilation>::offset() noexcept
{
   return I;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns the current size/dimension of the dilatedsubvector.
//
// \return The size of the dilatedsubvector.
*/
template< size_t I            // Index of the first element
        , size_t N            // Number of elements
        , size_t Dilation >   // Step-size in between of elements
inline constexpr size_t DilatedSubvectorData<I,N,Dilation>::size() noexcept
{
   return N;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns the current step-size of the dilatedsubvector.
//
// \return The step-size of the dilatedsubvector.
*/
template< size_t I            // Index of the first element
        , size_t N            // Number of elements
        , size_t Dilation >   // Step-size in between of elements
inline constexpr size_t DilatedSubvectorData<I,N,Dilation>::dilation() noexcept
{
   return Dilation;
}
/*! \endcond */
//*************************************************************************************************

} // namespace blaze

#endif
