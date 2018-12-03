//=================================================================================================
/*!
//  \file blazetest/mathtest/creator/DynamicTensor.h
//  \brief Specialization of the Creator class template for DynamicTensor
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

#ifndef _BLAZETEST_MATHTEST_CREATOR_DYNAMICTENSOR_H_
#define _BLAZETEST_MATHTEST_CREATOR_DYNAMICTENSOR_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <blazetest/mathtest/creator/Default.h>
#include <blazetest/mathtest/creator/Policies.h>
#include <blazetest/system/Types.h>

#include <blaze_tensor/math/DynamicTensor.h>

namespace blazetest {

//=================================================================================================
//
//  CLASS DEFINITION
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Specialization of the Creator class template for dynamic \f$ M \times N \f$ tensors.
//
// This specialization of the Creator class template is able to create random \f$ M \times N \f$
// tensors.
*/
template< typename T > // Element type of the dynamic tensor
class Creator< blaze::DynamicTensor<T> >
{
 public:
   //**Type definitions****************************************************************************
   using Type = blaze::DynamicTensor<T>;  //!< Type to be created by the Creator.
   //**********************************************************************************************

   //**Constructors********************************************************************************
   /*!\name Constructors */
   //@{
   explicit inline Creator( const Creator<T>& elementCreator = Creator<T>() );
   explicit inline Creator( size_t o, size_t m, size_t n, const Creator<T>& elementCreator = Creator<T>() );
   // No explicitly declared copy constructor.
   //@}
   //**********************************************************************************************

   //**Destructor**********************************************************************************
   // No explicitly declared destructor.
   //**********************************************************************************************

   //**Operators***********************************************************************************
   /*!\name Operators */
   //@{
   // No explicitly declared copy assignment operator.

   const blaze::DynamicTensor<T> operator()() const;

   template< typename CP >
   const blaze::DynamicTensor<T> operator()( const CP& policy ) const;
   //@}
   //**********************************************************************************************

 private:
   //**Member variables****************************************************************************
   /*!\name Member variables */
   //@{
   size_t o_;       //!< The number of pages of the dynamic tensor.
   size_t m_;       //!< The number of rows of the dynamic tensor.
   size_t n_;       //!< The number of columns of the dynamic tensor.
   Creator<T> ec_;  //!< Creator for the elements of the dynamic tensor.
   //@}
   //**********************************************************************************************
};
//*************************************************************************************************




//=================================================================================================
//
//  CONSTRUCTORS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Constructor for the creator specialization for DynamicTensor.
//
// \param elementCreator The creator for the elements of the dynamic tensor.
*/
template< typename T > // Element type of the dynamic tensor
inline Creator< blaze::DynamicTensor<T> >::Creator( const Creator<T>& elementCreator )
   : o_( 2UL )              // The number of pages of the dynamic tensor
   , m_( 3UL )              // The number of rows of the dynamic tensor
   , n_( 3UL )              // The number of columns of the dynamic tensor
   , ec_( elementCreator )  // Creator for the elements of the dynamic tensor
{}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Constructor for the creator specialization for DynamicTensor.
//
// \param o The number of pages of the dynamic tensor.
// \param m The number of rows of the dynamic tensor.
// \param n The number of columns of the dynamic tensor.
// \param elementCreator The creator for the elements of the dynamic tensor.
*/
template< typename T > // Element type of the dynamic tensor
inline Creator< blaze::DynamicTensor<T> >::Creator( size_t o, size_t m, size_t n, const Creator<T>& elementCreator )
   : o_( o )                // The number of pages of the dynamic tensor
   , m_( m )                // The number of rows of the dynamic tensor
   , n_( n )                // The number of columns of the dynamic tensor
   , ec_( elementCreator )  // Creator for the elements of the dynamic tensor
{}
//*************************************************************************************************




//=================================================================================================
//
//  OPERATORS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Returns a randomly created dynamic tensor.
//
// \return The randomly generated dynamic tensor.
*/
template< typename T > // Element type of the dynamic tensor
inline const blaze::DynamicTensor<T> Creator< blaze::DynamicTensor<T> >::operator()() const
{
   return (*this)( Default() );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns a randomly created dynamic tensor.
//
// \param policy The creation policy for the elements of fundamental data type.
// \return The randomly generated dynamic tensor.
*/
template< typename T > // Element type of the dynamic tensor
template< typename CP >  // Creation policy
inline const blaze::DynamicTensor<T>
   Creator< blaze::DynamicTensor<T> >::operator()( const CP& policy ) const
{
   blaze::DynamicTensor<T> tensor( o_, m_, n_ );

   for( size_t k=0UL; k<o_; ++k )
      for( size_t i=0UL; i<m_; ++i )
         for( size_t j=0UL; j<n_; ++j )
            tensor(k,i,j) = ec_( policy );

   return tensor;
}
//*************************************************************************************************

} // namespace blazetest

#endif
