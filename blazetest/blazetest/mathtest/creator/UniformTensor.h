//=================================================================================================
/*!
//  \file blazetest/mathtest/creator/UniformTensor.h
//  \brief Specialization of the Creator class template for UniformTensor
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

#ifndef _BLAZETEST_MATHTEST_CREATOR_UNIFORMTENSOR_H_
#define _BLAZETEST_MATHTEST_CREATOR_UNIFORMTENSOR_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <blazetest/mathtest/creator/Default.h>
#include <blazetest/mathtest/creator/Policies.h>
#include <blazetest/system/Types.h>

#include <blaze_tensor/math/UniformTensor.h>

namespace blazetest {

//=================================================================================================
//
//  CLASS DEFINITION
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Specialization of the Creator class template for uniform tensors.
//
// This specialization of the Creator class template is able to create random static matrices.
*/
template< typename T > // Element type of the uniform tensor
class Creator< blaze::UniformTensor<T> >
{
 public:
   //**Type definitions****************************************************************************
   using Type = blaze::UniformTensor<T>;  //!< Type to be created by the Creator.
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

   const blaze::UniformTensor<T> operator()() const;

   template< typename CP >
   const blaze::UniformTensor<T> operator()( const CP& policy ) const;
   //@}
   //**********************************************************************************************

 private:
   //**Member variables****************************************************************************
   /*!\name Member variables */
   //@{
   size_t o_;       //!< The number of pages of the uniform tensor.
   size_t m_;       //!< The number of rows of the uniform tensor.
   size_t n_;       //!< The number of columns of the uniform tensor.
   Creator<T> ec_;  //!< Creator for the elements of the static tensor.
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
/*!\brief Constructor for the creator specialization for UniformTensor.
//
// \param elementCreator The creator for the elements of the static tensor.
*/
template< typename T > // Element type of the uniform tensor
inline Creator< blaze::UniformTensor<T> >::Creator( const Creator<T>& elementCreator )
   : o_( 2UL )              // The number of pages of the uniform tensor
   , m_( 3UL )              // The number of rows of the uniform tensor
   , n_( 3UL )              // The number of columns of the uniform tensor
   , ec_( elementCreator )  // Creator for the elements of the static tensor
{}
//*************************************************************************************************




//*************************************************************************************************
/*!\brief Constructor for the creator specialization for UniformTensor.
//
// \param elementCreator The creator for the elements of the static tensor.
*/
template< typename T > // Element type of the uniform tensor
inline Creator< blaze::UniformTensor<T> >::Creator( size_t o, size_t m, size_t n, const Creator<T>& elementCreator )
   : o_( o )              // The number of pages of the uniform tensor
   , m_( m )              // The number of rows of the uniform tensor
   , n_( n )              // The number of columns of the uniform tensor
   , ec_( elementCreator )  // Creator for the elements of the static tensor
{}
//*************************************************************************************************




//=================================================================================================
//
//  OPERATORS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Returns a randomly created static tensor.
//
// \return The randomly generated static tensor.
*/
template< typename T > // Element type of the uniform tensor
inline const blaze::UniformTensor<T>
   Creator< blaze::UniformTensor<T> >::operator()() const
{
   return (*this)( Default() );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns a randomly created static tensor.
//
// \param policy The creation policy for the elements of fundamental data type.
// \return The randomly generated static tensor.
*/
template< typename T > // Element type of the uniform tensor
template< typename CP >  // Creation policy
inline const blaze::UniformTensor<T>
   Creator< blaze::UniformTensor<T> >::operator()( const CP& policy ) const
{
   return blaze::UniformTensor<T>( o_, m_, n_, ec_( policy ) );
}
//*************************************************************************************************

} // namespace blazetest

#endif
