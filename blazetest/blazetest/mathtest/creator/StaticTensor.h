//=================================================================================================
/*!
//  \file blazetest/mathtest/creator/StaticTensor.h
//  \brief Specialization of the Creator class template for StaticTensor
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

#ifndef _BLAZETEST_MATHTEST_CREATOR_STATICTENSOR_H_
#define _BLAZETEST_MATHTEST_CREATOR_STATICTENSOR_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <blazetest/mathtest/creator/Default.h>
#include <blazetest/mathtest/creator/Policies.h>
#include <blazetest/system/Types.h>

#include <blaze_tensor/math/StaticTensor.h>

namespace blazetest {

//=================================================================================================
//
//  CLASS DEFINITION
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Specialization of the Creator class template for static matrices.
//
// This specialization of the Creator class template is able to create random static matrices.
*/
template< typename T  // Element type of the static tensor
        , size_t O    // Number of rows of the static tensor
        , size_t M    // Number of rows of the static tensor
        , size_t N >  // Number of columns of the static tensor
class Creator< blaze::StaticTensor<T,O,M,N> >
{
 public:
   //**Type definitions****************************************************************************
   using Type = blaze::StaticTensor<T,O,M,N>;  //!< Type to be created by the Creator.
   //**********************************************************************************************

   //**Constructors********************************************************************************
   /*!\name Constructors */
   //@{
   explicit inline Creator( const Creator<T>& elementCreator = Creator<T>() );
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

   const blaze::StaticTensor<T,O,M,N> operator()() const;

   template< typename CP >
   const blaze::StaticTensor<T,O,M,N> operator()( const CP& policy ) const;
   //@}
   //**********************************************************************************************

 private:
   //**Member variables****************************************************************************
   /*!\name Member variables */
   //@{
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
/*!\brief Constructor for the creator specialization for StaticTensor.
//
// \param elementCreator The creator for the elements of the static tensor.
*/
template< typename T  // Element type of the static tensor
        , size_t O    // Number of rows of the static tensor
        , size_t M    // Number of rows of the static tensor
        , size_t N >  // Number of columns of the static tensor
inline Creator< blaze::StaticTensor<T,O,M,N> >::Creator( const Creator<T>& elementCreator )
   : ec_( elementCreator )  // Creator for the elements of the static tensor
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
template< typename T  // Element type of the static tensor
        , size_t O    // Number of rows of the static tensor
        , size_t M    // Number of rows of the static tensor
        , size_t N >  // Number of columns of the static tensor
inline const blaze::StaticTensor<T,O,M,N>
   Creator< blaze::StaticTensor<T,O,M,N> >::operator()() const
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
template< typename T  // Element type of the static tensor
        , size_t O    // Number of rows of the static tensor
        , size_t M    // Number of rows of the static tensor
        , size_t N >  // Number of columns of the static tensor
template< typename CP >  // Creation policy
inline const blaze::StaticTensor<T,O,M,N>
   Creator< blaze::StaticTensor<T,O,M,N> >::operator()( const CP& policy ) const
{
   blaze::StaticTensor<T,O,M,N> tensor;

   for( size_t k=0UL; k<O; ++k )
      for( size_t i=0UL; i<M; ++i )
         for( size_t j=0UL; j<N; ++j )
            tensor(k,i,j) = ec_( policy );

   return tensor;
}
//*************************************************************************************************

} // namespace blazetest

#endif
