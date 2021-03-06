//=================================================================================================
/*!
//  \file blaze_tensor/math/Tensor.h
//  \brief Header file for all basic Tensor functionality
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

#ifndef _BLAZE_TENSOR_MATH_TENSOR_H_
#define _BLAZE_TENSOR_MATH_TENSOR_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <iomanip>
#include <iosfwd>

#include <blaze/math/Matrix.h>

#include <blaze_tensor/math/expressions/Forward.h>
#include <blaze_tensor/math/expressions/Tensor.h>


namespace blaze {

//=================================================================================================
//
//  GLOBAL FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*!\name Tensor functions */
//@{
template< typename MT >
bool isUniform( const Tensor<MT>& m );
//@}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Checks if the given tensor is a uniform tensor.
// \ingroup tensor
//
// \param m The tensor to be checked.
// \return \a true if the tensor is a uniform tensor, \a false if not.
//
// This function checks if the given dense or sparse tensor is a uniform tensor. The tensor
// is considered to be uniform if all its elements are identical. The following code example
// demonstrates the use of the function:

   \code
   blaze::Dynamictensor<int,blaze::rowMajor> A, B;
   // ... Initialization
   if( isUniform( A ) ) { ... }
   \endcode

// Optionally, it is possible to switch between strict semantics (blaze::strict) and relaxed
// semantics (blaze::relaxed):

   \code
   if( isUniform<relaxed>( A ) ) { ... }
   \endcode

// It is also possible to check if a tensor expression results in a uniform tensor:

   \code
   if( isUniform( A * B ) ) { ... }
   \endcode

// However, note that this might require the complete evaluation of the expression, including
// the generation of a temporary tensor.
*/
template< typename MT > // Type of the tensor
inline bool isUniform( const Tensor<MT>& t )
{
   return isUniform<relaxed>( *t );
}
//*************************************************************************************************


//=================================================================================================
//
//  GLOBAL OPERATORS
//
//=================================================================================================

//*************************************************************************************************
/*!\name Tensor operators */
//@{
template< typename MT >
inline std::ostream& operator<<( std::ostream& os, const Tensor<MT>& m );
//@}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Global output operator for dense and sparse tensors.
// \ingroup tensor
//
// \param os Reference to the output stream.
// \param m Reference to a constant tensor object.
// \return Reference to the output stream.
*/
template< typename MT >
inline std::ostream& operator<<( std::ostream& os, const Tensor<MT>& m )
{
   CompositeType_t<MT> tmp( *m );

   for (size_t k = 0UL; k < tmp.pages(); ++k) {
      os << "(";
      for (size_t i = 0UL; i < tmp.rows(); ++i) {
         os << "(";
         for (size_t j = 0UL; j < tmp.columns(); ++j) {
            os << std::setw(12) << tmp(k, i, j) << " ";
         }
         os << ") ";
      }
      os << ")\n";
   }

   return os;
}
//*************************************************************************************************

} // namespace blaze

#endif
