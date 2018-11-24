//=================================================================================================
/*!
//  \file blaze_tensor/math/Subtensor.h
//  \brief Header file for the complete Subtensor implementation
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

#ifndef _BLAZE_TENSOR_MATH_SUBTENSOR_H_
#define _BLAZE_TENSOR_MATH_SUBTENSOR_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <blaze/math/Aliases.h>
#include <blaze/math/Exception.h>
#include <blaze/util/Random.h>
#include <blaze/util/typetraits/RemoveReference.h>

#include <blaze_tensor/math/constraints/DenseTensor.h>
#include <blaze_tensor/math/constraints/Subtensor.h>
#include <blaze_tensor/math/smp/DenseTensor.h>
#include <blaze_tensor/math/views/Subtensor.h>

namespace blaze {

//=================================================================================================
//
//  RAND SPECIALIZATION FOR DENSE SUBTENSORS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Specialization of the Rand class template for dense subtensors.
// \ingroup random
//
// This specialization of the Rand class randomizes dense subtensors.
*/
template< typename MT       // Type of the tensor
        , AlignmentFlag AF  // Alignment flag
        , bool SO           // Storage order
        , size_t... CSAs >  // Compile time subtensor arguments
class Rand< Subtensor<MT,AF,SO,true,CSAs...> >
{
 public:
   //**Randomize functions*************************************************************************
   /*!\name Randomize functions */
   //@{
   template< typename SMT >
   inline void randomize( SMT&& subtensor ) const;

   template< typename SMT, typename Arg >
   inline void randomize( SMT&& subtensor, const Arg& min, const Arg& max ) const;
   //@}
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Randomization of a dense subtensor.
//
// \param subtensor The subtensor to be randomized.
// \return void
*/
template< typename MT       // Type of the dense tensor
        , AlignmentFlag AF  // Alignment flag
        , bool SO           // Storage order
        , size_t... CSAs >  // Compile time subtensor arguments
template< typename SMT >    // Type of the subtensor
inline void Rand< Subtensor<MT,AF,SO,true,CSAs...> >::randomize( SMT&& subtensor ) const
{
   using blaze::randomize;

   using SubtensorType = RemoveReference_t<SMT>;

   BLAZE_CONSTRAINT_MUST_BE_SUBTENSOR_TYPE( SubtensorType );
   BLAZE_CONSTRAINT_MUST_BE_DENSE_TENSOR_TYPE( SubtensorType );

   if( SO == rowMajor ) {
      for( size_t i=0UL; i<subtensor.rows(); ++i ) {
         for( size_t j=0UL; j<subtensor.columns(); ++j ) {
            randomize( subtensor(i,j) );
         }
      }
   }
   else {
      for( size_t j=0UL; j<subtensor.columns(); ++j ) {
         for( size_t i=0UL; i<subtensor.rows(); ++i ) {
            randomize( subtensor(i,j) );
         }
      }
   }
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Randomization of a dense subtensor.
//
// \param subtensor The subtensor to be randomized.
// \param min The smallest possible value for a tensor element.
// \param max The largest possible value for a tensor element.
// \return void
*/
template< typename MT       // Type of the dense tensor
        , AlignmentFlag AF  // Alignment flag
        , bool SO           // Storage order
        , size_t... CSAs >  // Compile time subtensor arguments
template< typename SMT      // Type of the subtensor
        , typename Arg >    // Min/max argument type
inline void Rand< Subtensor<MT,AF,SO,true,CSAs...> >::randomize( SMT&& subtensor,
                                                                 const Arg& min, const Arg& max ) const
{
   using blaze::randomize;

   using SubtensorType = RemoveReference_t<SMT>;

   BLAZE_CONSTRAINT_MUST_BE_SUBTENSOR_TYPE( SubtensorType );
   BLAZE_CONSTRAINT_MUST_BE_DENSE_TENSOR_TYPE( SubtensorType );

   if( SO == rowMajor ) {
      for( size_t i=0UL; i<subtensor.rows(); ++i ) {
         for( size_t j=0UL; j<subtensor.columns(); ++j ) {
            randomize( subtensor(i,j), min, max );
         }
      }
   }
   else {
      for( size_t j=0UL; j<subtensor.columns(); ++j ) {
         for( size_t i=0UL; i<subtensor.rows(); ++i ) {
            randomize( subtensor(i,j), min, max );
         }
      }
   }
}
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  RAND SPECIALIZATION FOR SPARSE SUBTENSORS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Specialization of the Rand class template for sparse subtensors.
// \ingroup random
//
// This specialization of the Rand class randomizes sparse subtensors.
*/
template< typename MT       // Type of the dense tensor
        , AlignmentFlag AF  // Alignment flag
        , bool SO           // Storage order
        , size_t... CSAs >  // Compile time subtensor arguments
class Rand< Subtensor<MT,AF,SO,false,CSAs...> >
{
 public:
   //**Randomize functions*************************************************************************
   /*!\name Randomize functions */
   //@{
   template< typename SMT >
   inline void randomize( SMT&& subtensor ) const;

   template< typename SMT >
   inline void randomize( SMT&& subtensor, size_t nonzeros ) const;

   template< typename SMT, typename Arg >
   inline void randomize( SMT&& subtensor, const Arg& min, const Arg& max ) const;

   template< typename SMT, typename Arg >
   inline void randomize( SMT&& subtensor, size_t nonzeros, const Arg& min, const Arg& max ) const;
   //@}
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Randomization of a sparse subtensor.
//
// \param subtensor The subtensor to be randomized.
// \return void
*/
template< typename MT       // Type of the dense tensor
        , AlignmentFlag AF  // Alignment flag
        , bool SO           // Storage order
        , size_t... CSAs >  // Compile time subtensor arguments
template< typename SMT >    // Type of the subtensor
inline void Rand< Subtensor<MT,AF,SO,false,CSAs...> >::randomize( SMT&& subtensor ) const
{
   using SubtensorType = RemoveReference_t<SMT>;
   using ElementType   = ElementType_t<SubtensorType>;

   BLAZE_CONSTRAINT_MUST_BE_SUBTENSOR_TYPE( SubtensorType );
   BLAZE_CONSTRAINT_MUST_BE_SPARSE_TENSOR_TYPE( SubtensorType );

   const size_t m( subtensor.rows()    );
   const size_t n( subtensor.columns() );

   if( m == 0UL || n == 0UL ) return;

   const size_t nonzeros( rand<size_t>( 1UL, std::ceil( 0.5*m*n ) ) );

   subtensor.reset();
   subtensor.reserve( nonzeros );

   while( subtensor.nonZeros() < nonzeros ) {
      subtensor( rand<size_t>( 0UL, m-1UL ), rand<size_t>( 0UL, n-1UL ) ) = rand<ElementType>();
   }
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Randomization of a sparse subtensor.
//
// \param subtensor The subtensor to be randomized.
// \param nonzeros The number of non-zero elements of the random subtensor.
// \return void
// \exception std::invalid_argument Invalid number of non-zero elements.
*/
template< typename MT       // Type of the dense tensor
        , AlignmentFlag AF  // Alignment flag
        , bool SO           // Storage order
        , size_t... CSAs >  // Compile time subtensor arguments
template< typename SMT >    // Type of the subtensor
inline void Rand< Subtensor<MT,AF,SO,false,CSAs...> >::randomize( SMT&& subtensor, size_t nonzeros ) const
{
   using SubtensorType = RemoveReference_t<SMT>;
   using ElementType   = ElementType_t<SubtensorType>;

   BLAZE_CONSTRAINT_MUST_BE_SUBTENSOR_TYPE( SubtensorType );
   BLAZE_CONSTRAINT_MUST_BE_SPARSE_TENSOR_TYPE( SubtensorType );

   const size_t m( subtensor.rows()    );
   const size_t n( subtensor.columns() );

   if( nonzeros > m*n ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid number of non-zero elements" );
   }

   if( m == 0UL || n == 0UL ) return;

   subtensor.reset();
   subtensor.reserve( nonzeros );

   while( subtensor.nonZeros() < nonzeros ) {
      subtensor( rand<size_t>( 0UL, m-1UL ), rand<size_t>( 0UL, n-1UL ) ) = rand<ElementType>();
   }
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Randomization of a sparse subtensor.
//
// \param subtensor The subtensor to be randomized.
// \param min The smallest possible value for a subtensor element.
// \param max The largest possible value for a subtensor element.
// \return void
*/
template< typename MT       // Type of the dense tensor
        , AlignmentFlag AF  // Alignment flag
        , bool SO           // Storage order
        , size_t... CSAs >  // Compile time subtensor arguments
template< typename SMT      // Type of the subtensor
        , typename Arg >    // Min/max argument type
inline void Rand< Subtensor<MT,AF,SO,false,CSAs...> >::randomize( SMT&& subtensor,
                                                                  const Arg& min, const Arg& max ) const
{
   using SubtensorType = RemoveReference_t<SMT>;
   using ElementType   = ElementType_t<SubtensorType>;

   BLAZE_CONSTRAINT_MUST_BE_SUBTENSOR_TYPE( SubtensorType );
   BLAZE_CONSTRAINT_MUST_BE_SPARSE_TENSOR_TYPE( SubtensorType );

   const size_t m( subtensor.rows()    );
   const size_t n( subtensor.columns() );

   if( m == 0UL || n == 0UL ) return;

   const size_t nonzeros( rand<size_t>( 1UL, std::ceil( 0.5*m*n ) ) );

   subtensor.reset();
   subtensor.reserve( nonzeros );

   while( subtensor.nonZeros() < nonzeros ) {
      subtensor( rand<size_t>( 0UL, m-1UL ), rand<size_t>( 0UL, n-1UL ) ) = rand<ElementType>( min, max );
   }
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Randomization of a sparse subtensor.
//
// \param subtensor The subtensor to be randomized.
// \param nonzeros The number of non-zero elements of the random subtensor.
// \param min The smallest possible value for a subtensor element.
// \param max The largest possible value for a subtensor element.
// \return void
// \exception std::invalid_argument Invalid number of non-zero elements.
*/
template< typename MT       // Type of the dense tensor
        , AlignmentFlag AF  // Alignment flag
        , bool SO           // Storage order
        , size_t... CSAs >  // Compile time subtensor arguments
template< typename SMT      // Type of the subtensor
        , typename Arg >    // Min/max argument type
inline void Rand< Subtensor<MT,AF,SO,false,CSAs...> >::randomize( SMT&& subtensor, size_t nonzeros,
                                                                  const Arg& min, const Arg& max ) const
{
   using SubtensorType = RemoveReference_t<SMT>;
   using ElementType   = ElementType_t<SubtensorType>;

   BLAZE_CONSTRAINT_MUST_BE_SUBTENSOR_TYPE( SubtensorType );
   BLAZE_CONSTRAINT_MUST_BE_SPARSE_TENSOR_TYPE( SubtensorType );

   const size_t m( subtensor.rows()    );
   const size_t n( subtensor.columns() );

   if( nonzeros > m*n ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid number of non-zero elements" );
   }

   if( m == 0UL || n == 0UL ) return;

   subtensor.reset();
   subtensor.reserve( nonzeros );

   while( subtensor.nonZeros() < nonzeros ) {
      subtensor( rand<size_t>( 0UL, m-1UL ), rand<size_t>( 0UL, n-1UL ) ) = rand<ElementType>( min, max );
   }
}
/*! \endcond */
//*************************************************************************************************

} // namespace blaze

#endif
