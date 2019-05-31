//=================================================================================================
/*!
//  \file blaze_array/math/expressions/DArrNormExpr.h
//  \brief Header file for the dense array norm expression
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

#ifndef _BLAZE_TENSOR_MATH_EXPRESSIONS_DARRNORMEXPR_H_
#define _BLAZE_TENSOR_MATH_EXPRESSIONS_DARRNORMEXPR_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <utility>

#include <blaze/math/Aliases.h>
#include <blaze/math/SIMD.h>
#include <blaze/math/functors/Abs.h>
#include <blaze/math/functors/Cbrt.h>
#include <blaze/math/functors/L1Norm.h>
#include <blaze/math/functors/L2Norm.h>
#include <blaze/math/functors/L3Norm.h>
#include <blaze/math/functors/L4Norm.h>
#include <blaze/math/functors/LpNorm.h>
#include <blaze/math/functors/Noop.h>
#include <blaze/math/functors/Pow2.h>
#include <blaze/math/functors/Pow3.h>
#include <blaze/math/functors/Qdrt.h>
#include <blaze/math/functors/SqrAbs.h>
#include <blaze/math/functors/Sqrt.h>
#include <blaze/math/functors/UnaryPow.h>
#include <blaze/math/shims/Evaluate.h>
#include <blaze/math/shims/Invert.h>
#include <blaze/math/shims/IsZero.h>
#include <blaze/math/traits/MultTrait.h>
#include <blaze/math/typetraits/HasLoad.h>
#include <blaze/math/typetraits/HasSIMDAdd.h>
#include <blaze/math/typetraits/IsPadded.h>
#include <blaze/math/typetraits/IsSIMDEnabled.h>
#include <blaze/math/typetraits/UnderlyingBuiltin.h>
#include <blaze/system/Optimizations.h>
#include <blaze/util/Assert.h>
#include <blaze/util/FalseType.h>
#include <blaze/util/FunctionTrace.h>
#include <blaze/util/StaticAssert.h>
#include <blaze/util/TrueType.h>
#include <blaze/util/TypeList.h>
#include <blaze/util/Types.h>
#include <blaze/util/mpl/And.h>
#include <blaze/util/mpl/Bool.h>
#include <blaze/util/mpl/If.h>
#include <blaze/util/typetraits/HasMember.h>
#include <blaze/util/typetraits/RemoveConst.h>
#include <blaze/util/typetraits/RemoveReference.h>

#include <blaze_tensor/math/expressions/DenseArray.h>
#include <blaze_tensor/util/ArrayForEach.h>

namespace blaze {

//=================================================================================================
//
//  CLASS DEFINITION
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Auxiliary helper struct for the dense array norms.
// \ingroup dense_array
*/
template< typename MT       // Type of the dense array
        , typename Abs      // Type of the abs operation
        , typename Power >  // Type of the power operation
struct DArrNormHelper
{
   //**Type definitions****************************************************************************
   //! Element type of the dense array expression.
   using ET = ElementType_t<MT>;

   //! Composite type of the dense array expression.
   using CT = RemoveReference_t< CompositeType_t<MT> >;
   //**********************************************************************************************

   //**********************************************************************************************
   static constexpr bool value =
      ( useOptimizedKernels &&
        CT::simdEnabled &&
        If_t< HasSIMDEnabled_v<Abs> && HasSIMDEnabled_v<Power>
            , And< GetSIMDEnabled<Abs,ET>, GetSIMDEnabled<Power,ET> >
            , And< HasLoad<Abs>, HasLoad<Power> > >::value &&
        HasSIMDAdd_v< ElementType_t<CT>, ElementType_t<CT> > );
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  GLOBAL FUNCTIONS
//
//=================================================================================================


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Computes a custom norm for the given dense array.
// \ingroup dense_array
//
// \param dm The given dense array for the norm computation.
// \param abs The functor for the abs operation.
// \param power The functor for the power operation.
// \param root The functor for the root operation.
// \return The norm of the given dense array.
//
// This function computes a custom norm of the given dense array by means of the given functors.
// The following example demonstrates the computation of the L2 norm by means of the blaze::Pow2
// and blaze::Sqrt functors:

   \code
   blaze::DenseArray<double> A;
   // ... Resizing and initialization
   const double l2 = norm( A, blaze::Pow2(), blaze::Sqrt() );
   \endcode
*/
template< typename MT      // Type of the dense array
        , typename Abs     // Type of the abs operation
        , typename Power   // Type of the power operation
        , typename Root >  // Type of the root operation
decltype(auto) norm_backend( const DenseArray<MT>& dm, Abs abs, Power power, Root root )
{
   using CT = CompositeType_t<MT>;
   using ET = ElementType_t<MT>;
   using RT = decltype( evaluate( root( std::declval<ET>() ) ) );

   if( ArrayDimAnyOf( ( ~dm ).dimensions(),
          []( size_t i, size_t dim ) { return dim == 0; } ) ) {
      return RT{};
   }

   CT tmp( ~dm );

   BLAZE_INTERNAL_ASSERT( tmp.dimensions() == (~dm).dimensions(), "Invalid number of elements" );

   using AT = RemoveCV_t<RemoveReference_t< decltype( ~dm ) > >;
   constexpr size_t N = AT::num_dimensions();

   std::array< size_t, N > dims{};
   ET norm( power( abs( tmp( dims ) ) ) );

   ArrayForEachGrouped( ( ~dm ).dimensions(), 
      [&]( std::array< size_t, N > const& dims ) {
         norm += power( abs( tmp( dims ) ) );
      } );

   return evaluate( root( norm ) );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Computes the L2 norm for the given dense array.
// \ingroup dense_array
//
// \param dm The given dense array for the norm computation.
// \return The L2 norm of the given dense array.
//
// This function computes the L2 norm of the given dense array:

   \code
   blaze::DenseArray<double> A;
   // ... Resizing and initialization
   const double l2 = norm( A );
   \endcode
*/
template< typename MT > // Type of the dense array
decltype(auto) norm( const DenseArray<MT>& dm )
{
   BLAZE_FUNCTION_TRACE;

   return norm_backend( ~dm, SqrAbs(), Noop(), Sqrt() );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Computes the squared L2 norm for the given dense array.
// \ingroup dense_array
//
// \param dm The given dense array for the norm computation.
// \return The squared L2 norm of the given dense array.
//
// This function computes the squared L2 norm of the given dense array:

   \code
   blaze::DenseArray<double> A;
   // ... Resizing and initialization
   const double l2 = sqrNorm( A );
   \endcode
*/
template< typename MT > // Type of the dense array
decltype(auto) sqrNorm( const DenseArray<MT>& dm )
{
   BLAZE_FUNCTION_TRACE;

   return norm_backend( ~dm, SqrAbs(), Noop(), Noop() );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Computes the L1 norm for the given dense array.
// \ingroup dense_array
//
// \param dm The given dense array for the norm computation.
// \return The L1 norm of the given dense array.
//
// This function computes the L1 norm of the given dense array:

   \code
   blaze::DenseArray<double> A;
   // ... Resizing and initialization
   const double l1 = l1Norm( A );
   \endcode
*/
template< typename MT > // Type of the dense array
decltype(auto) l1Norm( const DenseArray<MT>& dm )
{
   BLAZE_FUNCTION_TRACE;

   return norm_backend( ~dm, Abs(), Noop(), Noop() );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Computes the L2 norm for the given dense array.
// \ingroup dense_array
//
// \param dm The given dense array for the norm computation.
// \return The L2 norm of the given dense array.
//
// This function computes the L2 norm of the given dense array:

   \code
   blaze::DenseArray<double> A;
   // ... Resizing and initialization
   const double l2 = l2Norm( A );
   \endcode
*/
template< typename MT > // Type of the dense array
decltype(auto) l2Norm( const DenseArray<MT>& dm )
{
   BLAZE_FUNCTION_TRACE;

   return norm_backend( ~dm, SqrAbs(), Noop(), Sqrt() );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Computes the L3 norm for the given dense array.
// \ingroup dense_array
//
// \param dm The given dense array for the norm computation.
// \return The L3 norm of the given dense array.
//
// This function computes the L3 norm of the given dense array:

   \code
   blaze::DenseArray<double> A;
   // ... Resizing and initialization
   const double l3 = l3Norm( A );
   \endcode
*/
template< typename MT > // Type of the dense array
decltype(auto) l3Norm( const DenseArray<MT>& dm )
{
   BLAZE_FUNCTION_TRACE;

   return norm_backend( ~dm, Abs(), Pow3(), Cbrt() );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Computes the L4 norm for the given dense array.
// \ingroup dense_array
//
// \param dm The given dense array for the norm computation.
// \return The L4 norm of the given dense array.
//
// This function computes the L4 norm of the given dense array:

   \code
   blaze::DenseArray<double> A;
   // ... Resizing and initialization
   const double l4 = l4Norm( A );
   \endcode
*/
template< typename MT > // Type of the dense array
decltype(auto) l4Norm( const DenseArray<MT>& dm )
{
   BLAZE_FUNCTION_TRACE;

   return norm_backend( ~dm, SqrAbs(), Pow2(), Qdrt() );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Computes the Lp norm for the given dense array.
// \ingroup dense_array
//
// \param dm The given dense array for the norm computation.
// \param p The norm parameter (p > 0).
// \return The Lp norm of the given dense array.
//
// This function computes the Lp norm of the given dense array, where the norm is specified by
// the runtime argument \a p:

   \code
   blaze::DenseArray<double> A;
   // ... Resizing and initialization
   const double lp = lpNorm( A, 2.3 );
   \endcode

// \note The norm parameter \a p is expected to be larger than 0. This precondition is only checked
// by a user assertion.
*/
template< typename MT    // Type of the dense array
        , typename ST >  // Type of the norm parameter
decltype(auto) lpNorm( const DenseArray<MT>& dm, ST p )
{
   BLAZE_FUNCTION_TRACE;

   BLAZE_USER_ASSERT( !isZero( p ), "Invalid p for Lp norm detected" );

   using ScalarType = MultTrait_t< UnderlyingBuiltin_t<MT>, decltype( inv( p ) ) >;
   return norm_backend( ~dm, Abs(), UnaryPow<ScalarType>( p ), UnaryPow<ScalarType>( inv( p ) ) );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Computes the Lp norm for the given dense array.
// \ingroup dense_array
//
// \param dm The given dense array for the norm computation.
// \return The Lp norm of the given dense array.
//
// This function computes the Lp norm of the given dense array, where the norm is specified by
// the runtime argument \a P:

   \code
   blaze::DenseArray<double> A;
   // ... Resizing and initialization
   const double lp = lpNorm<2>( A );
   \endcode

// \note The norm parameter \a P is expected to be larger than 0. A value of 0 results in a
// compile time error!.
*/
template< size_t P     // Compile time norm parameter
        , typename MT > // Type of the dense array
inline decltype(auto) lpNorm( const DenseArray<MT>& dm )
{
   BLAZE_STATIC_ASSERT_MSG( P > 0UL, "Invalid norm parameter detected" );

   using Norms = TypeList< L1Norm, L2Norm, L3Norm, L4Norm, LpNorm<P> >;
   using Norm  = typename TypeAt< Norms, min( P-1UL, 4UL ) >::Type;

   return Norm()( ~dm );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Computes the maximum norm for the given dense array.
// \ingroup dense_array
//
// \param dm The given dense array for the norm computation.
// \return The maximum norm of the given dense array.
//
// This function computes the maximum norm of the given dense array:

   \code
   blaze::DenseArray<double> A;
   // ... Resizing and initialization
   const double max = maxNorm( A );
   \endcode
*/
template< typename MT > // Type of the dense array
decltype(auto) maxNorm( const DenseArray<MT>& dm )
{
   BLAZE_FUNCTION_TRACE;

   return max( abs( ~dm ) );
}
//*************************************************************************************************

} // namespace blaze

#endif
