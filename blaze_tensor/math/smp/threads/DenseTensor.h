//=================================================================================================
/*!
//  \file blaze/math/smp/threads/DenseTensor.h
//  \brief Header file for the C++11/Boost thread-based dense tensor SMP implementation
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

#ifndef _BLAZE_TENSOR_MATH_SMP_THREADS_DENSETENSOR_H_
#define _BLAZE_TENSOR_MATH_SMP_THREADS_DENSETENSOR_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <blaze/math/Aliases.h>
#include <blaze/math/StorageOrder.h>
#include <blaze/math/constraints/SMPAssignable.h>
#include <blaze/math/simd/SIMDTrait.h>
#include <blaze/math/smp/ParallelSection.h>
#include <blaze/math/smp/SerialSection.h>
#include <blaze/math/smp/ThreadMapping.h>
#include <blaze/math/smp/threads/ThreadBackend.h>
#include <blaze/math/typetraits/IsSIMDCombinable.h>
#include <blaze/math/typetraits/IsSMPAssignable.h>
#include <blaze/math/views/Submatrix.h>
#include <blaze/system/SMP.h>
#include <blaze/util/Assert.h>
#include <blaze/util/EnableIf.h>
#include <blaze/util/FunctionTrace.h>
#include <blaze/util/StaticAssert.h>
#include <blaze/util/Types.h>
#include <blaze/util/algorithms/Min.h>

#include <blaze_tensor/math/expressions/DenseTensor.h>
#include <blaze_tensor/math/smp/TensorThreadMapping.h>
#include <blaze_tensor/math/typetraits/IsDenseTensor.h>
#include <blaze_tensor/math/views/PageSlice.h>

namespace blaze {

//=================================================================================================
//
//  THREAD-BASED ASSIGNMENT KERNELS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Backend of the C++11/Boost thread-based SMP assignment of a dense tensor to a dense tensor.
// \ingroup math
//
// \param lhs The target left-hand side dense tensor.
// \param rhs The right-hand side dense tensor to be assigned.
// \param op The (compound) assignment operation.
// \return void
//
// This function is the backend implementation of the C++11/Boost thread-based SMP assignment
// of a dense tensor to a dense tensor.\n
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename MT1   // Type of the left-hand side dense tensor
        , typename MT2   // Type of the right-hand side dense tensor
        , typename OP >  // Type of the assignment operation
void threadAssign( DenseTensor<MT1>& lhs, const DenseTensor<MT2>& rhs, OP op )
{
   BLAZE_FUNCTION_TRACE;

   BLAZE_INTERNAL_ASSERT( isParallelSectionActive(), "Invalid call outside a parallel section" );

   using ET1 = ElementType_t<MT1>;
   using ET2 = ElementType_t<MT2>;

   constexpr bool simdEnabled( MT1::simdEnabled && MT2::simdEnabled && IsSIMDCombinable_v<ET1,ET2> );
   constexpr size_t SIMDSIZE( SIMDTrait< ElementType_t<MT1> >::size );

   const bool lhsAligned( (*lhs).isAligned() );
   const bool rhsAligned( (*rhs).isAligned() );

   const ThreadMapping threads( createThreadMapping( TheThreadBackend::size(), *rhs ) );

   const size_t addon1     ( ( ( (*rhs).rows() % threads.first ) != 0UL )? 1UL : 0UL );
   const size_t equalShare1( (*rhs).rows() / threads.first + addon1 );
   const size_t rest1      ( equalShare1 & ( SIMDSIZE - 1UL ) );
   const size_t rowsPerThread( ( simdEnabled && rest1 )?( equalShare1 - rest1 + SIMDSIZE ):( equalShare1 ) );

   const size_t addon2     ( ( ( (*rhs).columns() % threads.second ) != 0UL )? 1UL : 0UL );
   const size_t equalShare2( (*rhs).columns() / threads.second + addon2 );
   const size_t rest2      ( equalShare2 & ( SIMDSIZE - 1UL ) );
   const size_t colsPerThread( ( simdEnabled && rest2 )?( equalShare2 - rest2 + SIMDSIZE ):( equalShare2 ) );

   for( size_t i=0UL; i<threads.first; ++i )
   {
      const size_t row( i*rowsPerThread );

      if( row >= (*lhs).rows() )
         continue;

      for( size_t j=0UL; j<threads.second; ++j )
      {
         const size_t column( j*colsPerThread );

         if( column >= (*rhs).columns() )
            continue;

         for (size_t k = 0; k != (*rhs).pages(); ++k)
         {
            const size_t m( min( rowsPerThread, (*lhs).rows()    - row    ) );
            const size_t n( min( colsPerThread, (*rhs).columns() - column ) );

            auto lhs_slice = pageslice( *lhs, k );
            auto rhs_slice = pageslice( *rhs, k );

            if( simdEnabled && lhsAligned && rhsAligned ) {
               auto       target( submatrix<aligned>( *lhs_slice, row, column, m, n, unchecked ) );
               const auto source( submatrix<aligned>( *rhs_slice, row, column, m, n, unchecked ) );
               TheThreadBackend::schedule( target, source, op );
            }
            else if( simdEnabled && lhsAligned ) {
               auto       target( submatrix<aligned>  ( *lhs_slice, row, column, m, n, unchecked ) );
               const auto source( submatrix<unaligned>( *rhs_slice, row, column, m, n, unchecked ) );
               TheThreadBackend::schedule( target, source, op );
            }
            else if( simdEnabled && rhsAligned ) {
               auto       target( submatrix<unaligned>( *lhs_slice, row, column, m, n, unchecked ) );
               const auto source( submatrix<aligned>  ( *rhs_slice, row, column, m, n, unchecked ) );
               TheThreadBackend::schedule( target, source, op );
            }
            else {
               auto       target( submatrix<unaligned>( *lhs_slice, row, column, m, n, unchecked ) );
               const auto source( submatrix<unaligned>( *rhs_slice, row, column, m, n, unchecked ) );
               TheThreadBackend::schedule( target, source, op );
            }
         }
      }
   }

   TheThreadBackend::wait();
}
/*! \endcond */
//*************************************************************************************************


//=================================================================================================
//
//  PLAIN ASSIGNMENT
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Default implementation of the C++11/Boost thread-based SMP assignment to a dense tensor.
// \ingroup smp
//
// \param lhs The target left-hand side dense tensor.
// \param rhs The right-hand side tensor to be assigned.
// \return void
//
// This function implements the default C++11/Boost thread-based SMP assignment to a dense tensor.
// Due to the explicit application of the SFINAE principle, this function can only be selected by
// the compiler in case both operands are SMP-assignable and the element types of both operands
// are not SMP-assignable.\n
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename MT1  // Type of the left-hand side dense tensor
        , typename MT2 > // Type of the right-hand side tensor
inline EnableIf_t< IsDenseTensor_v<MT1> && ( !IsSMPAssignable_v<MT1> || !IsSMPAssignable_v<MT2> ) >
   smpAssign( Tensor<MT1>& lhs, const Tensor<MT2>& rhs )
{
   BLAZE_FUNCTION_TRACE;

   BLAZE_INTERNAL_ASSERT( (*lhs).rows()    == (*rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( (*lhs).columns() == (*rhs).columns(), "Invalid number of columns" );
   BLAZE_INTERNAL_ASSERT( (*lhs).pages()   == (*rhs).pages(),   "Invalid number of pages"   );

   assign( *lhs, *rhs );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Implementation of the C++11/Boost thread-based SMP assignment to a dense tensor.
// \ingroup math
//
// \param lhs The target left-hand side dense tensor.
// \param rhs The right-hand side tensor to be assigned.
// \return void
//
// This function implements the C++11/Boost thread-based SMP assignment to a dense tensor. Due
// to the explicit application of the SFINAE principle, this function can only be selected by
// the compiler in case both operands are SMP-assignable and the element types of both operands
// are not SMP-assignable.\n
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename MT1  // Type of the left-hand side dense tensor
        , typename MT2 > // Type of the right-hand side tensor
inline EnableIf_t< IsDenseTensor_v<MT1> && IsSMPAssignable_v<MT1> && IsSMPAssignable_v<MT2> >
   smpAssign( Tensor<MT1>& lhs, const Tensor<MT2>& rhs )
{
   BLAZE_FUNCTION_TRACE;

   BLAZE_CONSTRAINT_MUST_NOT_BE_SMP_ASSIGNABLE( ElementType_t<MT1> );
   BLAZE_CONSTRAINT_MUST_NOT_BE_SMP_ASSIGNABLE( ElementType_t<MT2> );

   BLAZE_INTERNAL_ASSERT( (*lhs).rows()    == (*rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( (*lhs).columns() == (*rhs).columns(), "Invalid number of columns" );
   BLAZE_INTERNAL_ASSERT( (*lhs).pages()   == (*rhs).pages(),   "Invalid number of pages"   );

   BLAZE_PARALLEL_SECTION
   {
      if( isSerialSectionActive() || !(*rhs).canSMPAssign() ) {
         assign( *lhs, *rhs );
      }
      else {
         threadAssign( *lhs, *rhs, []( auto& a, const auto& b ){ assign( a, b ); } );
      }
   }
}
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  ADDITION ASSIGNMENT
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Default implementation of the C++11/Boost thread-based SMP addition assignment to a
//        dense tensor.
// \ingroup smp
//
// \param lhs The target left-hand side dense tensor.
// \param rhs The right-hand side tensor to be added.
// \return void
//
// This function implements the default C++11/Boost thread-based SMP addition assignment to a
// dense tensor. Due to the explicit application of the SFINAE principle, this function can only
// be selected by the compiler in case both operands are SMP-assignable and the element types of
// both operands are not SMP-assignable.\n
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename MT1  // Type of the left-hand side dense tensor
        , typename MT2 > // Type of the right-hand side tensor
inline EnableIf_t< IsDenseTensor_v<MT1> && ( !IsSMPAssignable_v<MT1> || !IsSMPAssignable_v<MT2> ) >
   smpAddAssign( Tensor<MT1>& lhs, const Tensor<MT2>& rhs )
{
   BLAZE_FUNCTION_TRACE;

   BLAZE_INTERNAL_ASSERT( (*lhs).rows()    == (*rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( (*lhs).columns() == (*rhs).columns(), "Invalid number of columns" );
   BLAZE_INTERNAL_ASSERT( (*lhs).pages()   == (*rhs).pages(),   "Invalid number of pages"   );

   addAssign( *lhs, *rhs );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Implementation of the C++11/Boost thread-based SMP addition assignment to a dense tensor.
// \ingroup math
//
// \param lhs The target left-hand side dense tensor.
// \param rhs The right-hand side tensor to be added.
// \return void
//
// This function implements the C++11/Boost thread-based SMP addition assignment to a dense tensor.
// Due to the explicit application of the SFINAE principle, this function can only be selected by
// the compiler in case both operands are SMP-assignable and the element types of both operands
// are not SMP-assignable.\n
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename MT1  // Type of the left-hand side dense tensor
        , typename MT2 > // Type of the right-hand side tensor
inline EnableIf_t< IsDenseTensor_v<MT1> && IsSMPAssignable_v<MT1> && IsSMPAssignable_v<MT2> >
   smpAddAssign( Tensor<MT1>& lhs, const Tensor<MT2>& rhs )
{
   BLAZE_FUNCTION_TRACE;

   BLAZE_CONSTRAINT_MUST_NOT_BE_SMP_ASSIGNABLE( ElementType_t<MT1> );
   BLAZE_CONSTRAINT_MUST_NOT_BE_SMP_ASSIGNABLE( ElementType_t<MT2> );

   BLAZE_INTERNAL_ASSERT( (*lhs).rows()    == (*rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( (*lhs).columns() == (*rhs).columns(), "Invalid number of columns" );
   BLAZE_INTERNAL_ASSERT( (*lhs).pages()   == (*rhs).pages(),   "Invalid number of pages"   );

   BLAZE_PARALLEL_SECTION
   {
      if( isSerialSectionActive() || !(*rhs).canSMPAssign() ) {
         addAssign( *lhs, *rhs );
      }
      else {
         threadAssign( *lhs, *rhs, []( auto& a, const auto& b ){ addAssign( a, b ); } );
      }
   }
}
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  SUBTRACTION ASSIGNMENT
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Default implementation of the C++11/Boost thread-based SMP subtracction assignment to a
//        dense tensor.
// \ingroup smp
//
// \param lhs The target left-hand side dense tensor.
// \param rhs The right-hand side tensor to be subtracted.
// \return void
//
// This function implements the default C++11/Boost thread-based SMP subtraction assignment to a
// dense tensor. Due to the explicit application of the SFINAE principle, this function can only
// be selected by the compiler in case both operands are SMP-assignable and the element types of
// both operands are not SMP-assignable.\n
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename MT1  // Type of the left-hand side dense tensor
        , typename MT2 > // Type of the right-hand side tensor
inline EnableIf_t< IsDenseTensor_v<MT1> && ( !IsSMPAssignable_v<MT1> || !IsSMPAssignable_v<MT2> ) >
   smpSubAssign( Tensor<MT1>& lhs, const Tensor<MT2>& rhs )
{
   BLAZE_FUNCTION_TRACE;

   BLAZE_INTERNAL_ASSERT( (*lhs).rows()    == (*rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( (*lhs).columns() == (*rhs).columns(), "Invalid number of columns" );
   BLAZE_INTERNAL_ASSERT( (*lhs).pages()   == (*rhs).pages(),   "Invalid number of pages"   );

   subAssign( *lhs, *rhs );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Implementation of the C++11/Boost thread-based SMP subtracction assignment to a dense
//        tensor.
// \ingroup smp
//
// \param lhs The target left-hand side dense tensor.
// \param rhs The right-hand side tensor to be subtracted.
// \return void
//
// This function implements the default C++11/Boost thread-based SMP subtraction assignment of a
// tensor to a dense tensor. Due to the explicit application of the SFINAE principle, this function
// can only be selected by the compiler in case both operands are SMP-assignable and the element
// types of both operands are not SMP-assignable.\n
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename MT1  // Type of the left-hand side dense tensor
        , typename MT2 > // Type of the right-hand side tensor
inline EnableIf_t< IsDenseTensor_v<MT1> && IsSMPAssignable_v<MT1> && IsSMPAssignable_v<MT2> >
   smpSubAssign( Tensor<MT1>& lhs, const Tensor<MT2>& rhs )
{
   BLAZE_FUNCTION_TRACE;

   BLAZE_CONSTRAINT_MUST_NOT_BE_SMP_ASSIGNABLE( ElementType_t<MT1> );
   BLAZE_CONSTRAINT_MUST_NOT_BE_SMP_ASSIGNABLE( ElementType_t<MT2> );

   BLAZE_INTERNAL_ASSERT( (*lhs).rows()    == (*rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( (*lhs).columns() == (*rhs).columns(), "Invalid number of columns" );
   BLAZE_INTERNAL_ASSERT( (*lhs).pages()   == (*rhs).pages(),   "Invalid number of pages"   );

   BLAZE_PARALLEL_SECTION
   {
      if( isSerialSectionActive() || !(*rhs).canSMPAssign() ) {
         subAssign( *lhs, *rhs );
      }
      else {
         threadAssign( *lhs, *rhs, []( auto& a, const auto& b ){ subAssign( a, b ); } );
      }
   }
}
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  SCHUR PRODUCT ASSIGNMENT
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Default implementation of the C++11/Boost thread-based SMP Schur product assignment to
//        a dense tensor.
// \ingroup smp
//
// \param lhs The target left-hand side dense tensor.
// \param rhs The right-hand side tensor for the Schur product.
// \return void
//
// This function implements the default C++11/Boost thread-based SMP Schur product assignment to
// a dense tensor. Due to the explicit application of the SFINAE principle, this function can only
// be selected by the compiler in case both operands are SMP-assignable and the element types of
// both operands are not SMP-assignable.\n
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename MT1  // Type of the left-hand side dense tensor
        , typename MT2 > // Type of the right-hand side tensor
inline EnableIf_t< IsDenseTensor_v<MT1> && ( !IsSMPAssignable_v<MT1> || !IsSMPAssignable_v<MT2> ) >
   smpSchurAssign( Tensor<MT1>& lhs, const Tensor<MT2>& rhs )
{
   BLAZE_FUNCTION_TRACE;

   BLAZE_INTERNAL_ASSERT( (*lhs).rows()    == (*rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( (*lhs).columns() == (*rhs).columns(), "Invalid number of columns" );
   BLAZE_INTERNAL_ASSERT( (*lhs).pages()   == (*rhs).pages(),   "Invalid number of pages"   );

   schurAssign( *lhs, *rhs );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Implementation of the C++11/Boost thread-based SMP Schur product assignment to a dense
//        tensor.
// \ingroup math
//
// \param lhs The target left-hand side dense tensor.
// \param rhs The right-hand side tensor for the Schur product.
// \return void
//
// This function implements the C++11/Boost thread-based SMP Schur product assignment to a dense
// tensor. Due to the explicit application of the SFINAE principle, this function can only be
// selected by the compiler in case both operands are SMP-assignable and the element types of
// both operands are not SMP-assignable.\n
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename MT1  // Type of the left-hand side dense tensor
        , typename MT2 > // Type of the right-hand side tensor
inline EnableIf_t< IsDenseTensor_v<MT1> && IsSMPAssignable_v<MT1> && IsSMPAssignable_v<MT2> >
   smpSchurAssign( Tensor<MT1>& lhs, const Tensor<MT2>& rhs )
{
   BLAZE_FUNCTION_TRACE;

   BLAZE_CONSTRAINT_MUST_NOT_BE_SMP_ASSIGNABLE( ElementType_t<MT1> );
   BLAZE_CONSTRAINT_MUST_NOT_BE_SMP_ASSIGNABLE( ElementType_t<MT2> );

   BLAZE_INTERNAL_ASSERT( (*lhs).rows()    == (*rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( (*lhs).columns() == (*rhs).columns(), "Invalid number of columns" );
   BLAZE_INTERNAL_ASSERT( (*lhs).pages()   == (*rhs).pages(),   "Invalid number of pages"   );

   BLAZE_PARALLEL_SECTION
   {
      if( isSerialSectionActive() || !(*rhs).canSMPAssign() ) {
         schurAssign( *lhs, *rhs );
      }
      else {
         threadAssign( *lhs, *rhs, []( auto& a, const auto& b ){ schurAssign( a, b ); } );
      }
   }
}
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  MULTIPLICATION ASSIGNMENT
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Default implementation of the C++11/Boost thread-based SMP multiplication assignment
//        to a dense tensor.
// \ingroup smp
//
// \param lhs The target left-hand side dense tensor.
// \param rhs The right-hand side tensor to be multiplied.
// \return void
//
// This function implements the default C++11/Boost thread-based SMP multiplication assignment
// to a dense tensor.\n
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename MT1  // Type of the left-hand side dense tensor
        , typename MT2 > // Type of the right-hand side tensor
inline EnableIf_t< IsDenseTensor_v<MT1> >
   smpMultAssign( Tensor<MT1>& lhs, const Tensor<MT2>& rhs )
{
   BLAZE_FUNCTION_TRACE;

   BLAZE_INTERNAL_ASSERT( (*lhs).rows()    == (*rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( (*lhs).columns() == (*rhs).columns(), "Invalid number of columns" );
   BLAZE_INTERNAL_ASSERT( (*lhs).pages()   == (*rhs).pages(),   "Invalid number of pages"   );

   multAssign( *lhs, *rhs );
}
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  COMPILE TIME CONSTRAINT
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
namespace {

BLAZE_STATIC_ASSERT( BLAZE_CPP_THREADS_PARALLEL_MODE || BLAZE_BOOST_THREADS_PARALLEL_MODE );

}
/*! \endcond */
//*************************************************************************************************

} // namespace blaze

#endif
