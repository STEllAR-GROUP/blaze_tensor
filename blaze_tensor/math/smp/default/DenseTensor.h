//=================================================================================================
/*!
//  \file blaze_tensor/math/smp/default/DenseTensor.h
//  \brief Header file for the default dense tensor SMP implementation
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

#ifndef _BLAZE_TENSOR_MATH_SMP_DEFAULT_DENSETENSOR_H_
#define _BLAZE_TENSOR_MATH_SMP_DEFAULT_DENSETENSOR_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <blaze/system/SMP.h>
#include <blaze/util/Assert.h>
#include <blaze/util/EnableIf.h>
#include <blaze/util/FunctionTrace.h>
#include <blaze/util/StaticAssert.h>

#include <blaze_tensor/math/expressions/Tensor.h>
#include <blaze_tensor/math/typetraits/IsDenseTensor.h>

namespace blaze {

//=================================================================================================
//
//  GLOBAL FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*!\name Dense tensor SMP functions */
//@{
template< typename TT1, typename TT2 >
inline EnableIf_t< IsDenseTensor_v<TT1> >
   smpAssign( Tensor<TT1>& lhs, const Tensor<TT2>& rhs );

template< typename TT1, typename TT2 >
inline EnableIf_t< IsDenseTensor_v<TT1> >
   smpAddAssign( Tensor<TT1>& lhs, const Tensor<TT2>& rhs );

template< typename TT1, typename TT2 >
inline EnableIf_t< IsDenseTensor_v<TT1> >
   smpSubAssign( Tensor<TT1>& lhs, const Tensor<TT2>& rhs );

template< typename TT1, typename TT2 >
inline EnableIf_t< IsDenseTensor_v<TT1> >
   smpSchurAssign( Tensor<TT1>& lhs, const Tensor<TT2>& rhs );
//@}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Default implementation of the SMP assignment of a tensor to a dense tensor.
// \ingroup smp
//
// \param lhs The target left-hand side dense tensor.
// \param rhs The right-hand side tensor to be assigned.
// \return void
//
// This function implements the default SMP assignment of a tensor to a dense tensor.\n
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename TT1  // Type of the left-hand side dense tensor
        , typename TT2 > // Type of the right-hand side tensor
inline EnableIf_t< IsDenseTensor_v<TT1> >
   smpAssign( Tensor<TT1>& lhs, const Tensor<TT2>& rhs )
{
   BLAZE_FUNCTION_TRACE;

   BLAZE_INTERNAL_ASSERT( (~lhs).rows()    == (~rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( (~lhs).columns() == (~rhs).columns(), "Invalid number of columns" );
   BLAZE_INTERNAL_ASSERT( (~lhs).pages()   == (~rhs).pages(),   "Invalid number of pages"   );

   assign( ~lhs, ~rhs );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Default implementation of the SMP addition assignment of a tensor to a dense tensor.
// \ingroup smp
//
// \param lhs The target left-hand side dense tensor.
// \param rhs The right-hand side tensor to be added.
// \return void
//
// This function implements the default SMP addition assignment of a tensor to a dense tensor.\n
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename TT1  // Type of the left-hand side dense tensor
        , typename TT2 > // Type of the right-hand side tensor
inline EnableIf_t< IsDenseTensor_v<TT1> >
   smpAddAssign( Tensor<TT1>& lhs, const Tensor<TT2>& rhs )
{
   BLAZE_FUNCTION_TRACE;

   BLAZE_INTERNAL_ASSERT( (~lhs).rows()    == (~rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( (~lhs).columns() == (~rhs).columns(), "Invalid number of columns" );
   BLAZE_INTERNAL_ASSERT( (~lhs).pages()   == (~rhs).pages(),   "Invalid number of pages"   );

   addAssign( ~lhs, ~rhs );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Default implementation of the SMP subtraction assignment of a tensor to dense tensor.
// \ingroup smp
//
// \param lhs The target left-hand side dense tensor.
// \param rhs The right-hand side tensor to be subtracted.
// \return void
//
// This function implements the default SMP subtraction assignment of a tensor to a dense tensor.\n
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename TT1  // Type of the left-hand side dense tensor
        , typename TT2 > // Type of the right-hand side tensor
inline EnableIf_t< IsDenseTensor_v<TT1> >
   smpSubAssign( Tensor<TT1>& lhs, const Tensor<TT2>& rhs )
{
   BLAZE_FUNCTION_TRACE;

   BLAZE_INTERNAL_ASSERT( (~lhs).rows()    == (~rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( (~lhs).columns() == (~rhs).columns(), "Invalid number of columns" );
   BLAZE_INTERNAL_ASSERT( (~lhs).pages()   == (~rhs).pages(),   "Invalid number of pages"   );

   subAssign( ~lhs, ~rhs );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Default implementation of the SMP Schur product assignment of a tensor to dense tensor.
// \ingroup smp
//
// \param lhs The target left-hand side dense tensor.
// \param rhs The right-hand side tensor for the Schur product.
// \return void
//
// This function implements the default SMP Schur product assignment of a tensor to a dense
// tensor.\n
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename TT1  // Type of the left-hand side dense tensor
        , typename TT2 > // Type of the right-hand side tensor
inline EnableIf_t< IsDenseTensor_v<TT1> >
   smpSchurAssign( Tensor<TT1>& lhs, const Tensor<TT2>& rhs )
{
   BLAZE_FUNCTION_TRACE;

   BLAZE_INTERNAL_ASSERT( (~lhs).rows()    == (~rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( (~lhs).columns() == (~rhs).columns(), "Invalid number of columns" );
   BLAZE_INTERNAL_ASSERT( (~lhs).pages()   == (~rhs).pages(),   "Invalid number of pages"   );

   schurAssign( ~lhs, ~rhs );
}
//*************************************************************************************************




//=================================================================================================
//
//  COMPILE TIME CONSTRAINT
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
namespace {

//BLAZE_STATIC_ASSERT( !BLAZE_HPX_PARALLEL_MODE           );
BLAZE_STATIC_ASSERT( !BLAZE_CPP_THREADS_PARALLEL_MODE   );
BLAZE_STATIC_ASSERT( !BLAZE_BOOST_THREADS_PARALLEL_MODE );
BLAZE_STATIC_ASSERT( !BLAZE_OPENMP_PARALLEL_MODE        );

}
/*! \endcond */
//*************************************************************************************************

} // namespace blaze

#endif
