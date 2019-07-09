//=================================================================================================
/*!
//  \file blaze_tensor/math/smp/default/DenseArray.h
//  \brief Header file for the default dense array SMP implementation
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

#ifndef _BLAZE_TENSOR_MATH_SMP_DEFAULT_DENSEARRAY_H_
#define _BLAZE_TENSOR_MATH_SMP_DEFAULT_DENSEARRAY_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <blaze/system/SMP.h>
#include <blaze/util/Assert.h>
#include <blaze/util/EnableIf.h>
#include <blaze/util/FunctionTrace.h>
#include <blaze/util/StaticAssert.h>

#include <blaze_tensor/math/expressions/Array.h>
#include <blaze_tensor/math/typetraits/IsDenseArray.h>

namespace blaze {

//=================================================================================================
//
//  GLOBAL FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*!\name Dense array SMP functions */
//@{
template< typename TT1, typename TT2 >
inline EnableIf_t< IsDenseArray_v<TT1> >
   smpAssign( Array<TT1>& lhs, const Array<TT2>& rhs );

template< typename TT1, typename TT2 >
inline EnableIf_t< IsDenseArray_v<TT1> >
   smpAddAssign( Array<TT1>& lhs, const Array<TT2>& rhs );

template< typename TT1, typename TT2 >
inline EnableIf_t< IsDenseArray_v<TT1> >
   smpSubAssign( Array<TT1>& lhs, const Array<TT2>& rhs );

template< typename TT1, typename TT2 >
inline EnableIf_t< IsDenseArray_v<TT1> >
   smpSchurAssign( Array<TT1>& lhs, const Array<TT2>& rhs );
//@}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Default implementation of the SMP assignment of a array to a dense array.
// \ingroup smp
//
// \param lhs The target left-hand side dense array.
// \param rhs The right-hand side array to be assigned.
// \return void
//
// This function implements the default SMP assignment of a array to a dense array.\n
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename TT1  // Type of the left-hand side dense array
        , typename TT2 > // Type of the right-hand side array
inline EnableIf_t< IsDenseArray_v<TT1> >
   smpAssign( Array<TT1>& lhs, const Array<TT2>& rhs )
{
   BLAZE_FUNCTION_TRACE;

   BLAZE_INTERNAL_ASSERT( (~lhs).dimensions() == (~rhs).dimensions(), "Invalid dimensions"    );

   assign( ~lhs, ~rhs );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Default implementation of the SMP addition assignment of a array to a dense array.
// \ingroup smp
//
// \param lhs The target left-hand side dense array.
// \param rhs The right-hand side array to be added.
// \return void
//
// This function implements the default SMP addition assignment of a array to a dense array.\n
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename TT1  // Type of the left-hand side dense array
        , typename TT2 > // Type of the right-hand side array
inline EnableIf_t< IsDenseArray_v<TT1> >
   smpAddAssign( Array<TT1>& lhs, const Array<TT2>& rhs )
{
   BLAZE_FUNCTION_TRACE;

   BLAZE_INTERNAL_ASSERT( (~lhs).dimensions() == (~rhs).dimensions(), "Invalid dimensions"    );

   addAssign( ~lhs, ~rhs );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Default implementation of the SMP subtraction assignment of a array to dense array.
// \ingroup smp
//
// \param lhs The target left-hand side dense array.
// \param rhs The right-hand side array to be subtracted.
// \return void
//
// This function implements the default SMP subtraction assignment of a array to a dense array.\n
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename TT1  // Type of the left-hand side dense array
        , typename TT2 > // Type of the right-hand side array
inline EnableIf_t< IsDenseArray_v<TT1> >
   smpSubAssign( Array<TT1>& lhs, const Array<TT2>& rhs )
{
   BLAZE_FUNCTION_TRACE;

   BLAZE_INTERNAL_ASSERT( (~lhs).dimensions() == (~rhs).dimensions(), "Invalid dimensions"    );

   subAssign( ~lhs, ~rhs );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Default implementation of the SMP Schur product assignment of a array to dense array.
// \ingroup smp
//
// \param lhs The target left-hand side dense array.
// \param rhs The right-hand side array for the Schur product.
// \return void
//
// This function implements the default SMP Schur product assignment of a array to a dense
// array.\n
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename TT1  // Type of the left-hand side dense array
        , typename TT2 > // Type of the right-hand side array
inline EnableIf_t< IsDenseArray_v<TT1> >
   smpSchurAssign( Array<TT1>& lhs, const Array<TT2>& rhs )
{
   BLAZE_FUNCTION_TRACE;

   BLAZE_INTERNAL_ASSERT( (~lhs).dimensions() == (~rhs).dimensions(), "Invalid dimensions"    );

   schurAssign( ~lhs, ~rhs );
}
//*************************************************************************************************


//=================================================================================================
//
//  MULTIPLICATION ASSIGNMENT
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Default implementation of the SMP multiplication assignment to a dense array.
// \ingroup smp
//
// \param lhs The target left-hand side dense array.
// \param rhs The right-hand side array to be multiplied.
// \return void
//
// This function implements the default HPX-based SMP multiplication assignment to a dense
// array.\n
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename AT1  // Type of the left-hand side dense array
        , typename AT2 > // Type of the right-hand side array
inline EnableIf_t< IsDenseArray_v<AT1> >
   smpMultAssign( Array<AT1>& lhs, const Array<AT2>& rhs )
{
   BLAZE_FUNCTION_TRACE;

   BLAZE_INTERNAL_ASSERT( (~lhs).dimensions() == (~rhs).dimensions(), "Invalid dimensions"    );

   multAssign( ~lhs, ~rhs );
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

BLAZE_STATIC_ASSERT( !BLAZE_HPX_PARALLEL_MODE           );
BLAZE_STATIC_ASSERT( !BLAZE_CPP_THREADS_PARALLEL_MODE   );
BLAZE_STATIC_ASSERT( !BLAZE_BOOST_THREADS_PARALLEL_MODE );
BLAZE_STATIC_ASSERT( !BLAZE_OPENMP_PARALLEL_MODE        );

}
/*! \endcond */
//*************************************************************************************************

} // namespace blaze

#endif
