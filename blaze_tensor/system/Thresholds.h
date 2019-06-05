//=================================================================================================
/*!
//  \file blaze/system/Thresholds.h
//  \brief Header file for the thresholds for matrix/vector and matrix/matrix multiplications
//
//  Copyright (C) 2012-2019 Klaus Iglberger - All Rights Reserved
//  Copyright (C) 2018-2019 Hartmut Kaiser - All Rights Reserved
//  Copyright (C) 2019 Bita Hasheminezhad - All Rights Reserved
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

#ifndef _BLAZE_TENSOR_SYSTEM_THRESHOLDS_H_
#define _BLAZE_TENSOR_SYSTEM_THRESHOLDS_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <blaze/system/Debugging.h>
#include <blaze/util/StaticAssert.h>
#include <blaze/util/Types.h>




//=================================================================================================
//
//  THRESHOLDS
//
//=================================================================================================

#include <blaze/config/Thresholds.h>
#include <blaze_tensor/config/Thresholds.h>



namespace blaze {

//=================================================================================================
//
//  BLAS THRESHOLDS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Row-major dense matrix/dense vector multiplication threshold.
// \ingroup config
//
// This debug value is used instead of the BLAZE_DTENSDVECMULT_THRESHOLD while the Blaze debug
// mode is active. It specifies the threshold between the application of the custom Blaze kernels
// and the BLAS kernels for the row-major dense matrix/dense vector multiplication. In case the
// number of elements in the dense matrix is equal or higher than this value, the BLAS kernels
// are preferred over the custom Blaze kernels. In case the number of elements in the dense
// matrix is smaller, the Blaze kernels are used.
*/
constexpr size_t DTENSDVECMULT_DEBUG_THRESHOLD = 256UL;
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
constexpr size_t DTENSDVECMULT_THRESHOLD  = ( BLAZE_DEBUG_MODE ? DTENSDVECMULT_DEBUG_THRESHOLD  : BLAZE_DTENSDVECMULT_THRESHOLD  );

/*! \endcond */
//*************************************************************************************************


//=================================================================================================
//
//  SMP THRESHOLDS
//
//=================================================================================================


//*************************************************************************************************
/*!\brief SMP dense matrix assignment threshold.
// \ingroup config
//
// This debug value is used instead of the BLAZE_SMP_DMATASSIGN_THRESHOLD while the Blaze
// debug mode is active. It specifies when an assignment with a simple dense matrix can be executed
// in parallel. In case the number of elements of the target matrix is larger or equal to this
// threshold, the operation is executed in parallel. If the number of elements is below this
// threshold the operation is executed single-threaded.
*/
constexpr size_t SMP_DTENSASSIGN_DEBUG_THRESHOLD = 256UL;
//*************************************************************************************************


//*************************************************************************************************
/*!\brief SMP row-major dense matrix/dense vector multiplication threshold.
// \ingroup config
//
// This debug value is used instead of the BLAZE_SMP_DTENSDVECMULT_THRESHOLD while the Blaze
// debug mode is active. It specifies when a row-major dense matrix/dense vector multiplication
// can be executed in parallel. In case the number of elements of the target vector is larger or
// equal to this threshold, the operation is executed in parallel. If the number of elements is
// below this threshold the operation is executed single-threaded.
*/
constexpr size_t SMP_DTENSDVECMULT_DEBUG_THRESHOLD = 16UL;
//*************************************************************************************************


//*************************************************************************************************
/*!\brief SMP row-major dense matrix/row-major dense matrix Schur product threshold.
// \ingroup config
//
// This debug value is used instead of the BLAZE_SMP_DMATDMATSCHUR_THRESHOLD while the
// Blaze debug mode is active. It specifies when a row-major dense matrix/row-major dense matrix
// Schur product can be executed in parallel. This threshold affects both Schur products between
// two row-major matrices or two column-major dense matrices. In case the number of elements of
// the target matrix is larger or equal to this threshold, the operation is executed in parallel.
// If the number of elements is below this threshold the operation is executed single-threaded.
*/
constexpr size_t SMP_DTENSDMATSCHUR_DEBUG_THRESHOLD = 256UL;
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
constexpr size_t SMP_DTENSASSIGN_THRESHOLD    = ( BLAZE_DEBUG_MODE ? SMP_DTENSASSIGN_DEBUG_THRESHOLD    : BLAZE_SMP_DTENSASSIGN_THRESHOLD     );
constexpr size_t SMP_DTENSDMATSCHUR_THRESHOLD = ( BLAZE_DEBUG_MODE ? SMP_DTENSDMATSCHUR_DEBUG_THRESHOLD : BLAZE_SMP_DTENSDMATSCHUR_THRESHOLD  );
constexpr size_t SMP_DTENSDVECMULT_THRESHOLD  = ( BLAZE_DEBUG_MODE ? SMP_DTENSDVECMULT_DEBUG_THRESHOLD  : BLAZE_SMP_DTENSDVECMULT_THRESHOLD  );
/*! \endcond */
//*************************************************************************************************

} // namespace blaze




//=================================================================================================
//
//  COMPILE TIME CONSTRAINT
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
namespace {

BLAZE_STATIC_ASSERT( blaze::DTENSDVECMULT_THRESHOLD  > 0UL );

BLAZE_STATIC_ASSERT( blaze::SMP_DTENSASSIGN_THRESHOLD    >= 0UL );
BLAZE_STATIC_ASSERT( blaze::SMP_DTENSDMATSCHUR_THRESHOLD >= 0UL );
BLAZE_STATIC_ASSERT( blaze::SMP_DTENSDVECMULT_THRESHOLD  >= 0UL );

}
/*! \endcond */
//*************************************************************************************************

#endif
