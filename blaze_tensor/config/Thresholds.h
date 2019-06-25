//=================================================================================================
/*!
//  \file blaze_tensor/config/Thresholds.h
//  \brief Configuration of the thresholds for matrix/vector and matrix/matrix multiplications
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


//=================================================================================================
//
//  BLAS THRESHOLDS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Row-major dense matrix/dense vector multiplication threshold.
// \ingroup config
//
// This setting specifies the threshold between the application of the custom Blaze kernels
// and the BLAS kernels for the row-major dense matrix/dense vector multiplication. In case
// the number of elements in the dense matrix is equal or higher than this value, the BLAS
// kernels are preferred over the custom Blaze kernels. In case the number of elements in the
// dense matrix is smaller, the Blaze kernels are used.
//
// The default setting for this threshold is 4000000 (which for instance corresponds to a matrix
// size of \f$ 2000 \times 2000 \f$). Note that in case the Blaze debug mode is active, this
// threshold will be replaced by the blaze::DMATDVECMULT_DEBUG_THRESHOLD value.
//
// \note It is possible to specify this threshold via command line or by defining this symbol
// manually before including any Blaze header file:

   \code
   #define BLAZE_DTENSDVECMULT_THRESHOLD 4000000UL
   #include <blaze/Blaze.h>
   \endcode
*/
#ifndef BLAZE_DTENSDVECMULT_THRESHOLD
#define BLAZE_DTENSDVECMULT_THRESHOLD 4000000UL
#endif
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
// This threshold specifies when an assignment with a simple dense matrix can be executed in
// parallel. In case the number of elements of the target matrix is larger or equal to this
// threshold, the operation is executed in parallel. If the number of elements is below this
// threshold the operation is executed single-threaded.
//
// Please note that this threshold is highly sensitiv to the used system architecture and the
// shared memory parallelization technique. Therefore the default value cannot guarantee maximum
// performance for all possible situations and configurations. It merely provides a reasonable
// standard for the current generation of CPUs. Also note that the provided default has been
// determined using the OpenMP parallelization and requires individual adaption for the C++11
// and Boost thread parallelization or the HPX-based parallelization.
//
// The default setting for this threshold is 48400 (which corresponds to a matrix size of
// \f$ 220 \times 220 \f$). In case the threshold is set to 0, the operation is unconditionally
// executed in parallel.
//
// \note It is possible to specify this threshold via command line or by defining this symbol
// manually before including any Blaze header file:

   \code
   #define BLAZE_SMP_DMATASSIGN_THRESHOLD 48400UL
   #include <blaze/Blaze.h>
   \endcode
*/
#ifndef BLAZE_SMP_DTENSASSIGN_THRESHOLD
#define BLAZE_SMP_DTENSASSIGN_THRESHOLD 48400UL
#endif
//*************************************************************************************************


//*************************************************************************************************
/*!\brief SMP row-major dense matrix/dense vector multiplication threshold.
// \ingroup config
//
// This threshold specifies when a row-major dense matrix/dense vector multiplication can be
// executed in parallel. In case the number of elements of the target vector is larger or equal
// to this threshold, the operation is executed in parallel. If the number of elements is below
// this threshold the operation is executed single-threaded.
//
// Please note that this threshold is highly sensitiv to the used system architecture and the
// shared memory parallelization technique. Therefore the default value cannot guarantee maximum
// performance for all possible situations and configurations. It merely provides a reasonable
// standard for the current generation of CPUs. Also note that the provided default has been
// determined using the OpenMP parallelization and requires individual adaption for the C++11
// and Boost thread parallelization or the HPX-based parallelization.
//
// The default setting for this threshold is 330. In case the threshold is set to 0, the operation
// is unconditionally executed in parallel.
//
// \note It is possible to specify this threshold via command line or by defining this symbol
// manually before including any Blaze header file:

   \code
   #define BLAZE_SMP_DTENSDVECMULT_THRESHOLD 330UL
   #include <blaze/Blaze.h>
   \endcode
*/
#ifndef BLAZE_SMP_DTENSDVECMULT_THRESHOLD
#define BLAZE_SMP_DTENSDVECMULT_THRESHOLD 330UL
#endif
//*************************************************************************************************



//*************************************************************************************************
/*!\brief SMP row-major dense matrix/row-major dense matrix Schur product threshold.
// \ingroup config
//
// This threshold specifies when a row-major dense matrix/row-major dense matrix Schur product
// can be executed in parallel. This threshold affects both Schur products between two row-major
// matrices or two column-major dense matrices. In case the number of elements of the target
// matrix is larger or equal to this threshold, the operation is executed in parallel. If the
// number of elements is below this threshold the operation is executed single-threaded.
//
// Please note that this threshold is highly sensitiv to the used system architecture and the
// shared memory parallelization technique. Therefore the default value cannot guarantee maximum
// performance for all possible situations and configurations. It merely provides a reasonable
// standard for the current generation of CPUs. Also note that the provided default has been
// determined using the OpenMP parallelization and requires individual adaption for the C++11
// and Boost thread parallelization or the HPX-based parallelization.
//
// The default setting for this threshold is 36100 (which corresponds to a matrix size of
// \f$ 190 \times 190 \f$). In case the threshold is set to 0, the operation is unconditionally
// executed in parallel.
//
// \note It is possible to specify this threshold via command line or by defining this symbol
// manually before including any Blaze header file:

   \code
   #define BLAZE_SMP_DMATDMATSCHUR_THRESHOLD 36100UL
   #include <blaze/Blaze.h>
   \endcode
*/
#ifndef BLAZE_SMP_DTENSDMATSCHUR_THRESHOLD
#define BLAZE_SMP_DTENSDMATSCHUR_THRESHOLD 36100UL
#endif
//*************************************************************************************************
