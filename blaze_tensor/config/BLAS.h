//=================================================================================================
/*!
//  \file blaze_tensor/config/BLAS.h
//  \brief Configuration of the BLAS mode
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


//*************************************************************************************************
/*!\brief Compilation switch for the BLAS mode.
// \ingroup config
//
// This compilation switch enables/disables the BLAS mode. In case the BLAS mode is enabled,
// several basic linear algebra functions (such as for instance matrix-matrix multiplications
// between two dense matrices) are handled by performance optimized BLAS functions. Note that
// in this case it is mandatory to provide the according BLAS header file for the compilation
// of the Blaze library. In case the BLAS mode is disabled, all linear algebra functions use
// the default implementations of the Blaze library and therefore BLAS is not a requirement
// for the compilation process.
//
// Possible settings for the BLAS switch:
//  - Disabled: \b 0
//  - Enabled : \b 1
//
// \warning Changing the setting of the BLAS mode requires a recompilation of all code using
// the Blaze library!
//
// \note It is possible to (de-)activate the BLAS mode via command line or by defining this
// symbol manually before including any Blaze header file:

   \code
   #define BLAZE_BLAS_MODE 1
   #include <blaze/Blaze.h>
   \endcode
*/
#ifndef BLAZE_BLAS_MODE
#define BLAZE_BLAS_MODE 0
#endif
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Compilation switch for the BLAS matrix/vector multiplication kernels (gemv).
// \ingroup config
//
// This compilation switch enables/disables the BLAS matrix/vector multiplication kernels. If the
// switch is enabled, multiplications between dense matrices and dense vectors are computed by
// BLAS kernels, if it is disabled the multiplications are handled by the default Blaze kernels.
//
// Possible settings for the switch:
//  - Disabled: \b 0 (default)
//  - Enabled : \b 1
//
// \warning Changing the setting of this compilation switch requires a recompilation of all code
// using the Blaze library!
//
// \note It is possible to (de-)activate the use of the BLAS matrix/vector multiplication kernels
// via command line or by defining this symbol manually before including any Blaze header file:

   \code
   #define BLAZE_USE_BLAS_MATRIX_VECTOR_MULTIPLICATION 1
   #include <blaze/Blaze.h>
   \endcode
*/
#ifndef BLAZE_USE_BLAS_TENSOR_VECTOR_MULTIPLICATION
#define BLAZE_USE_BLAS_TENSOR_VECTOR_MULTIPLICATION 0
#endif
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Compilation switch for the BLAS matrix/matrix multiplication kernels (gemv).
// \ingroup config
//
// This compilation switch enables/disables the BLAS matrix/matrix multiplication kernels. If the
// switch is enabled, multiplications between dense matrices are computed by BLAS kernels, if it
// is disabled the multiplications are handled by the default Blaze kernels.
//
// Possible settings for the switch:
//  - Disabled: \b 0
//  - Enabled : \b 1 (default)
//
// \warning Changing the setting of this compilation switch requires a recompilation of all code
// code using the Blaze library!
//
// \note It is possible to (de-)activate the use of the BLAS matrix/matrix multiplication kernels
// via command line or by defining this symbol manually before including any Blaze header file:

   \code
   #define BLAZE_USE_BLAS_MATRIX_MATRIX_MULTIPLICATION 1
   #include <blaze/Blaze.h>
   \endcode
*/
#ifndef BLAZE_USE_BLAS_TENSOR_TENSOR_MULTIPLICATION
#define BLAZE_USE_BLAS_TENSOR_TENSOR_MULTIPLICATION 1
#endif
//*************************************************************************************************
