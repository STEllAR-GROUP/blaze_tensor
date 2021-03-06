//=================================================================================================
/*!
//  \file blaze/config/HPX.h
//  \brief Configuration of the HPX parallelization
//
//  Copyright (C) 2012-2018 Klaus Iglberger - All Rights Reserved
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

// Configuration for vector operations
#ifndef BLAZE_HPX_VECTOR_THRESHOLD
#define BLAZE_HPX_VECTOR_THRESHOLD 1000
#endif

#ifndef BLAZE_HPX_VECTOR_CHUNK_SIZE
#define BLAZE_HPX_VECTOR_CHUNK_SIZE 10
#endif

#ifndef BLAZE_HPX_VECTOR_BLOCK_SIZE
#define BLAZE_HPX_VECTOR_BLOCK_SIZE 64
#endif

#ifndef BLAZE_HPX_MATRIX_CHUNK_SIZE
#define BLAZE_HPX_MATRIX_CHUNK_SIZE 10
#endif

#ifndef BLAZE_HPX_MATRIX_BLOCK_SIZE_ROW
#define BLAZE_HPX_MATRIX_BLOCK_SIZE_ROW 4
#endif

#ifndef BLAZE_HPX_MATRIX_BLOCK_SIZE_COLUMN
#define BLAZE_HPX_MATRIX_BLOCK_SIZE_COLUMN 1024
#endif

#ifndef BLAZE_HPX_TENSOR_CHUNK_SIZE
#define BLAZE_HPX_TENSOR_CHUNK_SIZE 10
#endif

#ifndef BLAZE_HPX_TENSOR_BLOCK_SIZE_PAGE
#define BLAZE_HPX_TENSOR_BLOCK_SIZE_PAGE 4
#endif

#ifndef BLAZE_HPX_TENSOR_BLOCK_SIZE_ROW
#define BLAZE_HPX_TENSOR_BLOCK_SIZE_ROW 4
#endif

#ifndef BLAZE_HPX_TENSOR_BLOCK_SIZE_COLUMN
#define BLAZE_HPX_TENSOR_BLOCK_SIZE_COLUMN 1024
#endif
