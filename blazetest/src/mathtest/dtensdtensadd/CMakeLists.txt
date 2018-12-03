# =================================================================================================
#
#   Copyright (C) 2012-2018 Klaus Iglberger - All Rights Reserved
#   Copyright (C) 2018 Hartmut Kaiser - All Rights Reserved
#
#   This file is part of the Blaze library. You can redistribute it and/or modify it under
#   the terms of the New (Revised) BSD License. Redistribution and use in source and binary
#   forms, with or without modification, are permitted provided that the following conditions
#   are met:
#
#   1. Redistributions of source code must retain the above copyright notice, this list of
#      conditions and the following disclaimer.
#   2. Redistributions in binary form must reproduce the above copyright notice, this list
#      of conditions and the following disclaimer in the documentation and/or other materials
#      provided with the distribution.
#   3. Neither the names of the Blaze development group nor the names of its contributors
#      may be used to endorse or promote products derived from this software without specific
#      prior written permission.
#
#   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY
#   EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
#   OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT
#   SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
#   INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
#   TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
#   BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
#   CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
#   ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
#   DAMAGE.
#
# =================================================================================================

set(category DTensDTensAdd)

set(tests
    AliasingTest
    TDaT12x16x8a
    TDaT12x16x8b
    TDaT2x2x2a
    TDaT2x2x2b
    TDaT3x3x3a
    TDaT3x3x3b
    TDaT3x7x13a
    TDaT3x7x13b
    TDaT4x4x4a
    TDaT4x4x4b
    TDaT5x5x5a
    TDaT5x5x5b
    TDaT6x6x6a
    TDaT6x6x6b
    TDaTDa
    TDaTDb
    TDbT12x16x8a
    TDbT12x16x8b
    TDbT2x2x2a
    TDbT2x2x2b
    TDbT3x3x3a
    TDbT3x3x3b
    TDbT3x7x13a
    TDbT3x7x13b
    TDbT4x4x4a
    TDbT4x4x4b
    TDbT5x5x5a
    TDbT5x5x5b
    TDbT6x6x6a
    TDbT6x6x6b
    T12x16x8aT12x16x8a
    T12x16x8aT12x16x8b
    T12x16x8aTDa
    T12x16x8aTDb
    T12x16x8bT12x16x8a
    T12x16x8bT12x16x8b
    T12x16x8bTDa
    T12x16x8bTDb
    T2x2x2aT2x2x2a
    T2x2x2aT2x2x2b
    T2x2x2aTDa
    T2x2x2aTDb
    T2x2x2bT2x2x2a
    T2x2x2bT2x2x2b
    T2x2x2bTDa
    T2x2x2bTDb
    T3x3x3aT3x3x3a
    T3x3x3aT3x3x3b
    T3x3x3aTDa
    T3x3x3aTDb
    T3x3x3bT3x3x3a
    T3x3x3bT3x3x3b
    T3x3x3bTDa
    T3x3x3bTDb
    T3x7x13aT3x7x13a
    T3x7x13aT3x7x13b
    T3x7x13aTDa
    T3x7x13aTDb
    T3x7x13bT3x7x13a
    T3x7x13bT3x7x13b
    T3x7x13bTDa
    T3x7x13bTDb
    T4x4x4aT4x4x4a
    T4x4x4aT4x4x4b
    T4x4x4aTDa
    T4x4x4aTDb
    T4x4x4bT4x4x4a
    T4x4x4bT4x4x4b
    T4x4x4bTDa
    T4x4x4bTDb
    T5x5x5aT5x5x5a
    T5x5x5aT5x5x5b
    T5x5x5aTDa
    T5x5x5aTDb
    T5x5x5bT5x5x5a
    T5x5x5bT5x5x5b
    T5x5x5bTDa
    T5x5x5bTDb
    T6x6x6aT6x6x6a
    T6x6x6aT6x6x6b
    T6x6x6aTDa
    T6x6x6aTDb
    T6x6x6bT6x6x6a
    T6x6x6bT6x6x6b
    T6x6x6bTDa
    T6x6x6bTDb
)

foreach(test ${tests})
   add_blaze_tensor_test(${category}${test}
      SOURCES ${test}
      FOLDER "Tests/${category}")
endforeach()
