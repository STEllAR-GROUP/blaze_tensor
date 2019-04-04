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

# if no git commit is set, try to get it from the source directory
if(NOT BLAZE_TENSOR_WITH_GIT_COMMIT OR "${BLAZE_TENSOR_WITH_GIT_COMMIT}" STREQUAL "None")

  find_package(Git)

  if(GIT_FOUND)
    execute_process(
      COMMAND "${GIT_EXECUTABLE}" "log" "--pretty=%H" "-1" "${PROJECT_SOURCE_DIR}"
      WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
      OUTPUT_VARIABLE BLAZE_TENSOR_WITH_GIT_COMMIT ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)
  endif()

endif()

if(NOT BLAZE_TENSOR_WITH_GIT_COMMIT OR "${BLAZE_TENSOR_WITH_GIT_COMMIT}" STREQUAL "None")
#  message(STATUS "GIT commit not found (set to 'unknown').")
  set(BLAZE_TENSOR_WITH_GIT_COMMIT "unknown")
  set(BLAZE_TENSOR_WITH_GIT_COMMIT_SHORT "unknown")
else()
#  message(STATUS "GIT commit is ${BLAZE_TENSOR_WITH_GIT_COMMIT}.")
  if(NOT BLAZE_TENSOR_WITH_GIT_COMMIT_SHORT OR "${BLAZE_TENSOR_WITH_GIT_COMMIT_SHORT}" STREQUAL "None")
    string(SUBSTRING "${BLAZE_TENSOR_WITH_GIT_COMMIT}" 0 7 BLAZE_TENSOR_WITH_GIT_COMMIT_SHORT)
  endif()
endif()

