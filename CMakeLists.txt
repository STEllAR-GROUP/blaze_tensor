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

cmake_minimum_required(VERSION 3.5)
project(BlazeTensor CXX)

set(CMAKE_CXX_STANDARD 14)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# Add cmake directory
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/cmake")

# Set up BlazeTensor target
add_library(BlazeTensor INTERFACE)

# Set BlazeTensor source files
set(SOURCE_FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/blaze_tensor/Blaze.h
  ${CMAKE_CURRENT_SOURCE_DIR}/blaze_tensor/Math.h
)

target_sources(BlazeTensor INTERFACE $<BUILD_INTERFACE:${SOURCE_FILES}>)

target_include_directories(
   BlazeTensor
      INTERFACE
         $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
         $<INSTALL_INTERFACE:blaze_tensor/>)

# Import Blaze and "link" to BlazeTensor
find_package(blaze REQUIRED NO_CMAKE_PACKAGE_REGISTRY)
target_link_libraries(BlazeTensor INTERFACE blaze::blaze)

# setup HPX, if needed
get_target_property(blaze_parallelization_mode blaze::blaze INTERFACE_COMPILE_DEFINITIONS)
if(blaze_parallelization_mode AND "${blaze_parallelization_mode}" STREQUAL "BLAZE_USE_HPX_THREADS")
   find_package(HPX REQUIRED NO_CMAKE_PACKAGE_REGISTRY)
   target_include_directories(BlazeTensor INTERFACE ${HPX_INCLUDE_DIRS})
   target_link_libraries(BlazeTensor INTERFACE ${HPX_LIBRARIES})
endif()

install(TARGETS BlazeTensor EXPORT BlazeTensorTargets)

export(
   EXPORT BlazeTensorTargets
   FILE "${CMAKE_CURRENT_BINARY_DIR}/cmake/BlazeTensorTargets.cmake"
   NAMESPACE BlazeTensor::
)

export(PACKAGE BlazeTensor)

configure_file(
   "${CMAKE_CURRENT_SOURCE_DIR}/cmake/BlazeTensorConfig.cmake"
   "${CMAKE_CURRENT_BINARY_DIR}/cmake/BlazeTensorConfig.cmake"
   COPYONLY
)

# Install library
install(
   EXPORT BlazeTensorTargets
   FILE BlazeTensorTargets.cmake
   NAMESPACE BlazeTensor::
   DESTINATION share/BlazeTensor/cmake
)

install(
  DIRECTORY blaze_tensor/
  DESTINATION blaze_tensor/
  FILES_MATCHING PATTERN "*.h")

install(
   FILES "${CMAKE_CURRENT_BINARY_DIR}/cmake/BlazeTensorConfig.cmake"
   DESTINATION share/BlazeTensor/cmake
)

# Optionally build tests (primarily for devs)
option(BLAZETENSOR_WITH_TESTS "Build BlazeTensor tests" OFF)
if(BLAZETENSOR_WITH_TESTS)
   enable_testing()
   include(CTest)

   add_subdirectory(blazetest)
endif()

