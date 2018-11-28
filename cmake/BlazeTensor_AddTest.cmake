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

function(add_blaze_tensor_test name)
   # retrieve arguments
   set(options EXCLUDE_FROM_ALL EXCLUDE_FROM_DEFAULT_BUILD)
   set(one_value_args FOLDER)
   set(multi_value_args SOURCES HEADERS COMPILE_FLAGS LINK_FLAGS)
   cmake_parse_arguments(${name} "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN})

   add_executable(${name} "${${name}_SOURCES}")
   target_link_libraries(${name} BlazeTensor)
   target_compile_definitions(${name} INTERFACE blaze::blaze)

   if(${name}_FOLDER)
      set_target_properties(${name} PROPERTIES FOLDER ${${name}_FOLDER})
   endif()

   get_target_property(blaze_parallelization_mode blaze::blaze INTERFACE_COMPILE_DEFINITIONS)
   if(blaze_parallelization_mode AND "${blaze_parallelization_mode}" STREQUAL "BLAZE_USE_HPX_THREADS")
      set(working_directory)
      if(MSVC)
         get_filename_component(__working_directory ${HPX_DIR} DIRECTORY)
         get_filename_component(__working_directory ${__working_directory} DIRECTORY)
         get_filename_component(__working_directory ${__working_directory} DIRECTORY)
         set(working_directory WORKING_DIRECTORY ${__working_directory}/$<CONFIG>/bin)
      endif()

      add_test(
         NAME ${category}${test}
         COMMAND ${category}${test} $<$<CONFIG:DEBUG>:--hpx:threads=1>
         ${working_directory})

      set(compile_flags)
      if(MSVC)
         set(compile_flags COMPILE_FLAGS "-wd4146 -wd4244 -bigobj")
      endif()
      hpx_setup_target(${name} TYPE EXECUTABLE ${compile_flags})
   else()
      if(MSVC)
         target_compile_options(${name} PRIVATE -wd4146 -wd4244 -bigobj)
      endif()
      add_test(NAME ${category}${test} COMMAND ${category}${test})
   endif()
endfunction()

