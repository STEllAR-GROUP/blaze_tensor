# Copyright (c) 2018 Hartmut Kaiser
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

function(add_blaze_tensor_test name)
   # retrieve arguments
   set(options EXCLUDE_FROM_ALL EXCLUDE_FROM_DEFAULT_BUILD)
   set(one_value_args FOLDER)
   set(multi_value_args SOURCES HEADERS COMPILE_FLAGS LINK_FLAGS)
   cmake_parse_arguments(${name} "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN})

   add_executable(${name} "${${name}_SOURCES}")
   target_link_libraries(${name} BlazeTensor)
   if(${name}_FOLDER)
      set_target_properties(${name} PROPERTIES FOLDER ${${name}_FOLDER})
   endif()
   add_test(NAME ${category}${test} COMMAND ${category}${test})

   if(BLAZE_SHARED_MEMORY_PARALLELIZATION AND "${BLAZE_SMP_THREADS}" STREQUAL "HPX")
      hpx_setup_target(${name} TYPE EXECUTABLE)
   endif()
endfunction()

