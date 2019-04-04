#---------------------------------------------------------------------------
#
# BlazeTensorConfig.cmake - CMake configuration file for external projects.
# Use this by invoking
#
#   find_package(BlazeTensor)
#
# The module defines BlazeTensor::BlazeTensor IMPORTED target

include("${CMAKE_CURRENT_LIST_DIR}/BlazeTensorTargets.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/BlazeTensorConfigVersion.cmake")
message(STATUS "Found BlazeTensor")
