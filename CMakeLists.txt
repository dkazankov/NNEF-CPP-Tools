cmake_minimum_required(VERSION 3.5)
#set(CMAKE_VERBOSE_MAKEFILE ON)

project(nnef-tools CXX)

# build information
message(STATUS "Build Configuration: ${CMAKE_BUILD_TYPE}")
message(STATUS "Build executables in: ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")

include_directories("../NNEF-Tools/nnef-pyproject/nnef/cpp/include")

add_executable(infer infer.cpp)
add_executable(nnef_tff_info nnef_tff_info.cpp)
add_executable(nnef2ada nnef2ada.cpp)

add_library(nnef STATIC IMPORTED)
set_target_properties(nnef PROPERTIES
    IMPORTED_LOCATION "../../NNEF-Tools/build/Release/nnef.lib"
    IMPORTED_LOCATION_DEBUG "../../NNEF-Tools/build/Debug/nnef_d.lib"
    IMPORTED_CONFIGURATIONS "RELEASE;DEBUG"
)

set_target_properties(infer PROPERTIES CXX_STANDARD 11)
set_target_properties(nnef_tff_info PROPERTIES CXX_STANDARD 11)
set_target_properties(nnef2ada PROPERTIES CXX_STANDARD 11)

target_link_libraries(infer PRIVATE nnef)
target_link_libraries(nnef_tff_info PRIVATE nnef)
target_link_libraries(nnef2ada PRIVATE nnef)
