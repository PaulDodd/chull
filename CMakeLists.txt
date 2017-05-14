CMAKE_MINIMUM_REQUIRED(VERSION 2.8.0 FATAL_ERROR)

if (CMAKE_MAJOR_VERSION VERSION_GREATER 2)
if(COMMAND cmake_policy)
    cmake_policy(SET CMP0003 NEW)
    cmake_policy(SET CMP0026 OLD)
    cmake_policy(SET CMP0042 OLD)
    set(CMAKE_MACOSX_RPATH ON)
endif(COMMAND cmake_policy)
endif()

project (CHULL)
add_subdirectory (src)
