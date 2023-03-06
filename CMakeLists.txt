cmake_minimum_required(VERSION 3.13 FATAL_ERROR)

set(CMAKE_USER_MAKE_RULES_OVERRIDE ${CMAKE_SOURCE_DIR}/cmake/UserOverride.cmake)

project(fastGPT)
enable_language(Fortran)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
set(CMAKE_Fortran_MODULE_DIRECTORY ${CMAKE_BINARY_DIR}/mod_files)

# Make sure that CMAKE_BUILD_TYPE is either Debug or Release:
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release
        CACHE STRING "Build type (Debug, Release)" FORCE)
endif ()
if (NOT (CMAKE_BUILD_TYPE STREQUAL "Debug" OR
        CMAKE_BUILD_TYPE STREQUAL "Release"))
    message("${CMAKE_BUILD_TYPE}")
    message(FATAL_ERROR "CMAKE_BUILD_TYPE must be one of: Debug, Release (current value: '${CMAKE_BUILD_TYPE}')")
endif ()

if (APPLE)
    set(DEFAULT_ACCELERATE_FRAMEWORK yes)
else()
    set(DEFAULT_ACCELERATE_FRAMEWORK no)
endif()
set(WITH_ACCELERATE_FRAMEWORK ${DEFAULT_ACCELERATE_FRAMEWORK}
    CACHE BOOL "Build with macoS Accelerate framework")

enable_testing()

set(SRC
    main.f90
    gpt2.f90
    )
if(WITH_ACCELERATE_FRAMEWORK)
    set(SRC ${SRC} linalg_accelerate.c)
    set(SRC ${SRC} linalg_c.f90)
else()
    set(SRC ${SRC} linalg_f.f90)
endif()
add_executable(gpt2 ${SRC})
if(WITH_ACCELERATE_FRAMEWORK)
    target_link_options(gpt2 PRIVATE -framework accelerate)
endif()


message("\n")
message("Configuration results")
message("---------------------")
message("Fortran compiler: ${CMAKE_Fortran_COMPILER}")
message("Build type: ${CMAKE_BUILD_TYPE}")
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    message("Fortran compiler flags: ${CMAKE_Fortran_FLAGS_DEBUG}")
else ()
    message("Fortran compiler flags: ${CMAKE_Fortran_FLAGS_RELEASE}")
endif ()
message("Installation prefix: ${CMAKE_INSTALL_PREFIX}")
message("WITH_ACCELERATE_FRAMEWORK: ${WITH_ACCELERATE_FRAMEWORK}")
