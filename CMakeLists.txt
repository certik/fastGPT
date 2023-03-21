cmake_minimum_required(VERSION 3.13 FATAL_ERROR)

set(CMAKE_USER_MAKE_RULES_OVERRIDE ${CMAKE_SOURCE_DIR}/cmake/UserOverride.cmake)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release
        CACHE STRING "Build type (Debug, Release)")
endif()

project(fastGPT)
enable_language(Fortran)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
set(CMAKE_Fortran_MODULE_DIRECTORY ${CMAKE_BINARY_DIR}/mod_files)

# Make sure that CMAKE_BUILD_TYPE is either Debug or Release:
if (NOT CMAKE_BUILD_TYPE MATCHES "Debug|Release")
    message(FATAL_ERROR "CMAKE_BUILD_TYPE must be one of: Debug, Release (current value: '${CMAKE_BUILD_TYPE}')")
endif ()

if (APPLE)
    set(DEFAULT_FASTGPT_BLAS "Accelerate")
else()
    set(DEFAULT_FASTGPT_BLAS "Fortran")
endif()
set(FASTGPT_BLAS ${DEFAULT_FASTGPT_BLAS}
    CACHE STRING "The BLAS library that fastGPT should use")
if (NOT FASTGPT_BLAS MATCHES "Accelerate|OpenBLAS|Fortran")
    message(FATAL_ERROR "FASTGPT_BLAS must be one of: OpenBLAS, Accelerate, Fortran (current value: '${FASTGPT_BLAS}')")
endif ()
if (FASTGPT_BLAS STREQUAL "Accelerate")
    find_package(OMP)
    if(NOT OMP_FOUND)
        find_package(OpenMP REQUIRED COMPONENTS Fortran)
    endif()
elseif (FASTGPT_BLAS STREQUAL "OpenBLAS")
    find_package(OPENBLAS REQUIRED)
    find_package(OMP)
    if(NOT OMP_FOUND)
        find_package(OpenMP REQUIRED COMPONENTS Fortran)
    endif()
else()
    # pass
endif()

enable_testing()

set(SRC
    main.f90
    gpt2.f90
    tokenizer.f90
    )
if (FASTGPT_BLAS STREQUAL "Accelerate")
    list(APPEND SRC
        linalg_accelerate.c
        linalg_c.f90
        omp.f90
    )
elseif (FASTGPT_BLAS STREQUAL "OpenBLAS")
    list(APPEND SRC
        linalg_openblas.c
        linalg_c.f90
        omp.f90
    )
else()
    list(APPEND SRC
        linalg_f.f90
        omp_dummy.f90
    )
endif()
add_executable(gpt2 ${SRC})
if (FASTGPT_BLAS STREQUAL "Accelerate")
    target_link_options(gpt2 PRIVATE -framework accelerate)
    target_link_libraries(gpt2 PRIVATE "$<IF:$<BOOL:${OMP_FOUND}>,p::omp,OpenMP::OpenMP_Fortran>")
elseif (FASTGPT_BLAS STREQUAL "OpenBLAS")
    target_link_libraries(gpt2 p::openblas
        "$<IF:$<BOOL:${OMP_FOUND}>,p::omp,OpenMP::OpenMP_Fortran>")
endif()

if(NOT CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
    # Git auto-ignore out-of-source build directory
    file(GENERATE OUTPUT .gitignore CONTENT "*")
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
message("FASTGPT_BLAS: ${FASTGPT_BLAS}")
