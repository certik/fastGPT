find_path(OMP_INCLUDE_DIR omp.h)
find_library(OMP_LIBRARY omp)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(OMP DEFAULT_MSG OMP_INCLUDE_DIR
    OMP_LIBRARY)

add_library(p::omp INTERFACE IMPORTED)
set_property(TARGET p::omp PROPERTY INTERFACE_INCLUDE_DIRECTORIES
    ${OMP_INCLUDE_DIR})
set_property(TARGET p::omp PROPERTY INTERFACE_LINK_LIBRARIES
    ${OMP_LIBRARY})
