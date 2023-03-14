module omp
implicit none
private
public :: omp_get_wtime

integer, parameter :: dp = kind(0.d0)

interface
    real(dp) function omp_get_wtime()
    import :: dp
    end function
end interface

end module
