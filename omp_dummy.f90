module omp
implicit none
private
public :: omp_get_wtime

integer, parameter :: dp = kind(0.d0)

contains

real(dp) function omp_get_wtime()
omp_get_wtime = 0
end function

end module
