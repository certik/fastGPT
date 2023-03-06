module linalg
! Pure Fortran implementation of the matmul routines
implicit none

integer, parameter :: sp = kind(0.0)

contains

    subroutine matmul_2d(A, B, C)
    real(sp), intent(in) :: A(:,:), B(:,:)
    real(sp), intent(out) :: C(:,:)
    C = matmul(A, B)
    end subroutine

    subroutine matmul_2d_t(A, B, C)
    real(sp), intent(in) :: A(:,:), B(:,:)
    real(sp), intent(out) :: C(:,:)
    C = matmul(transpose(A), B)
    end subroutine

end module
