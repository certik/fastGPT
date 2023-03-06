module linalg
! C implementation of the matmul routines
use iso_c_binding, only: c_int, c_float
implicit none

integer, parameter :: sp = kind(0.0)

interface
    subroutine acc_sgemm(m, n, k, A, B, C) bind(c)
    import :: c_int, c_float
    implicit none
    integer(c_int), value, intent(in) :: m, n, k
    real(c_float), intent(in) :: A(m,k), B(k,n)
    real(c_float), intent(out) :: C(m,n)
    end subroutine

    subroutine acc_sgemm_t(m, n, k, A, B, C) bind(c)
    import :: c_int, c_float
    implicit none
    integer(c_int), value, intent(in) :: m, n, k
    real(c_float), intent(in) :: A(k,m), B(k,n)
    real(c_float), intent(out) :: C(m,n)
    end subroutine
end interface

contains

    subroutine matmul_2d(A, B, C)
    ! C = matmul(A, B)
    real(sp), intent(in) :: A(:,:), B(:,:)
    real(sp), intent(out) :: C(:,:)
    call acc_sgemm(size(A,1), size(B,2), size(A,2), A, B, C)
    end subroutine

    subroutine matmul_2d_t(A, B, C)
    ! C = matmul(transpose(A), B)
    real(sp), intent(in) :: A(:,:), B(:,:)
    real(sp), intent(out) :: C(:,:)
    call acc_sgemm_t(size(A,2), size(B,2), size(A,1), A, B, C)
    end subroutine

end module
