module linalg

implicit none
integer, parameter :: sp = kind(0.0)

contains

    subroutine matmul_2d(A, B, C)
    ! C = matmul(A, B)
    real(sp), intent(in) :: A(:,:), B(:,:)
    real(sp), intent(out) :: C(:,:)
    
    integer :: m, n, k
    external :: sgemm

    m = size(A,1)
    n = size(B,2)
    k = size(A,2)

    call sgemm('N','N',m,n,k,1.0_sp,A,size(A,1),B,size(B,1),0.0_sp,C,size(C,1))

    end subroutine

    subroutine matmul_2d_t(A, B, C)
    ! C = matmul(transpose(A), B)
    real(sp), intent(in) :: A(:,:), B(:,:)
    real(sp), intent(out) :: C(:,:)
    integer :: m, n, k
    external :: sgemm

    m = size(A,1)
    n = size(B,2)
    k = size(A,2)

    call sgemm('T','N',m,n,k,1.0_sp,A,size(A,1),B,size(B,1),0.0_sp,C,size(C,1))

    end subroutine

end module
