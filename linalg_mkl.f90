module linalg

implicit none
integer, parameter :: sp = kind(0.0)

contains

    subroutine matmul_2d(A, B, C)
    ! C = matmul(A, B)
    real(sp), intent(in) :: A(:,:), B(:,:)
    real(sp), intent(out) :: C(:,:)
    
    integer :: m, n, k, lda, ldb, ldc
    external :: sgemm

    m = size(A,1)
    n = size(B,2)
    k = size(A,2)

    lda = size(A,1)
    ldb = size(B,1)
    ldc = size(C,1)
    
    call sgemm('N','N',m,n,k,1.0_sp,A,lda,B,ldb,0.0_sp,C,ldc)

    end subroutine

    subroutine matmul_2d_t(A, B, C)
    ! C = matmul(transpose(A), B)
    real(sp), intent(in) :: A(:,:), B(:,:)
    real(sp), intent(out) :: C(:,:)
    integer :: m, n, k, lda, ldb, ldc
    external :: sgemm

    m = size(A,1)
    n = size(B,2)
    k = size(A,2)

    lda = size(A,1)
    ldb = size(B,1)
    ldc = size(C,1)

    call sgemm('T','N',m,n,k,1.0_sp,A,lda,B,ldb,0.0_sp,C,ldc)

    end subroutine

end module
