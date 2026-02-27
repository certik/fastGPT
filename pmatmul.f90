! Compile with:
!
! $ mpif90 -fcheck=all pmatmul.f90
! $ mpiexec -n 1 ./a.out

program pmatmul
use mpi
implicit none
integer, parameter :: dp = kind(0.d0)
integer :: n ! matrix size
integer :: i, j, ierr, myrank, numprocs, rowstart, rowend
real(dp) :: temp_sum
real(dp), allocatable :: A(:,:), x(:), y(:)

n = 1000

! Initialize MPI environment
call MPI_INIT(ierr)
call MPI_COMM_RANK(MPI_COMM_WORLD, myrank, ierr)
call MPI_COMM_SIZE(MPI_COMM_WORLD, numprocs, ierr)

if (myrank == 0) then
    print *, "nproc =", numprocs
end if

allocate(A(n,n), x(n), y(n))

! Divide rows of matrix A among processes
rowstart = ((n-1)*myrank)/numprocs + 1
rowend = ((n-1)*(myrank+1))/numprocs + 1

! Initialize matrix A and vector x
do i = 1, n
    x(i) = 1
    do j = 1, n
        A(i,j) = 1.0d0/(i+j-1.0d0)
    end do
end do

! Scatter matrix A and vector x among processes
call MPI_SCATTER(A(:,rowstart:), (rowend-rowstart+1)*n, &
  MPI_DOUBLE_PRECISION, A(:,rowstart), (rowend-rowstart+1)*n, &
  MPI_DOUBLE_PRECISION, 0, MPI_COMM_WORLD, ierr)
call MPI_BCAST(x, n, MPI_DOUBLE_PRECISION, 0, MPI_COMM_WORLD, ierr)

! Compute partial dot products
do i = rowstart, rowend
    y(i) = dot_product(A(:,i), x(:))
end do

! Gather result vectors from all processes
call MPI_GATHER(y(rowstart:rowend), (rowend-rowstart+1), &
  MPI_DOUBLE_PRECISION, y, (rowend-rowstart+1), MPI_DOUBLE_PRECISION, &
  0, MPI_COMM_WORLD, ierr)

! Finalize MPI environment
call MPI_FINALIZE(ierr)

! Print the result vector y from rank 0
if (myrank == 0) then
    write(*,*) 'Result vector y:'
    write(*,*) y(:10)
end if

end program
