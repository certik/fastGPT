program gpt2
use driver, only: gpt2_driver
implicit none
integer, allocatable :: input(:), output(:)
call gpt2_driver(input, output)
end program
