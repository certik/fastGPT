program gpt2
use driver, only: gpt2_driver, model_t
implicit none
integer, allocatable :: input(:), output(:)
type(model_t) :: m
call gpt2_driver(input, output, m)
end program
