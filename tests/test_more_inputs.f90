program test_more_inputs
use driver, only: gpt2_driver2, model_t, load_model
implicit none

type(model_t) :: m
integer, parameter :: input_ref(*) = [46, 358, 129, 247, 68, 73, 34754, 234, &
    861, 8836, 74, 373, 4642, 287]
integer, parameter :: output_ref(*) = [1248, 5332, 287, 262, 7404, 286, &
    25370, 254, 368, 83, 6557, 81, 11]
integer, allocatable :: input(:), output(:)

call load_model("model.dat", m)
call gpt2_driver2("Ondřej Čertík was born in ", 13, m, input, output)

print *
print *, "TESTS:"

if (all(input == input_ref)) then
    print *, "Input tokens agree with reference results"
else
    print *, "Input tokens DO NOT agree with reference results"
    error stop
end if

if (all(output == output_ref)) then
    print *, "Output tokens agree with reference results"
else
    print *, "Output tokens DO NOT agree with reference results"
    error stop
end if


end program
