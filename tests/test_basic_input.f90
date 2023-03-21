program test_basic_input
use driver, only: gpt2_driver
implicit none

integer, parameter :: input_ref(*) = [36235, 39141, 18765, 1143, 326, 9061, &
    561, 530, 1110, 1716, 845, 3665, 11, 475, 772, 339, 714, 407, 5967]
integer, parameter :: output_ref(*) = [703, 484, 561, 307, 1498, 284, 466, &
    523, 13, 198, 198, 1, 40, 892, 326, 262, 749, 1593, 1517, 318]
integer, allocatable :: input(:), output(:)

call gpt2_driver(input, output)

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
