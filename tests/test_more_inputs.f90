program test_more_inputs
use driver, only: gpt2_driver2, model_t, load_model
implicit none

type(model_t) :: m
integer, parameter :: input_ref(*) = [46, 358, 129, 247, 68, 73, 34754, 234, &
    861, 8836, 74, 373, 4642, 287]
integer, parameter :: output_ref(*) = [1248, 5332, 287, 262, 7404, 286, &
    25370, 254, 368, 83, 6557, 81, 11]
integer, allocatable :: input(:), output(:)

call load_model("model.gguf", m)

call gpt2_driver2("Ondřej Čertík was born in", 13, m, input, output)
print *
print *, "TESTS:"
call test(input, input_ref, "Input")
call test(output, output_ref, "Output")

call gpt2_driver2("San Francisco is", 8, m, input, output)
print *
print *, "TESTS:"
call test(input, [15017, 6033, 318], "Input")
call test(output, [257, 1748, 286, 517, 621, 352, 1510, 661], "Output")

call gpt2_driver2("Cars are", 13, m, input, output)
print *
print *, "TESTS:"
call test(input, [34, 945, 389], "Input")
call test(output, [407, 3142, 284, 307, 973, 287, 262, 7647, 1256, 286, &
    257, 7072, 13], "Output")

contains

subroutine test(a, a_ref, text)
integer, intent(in) :: a(:), a_ref(:)
character(*), intent(in) :: text
if (all(a == a_ref)) then
    print *, text, ": OK"
else
    print *, text, ": FAIL"
    error stop
end if
end subroutine

end program
