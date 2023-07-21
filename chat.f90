program chatgpt2
use driver, only: chat
use tokenizer, only: string
implicit none
type(string), allocatable :: inputs(:)
call chat(.false., inputs)
end program
