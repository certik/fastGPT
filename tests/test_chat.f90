program test_chat
use driver, only: chat
use tokenizer, only: string
implicit none
type(string), allocatable :: inputs(:)
allocate(inputs(14))
inputs = [ &
    string("What color does the sky have?"), &
    string("What can you type a document on?"), &
    string("What can you drive in?"), &
    string("What can you fly in?"), &
    string("What continent is Germany in?"), &
    string("When did Second World War start?"), &
    string("When did it end?"), &
    string("When did the U.S. enter the Second World War?"), &
    string("When did the First World War start?"), &
    string("When did it end?"), &
    string("When did the Mexican-American war start?"), &
    string("When did it end?"), &
    string("What color is snow?"), &
    string("What color do plants usually have?") &
    ]
call chat(inputs(:2))
end program
