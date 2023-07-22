module tokenizer
implicit none

type :: string
    character(:), allocatable :: s
end type

contains

function c2s(x) result(y)
character, intent(in) :: x(:)
character(:), allocatable :: y
integer :: i
allocate(character(size(x)) :: y)
do i = 1, size(x)
    y(i:i) = x(i)
end do
end function

function next_token(input, i) result(y)
! TODO: tokenize exactly according to this regex:
! re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
! Right now we are more greedy, but the bpe() tokenizer seems to still return
! exactly the same tokens for most inputs (it is not clear if for all inputs).
character(*), intent(in) :: input
integer, intent(inout) :: i
character(:), allocatable :: y
if (i > len(input)) then
    y = ""
else if (input(i:i) == " ") then
    y = tokenize_word(input, i)
else if (input(i:i) == "," .or. input(i:i) == ".") then
    y = input(i:i)
    i = i + 1
else
    y = tokenize_word(input, i)
end if
end function

function tokenize_word(input, i) result(y)
character(*), intent(in) :: input
integer, intent(inout) :: i
character(:), allocatable :: y
integer :: i0
i0 = i
if (input(i:i) == " ") then
    i = i + 1
end if
do
    if (i > len(input)) then
        y = input(i0:i-1)
        exit
    end if
    if (input(i:i) == " " .or. input(i:i) == "," .or. input(i:i) == ".") then
        y = input(i0:i-1)
        exit
    end if
    i = i + 1
end do
end function

function word_idx(word, idx, decoder_txt) result(token)
character(*), intent(in) :: word
integer, intent(in) :: idx(0:)
character, intent(in) :: decoder_txt(:)
integer :: token
integer :: i
! This is O(n) search instead of O(1) lookup in a dictionary, so it is slow
do i = 0, ubound(idx,1)-1
    if (c2s(decoder_txt(idx(i)+1:idx(i+1))) == word) then
        token = i
        return
    end if
end do
token = -1
end function

subroutine codepoint_to_utf8(s, c)
! UTF-32 -> UTF-8
character(:), allocatable, intent(inout) :: s
integer, intent(in) :: c
integer :: d1, d2
if (c < 128) then
    s = s // achar(c)
else if (c < 2048) then
    d1 = ior(ishft(c, -6), 192)
    d2 = iand(ior(c, 128), 191)
    s = s // achar(d1) // achar(d2)
else
    error stop "UTF-32 range not supported"
end if
end subroutine

function utf8_to_codepoint(s, i) result(c)
! UTF-8 -> UTF-32
character(*), intent(in) :: s
integer, intent(inout) :: i
integer :: c, d
c = iachar(s(i:i))
if (c >= 128) then
    i = i + 1
    d = iachar(s(i:i))
    c = ior(ishft(iand(c, 31), 6), iand(d, 63))
end if
if (c >= 2048) then
    error stop "UTF-8 range not supported"
end if
end function

function merge_pair(intokens, idx) result(tokens)
! Merge the pair `idx`
type(string), intent(in) :: intokens(:)
integer, intent(in) :: idx
type(string), allocatable :: tokens(:)
type(string) :: merged_token
merged_token%s = intokens(idx)%s // intokens(idx+1)%s
tokens = [intokens(:idx-1), merged_token, intokens(idx+2:)]
end function

function merge_utf8_pairs(intokens) result(tokens)
! Merge all UTF-8 character pairs
type(string), intent(in) :: intokens(:)
type(string), allocatable :: tokens(:)
integer :: i, j
logical :: one_more_pass
allocate(tokens(size(intokens)))
tokens = intokens
one_more_pass = .true.
!print *, "merge_utf8_pairs:", size(tokens)
!print *, "tokens = ", (tokens(i)%s // " ", i=1,size(tokens))
j = 1
do while(one_more_pass)
    one_more_pass = .false.
    do i = j, size(tokens)-1
        if (len(tokens(i)%s) == 1 .and. iachar(tokens(i)%s(1:1)) >= 128) then
            tokens = merge_pair(tokens, i)
            one_more_pass = .true.
            j = i + 1
!            print *, "pass"
            exit
        end if
    end do
end do
!print *, "tokens = ", (tokens(i)%s // " ", i=1,size(tokens))
end function

function bpe(token, vocab_idx, vocab_txt) result(tokens)
! Takes a token as a string, and returns bpe tokens as an array of strings
character(*), intent(in) :: token
integer, intent(in) :: vocab_idx(0:)
character, intent(in) :: vocab_txt(:)
type(string), allocatable :: tokens(:)
integer, allocatable :: pair_scores(:)
integer :: not_found, merge_pair_idx
integer :: i
not_found = size(vocab_idx) + 10
allocate(tokens(len(token)))
do i = 1, len(token)
    tokens(i)%s = token(i:i)
end do
tokens = merge_utf8_pairs(tokens)
do
    !print *, "tokens = ", (tokens(i)%s // " ", i=1,size(tokens))
    if (size(tokens) == 1) then
        ! The token pairs were either all merged into one word, or the input
        ! token was a one character word, either way we are done:
        exit
    end if
    allocate(pair_scores(size(tokens)-1))
    ! Loop over pairs
    do i = 1, size(tokens)-1
        pair_scores(i) = word_idx(tokens(i)%s // " " // tokens(i+1)%s, vocab_idx, vocab_txt)
        if (pair_scores(i) == -1) pair_scores(i) = not_found
    end do
    merge_pair_idx = minloc(pair_scores, 1)
    if (pair_scores(merge_pair_idx) == not_found) then
        ! No token pair can be merged, so we are done:
        exit
    end if
    !print *, pair_scores
    !print *, merge_pair_idx, pair_scores(merge_pair_idx)
    tokens = merge_pair(tokens, merge_pair_idx)
    deallocate(pair_scores)
end do
!print *, "final tokens = ", (tokens(i)%s // " ", i=1,size(tokens))
end function

function encode(input, idx, decoder_txt, vocab_idx, vocab_txt, byte_encoder) &
        result(tokens2)
character(*), intent(in) :: input
integer, intent(in) :: idx(0:), vocab_idx(0:), byte_encoder(0:)
character, intent(in) :: decoder_txt(:), vocab_txt(:)
integer, parameter :: max_tokens = 2048
integer :: tokens(max_tokens)
integer, allocatable :: tokens2(:)
character(:), allocatable :: tmp, tmp2
type(string), allocatable :: bpe_tokens(:)
integer :: i, j, c, n_tokens
n_tokens = 0
i = 1
do
    tmp = next_token(input, i)
    if (tmp == "") exit
    tmp2 = ""
    do j = 1, len(tmp)
        c = iachar(tmp(j:j))
        c = byte_encoder(c)
        ! c is UTF-32 (4 bytes), but only the range [0, 324] is used
        ! Encode c from UTF-32 to UTF-8. Due to the limited range
        ! either one or two bytes of UTF-8 are appended to tmp2:
        call codepoint_to_utf8(tmp2, c)
    end do
    bpe_tokens = bpe(tmp2, vocab_idx, vocab_txt)
    do j = 1, size(bpe_tokens)
        n_tokens = n_tokens + 1
        if (n_tokens > max_tokens) error stop "exceeded max_tokens"
        tokens(n_tokens) = word_idx(bpe_tokens(j)%s, idx, decoder_txt)
    end do
    deallocate(tmp2)
end do
allocate(tokens2(n_tokens))
tokens2 = tokens(:n_tokens)
end function

function decode(tokens, idx, decoder_txt, byte_decoder) result(output)
integer, intent(in) :: tokens(:), idx(0:), byte_decoder(0:)
character, intent(in) :: decoder_txt(:)
character(:), allocatable :: output
character(:), allocatable :: output2, tmp
integer :: i, c
allocate(character(0) :: output2) ! Fix GFortran warning
output2 = ""
do i = 1, size(tokens)
    output2 = output2 // c2s(decoder_txt(idx(tokens(i))+1:idx(tokens(i)+1)))
end do
i = 1
output = ""
do
    ! Decode UTF-8 (one or more bytes) to UTF-32 code point (always 4 bytes),
    ! However for GPT-2 it seems only range 0-323 is used from UTF-32.
    c = utf8_to_codepoint(output2, i)
    ! [0,324] -> [0,255]
    if (c < 0 .or. c > ubound(byte_decoder,1)) error stop "Codepoint out of range for byte decoder"
    tmp = achar(byte_decoder(c))
    output = output // tmp
    if (i == len(output2)) exit
    i = i + 1
end do
end function

end module
