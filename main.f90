module omp
implicit none
private
public :: omp_get_wtime

integer, parameter :: dp = kind(0.d0)

contains

real(dp) function omp_get_wtime()
omp_get_wtime = 0
end function

end module
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
tokens = intokens
one_more_pass = .true.
!print *, "merge_utf8_pairs:", size(tokens)
!print *, "tokens = ", (tokens(i)%s // " ", i=1,size(tokens))
j = 1
do while(one_more_pass)
    one_more_pass = .false.
    do i = j, size(tokens)-1
!        if (len(tokens(i)%s) == 1 .and. iachar(tokens(i)%s(1:1)) >= 128) then
            tokens = merge_pair(tokens, i)
            one_more_pass = .true.
            j = i + 1
!            print *, "pass"
            exit
!        end if
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
!    tokens(i)%s = token(i:i)
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
!        pair_scores(i) = word_idx(tokens(i)%s // " " // tokens(i+1)%s, vocab_idx, vocab_txt)
        if (pair_scores(i) == -1) pair_scores(i) = not_found
    end do
!    merge_pair_idx = minloc(pair_scores, 1)
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
        result(tokens)
character(*), intent(in) :: input
integer, intent(in) :: idx(0:), vocab_idx(0:), byte_encoder(0:)
character, intent(in) :: decoder_txt(:), vocab_txt(:)
integer, allocatable :: tokens(:)
character(:), allocatable :: tmp, tmp2
type(string), allocatable :: bpe_tokens(:)
integer :: i, j, c
i = 1
allocate(tokens(0))
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
!        tokens = [tokens, word_idx(bpe_tokens(j)%s, idx, decoder_txt)]
    end do
    deallocate(tmp2)
end do
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

module linalg
! C implementation of the matmul routines
use iso_c_binding, only: c_int, c_float
implicit none

integer, parameter :: sp = kind(0.0)

interface
    subroutine acc_sgemm(m, n, k, A, B, C) bind(c)
    import :: c_int, c_float
    implicit none
    integer(c_int), value, intent(in) :: m, n, k
    real(c_float), intent(in) :: A(m,k), B(k,n)
    real(c_float), intent(out) :: C(m,n)
    end subroutine

    subroutine acc_sgemm_t(m, n, k, A, B, C) bind(c)
    import :: c_int, c_float
    implicit none
    integer(c_int), value, intent(in) :: m, n, k
    real(c_float), intent(in) :: A(k,m), B(k,n)
    real(c_float), intent(out) :: C(m,n)
    end subroutine
end interface

contains

    subroutine matmul_2d(A, B, C)
    ! C = matmul(A, B)
    real(sp), intent(in) :: A(:,:), B(:,:)
    real(sp), intent(out) :: C(:,:)
    call acc_sgemm(size(A,1), size(B,2), size(A,2), A, B, C)
    end subroutine

    subroutine matmul_2d_t(A, B, C)
    ! C = matmul(transpose(A), B)
    real(sp), intent(in) :: A(:,:), B(:,:)
    real(sp), intent(out) :: C(:,:)
    call acc_sgemm_t(size(A,2), size(B,2), size(A,1), A, B, C)
    end subroutine

end module
module gpt2_mod
use linalg, only: matmul_2d, matmul_2d_t
use tokenizer, only: decode
implicit none

integer, parameter :: sp = kind(0.0)
real(sp), parameter :: pi = 3.14159265358979323846_sp

! This derived type contains all the data of the GPT-2 model, including all
! weights, model parameters, and encoder/decoder data
type :: model_t
    integer :: n_vocab, n_ctx, n_embd, n_layer, n_head, &
        n_decoder_idx, n_decoder_txt, &
        n_vocab_idx, n_vocab_txt, n_byte_encoder
    real(sp), allocatable :: wte(:,:), wpe(:,:), &
        mlp_fc_w(:,:,:), mlp_fc_b(:,:), &
        mlp_proj_w(:,:,:), mlp_proj_b(:,:), &
        attn_w(:,:,:), attn_b(:,:), &
        attn_proj_w(:,:,:), attn_proj_b(:,:), &
        ln1_b(:,:), ln1_g(:,:), &
        ln2_b(:,:), ln2_g(:,:), &
        lnf_b(:), lnf_g(:)
    integer, allocatable :: decoder_idx(:), vocab_idx(:), byte_encoder(:)
    character, allocatable :: decoder_txt(:), vocab_txt(:)
end type

contains

elemental real(sp) function fast_tanh(x) result(y)
real(sp), intent(in) :: x
real(sp) :: x2
if (x > 5) then
    y = 1
elseif (x < -5) then
    y = -1
else
    x2 = x*x
    y = x * (0.98569772605911309407 + x2 *(-0.2794500993392901382 &
        + x2 * (6.8280504526399188164e-2 + x2 * (-1.0972014877337651823e-2 &
        + x2 * (1.1132367134444316902e-3 + x2 * (-7.018851897305717565e-5 &
        + x2 * (2.656616768082727089e-6 + x2 * (-5.5138381821615909058e-8 &
        + x2 * 4.8162484477588665996e-10))))))))
end if
end function

elemental real(sp) function gelu(x) result(y)
real(sp), intent(in) :: x
y = 0.5_sp * x * (1 + tanh(sqrt(2 / pi) * (x + 0.044715_sp * x**3)))
end function

function softmax(x) result(y)
real(sp), intent(in) :: x(:,:)
real(sp) :: y(size(x,1),size(x,2))
integer :: i
do i = 1, size(x,2)
    y(:,i) = exp(x(:,i))
    y(:,i) = y(:,i) / sum(y(:,i))
end do
end function

function layer_norm(x, g, b, eps) result(y)
real(sp), intent(in) :: x(:,:), g(:), b(:), eps
real(sp) :: y(size(x,1),size(x,2))
real(sp) :: mean(size(x,2)), variance(size(x,2))
integer :: i
do i = 1, size(x,2)
    mean(i) = sum(x(:,i)) / size(x,1)
    variance(i) = sum((x(:,i) - mean(i))**2) / size(x,1)
end do
!do i = 1, size(x,1)
!    y(i,:) = (x(i,:) - mean(:)) / sqrt(variance(:) + eps)
!    y(i,:) = g(i) * y(i,:) + b(i)
!end do
do i = 1, size(x,2)
    y(:,i) = (x(:,i) - mean(i)) / sqrt(variance(i) + eps)
    y(:,i) = g(:) * y(:,i) + b(:)
end do
end function

function linear(x, w, b) result(y)
real(sp), intent(in) :: x(:,:), w(:,:), b(:)
real(sp) :: y(size(b,1),size(x,2))
integer :: i
!y = matmul(w, x) + spread(b, 2, size(x,2))
!y = matmul(w, x)
call matmul_2d(w, x, y)
do i = 1, size(y,2)
    y(:,i) = y(:,i) + b(:)
end do
end function

function ffn(x, fc_w, fc_b, proj_w, proj_b) result(y)
real(sp), intent(in) :: x(:,:), fc_w(:,:), fc_b(:), proj_w(:,:), proj_b(:)
real(sp) :: y(size(x,1),size(x,2))
!real(sp) :: a(4*size(x,1),size(x,2))
!a = gelu(linear(x, fc_w, fc_b))
y = linear(gelu(linear(x, fc_w, fc_b)), proj_w, proj_b)
end function

function attention(n_embd_head,n_seq,n_seq_x, q, k, v, mask) result(y)
integer, intent(in) :: n_embd_head, n_seq, n_seq_x
real(sp), intent(in) :: q(n_embd_head,n_seq_x), k(n_embd_head,n_seq), v(n_embd_head,n_seq), mask(n_seq,n_seq_x)
real(sp) :: y(n_embd_head,n_seq_x)
real(sp) :: tmp(n_seq,n_seq_x)
!tmp = matmul(transpose(k), q)
!call matmul_2d(transpose(k), q, tmp)
call matmul_2d_t(k, q, tmp)
call matmul_2d(v, softmax(tmp / sqrt(real(n_embd_head,sp)) + mask), y)
end function

function mha(n_seq, n_seq_x, n_embd, x, attn_w, attn_b, proj_w, proj_b, n_head, &
            use_kv_cache, kv_cache) &
        result(y)
integer, intent(in) :: n_seq, n_seq_x, n_embd
real(sp), intent(in) :: x(n_embd,n_seq_x), &
    attn_w(3*n_embd,n_embd), attn_b(3*n_embd), &
    proj_w(n_embd,n_embd), proj_b(n_embd)
real(sp), intent(inout) :: kv_cache(n_embd,n_seq,2)
integer, intent(in) :: n_head
logical, intent(in) :: use_kv_cache
real(sp) :: y(n_embd,n_seq_x)
real(sp) :: causal_mask(n_seq,n_seq_x)
real(sp) :: x2(3*n_embd,n_seq_x)
integer :: i, j
! Mask
if (use_kv_cache) then
    causal_mask = 0
else
    do j = 1, n_seq
    do i = 1, n_seq
        if (i > j) then
            causal_mask(i,j) = -1e10_sp
        else
            causal_mask(i,j) = 0
        end if
    end do
    end do
end if
x2 = linear(x, attn_w, attn_b)
associate ( &
        q => x2((1-1)*n_embd+1:1*n_embd,:), &
        k => x2((2-1)*n_embd+1:2*n_embd,:), &
        v => x2((3-1)*n_embd+1:3*n_embd,:)  &
    )
    if (use_kv_cache) then
        kv_cache(:,n_seq,1) = k(:,1)
        kv_cache(:,n_seq,2) = v(:,1)
    else
        kv_cache(:,:,1) = k
        kv_cache(:,:,2) = v
    end if
end associate
associate ( &
        q => x2((1-1)*n_embd+1:1*n_embd,:), &
        k => kv_cache(:,:,1), &
        v => kv_cache(:,:,2)  &
    )
    ! Perform attention over each head
    do i = 1, n_head
        y((i-1)*n_embd/n_head+1:i*n_embd/n_head,:) = attention( &
            n_embd/n_head, n_seq, n_seq_x, &
            q((i-1)*n_embd/n_head+1:i*n_embd/n_head,:), &
            k((i-1)*n_embd/n_head+1:i*n_embd/n_head,:), &
            v((i-1)*n_embd/n_head+1:i*n_embd/n_head,:), &
            causal_mask)
    end do
end associate
! Out projection
y = linear(y, proj_w, proj_b)
end function


function transformer_block(n_seq, n_seq_x, n_embd, x, mlp_fc_w, mlp_fc_b, mlp_proj_w, mlp_proj_b, &
        attn_w, attn_b, attn_proj_w, attn_proj_b, ln1_g, ln1_b, ln2_g, ln2_b, &
        n_head, use_kv_cache, kv_cache) result(y)
integer, intent(in) :: n_head
integer, intent(in) :: n_seq, n_seq_x, n_embd
real(sp), intent(in) :: x(n_embd,n_seq_x), &
    mlp_fc_w(:,:), mlp_fc_b(:), &
    mlp_proj_w(:,:), mlp_proj_b(:), &
    attn_w(:,:), attn_b(:), attn_proj_w(:,:), attn_proj_b(:), &
    ln1_g(:), ln1_b(:), ln2_g(:), ln2_b(:)
real(sp) :: y(n_embd,n_seq_x)
logical, intent(in) :: use_kv_cache
real(sp), intent(inout) :: kv_cache(n_embd,n_seq,2)
y = x + mha(n_seq, n_seq_x, n_embd, layer_norm(x, ln1_g, ln1_b, 1e-5_sp), &
    attn_w, attn_b, attn_proj_w, attn_proj_b, n_head, use_kv_cache, kv_cache)
y = y + ffn(layer_norm(y, ln2_g, ln2_b, 1e-5_sp), &
    mlp_fc_w, mlp_fc_b, mlp_proj_w, mlp_proj_b)
end function

function gpt2(n_vocab, n_ctx, n_seq, n_seq_x, n_embd, n_layer, n_head, input, &
        wte, wpe, &
        mlp_fc_w, mlp_fc_b, mlp_proj_w, mlp_proj_b, &
        attn_w, attn_b, attn_proj_w, attn_proj_b, &
        ln1_g, ln1_b, ln2_g, ln2_b, lnf_g, lnf_b, &
        use_kv_cache, kv_cache) result(y)
integer, intent(in) :: n_vocab, n_ctx, n_seq, n_seq_x, n_embd, n_layer, n_head
integer, intent(in) :: input(n_seq)
real(sp), intent(in) :: wte(n_embd,n_vocab), wpe(n_embd,n_ctx), &
    mlp_fc_w(4*n_embd,n_embd,n_layer), mlp_fc_b(4*n_embd,n_layer), &
    mlp_proj_w(n_embd,4*n_embd,n_layer), mlp_proj_b(n_embd,n_layer), &
    attn_w(3*n_embd,n_embd,n_layer), attn_b(3*n_embd,n_layer), &
    attn_proj_w(n_embd,n_embd,n_layer), attn_proj_b(n_embd,n_layer), &
    ln1_b(n_embd,n_layer), ln1_g(n_embd,n_layer), &
    ln2_b(n_embd,n_layer), ln2_g(n_embd,n_layer), &
    lnf_b(n_embd), lnf_g(n_embd)
logical, intent(in) :: use_kv_cache
real(sp), intent(inout) :: kv_cache(n_embd,n_seq,2,n_layer)
real(sp) :: y(n_vocab,n_seq_x)
real(sp) :: x(n_embd,n_seq_x)
integer :: i
if (use_kv_cache) then
    i = n_seq
    x(:,1) = wte(:,input(i)+1) + wpe(:,i)
else
    do i = 1, n_seq
        x(:,i) = wte(:,input(i)+1) + wpe(:,i)
    end do
end if
do i = 1, n_layer
    x = transformer_block(n_seq, n_seq_x, n_embd, x, &
        mlp_fc_w(:,:,i), mlp_fc_b(:,i), &
        mlp_proj_w(:,:,i), mlp_proj_b(:,i), &
        attn_w(:,:,i), attn_b(:,i), attn_proj_w(:,:,i), attn_proj_b(:,i), &
        ln1_g(:,i), ln1_b(:,i), ln2_g(:,i), ln2_b(:,i), &
        n_head, use_kv_cache, kv_cache(:,:,:,i))
end do
x = layer_norm(x, lnf_g, lnf_b, 1e-5)
!y = matmul(transpose(wte), x)
call matmul_2d_t(wte, x, y)
end function

function generate(n_tokens_to_generate, m, &
        n_seq, input, &
        use_cache, &
        byte_decoder, stop_text) result(output)
integer, intent(in) :: n_seq, n_tokens_to_generate
type(model_t), intent(in) :: m
integer, intent(in) :: input(n_seq)
logical, intent(in) :: use_cache
integer, intent(in) :: byte_decoder(:)
character(*), intent(in), optional :: stop_text ! Stop if you see this text
integer, allocatable :: output(:)
real(sp), allocatable :: logits(:,:)
integer :: i
integer :: n_seq2, n_seq_x
integer :: next_id
integer, allocatable :: input2(:)
logical :: use_kv_cache
real(sp) :: kv_cache(m%n_embd,n_seq+n_tokens_to_generate,2,m%n_layer)
character(:), allocatable :: output_txt, last_token
allocate(input2(size(input)))
if (present(stop_text)) then
    output_txt = ""
end if
input2 = input
do i = 1, n_tokens_to_generate
    if (use_cache) then
        use_kv_cache = (i > 1) ! Use cache for subsequent tokens
    else
        use_kv_cache = .false.
    end if
    n_seq2 = size(input2)
    if (use_kv_cache) then
        n_seq_x = 1
    else
        n_seq_x = n_seq2
    end if
    allocate(logits(m%n_vocab, n_seq_x))
    logits = gpt2(m%n_vocab, m%n_ctx, n_seq2, n_seq_x, m%n_embd, m%n_layer, &
            m%n_head, &
            input2, &
            m%wte, m%wpe, &
            m%mlp_fc_w, m%mlp_fc_b, m%mlp_proj_w, m%mlp_proj_b, &
            m%attn_w, m%attn_b, m%attn_proj_w, m%attn_proj_b, &
            m%ln1_g, m%ln1_b, m%ln2_g, m%ln2_b, m%lnf_g, m%lnf_b, use_kv_cache, kv_cache(:,:n_seq2,:,:))
!    next_id = maxloc(logits(:,n_seq_x), dim=1)-1
    input2 = [input2, next_id]
    last_token = decode([next_id], m%decoder_idx, &
        m%decoder_txt, byte_decoder)
    write(*, fmt="(a)", advance="no") last_token
    if (present(stop_text)) then
        output_txt = output_txt // last_token
        if (output_txt(len(output_txt)-len(stop_text)+1:len(output_txt)) == stop_text) then
            exit
        end if
    end if
    deallocate(logits)
end do
output = input2(n_seq+1:)
end function

end module
module driver
use gpt2_mod, only: generate, model_t
use tokenizer, only: encode, decode, string
use omp, only: omp_get_wtime
implicit none

integer, parameter :: sp = kind(0.0)
integer, parameter :: dp = kind(0.d0)
character(1), parameter :: LF = achar(10)

contains

subroutine load_input(filename, input_txt, n_tokens_to_generate)
! Load the input from a namelist `filename`
character(*), intent(in) :: filename
character(:), allocatable, intent(out) :: input_txt
integer, intent(out) :: n_tokens_to_generate
character(1024) :: input_txt2
integer :: u, ios
!namelist / input_fastGPT / n_tokens_to_generate
allocate(character(0) :: input_txt)
input_txt = ""
open(newunit=u, file=filename, status="old")
!read(u, input_fastGPT)
do
    read(u, "(a)", iostat=ios) input_txt2
    if (ios /= 0) exit
    if (len(input_txt) > 0) input_txt = input_txt // char(10)
    input_txt = input_txt // trim(input_txt2)
end do
close(u)
end subroutine

subroutine load_model(filename, m)
character(*), intent(in) :: filename
type(model_t), intent(out) :: m
integer :: u
open(newunit=u, file=filename, form="unformatted", access="stream", status="old")
!read(u) model_version
!                    fastGPT (digits look similar to the letters they represent)
! model_version /= 0xfa51697
read(u) m%n_vocab, m%n_ctx, m%n_embd, m%n_layer, m%n_head, m%n_decoder_idx, &
    m%n_decoder_txt, m%n_vocab_idx, m%n_vocab_txt, m%n_byte_encoder
allocate(m%wte(m%n_embd,m%n_vocab), m%wpe(m%n_embd,m%n_ctx), &
    m%mlp_fc_w(4*m%n_embd,m%n_embd,m%n_layer), m%mlp_fc_b(4*m%n_embd,m%n_layer), &
    m%mlp_proj_w(m%n_embd,4*m%n_embd,m%n_layer), m%mlp_proj_b(m%n_embd,m%n_layer), &
    m%attn_w(3*m%n_embd,m%n_embd,m%n_layer), m%attn_b(3*m%n_embd,m%n_layer), &
    m%attn_proj_w(m%n_embd,m%n_embd,m%n_layer), m%attn_proj_b(m%n_embd,m%n_layer), &
    m%ln1_b(m%n_embd,m%n_layer), m%ln1_g(m%n_embd,m%n_layer), &
    m%ln2_b(m%n_embd,m%n_layer), m%ln2_g(m%n_embd,m%n_layer), &
    m%lnf_b(m%n_embd), m%lnf_g(m%n_embd), &
    m%decoder_idx(0:m%n_decoder_idx-1), m%decoder_txt(m%n_decoder_txt), &
    m%vocab_idx(0:m%n_vocab_idx-1), m%vocab_txt(m%n_vocab_txt), &
    m%byte_encoder(0:m%n_byte_encoder-1))
close(u)
end subroutine

subroutine gpt2_driver(input, output, m)
integer, allocatable, intent(out) :: input(:), output(:)
type(model_t), intent(out) :: m
character(:), allocatable :: input_txt
integer :: n_tokens_to_generate
real(dp) :: t1, t2
call load_input("input", input_txt, n_tokens_to_generate)

! Load the model
print "(a)", "Loading the model..."
call cpu_time(t1)
call load_model("model.dat", m)
call cpu_time(t2)
print "(a,f8.3,a)", "    done. Time:", t2-t1, "s"
print *
print "(a)", "Model parameters:"
print "(a,i6)", "n_vocab =", m%n_vocab
print "(a,i6)", "n_ctx   =", m%n_ctx
print "(a,i6)", "n_embd  =", m%n_embd
print "(a,i6)", "n_layer =", m%n_layer
print "(a,i6)", "n_head  =", m%n_head
print *

call gpt2_driver2(input_txt, n_tokens_to_generate, m, input, output)
endsubroutine

subroutine gpt2_driver2(input_txt, n_tokens_to_generate, m, input, output)
character(*), intent(in) :: input_txt
integer, intent(in) :: n_tokens_to_generate
type(model_t), intent(in) :: m
integer, allocatable, intent(out) :: input(:), output(:)
integer, allocatable :: byte_decoder(:)
integer :: n_seq
character(:), allocatable :: output_txt
real(dp) :: t1, t2, t1o, t2o
integer :: i
logical :: use_cache

! Compute byte_decoder:
allocate(byte_decoder(0:5))
byte_decoder = 0
do i = 0, size(m%byte_encoder)-1
    byte_decoder(m%byte_encoder(i)) = i
end do

print "(a)", "Input text"
print "(a)", input_txt

print *
print "(a)",  "Encoding: tokenizing input text into tokens (currently slow)..."
call cpu_time(t1)
input = encode(input_txt, m%decoder_idx, m%decoder_txt, m%vocab_idx, m%vocab_txt, &
    m%byte_encoder)
call cpu_time(t2)
n_seq = size(input)
print "(a,f8.3,a)", "    done. Time:", t2-t1, "s"
print *
print "(a)", "Input parameters:"
print "(a,i4)", "n_seq                =", n_seq
print "(a,i4)", "n_tokens_to_generate =", n_tokens_to_generate
print *
print "(a)", "Input tokens:"
print "(1000(i6))", input
print *

if (n_seq + n_tokens_to_generate >= m%n_ctx) then
    print *, "The maximum sequence length of the model was surpassed."
    print *, "Make the input and/or number of tokens to generate shorter."
    error stop
end if

print "(a)", "Decoded input as text:"
!print "(a)", decode(input, decoder_idx, decoder_txt, byte_decoder)
allocate(character(0) :: output_txt) ! Fix GFortran warning
output_txt = decode(input, m%decoder_idx, m%decoder_txt, byte_decoder)
print "(a)", output_txt
print *

if (input_txt /= output_txt) then
    error stop "The decoded input text does not agree with the input text"
end if

allocate(output(n_tokens_to_generate))
print "(a)", "Running model..."
call cpu_time(t1)
t1o = omp_get_wtime()
use_cache = .true.
output = generate(n_tokens_to_generate, m, size(input), input, use_cache, &
    byte_decoder)
print *
t2o = omp_get_wtime()
call cpu_time(t2)
print "(a,f8.3,a,f4.2,a)", "    done. Time:", t2o-t1o, "s (", (t2-t1)/(t2o-t1o), "x)"
print *
print "(a)", "Output tokens:"
print "(1000(i6))", output
output_txt = decode(output, m%decoder_idx, m%decoder_txt, byte_decoder)
print *
print "(a)", "Decoded output as text:"
print "(a)", output_txt
end subroutine

subroutine gpt2_driver3(input_txt, n_tokens_to_generate, stop_text, m, output_txt)
character(*), intent(in) :: input_txt, stop_text
integer, intent(in) :: n_tokens_to_generate
type(model_t), intent(in) :: m
integer, allocatable :: input(:), output(:)
integer, allocatable :: byte_decoder(:)
integer :: n_seq
character(:), allocatable, intent(out) :: output_txt
integer :: i
logical :: use_cache
! TODO: move the decoder into model_t
! Compute byte_decoder:
allocate(byte_decoder(0:5))
byte_decoder = 0
do i = 0, size(m%byte_encoder)-1
    byte_decoder(m%byte_encoder(i)) = i
end do
input = encode(input_txt, m%decoder_idx, m%decoder_txt, m%vocab_idx, m%vocab_txt, &
    m%byte_encoder)
n_seq = size(input)
if (n_seq + n_tokens_to_generate >= m%n_ctx) then
    print *, "The maximum sequence length of the model was surpassed."
    print *, "Make the input and/or number of tokens to generate shorter."
    error stop
end if
allocate(character(0) :: output_txt) ! Fix GFortran warning
output_txt = decode(input, m%decoder_idx, m%decoder_txt, byte_decoder)
if (input_txt /= output_txt) then
    error stop "The decoded input text does not agree with the input text"
end if
use_cache = .true.
output = generate(n_tokens_to_generate, m, size(input), input, use_cache, &
    byte_decoder, stop_text)
output_txt = decode(output, m%decoder_idx, m%decoder_txt, byte_decoder)
end subroutine

function get_prompt() result(input)
character(:), allocatable :: input
character(1024) :: tmp
integer ::ios
read(*,"(a)",iostat=ios) tmp
if (ios == 0) then
    input = trim(tmp)
else
    input = ""
end if
end function

subroutine chat(inputs)
type(string), optional, intent(in) :: inputs(:)
type(model_t) :: m
character(:), allocatable :: prompt, input, output
integer :: i, n_prompts
call load_model("model.dat", m)
prompt = "Your name is fastGPT and you are an AI bot. The user will ask you &
&questions and you answer in a nice, truthful, short way." // LF // "&
&User: What is the capital of Czechia?" // LF // "&
&fastGPT: Prague." // LF // "&
&User: How many legs does a dog have?" // LF // "&
&fastGPT: Four." // LF // "&
&User:"
write(*,"(a)",advance="no") prompt
if (present(inputs)) then
    n_prompts = size(inputs)
else
    n_prompts = 1024
end if
do i = 1, n_prompts
    write(*,"(a)",advance="no")  " "
    if (present(inputs)) then
        input = inputs(i)%s
        write(*,"(a)") input
    else
        input = get_prompt()
        if (input == "") exit
    end if
    write(*,"(a)",advance="no") "fastGPT:"
    prompt = prompt // " " // input // LF // "fastGPT:"
    call gpt2_driver3(prompt, 200, "User:", m, output)
    prompt = prompt // output
end do
print *
end subroutine

end module
program gpt2
use driver, only: gpt2_driver, model_t
implicit none
integer, allocatable :: input(:), output(:)
type(model_t) :: m
call gpt2_driver(input, output, m)
end program
