module gpt2_mod
use linalg, only: matmul_2d, matmul_2d_t
implicit none

integer, parameter :: sp = kind(0.0)
real(sp), parameter :: pi = 3.14159265358979323846_sp

type :: string
    character(:), allocatable :: s
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
    y(:,i) = exp(x(:,i) - maxval(x(:,i)))
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
real(sp), intent(in) :: x(n_embd,n_seq_x), &
    mlp_fc_w(:,:), mlp_fc_b(:), &
    mlp_proj_w(:,:), mlp_proj_b(:), &
    attn_w(:,:), attn_b(:), attn_proj_w(:,:), attn_proj_b(:), &
    ln1_g(:), ln1_b(:), ln2_g(:), ln2_b(:)
integer, intent(in) :: n_head
integer, intent(in) :: n_seq, n_seq_x, n_embd
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

function generate(n_tokens_to_generate, &
        n_vocab, n_ctx, n_seq, n_embd, n_layer, n_head, input, &
        wte, wpe, &
        mlp_fc_w, mlp_fc_b, mlp_proj_w, mlp_proj_b, &
        attn_w, attn_b, attn_proj_w, attn_proj_b, &
        ln1_g, ln1_b, ln2_g, ln2_b, lnf_g, lnf_b, use_cache, &
        decoder_idx, decoder_txt, byte_decoder) result(output)
integer, intent(in) :: n_vocab, n_ctx, n_seq, n_embd, n_layer, n_head, &
    n_tokens_to_generate
integer, intent(in) :: input(n_seq)
real(sp), intent(in) :: wte(n_embd,n_vocab), wpe(n_embd,n_ctx), &
    mlp_fc_w(4*n_embd,n_embd,n_layer), mlp_fc_b(4*n_embd,n_layer), &
    mlp_proj_w(n_embd,4*n_embd,n_layer), mlp_proj_b(n_embd,n_layer), &
    attn_w(3*n_embd,n_embd,n_layer), attn_b(3*n_embd,n_layer), &
    attn_proj_w(n_embd,n_embd,n_layer), attn_proj_b(n_embd,n_layer), &
    ln1_b(n_embd,n_layer), ln1_g(n_embd,n_layer), &
    ln2_b(n_embd,n_layer), ln2_g(n_embd,n_layer), &
    lnf_b(n_embd), lnf_g(n_embd)
logical, intent(in) :: use_cache
integer, intent(in) :: decoder_idx(:), byte_decoder(:)
character, intent(in) :: decoder_txt(:)
integer :: output(n_tokens_to_generate)
real(sp), allocatable :: logits(:,:)
integer :: i
integer :: n_seq2, n_seq_x
integer :: next_id
integer, allocatable :: input2(:)
logical :: use_kv_cache
real(sp) :: kv_cache(n_embd,n_seq+n_tokens_to_generate,2,n_layer)
allocate(input2(size(input)))
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
    allocate(logits(n_vocab, n_seq_x))
    logits = gpt2(n_vocab, n_ctx, n_seq2, n_seq_x, n_embd, n_layer, n_head, &
            input2, &
            wte, wpe, &
            mlp_fc_w, mlp_fc_b, mlp_proj_w, mlp_proj_b, &
            attn_w, attn_b, attn_proj_w, attn_proj_b, &
            ln1_g, ln1_b, ln2_g, ln2_b, lnf_g, lnf_b, use_kv_cache, kv_cache(:,:n_seq2,:,:))
    next_id = maxloc(logits(:,n_seq_x), dim=1)-1
    write(*, fmt="(a)", advance="no") decode([next_id], decoder_idx, decoder_txt, byte_decoder)
    input2 = [input2, next_id]
    deallocate(logits)
end do
output = input2(n_seq+1:)
print *
end function

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

function word_to_token(word, idx, decoder_txt) result(token)
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
token = 0
!error stop "Word not found in decoder_txt"
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

function bpe(token, vocab_idx, vocab_txt) result(tokens)
! Takes a token as a string, and returns bpe tokens as an array of strings
character(*), intent(in) :: token
integer, intent(in) :: vocab_idx(0:)
character, intent(in) :: vocab_txt(:)
type(string), allocatable :: tokens(:)
type(string) :: s, s2
if (token == "Ġtheorized") then
    s%s = "Ġtheor"
    s2%s = "ized"
    tokens = [s, s2]
else
    s%s = token
    tokens = [s]
end if
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
        tokens = [tokens, word_to_token(bpe_tokens(j)%s, idx, decoder_txt)]
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
