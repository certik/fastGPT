module gpt2_mod
use linalg, only: matmul_2d, matmul_2d_t
implicit none

integer, parameter :: sp = kind(0.0)
real(sp), parameter :: pi = 3.14159265358979323846_sp

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

    ! Alternative implementation (division is expensive)
    !a = x * (135135.0 + x2 * (17325.0 + x2 * (378.0 + x2)))
    !b = 135135.0 + x2 * (62370.0 + x2 * (3150.0 + x2 * 28.0))
    !y = a / b
end if
end function

elemental real(sp) function gelu(x) result(y)
real(sp), intent(in) :: x
y = 0.5_sp * x * (1 + fast_tanh(sqrt(2 / pi) * (x + 0.044715_sp * x**3)))
end function

elemental real(sp) function fast_gelu(x) result(y)
real(sp), intent(in) :: x
real(sp) :: x2, x_
x_ = sqrt(2 / pi) * (x + 0.044715_sp * x**3)
if (x_ > 5) then
    y = x
elseif (x_ < -5) then
    y = 0
else
    x2 = x_*x_
    y = x_ * (0.98569772605911309407 + x2 *(-0.2794500993392901382 &
        + x2 * (6.8280504526399188164e-2 + x2 * (-1.0972014877337651823e-2 &
        + x2 * (1.1132367134444316902e-3 + x2 * (-7.018851897305717565e-5 &
        + x2 * (2.656616768082727089e-6 + x2 * (-5.5138381821615909058e-8 &
        + x2 * 4.8162484477588665996e-10))))))))
    y = 0.5_sp * x * (1 + y)

    ! Alternative implementation (division is expensive)
    !a = x * (135135.0 + x2 * (17325.0 + x2 * (378.0 + x2)))
    !b = 135135.0 + x2 * (62370.0 + x2 * (3150.0 + x2 * 28.0))
    !y = a / b
end if
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
real(sp) :: a(4*size(x,1),size(x,2)), x0, x_, x2
integer :: i, j
!a = linear(x, fc_w, fc_b)
call matmul_2d(fc_w, x, a)
do i = 1, size(a,2)
do j = 1, size(a,1)
    !a(j,i) = fast_gelu(a(j,i) + fc_b(j))
    x0 = a(j,i) + fc_b(j)
    x_ = sqrt(2 / pi) * (x0 + 0.044715_sp * x0**3)
    if (x_ > 5) then
        a(j,i) = x0
    elseif (x_ < -5) then
        a(j,i) = 0
    else
        x2 = x_*x_
        a(j,i) = x_ * (0.98569772605911309407 + x2 *(-0.2794500993392901382 &
            + x2 * (6.8280504526399188164e-2 + x2 * (-1.0972014877337651823e-2 &
            + x2 * (1.1132367134444316902e-3 + x2 * (-7.018851897305717565e-5 &
            + x2 * (2.656616768082727089e-6 + x2 * (-5.5138381821615909058e-8 &
            + x2 * 4.8162484477588665996e-10))))))))
        a(j,i) = 0.5_sp * x0 * (1 + a(j,i))
    end if
end do
end do
y = linear(a, proj_w, proj_b)
end function

function attention(q, k, v, mask) result(y)
real(sp), intent(in) :: q(:,:), k(:,:), v(:,:), mask(:,:)
real(sp) :: y(size(v,1),size(q,2))
real(sp) :: tmp(size(k,2),size(q,2))
!tmp = matmul(transpose(k), q)
!call matmul_2d(transpose(k), q, tmp)
call matmul_2d_t(k, q, tmp)
call matmul_2d(v, softmax(tmp / sqrt(real(size(q,1),sp)) + mask), y)
end function

function mha(n_seq, n_embd, x, attn_w, attn_b, proj_w, proj_b, n_head) &
        result(y)
integer, intent(in) :: n_seq, n_embd
real(sp), intent(in) :: x(n_embd,n_seq), &
    attn_w(3*n_embd,n_embd), attn_b(3*n_embd), &
    proj_w(n_embd,n_embd), proj_b(n_embd)
integer, intent(in) :: n_head
real(sp) :: y(n_embd,n_seq)
real(sp) :: causal_mask(n_seq,n_seq)
real(sp) :: x2(3*n_embd,n_seq)
integer :: i, j
! Mask
do j = 1, n_seq
do i = 1, n_seq
    if (i > j) then
        causal_mask(i,j) = -1e10_sp
    else
        causal_mask(i,j) = 0
    end if
end do
end do
x2 = linear(x, attn_w, attn_b)
associate ( &
        q => x2((1-1)*n_embd+1:1*n_embd,:), &
        k => x2((2-1)*n_embd+1:2*n_embd,:), &
        v => x2((3-1)*n_embd+1:3*n_embd,:)  &
    )
    ! Perform attention over each head
    do i = 1, n_head
        y((i-1)*n_embd/n_head+1:i*n_embd/n_head,:) = attention( &
            q((i-1)*n_embd/n_head+1:i*n_embd/n_head,:), &
            k((i-1)*n_embd/n_head+1:i*n_embd/n_head,:), &
            v((i-1)*n_embd/n_head+1:i*n_embd/n_head,:), &
            causal_mask)
    end do
end associate
! Out projection
y = linear(y, proj_w, proj_b)
end function


function transformer_block(x, mlp_fc_w, mlp_fc_b, mlp_proj_w, mlp_proj_b, &
        attn_w, attn_b, attn_proj_w, attn_proj_b, ln1_g, ln1_b, ln2_g, ln2_b, &
        n_head) result(y)
real(sp), intent(in) :: x(:,:), &
    mlp_fc_w(:,:), mlp_fc_b(:), &
    mlp_proj_w(:,:), mlp_proj_b(:), &
    attn_w(:,:), attn_b(:), attn_proj_w(:,:), attn_proj_b(:), &
    ln1_g(:), ln1_b(:), ln2_g(:), ln2_b(:)
integer, intent(in) :: n_head
real(sp) :: y(size(x,1),size(x,2))
integer :: n_seq, n_embd
n_embd = size(x,1)
n_seq = size(x,2)
y = x + mha(n_seq, n_embd, layer_norm(x, ln1_g, ln1_b, 1e-5_sp), &
    attn_w, attn_b, attn_proj_w, attn_proj_b, n_head)
y = y + ffn(layer_norm(y, ln2_g, ln2_b, 1e-5_sp), &
    mlp_fc_w, mlp_fc_b, mlp_proj_w, mlp_proj_b)
end function

function gpt2(n_vocab, n_ctx, n_seq, n_embd, n_layer, n_head, input, &
        wte, wpe, &
        mlp_fc_w, mlp_fc_b, mlp_proj_w, mlp_proj_b, &
        attn_w, attn_b, attn_proj_w, attn_proj_b, &
        ln1_g, ln1_b, ln2_g, ln2_b, lnf_g, lnf_b) result(y)
integer, intent(in) :: n_vocab, n_ctx, n_seq, n_embd, n_layer, n_head
integer, intent(in) :: input(n_seq)
real(sp), intent(in) :: wte(n_embd,n_vocab), wpe(n_embd,n_ctx), &
    mlp_fc_w(4*n_embd,n_embd,n_layer), mlp_fc_b(4*n_embd,n_layer), &
    mlp_proj_w(n_embd,4*n_embd,n_layer), mlp_proj_b(n_embd,n_layer), &
    attn_w(3*n_embd,n_embd,n_layer), attn_b(3*n_embd,n_layer), &
    attn_proj_w(n_embd,n_embd,n_layer), attn_proj_b(n_embd,n_layer), &
    ln1_b(n_embd,n_layer), ln1_g(n_embd,n_layer), &
    ln2_b(n_embd,n_layer), ln2_g(n_embd,n_layer), &
    lnf_b(n_embd), lnf_g(n_embd)
real(sp) :: y(n_vocab,n_seq)
real(sp) :: x(n_embd,n_seq)
integer :: i
do i = 1, n_seq
    x(:,i) = wte(:,input(i)+1) + wpe(:,i)
end do
do i = 1, n_layer
    x = transformer_block(x, &
        mlp_fc_w(:,:,i), mlp_fc_b(:,i), &
        mlp_proj_w(:,:,i), mlp_proj_b(:,i), &
        attn_w(:,:,i), attn_b(:,i), attn_proj_w(:,:,i), attn_proj_b(:,i), &
        ln1_g(:,i), ln1_b(:,i), ln2_g(:,i), ln2_b(:,i), &
        n_head)
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
        ln1_g, ln1_b, ln2_g, ln2_b, lnf_g, lnf_b) result(output)
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
integer :: output(n_tokens_to_generate)
real(sp), allocatable :: logits(:,:)
integer :: i
integer :: next_id
integer, allocatable :: input2(:)
allocate(input2(size(input)))
input2 = input
do i = 1, n_tokens_to_generate
    allocate(logits(n_vocab, size(input2)))
    logits = gpt2(n_vocab, n_ctx, size(input2), n_embd, n_layer, n_head, &
            input2, &
            wte, wpe, &
            mlp_fc_w, mlp_fc_b, mlp_proj_w, mlp_proj_b, &
            attn_w, attn_b, attn_proj_w, attn_proj_b, &
            ln1_g, ln1_b, ln2_g, ln2_b, lnf_g, lnf_b)
    next_id = maxloc(logits(:,size(logits,2)), dim=1)-1
    print *, i, next_id
    input2 = [input2, next_id]
    deallocate(logits)
end do
output = input2(n_seq+1:)
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

function decode(tokens, idx, decoder_txt, byte_decoder) result(output)
integer, intent(in) :: tokens(:), idx(0:), byte_decoder(0:)
character, intent(in) :: decoder_txt(:)
character(:), allocatable :: output
character(:), allocatable :: output2, tmp
integer :: i, c, d
output2 = ""
do i = 1, size(tokens)
    output2 = output2 // c2s(decoder_txt(idx(tokens(i))+1:idx(tokens(i)+1)))
end do
i = 1
output = ""
do
    c = iachar(output2(i:i))
    if (c >= 128) then
        i = i + 1
        d = iachar(output2(i:i))
        c = ior(ishft(iand(c, 31), 6), iand(d, 63))
    end if
    tmp = achar(byte_decoder(c))
    output = output // tmp
    if (i == len(output2)) exit
    i = i + 1
end do
end function

end module
