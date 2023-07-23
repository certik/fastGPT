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
    integer(1), allocatable :: decoder_txt(:), vocab_txt(:)
    integer :: model_file_version
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

subroutine layer_norm(y, x, g, b, eps)
real(sp), intent(in) :: x(:,:), g(:), b(:), eps
real(sp), intent(out) :: y(size(x,1),size(x,2))
real(sp) :: mean(size(x,2)), variance(size(x,2))
real(sp) :: xi(size(x,1))
integer :: i, j
do i = 1, size(x,2)
    do j = 1, size(x,1)
        xi(j) = x(j,i)
    end do
    mean(i) = sum(xi) / size(x,1)
    do j = 1, size(x,1)
        xi(j) = (xi(j) - mean(i))**2
    end do
    variance(i) = sum(xi) / size(x,1)
end do
!do i = 1, size(x,1)
!    y(i,:) = (x(i,:) - mean(:)) / sqrt(variance(:) + eps)
!    y(i,:) = g(i) * y(i,:) + b(i)
!end do
do i = 1, size(x,2)
    do j = 1, size(x,1)
        y(j,i) = (x(j,i) - mean(i)) / sqrt(variance(i) + eps)
        y(j,i) = g(j) * y(j,i) + b(j)
    end do
end do
end subroutine

function linear(x, w, b) result(y)
real(sp), intent(in) :: x(:,:), w(:,:), b(:)
real(sp) :: y(size(b,1),size(x,2))
integer :: i, j
!y = matmul(w, x) + spread(b, 2, size(x,2))
!y = matmul(w, x)
call matmul_2d(w, x, y)
do i = 1, size(y,2)
    do j = 1, size(y,1)
        y(j,i) = y(j,i) + b(j)
    end do
end do
end function

subroutine ffn(y, x, fc_w, fc_b, proj_w, proj_b)
real(sp), intent(in) :: x(:,:), fc_w(:,:), fc_b(:), proj_w(:,:), proj_b(:)
real(sp), intent(inout) :: y(size(x,1),size(x,2))
real(sp) :: yy(size(x,1),size(x,2))
real(sp) :: a(4*size(x,1),size(x,2))
real(sp) :: aa(4*size(x,1),size(x,2))
aa = linear(x, fc_w, fc_b)
a = gelu(aa)
yy = linear(a, proj_w, proj_b)
y = y + yy
end subroutine

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

subroutine mha(y, n_seq, n_seq_x, n_embd, x, attn_w, attn_b, proj_w, proj_b, n_head, &
            use_kv_cache, kv_cache)
integer, intent(in) :: n_seq, n_seq_x, n_embd
real(sp), intent(in) :: x(n_embd,n_seq_x), &
    attn_w(3*n_embd,n_embd), attn_b(3*n_embd), &
    proj_w(n_embd,n_embd), proj_b(n_embd)
real(sp), intent(inout) :: kv_cache(n_embd,n_seq,2)
integer, intent(in) :: n_head
logical, intent(in) :: use_kv_cache
real(sp), intent(out) :: y(n_embd,n_seq_x)
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
if (use_kv_cache) then
    do j = 1, n_embd
        kv_cache(j,n_seq,1) = x2((2-1)*n_embd+j,1)
        kv_cache(j,n_seq,2) = x2((3-1)*n_embd+j,1)
    end do
else
    do i = 1, n_seq
    do j = 1, n_embd
        kv_cache(j,i,1) = x2((2-1)*n_embd+j,i)
        kv_cache(j,i,2) = x2((3-1)*n_embd+j,i)
    end do
    end do
end if
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
end subroutine


subroutine transformer_block(n_seq, n_seq_x, n_embd, x, mlp_fc_w, mlp_fc_b, mlp_proj_w, mlp_proj_b, &
        attn_w, attn_b, attn_proj_w, attn_proj_b, ln1_g, ln1_b, ln2_g, ln2_b, &
        n_head, use_kv_cache, kv_cache)
real(sp), intent(inout) :: x(n_embd,n_seq_x)
real(sp), intent(in) :: &
    mlp_fc_w(:,:), mlp_fc_b(:), &
    mlp_proj_w(:,:), mlp_proj_b(:), &
    attn_w(:,:), attn_b(:), attn_proj_w(:,:), attn_proj_b(:), &
    ln1_g(:), ln1_b(:), ln2_g(:), ln2_b(:)
integer, intent(in) :: n_head
integer, intent(in) :: n_seq, n_seq_x, n_embd
logical, intent(in) :: use_kv_cache
real(sp) :: y(n_embd,n_seq_x)
real(sp) :: yy(n_embd,n_seq_x)
real(sp), intent(inout) :: kv_cache(n_embd,n_seq,2)
call layer_norm(y, x, ln1_g, ln1_b, 1e-5_sp)
call mha(yy, n_seq, n_seq_x, n_embd, y, &
    attn_w, attn_b, attn_proj_w, attn_proj_b, n_head, use_kv_cache, kv_cache)
x = x + yy
!print *, "In: ", x(1,1), ln2_g(1), ln2_g(size(ln2_g)), ln2_b(1), ln2_b(size(ln2_b))
call layer_norm(y, x, ln2_g, ln2_b, 1e-5_sp)
!print *, "Out1:", y(1,1)
!x = y
!print *, "Out2:", x(1,1)
call ffn(x, y, mlp_fc_w, mlp_fc_b, mlp_proj_w, mlp_proj_b)
end subroutine

subroutine gpt2(y, n_vocab, n_ctx, n_seq, n_seq_x, n_embd, n_layer, n_head, input, &
        wte, wpe, &
        mlp_fc_w, mlp_fc_b, mlp_proj_w, mlp_proj_b, &
        attn_w, attn_b, attn_proj_w, attn_proj_b, &
        ln1_g, ln1_b, ln2_g, ln2_b, lnf_g, lnf_b, &
        use_kv_cache, kv_cache)
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
real(sp), intent(out) :: y(n_vocab,n_seq_x)
real(sp) :: x(n_embd,n_seq_x)
real(sp) :: yy(n_embd,n_seq_x)
integer :: i, j
if (use_kv_cache) then
    i = n_seq
    x(:,1) = wte(:,input(i)+1) + wpe(:,i)
else
    do i = 1, n_seq
        do j = 1, n_embd
            x(j,i) = wte(j,input(i)+1) + wpe(j,i)
        end do
    end do
end if
!print *, "It fails below:"
do j = 1, n_layer
    i = j
    !i = 1
!    print *, i ! Never gets printed
    call transformer_block(n_seq, n_seq_x, n_embd, x, &
        mlp_fc_w(:,:,i), mlp_fc_b(:,i), &
        mlp_proj_w(:,:,i), mlp_proj_b(:,i), &
        attn_w(:,:,i), attn_b(:,i), attn_proj_w(:,:,i), attn_proj_b(:,i), &
        ln1_g(:,i), ln1_b(:,i), ln2_g(:,i), ln2_b(:,i), &
        n_head, use_kv_cache, kv_cache(:,:,:,i))
!    print *, x(1,1)
end do
call layer_norm(yy, x, lnf_g, lnf_b, 1e-5)
x = yy
!y = matmul(transpose(wte), x)
call matmul_2d_t(wte, x, y)
end subroutine

subroutine generate(output, n_tokens_to_generate, m, &
        n_seq, input, &
        use_cache, &
        byte_decoder, stop_text)
integer, intent(in) :: n_seq, n_tokens_to_generate
type(model_t), intent(in) :: m
integer, intent(in) :: input(n_seq)
logical, intent(in) :: use_cache
integer, intent(in) :: byte_decoder(:)
character(*), intent(in), optional :: stop_text ! Stop if you see this text
integer, intent(out) :: output(:)
real(sp), allocatable :: logits(:,:)
integer :: i
integer :: n_seq2, n_seq_x
integer :: next_id
integer :: input2(size(input)+n_tokens_to_generate)
logical :: use_kv_cache
real(sp) :: kv_cache(m%n_embd,n_seq+n_tokens_to_generate,2,m%n_layer)
character(:), allocatable :: output_txt, last_token
if (present(stop_text)) then
    output_txt = ""
end if
input2(:n_seq) = input
do i = 1, n_tokens_to_generate
    if (use_cache) then
        use_kv_cache = (i > 1) ! Use cache for subsequent tokens
    else
        use_kv_cache = .false.
    end if
    n_seq2 = n_seq+i-1
    if (use_kv_cache) then
        n_seq_x = 1
    else
        n_seq_x = n_seq2
    end if
    allocate(logits(m%n_vocab, n_seq_x))
    call gpt2(logits, m%n_vocab, m%n_ctx, n_seq2, n_seq_x, m%n_embd, m%n_layer, &
            m%n_head, &
            input2(:n_seq2), &
            m%wte, m%wpe, &
            m%mlp_fc_w, m%mlp_fc_b, m%mlp_proj_w, m%mlp_proj_b, &
            m%attn_w, m%attn_b, m%attn_proj_w, m%attn_proj_b, &
            m%ln1_g, m%ln1_b, m%ln2_g, m%ln2_b, m%lnf_g, m%lnf_b, use_kv_cache, kv_cache(:,:n_seq2,:,:))
    next_id = maxloc(logits(:,n_seq_x), dim=1)-1
    input2(n_seq2+1) = next_id
    !last_token = decode([next_id], m%decoder_idx, &
    !    m%decoder_txt, byte_decoder)
    !write(*, fmt="(a)", advance="no") last_token
    if (present(stop_text)) then
        output_txt = output_txt // last_token
        if (output_txt(len(output_txt)-len(stop_text)+1:len(output_txt)) == stop_text) then
            exit
        end if
    end if
    deallocate(logits)
end do
do i = 1, n_tokens_to_generate
    output(i) = input2(n_seq+i)
end do
end subroutine

end module
