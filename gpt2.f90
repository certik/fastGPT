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
