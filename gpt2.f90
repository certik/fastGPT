module gpt2_mod
use linalg, only: matmul_2d, matmul_2d_t
use tokenizer, only: decode
implicit none

integer, parameter :: sp = kind(0.0)
real(sp), parameter :: pi = 3.14159265358979323846_sp

! Per-layer weights for a single transformer block
type :: layer_t
    real(sp), allocatable :: mlp_fc_w(:,:), mlp_fc_b(:)
    real(sp), allocatable :: mlp_proj_w(:,:), mlp_proj_b(:)
    real(sp), allocatable :: attn_w(:,:), attn_b(:)
    real(sp), allocatable :: attn_proj_w(:,:), attn_proj_b(:)
    real(sp), allocatable :: ln1_g(:), ln1_b(:)
    real(sp), allocatable :: ln2_g(:), ln2_b(:)
end type

! This derived type contains all the data of the GPT-2 model, including all
! weights, model parameters, and encoder/decoder data
type :: model_t
    integer :: n_vocab, n_ctx, n_embd, n_layer, n_head, &
        n_decoder_idx, n_decoder_txt, &
        n_vocab_idx, n_vocab_txt, n_byte_encoder
    real(sp), allocatable :: wte(:,:), wpe(:,:)
    type(layer_t), allocatable :: layers(:)
    real(sp), allocatable :: lnf_b(:), lnf_g(:)
    integer, allocatable :: decoder_idx(:), vocab_idx(:), byte_encoder(:)
    character, allocatable :: decoder_txt(:), vocab_txt(:)
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

! In-place softmax over columns
subroutine softmax(x)
real(sp), intent(inout) :: x(:,:)
integer :: i
do i = 1, size(x,2)
    x(:,i) = exp(x(:,i) - maxval(x(:,i)))
    x(:,i) = x(:,i) / sum(x(:,i))
end do
end subroutine

subroutine layer_norm(x, g, b, eps, y)
real(sp), intent(in) :: x(:,:), g(:), b(:), eps
real(sp), intent(out) :: y(size(x,1),size(x,2))
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
end subroutine

subroutine linear(x, w, b, y)
real(sp), intent(in) :: x(:,:), w(:,:), b(:)
real(sp), intent(out) :: y(size(b,1),size(x,2))
integer :: i
!y = matmul(w, x) + spread(b, 2, size(x,2))
!y = matmul(w, x)
if (size(x, 2) == 1) then
    ! Single-token path: plain GEMV avoids BLAS packing overhead
    y(:, 1) = matmul(w, x(:, 1)) + b(:)
else
    call matmul_2d(w, x, y)
    do i = 1, size(y,2)
        y(:,i) = y(:,i) + b(:)
    end do
end if
end subroutine

subroutine ffn(x, fc_w, fc_b, proj_w, proj_b, y)
real(sp), intent(in) :: x(:,:), fc_w(:,:), fc_b(:), proj_w(:,:), proj_b(:)
real(sp), intent(out) :: y(size(x,1),size(x,2))
!real(sp) :: a(4*size(x,1),size(x,2))
!a = gelu(linear(x, fc_w, fc_b))
real(sp) :: tmp_hidden(size(fc_b,1), size(x,2)) ! Explicit temporary avoids function return copy
call linear(x, fc_w, fc_b, tmp_hidden)
tmp_hidden = gelu(tmp_hidden)
call linear(tmp_hidden, proj_w, proj_b, y)
end subroutine

! Scaled dot-product attention. When n_seq_x == 1 (single new token during
! cached inference), uses a fused dot-product loop to avoid forming the full
! score matrix, which would otherwise trigger a costly BLAS packing operation.
subroutine attention_zerocopy(n_embd_head, n_seq, n_seq_x, q, k, v, mask, y)
integer, intent(in) :: n_embd_head, n_seq, n_seq_x
real(sp), intent(in) :: q(n_embd_head, n_seq_x)
real(sp), intent(in) :: k(n_embd_head, n_seq)
real(sp), intent(in) :: v(n_embd_head, n_seq)
real(sp), intent(in) :: mask(n_seq, n_seq_x)
real(sp), intent(out) :: y(n_embd_head, n_seq_x)
real(sp) :: tmp(n_seq, n_seq_x)
real(sp) :: tmp_t(n_seq_x, n_seq)
integer :: i
if (n_seq_x == 1) then
    do i = 1, n_seq
        tmp(i, 1) = dot_product(k(:, i), q(:, 1))
    end do
    tmp(:, 1) = tmp(:, 1) / sqrt(real(n_embd_head,sp)) + mask(:, 1)
    tmp(:, 1) = exp(tmp(:, 1) - maxval(tmp(:, 1)))
    tmp(:, 1) = tmp(:, 1) / sum(tmp(:, 1))
    y(:, 1) = 0.0_sp
    do i = 1, n_seq
        y(:, 1) = y(:, 1) + v(:, i) * tmp(i, 1)
    end do
else
    call matmul_2d_t(q, k, tmp_t)
    tmp = transpose(tmp_t)
    tmp = tmp / sqrt(real(n_embd_head,sp)) + mask
    call softmax(tmp)
    call matmul_2d(v, tmp, y)
end if
end subroutine

! Multi-head attention. k_cache and v_cache hold key/value vectors per head
! across the full sequence; they are written here and read by attention_zerocopy.
subroutine mha(n_seq, n_seq_x, n_embd, x, attn_w, attn_b, proj_w, proj_b, n_head, &
            use_kv_cache, max_seq_cache, k_cache, v_cache, y)
integer, intent(in) :: n_seq, n_seq_x, n_embd, max_seq_cache
real(sp), intent(in) :: x(n_embd,n_seq_x), &
    attn_w(3*n_embd,n_embd), attn_b(3*n_embd), &
    proj_w(n_embd,n_embd), proj_b(n_embd)
integer, intent(in) :: n_head
logical, intent(in) :: use_kv_cache
real(sp), intent(inout) :: k_cache(n_embd/n_head, max_seq_cache, n_head)
real(sp), intent(inout) :: v_cache(n_embd/n_head, max_seq_cache, n_head)
real(sp), intent(out) :: y(n_embd,n_seq_x)
real(sp) :: causal_mask(n_seq,n_seq_x)
real(sp) :: x2(3*n_embd,n_seq_x)
real(sp) :: tmp_y(n_embd, n_seq_x)
integer :: i, j, l, head_dim, istart, iend
! Mask
if (use_kv_cache) then
    causal_mask = 0
else
    do j = 1, n_seq_x
    do i = 1, n_seq
        if (i > j) then
            causal_mask(i,j) = -1e10_sp
        else
            causal_mask(i,j) = 0
        end if
    end do
    end do
end if
call linear(x, attn_w, attn_b, x2)
head_dim = n_embd / n_head
! Populate k/v caches from the projected input (x2 = [Q | K | V])
if (use_kv_cache) then
    do l = 1, n_head
        istart = (l-1) * head_dim + 1
        do j = 1, head_dim
            k_cache(j, n_seq, l) = x2(n_embd + istart - 1 + j, 1)
            v_cache(j, n_seq, l) = x2(2*n_embd + istart - 1 + j, 1)
        end do
    end do
else
    do l = 1, n_head
        istart = (l-1) * head_dim + 1
        do i = 1, n_seq
        do j = 1, head_dim
            k_cache(j, i, l) = x2(n_embd + istart - 1 + j, i)
            v_cache(j, i, l) = x2(2*n_embd + istart - 1 + j, i)
        end do
        end do
    end do
end if
! Perform attention over each head
!$omp parallel do default(none) private(l, istart, iend) &
!$omp shared(n_head, head_dim, n_seq, n_seq_x, x2, k_cache, v_cache, causal_mask, tmp_y)
do l = 1, n_head
    istart = (l-1) * head_dim + 1
    iend   = l * head_dim
    call attention_zerocopy(head_dim, n_seq, n_seq_x, &
        x2(istart:iend, :), &
        k_cache(:, 1:n_seq, l), &
        v_cache(:, 1:n_seq, l), &
        causal_mask, tmp_y(istart:iend, :))
end do
!$omp end parallel do
! Out projection
call linear(tmp_y, proj_w, proj_b, y)
end subroutine


! One decoder block: layer-norm -> MHA -> residual, layer-norm -> FFN -> residual.
! x is updated in place to avoid an extra copy of the full activation buffer.
subroutine transformer_block(n_seq, n_seq_x, n_embd, x, mlp_fc_w, mlp_fc_b, mlp_proj_w, mlp_proj_b, &
        attn_w, attn_b, attn_proj_w, attn_proj_b, ln1_g, ln1_b, ln2_g, ln2_b, &
        n_head, use_kv_cache, max_seq_cache, k_cache, v_cache)
real(sp), intent(inout) :: x(n_embd,n_seq_x)
real(sp), intent(in) :: mlp_fc_w(:,:), mlp_fc_b(:), &
    mlp_proj_w(:,:), mlp_proj_b(:), &
    attn_w(:,:), attn_b(:), attn_proj_w(:,:), attn_proj_b(:), &
    ln1_g(:), ln1_b(:), ln2_g(:), ln2_b(:)
integer, intent(in) :: n_head, max_seq_cache
integer, intent(in) :: n_seq, n_seq_x, n_embd
logical, intent(in) :: use_kv_cache
real(sp), intent(inout) :: k_cache(n_embd/n_head, max_seq_cache, n_head)
real(sp), intent(inout) :: v_cache(n_embd/n_head, max_seq_cache, n_head)
real(sp) :: norm_out(n_embd, n_seq_x)
real(sp) :: block_out(n_embd, n_seq_x)
call layer_norm(x, ln1_g, ln1_b, 1e-5_sp, norm_out)
call mha(n_seq, n_seq_x, n_embd, norm_out, attn_w, attn_b, attn_proj_w, attn_proj_b, &
         n_head, use_kv_cache, max_seq_cache, k_cache, v_cache, block_out)
x = x + block_out
call layer_norm(x, ln2_g, ln2_b, 1e-5_sp, norm_out)
call ffn(norm_out, mlp_fc_w, mlp_fc_b, mlp_proj_w, mlp_proj_b, block_out)
x = x + block_out
end subroutine

! Full GPT-2 forward pass. wte is stored transposed (n_vocab, n_embd) so that
! token embedding lookups are row accesses and the final logit projection is a
! plain matmul rather than matmul(transpose(wte), x).
subroutine gpt2(n_vocab, n_ctx, n_seq, n_seq_x, n_embd, n_layer, n_head, input, &
        wte, wpe, layers, lnf_g, lnf_b, use_kv_cache, max_seq_cache, k_cache, v_cache, logits)
integer, intent(in) :: n_vocab, n_ctx, n_seq, n_seq_x, n_embd, n_layer, n_head, max_seq_cache
integer, intent(in) :: input(n_seq)
real(sp), intent(in) :: wte(n_vocab,n_embd), wpe(n_embd,n_ctx)
type(layer_t), intent(in) :: layers(n_layer)
real(sp), intent(in) :: lnf_b(n_embd), lnf_g(n_embd)
logical, intent(in) :: use_kv_cache
real(sp), intent(inout) :: k_cache(n_embd/n_head, max_seq_cache, n_head, n_layer)
real(sp), intent(inout) :: v_cache(n_embd/n_head, max_seq_cache, n_head, n_layer)
real(sp), intent(out) :: logits(n_vocab,n_seq_x)
real(sp) :: x(n_embd,n_seq_x)
integer :: i
if (use_kv_cache) then
    i = n_seq
    x(:,1) = wte(input(i)+1,:) + wpe(:,i)
else
    do i = 1, n_seq
        x(:,i) = wte(input(i)+1,:) + wpe(:,i)
    end do
end if
do i = 1, n_layer
    call transformer_block(n_seq, n_seq_x, n_embd, x, &
        layers(i)%mlp_fc_w, layers(i)%mlp_fc_b, &
        layers(i)%mlp_proj_w, layers(i)%mlp_proj_b, &
        layers(i)%attn_w, layers(i)%attn_b, layers(i)%attn_proj_w, layers(i)%attn_proj_b, &
        layers(i)%ln1_g, layers(i)%ln1_b, layers(i)%ln2_g, layers(i)%ln2_b, &
        n_head, use_kv_cache, max_seq_cache, k_cache(:,:,:,i), v_cache(:,:,:,i))
end do
! FINAL HUGE GEMV BYPASS: Stop OpenBLAS from packing the massive vocab matrix!
block
    real(sp) :: x_norm(n_embd, n_seq_x)
    call layer_norm(x, lnf_g, lnf_b, 1e-5_sp, x_norm)
    if (n_seq_x == 1) then
        logits(:, 1) = matmul(wte, x_norm(:, 1))
    else
        call matmul_2d(wte, x_norm, logits)
    end if
end block
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
integer, allocatable, intent(out) :: output(:)
real(sp), allocatable :: logits(:,:)
integer :: i
integer :: n_seq2, n_seq_x
integer :: next_id
integer :: input2(size(input)+n_tokens_to_generate)
logical :: use_kv_cache
integer :: max_seq_cache
! Allocate full-length caches up front; pass directly each iteration
real(sp) :: k_cache(m%n_embd/m%n_head, n_seq+n_tokens_to_generate, m%n_head, m%n_layer)
real(sp) :: v_cache(m%n_embd/m%n_head, n_seq+n_tokens_to_generate, m%n_head, m%n_layer)
character(:), allocatable :: output_txt, last_token
max_seq_cache = n_seq + n_tokens_to_generate ! Fixed cache dimension passed to gpt2
if (present(stop_text)) then
    allocate(character(0) :: output_txt)
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
    call gpt2(m%n_vocab, m%n_ctx, n_seq2, n_seq_x, m%n_embd, m%n_layer, &
            m%n_head, &
            input2(:n_seq2), &
            m%wte, m%wpe, &
            m%layers, &
            m%lnf_g, m%lnf_b, use_kv_cache, &
            max_seq_cache, &
            k_cache, &
            v_cache, &
            logits)
    next_id = maxloc(logits(:,n_seq_x), dim=1)-1
    input2(n_seq2+1) = next_id
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
allocate(output(n_seq2 - n_seq + 1))
output(:) = input2(n_seq+1:n_seq2+1)
end subroutine

end module
