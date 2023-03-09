program gpt2
use gpt2_mod, only: generate, decode
implicit none

integer, parameter :: sp = kind(0.0)
integer, parameter :: dp = kind(0.d0)

integer :: n_vocab, n_ctx, n_seq, n_embd, n_layer, n_head, &
    n_tokens_to_generate, n_decoder_idx, n_decoder_txt, n_byte_decoder
integer, allocatable :: input(:), decoder_idx(:), byte_decoder(:)
real(sp), allocatable :: wte(:,:), wpe(:,:), &
    mlp_fc_w(:,:,:), mlp_fc_b(:,:), &
    mlp_proj_w(:,:,:), mlp_proj_b(:,:), &
    attn_w(:,:,:), attn_b(:,:), &
    attn_proj_w(:,:,:), attn_proj_b(:,:), &
    ln1_b(:,:), ln1_g(:,:), &
    ln2_b(:,:), ln2_g(:,:), &
    lnf_b(:), lnf_g(:)
character, allocatable :: decoder_txt(:)
integer, allocatable :: output(:)
character(:), allocatable :: output_txt
real(dp) :: t1, t2
integer :: u

! Load the model
print "(a)", "Loading the model..."
call cpu_time(t1)
open(newunit=u, file="model.dat", form="unformatted", access="stream", status="old")
read(u) n_vocab, n_ctx, n_embd, n_layer, n_head, n_decoder_idx, n_decoder_txt, &
    n_byte_decoder
allocate(wte(n_embd,n_vocab), wpe(n_embd,n_ctx), &
    mlp_fc_w(4*n_embd,n_embd,n_layer), mlp_fc_b(4*n_embd,n_layer), &
    mlp_proj_w(n_embd,4*n_embd,n_layer), mlp_proj_b(n_embd,n_layer), &
    attn_w(3*n_embd,n_embd,n_layer), attn_b(3*n_embd,n_layer), &
    attn_proj_w(n_embd,n_embd,n_layer), attn_proj_b(n_embd,n_layer), &
    ln1_b(n_embd,n_layer), ln1_g(n_embd,n_layer), &
    ln2_b(n_embd,n_layer), ln2_g(n_embd,n_layer), &
    lnf_b(n_embd), lnf_g(n_embd), &
    decoder_idx(n_decoder_idx), decoder_txt(n_decoder_txt), &
    byte_decoder(n_byte_decoder))
read(u) wte, wpe, &
    mlp_fc_w, mlp_fc_b, &
    mlp_proj_w, mlp_proj_b, &
    attn_w, attn_b, &
    attn_proj_w, attn_proj_b, &
    ln1_b, ln1_g, &
    ln2_b, ln2_g, &
    lnf_b, lnf_g, &
    decoder_idx, decoder_txt, byte_decoder
close(u)
call cpu_time(t2)
print "(a,f8.3,a)", "    done. Time:", t2-t1, "s"

! Load the input
open(newunit=u, file="input.dat", form="unformatted", access="stream", status="old")
read(u) n_seq, n_tokens_to_generate
allocate(input(n_seq))
read(u) input
close(u)

print "(a)", "Model parameters:"
print "(a,i6)", "n_vocab =", n_vocab
print "(a,i6)", "n_ctx   =", n_ctx
print "(a,i6)", "n_embd  =", n_embd
print "(a,i6)", "n_layer =", n_layer
print "(a,i6)", "n_head  =", n_head
print *
print "(a)", "Input parameters:"
print "(a,i4)", "n_seq                =", n_seq
print "(a,i4)", "n_tokens_to_generate =", n_tokens_to_generate

if (n_seq + n_tokens_to_generate >= n_ctx) then
    print *, "The maximum sequence length of the model was surpassed."
    print *, "Make the input and/or number of tokens to generate shorter."
    error stop
end if

print *
print "(a)", "Input tokens:"
print "(1000(i6))", input
print "(a)", "Decoded input as text:"
print "(a)", decode(input, decoder_idx, decoder_txt, byte_decoder)

allocate(output(n_tokens_to_generate))
print "(a)", "Running model..."
call cpu_time(t1)
output = generate(n_tokens_to_generate, n_vocab, n_ctx, size(input), n_embd, &
    n_layer, n_head, &
    input, &
    wte, wpe, &
    mlp_fc_w, mlp_fc_b, mlp_proj_w, mlp_proj_b, &
    attn_w, attn_b, attn_proj_w, attn_proj_b, &
    ln1_g, ln1_b, ln2_g, ln2_b, lnf_g, lnf_b)
call cpu_time(t2)
print "(a,f8.3,a)", "    done. Time:", t2-t1, "s"
print "(a)", "Output tokens:"
print "(1000(i6))", output
output_txt = decode(output, decoder_idx, decoder_txt, byte_decoder)
print "(a)", "Decoded output as text:"
print "(a)", output_txt
end program
