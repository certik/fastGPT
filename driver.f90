module driver
use gpt2_mod, only: generate
use tokenizer, only: encode, decode
use omp, only: omp_get_wtime
implicit none

integer, parameter :: sp = kind(0.0)
integer, parameter :: dp = kind(0.d0)

contains

subroutine load_input(filename, input_txt, n_tokens_to_generate)
! Load the input from a namelist `filename`
character(*), intent(in) :: filename
character(:), allocatable, intent(out) :: input_txt
integer, intent(out) :: n_tokens_to_generate
character(1024) :: input_txt2
integer :: u, ios
namelist / input_fastGPT / n_tokens_to_generate
allocate(character(0) :: input_txt)
input_txt = ""
open(newunit=u, file=filename, status="old")
read(u, input_fastGPT)
do
    read(u, "(a)", iostat=ios) input_txt2
    if (ios /= 0) exit
    if (len(input_txt) > 0) input_txt = input_txt // char(10)
    input_txt = input_txt // trim(input_txt2)
end do
close(u)
end subroutine

subroutine gpt2_driver(input, output)
integer, allocatable, intent(out) :: input(:), output(:)
character(:), allocatable :: input_txt
integer :: n_tokens_to_generate
call load_input("input", input_txt, n_tokens_to_generate)
call gpt2_driver2(input_txt, n_tokens_to_generate, input, output)
endsubroutine

subroutine gpt2_driver2(input_txt, n_tokens_to_generate, input, output)
character(*), intent(in) :: input_txt
integer, intent(in) :: n_tokens_to_generate
integer, allocatable, intent(out) :: input(:), output(:)
integer :: n_vocab, n_ctx, n_seq, n_embd, n_layer, n_head, &
    n_decoder_idx, n_decoder_txt, &
    n_vocab_idx, n_vocab_txt, n_byte_encoder
integer, allocatable :: decoder_idx(:), vocab_idx(:), byte_decoder(:)
integer :: byte_encoder(0:255)
real(sp), allocatable :: wte(:,:), wpe(:,:), &
    mlp_fc_w(:,:,:), mlp_fc_b(:,:), &
    mlp_proj_w(:,:,:), mlp_proj_b(:,:), &
    attn_w(:,:,:), attn_b(:,:), &
    attn_proj_w(:,:,:), attn_proj_b(:,:), &
    ln1_b(:,:), ln1_g(:,:), &
    ln2_b(:,:), ln2_g(:,:), &
    lnf_b(:), lnf_g(:)
character, allocatable :: decoder_txt(:), vocab_txt(:)
character(:), allocatable :: output_txt
real(dp) :: t1, t2, t1o, t2o
integer :: u, i
logical :: use_cache

! Load the model
print "(a)", "Loading the model..."
call cpu_time(t1)
open(newunit=u, file="model.dat", form="unformatted", access="stream", status="old")
!read(u) model_version
!                    fastGPT (digits look similar to the letters they represent)
! model_version /= 0xfa51697
read(u) n_vocab, n_ctx, n_embd, n_layer, n_head, n_decoder_idx, n_decoder_txt, &
    n_vocab_idx, n_vocab_txt, n_byte_encoder
allocate(wte(n_embd,n_vocab), wpe(n_embd,n_ctx), &
    mlp_fc_w(4*n_embd,n_embd,n_layer), mlp_fc_b(4*n_embd,n_layer), &
    mlp_proj_w(n_embd,4*n_embd,n_layer), mlp_proj_b(n_embd,n_layer), &
    attn_w(3*n_embd,n_embd,n_layer), attn_b(3*n_embd,n_layer), &
    attn_proj_w(n_embd,n_embd,n_layer), attn_proj_b(n_embd,n_layer), &
    ln1_b(n_embd,n_layer), ln1_g(n_embd,n_layer), &
    ln2_b(n_embd,n_layer), ln2_g(n_embd,n_layer), &
    lnf_b(n_embd), lnf_g(n_embd), &
    decoder_idx(n_decoder_idx), decoder_txt(n_decoder_txt), &
    vocab_idx(n_vocab_idx), vocab_txt(n_vocab_txt))
if (n_byte_encoder /= 256) error stop "n_byte_encoder must be 256"
read(u) wte, wpe, &
    mlp_fc_w, mlp_fc_b, &
    mlp_proj_w, mlp_proj_b, &
    attn_w, attn_b, &
    attn_proj_w, attn_proj_b, &
    ln1_b, ln1_g, &
    ln2_b, ln2_g, &
    lnf_b, lnf_g, &
    decoder_idx, decoder_txt, &
    vocab_idx, vocab_txt, &
    byte_encoder
close(u)
call cpu_time(t2)
print "(a,f8.3,a)", "    done. Time:", t2-t1, "s"
print *
print "(a)", "Model parameters:"
print "(a,i6)", "n_vocab =", n_vocab
print "(a,i6)", "n_ctx   =", n_ctx
print "(a,i6)", "n_embd  =", n_embd
print "(a,i6)", "n_layer =", n_layer
print "(a,i6)", "n_head  =", n_head
print *

! Compute byte_decoder:
allocate(byte_decoder(0:maxval(byte_encoder)))
byte_decoder = 0
do i = 0, size(byte_encoder)-1
    byte_decoder(byte_encoder(i)) = i
end do

print "(a)", "Input text"
print "(a)", input_txt

print *
print "(a)",  "Encoding: tokenizing input text into tokens (currently slow)..."
call cpu_time(t1)
input = encode(input_txt, decoder_idx, decoder_txt, vocab_idx, vocab_txt, &
    byte_encoder)
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

if (n_seq + n_tokens_to_generate >= n_ctx) then
    print *, "The maximum sequence length of the model was surpassed."
    print *, "Make the input and/or number of tokens to generate shorter."
    error stop
end if

print "(a)", "Decoded input as text:"
!print "(a)", decode(input, decoder_idx, decoder_txt, byte_decoder)
allocate(character(0) :: output_txt) ! Fix GFortran warning
output_txt = decode(input, decoder_idx, decoder_txt, byte_decoder)
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
output = generate(n_tokens_to_generate, n_vocab, n_ctx, size(input), n_embd, &
    n_layer, n_head, &
    input, &
    wte, wpe, &
    mlp_fc_w, mlp_fc_b, mlp_proj_w, mlp_proj_b, &
    attn_w, attn_b, attn_proj_w, attn_proj_b, &
    ln1_g, ln1_b, ln2_g, ln2_b, lnf_g, lnf_b, use_cache, &
    decoder_idx, decoder_txt, byte_decoder)
t2o = omp_get_wtime()
call cpu_time(t2)
print "(a,f8.3,a,f4.2,a)", "    done. Time:", t2o-t1o, "s (", (t2-t1)/(t2o-t1o), "x)"
print *
print "(a)", "Output tokens:"
print "(1000(i6))", output
output_txt = decode(output, decoder_idx, decoder_txt, byte_decoder)
print *
print "(a)", "Decoded output as text:"
print "(a)", output_txt
end subroutine

end module