module driver
use gpt2_mod, only: generate
use tokenizer, only: encode, decode
use omp, only: omp_get_wtime
implicit none

integer, parameter :: sp = kind(0.0)
integer, parameter :: dp = kind(0.d0)

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
read(u) m%wte, m%wpe, &
    m%mlp_fc_w, m%mlp_fc_b, &
    m%mlp_proj_w, m%mlp_proj_b, &
    m%attn_w, m%attn_b, &
    m%attn_proj_w, m%attn_proj_b, &
    m%ln1_b, m%ln1_g, &
    m%ln2_b, m%ln2_g, &
    m%lnf_b, m%lnf_g, &
    m%decoder_idx, m%decoder_txt, &
    m%vocab_idx, m%vocab_txt, &
    m%byte_encoder
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
type(model_t) :: m
integer, allocatable :: byte_decoder(:)
integer :: n_seq
character(:), allocatable :: output_txt
real(dp) :: t1, t2, t1o, t2o
integer :: i
logical :: use_cache

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

! Compute byte_decoder:
allocate(byte_decoder(0:maxval(m%byte_encoder)))
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
output = generate(n_tokens_to_generate, m%n_vocab, m%n_ctx, size(input), &
    m%n_embd, &
    m%n_layer, m%n_head, &
    input, &
    m%wte, m%wpe, &
    m%mlp_fc_w, m%mlp_fc_b, m%mlp_proj_w, m%mlp_proj_b, &
    m%attn_w, m%attn_b, m%attn_proj_w, m%attn_proj_b, &
    m%ln1_g, m%ln1_b, m%ln2_g, m%ln2_b, m%lnf_g, m%lnf_b, use_cache, &
    m%decoder_idx, m%decoder_txt, byte_decoder)
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

end module
