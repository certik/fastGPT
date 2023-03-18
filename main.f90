program gpt2
   use gpt2_mod, only: generate, decode
   use omp, only: omp_get_wtime
   implicit none

   integer, parameter :: sp = kind(0.0)
   integer, parameter :: dp = kind(0.d0)

   integer :: n_vocab, n_ctx, n_seq, n_embd, n_layer, n_head, &
              n_tokens_to_generate, n_decoder_idx, n_decoder_txt, n_byte_decoder
   integer, allocatable :: input(:), decoder_idx(:), byte_decoder(:)
   real(sp), allocatable :: wte(:, :), wpe(:, :), &
                            mlp_fc_w(:, :, :), mlp_fc_b(:, :), &
                            mlp_proj_w(:, :, :), mlp_proj_b(:, :), &
                            attn_w(:, :, :), attn_b(:, :), &
                            attn_proj_w(:, :, :), attn_proj_b(:, :), &
                            ln1_b(:, :), ln1_g(:, :), &
                            ln2_b(:, :), ln2_g(:, :), &
                            lnf_b(:), lnf_g(:)
   character, allocatable :: decoder_txt(:)
   integer, allocatable :: output(:)
   real(dp) :: t1, t2, t1o, t2o
   integer :: u, i, total_input, chunk, start, end_, dummy, j, w, stat
   logical :: use_cache, exists
   character(len=13) output_file

! Load the model
   call cpu_time(t1)
   open (newunit=u, file="model.dat", form="unformatted", access="stream", status="old")

   read (u) n_vocab, n_ctx, n_embd, n_layer, n_head, n_decoder_idx, n_decoder_txt, &
      n_byte_decoder
   allocate (wte(n_embd, n_vocab), wpe(n_embd, n_ctx), &
             mlp_fc_w(4*n_embd, n_embd, n_layer), mlp_fc_b(4*n_embd, n_layer), &
             mlp_proj_w(n_embd, 4*n_embd, n_layer), mlp_proj_b(n_embd, n_layer), &
             attn_w(3*n_embd, n_embd, n_layer), attn_b(3*n_embd, n_layer), &
             attn_proj_w(n_embd, n_embd, n_layer), attn_proj_b(n_embd, n_layer), &
             ln1_b(n_embd, n_layer), ln1_g(n_embd, n_layer), &
             ln2_b(n_embd, n_layer), ln2_g(n_embd, n_layer), &
             lnf_b(n_embd), lnf_g(n_embd), &
             decoder_idx(n_decoder_idx), decoder_txt(n_decoder_txt), &
             byte_decoder(n_byte_decoder))
   read (u) wte, wpe, &
      mlp_fc_w, mlp_fc_b, &
      mlp_proj_w, mlp_proj_b, &
      attn_w, attn_b, &
      attn_proj_w, attn_proj_b, &
      ln1_b, ln1_g, &
      ln2_b, ln2_g, &
      lnf_b, lnf_g, &
      decoder_idx, decoder_txt, byte_decoder
   close (u)
   call cpu_time(t2)

   ! print "(a, i4)", "Image Id:", this_image()
   if (this_image() == 1) then
      print "(a,f8.3,a)", "    done. Time:", t2 - t1, "s"
      print "(a)", "Model parameters:"
      print "(a,i6)", "n_vocab =", n_vocab
      print "(a,i6)", "n_ctx   =", n_ctx
      print "(a,i6)", "n_embd  =", n_embd
      print "(a,i6)", "n_layer =", n_layer
      print "(a,i6)", "n_head  =", n_head
      print *
      print "(a)", "Running model..."
   end if

! Load the input and prepare the output file
   open (newunit=u, file="input.dat", form="unformatted", access="stream", status="old")
   write (output_file, '(a,i2.2,a)') "output_", this_image(), ".txt"
   inquire (file=output_file, exist=exists)
   if (exists) then
      open (file=output_file, newunit=w, iostat=stat)
      if (stat == 0) close (w, status="delete", iostat=stat)
   end if

   open (newunit=w, file=output_file, position="append", status="new", action="write")
   ! Read total lines
   read (u) total_input
   chunk = (total_input + num_images() - 1)/num_images()
   start = (this_image() - 1)*chunk
   end_ = start + chunk
   print *, "Image Id:", this_image(), "Range:", start, end_ - 1
   do i = 0, start - 1
      read (u, end=1) n_seq, n_tokens_to_generate
      read (u) (dummy, j=1, n_seq)                   !Skip line
   end do
   do i = start, end_ - 1
      read (u, end=1) n_seq, n_tokens_to_generate
      if (allocated(input)) deallocate (input)
      allocate (input(n_seq))
      read (u, end=1) input
      ! Check n_ctx Bound
      if (n_seq + n_tokens_to_generate >= n_ctx) then
         print *, "The maximum sequence length of the model was surpassed."
         print *, "Make the input and/or number of tokens to generate shorter."
         error stop
      end if
      ! Set timers
      call cpu_time(t1)
      t1o = omp_get_wtime()
      ! Run generation
      use_cache = .true.
      if (allocated(output)) deallocate (output)
      allocate (output(n_tokens_to_generate))
      output = generate(n_tokens_to_generate, n_vocab, n_ctx, size(input), n_embd, &
                        n_layer, n_head, &
                        input, &
                        wte, wpe, &
                        mlp_fc_w, mlp_fc_b, mlp_proj_w, mlp_proj_b, &
                        attn_w, attn_b, attn_proj_w, attn_proj_b, &
                        ln1_g, ln1_b, ln2_g, ln2_b, lnf_g, lnf_b, use_cache)
      t2o = omp_get_wtime()
      call cpu_time(t2)
      ! FIXME: Time difference in seconds is always zero in ubuntu
      print "(a,i4,a,f8.3,a,f4.2,a)", "Sample:", i, "|Time:", t2o - t1o, "s (", (t2 - t1)/(t2o - t1o), "x)"
      write (w, '(a, a)') decode(input, decoder_idx, decoder_txt, byte_decoder), &
           decode(output, decoder_idx, decoder_txt, byte_decoder)
      flush (w)
   end do
1  close (u)
   close (w)
end program
