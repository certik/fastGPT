# fastGPT

The progression of GPT-2 codes from the original to "minimal", "nano" and
"pico":

* [openai/gpt-2](https://github.com/openai/gpt-2)
* [karpathy/minGPT](https://github.com/karpathy/mingpt)
* [karpathy/nanoGPT](https://github.com/karpathy/nanogpt)
* [jaymody/picoGPT](https://github.com/jaymody/picoGPT)

`fastGPT` is very similar to `picoGPT` (very small and readable), but it is
also fast (see the Benchmarks section below). The speed and readability is
achieved by using Fortran. I wrote a
[blog post](https://ondrejcertik.com/blog/2023/03/fastgpt-faster-than-pytorch-in-300-lines-of-fortran/)
introducing fastGPT.

`fastGPT` features:
* Fast? ✅
* Training code? ❌
* Batch inference? ❌
* top-p sampling? ❌ top-k? ❌ temperature? ❌ categorical sampling?! ❌ greedy? ✅
* Readable? ✅
* Small? ✅

A quick breakdown of each of the files:

* `gpt2.f90`: the actual GPT-2 model and a decoder
* `main.f90`: the main driver
* `create_model.py`: downloads the TensorFlow model and converts to our own
  format (`model.dat`)
* `encode_input.py`: encodes the text input into tokens (input file for `gpt2`)
* Matmul implementations
    * `linalg_f.f90` native Fortran
    * `linalg_c.f90`, `linalg_accelerate.c` macOS Accelerate Framework
* `pt.py`: a reference script to run PyTorch (returns the same answer)

## Build and Run

Install prerequisites:

    mamba env create -f environment.yml
    conda activate fastgpt

Configure and build:

    FC=gfortran cmake .
    make

Create the `model.dat` file from a given GPT-2 model. Supported sizes (and the
corresponding names to be used in `pt.py`, and the approximate download size):
"124M" (`gpt2`, 0.5GB), "355M" (`gpt-medium`, 1.5GB), "774M" (`gpt-large`,
3GB), "1558M" (`gpt-xl`, 6GB). This will download the model and cache it for
subsequent runs:

    python create_model.py --models_dir "models" --model_size "124M"

Now you can modify the `input` file to change the input string and set other
parameters.

Run (requires `model.dat` and `input` in the current directory):

    ./gpt2

### Example Output

The above `./gpt2` command prints on Apple M1 Max:
```
$ ./gpt2
Loading the model...
    done. Time:   0.111s

Model parameters:
n_vocab = 50257
n_ctx   =  1024
n_embd  =   768
n_layer =    12
n_head  =    12

Input text
Alan Turing theorized that computers would one day become very powerful, but even he could not imagine

Encoding: tokenizing input text into tokens (currently slow)...
    done. Time:   0.074s

Input parameters:
n_seq                =  19
n_tokens_to_generate =  20

Input tokens:
 36235 39141 18765  1143   326  9061   561   530  1110  1716   845  3665    11   475   772   339   714   407  5967

Decoded input as text:
Alan Turing theorized that computers would one day become very powerful, but even he could not imagine

Running model...
 how they would be able to do so.

"I think that the most important thing is
    done. Time:   0.304s (1.01x)

Output tokens:
   703   484   561   307  1498   284   466   523    13   198   198     1    40   892   326   262   749  1593  1517   318

Decoded output as text:
 how they would be able to do so.

"I think that the most important thing is
```

### Chat interface

Here is an example chat using the largest 1558M model:

```
$ ./chat
Your name is fastGPT and you are an AI bot. The user will ask you questions and you answer in a nice, truthful, short way.
User: What is the capital of Czechia?
fastGPT: Prague.
User: How many legs does a dog have?
fastGPT: Four.
User: What color does the sky have?
fastGPT: Blue.
User: What can you type a document on?
fastGPT: A typewriter.
User: What can you drive in?
fastGPT: A car.
User: What can you fly in?
fastGPT: A plane.
User: What continent is Germany in?
fastGPT: Europe.
User: When did Second World War start?
fastGPT: 1939.
User: When did it end?
fastGPT: 1945.
User: When did the U.S. enter the Second World War?
fastGPT: 1941.
User: When did the First World War start?
fastGPT: 1914.
User: When did it end?
fastGPT: 1918.
User: When did the Mexican-American war start?
fastGPT: 1846.
User: When did it end?
fastGPT: 1848.
User: What color is snow?
fastGPT: White.
User: What color do plants usually have?
fastGPT: Green.
User: What is your name?
fastGPT: fastGPT.
```


### BLAS Implementation

You can choose which BLAS implementation to use for `matmul` using:
* `-DFASTGPT_BLAS=OpenBLAS`: Use OpenBLAS
* `-DFASTGPT_BLAS=Accelerate`: Use the macOS Accelerate Framework
* `-DFASTGPT_BLAS=Fortran`: Use the default Fortran's intrinsic `matmul`

## Benchmarks

On Apple M1 Max, inference of the above input file (20 tokens):

                                    1 core  2 cores  4 cores  8 cores

    fastGPT (Accelerate, fast_tanh) 0.288s
    fastGPT (Accelerate)            0.299s
    fastGPT (OpenBLAS)              0.837s  0.514s    0.341s   0.339s
    PyTorch (OpenBLAS)              0.873s  0.539s    0.386s   0.392s
    fastGPT (Accelerate, no cache)  0.717s
    fastGPT (OpenBLAS, no cache)    2.343s  1.603s    1.209s   1.018s
    PyTorch (OpenBLAS, no cache)    2.356s  1.520s    1.104s   0.997s
    picoGPT (OpenBLAS, no cache)    2.427s  1.645s    1.272s   1.081s

Total run (includes loading the model and Python imports):

    fastGPT (Accelerate, fast_tanh): 0.401s
    picoGPT (8 cores):               3.445s
    PyTorch (OpenBLAS, 4 cores):     4.867s

## TODO

* [ ] Parallelization:
  * [ ] Over heads: https://github.com/certik/fastGPT/issues/2
  * [ ] MPI: https://github.com/certik/fastGPT/issues/5
* [ ] Other sampling methods: https://github.com/certik/fastGPT/issues/8
* [ ] Batching: https://github.com/certik/fastGPT/issues/7
* [x] Improve the UI:
  * [x] Implement the input tokenizer in Fortran: https://github.com/certik/fastGPT/issues/1
  * [x] Show the words as they are generated: https://github.com/certik/fastGPT/issues/6
