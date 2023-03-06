# fastGPT

The progression of GPT-2 codes from the original to "minimal", "nano" and
"pico":

* [openai/gpt-2](https://github.com/openai/gpt-2)
* [karpathy/minGPT](https://github.com/karpathy/mingpt)
* [karpathy/nanoGPT](https://github.com/karpathy/nanogpt)
* [jaymody/picoGPT](https://github.com/jaymody/picoGPT)

`fastGPT` is very similar to `picoGPT` (very small and readable), but it is
also fast (see the Benchmarks section below). The speed and readability is
achieved by using Fortran.

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
corresponding names to be used in `pt.py`): "124M" (`gpt2`), "355M"
(`gpt-medium`), "774M" (`gpt-large`), "1558M" (`gpt-xl`).

    python create_model.py --models_dir "models" --model_size "124M"

Create an input file:

    python encode_input.py \
        "Alan Turing theorized that computers would one day become very powerful, but even he could not imagine" \
        -n 20

Run (requires `model.dat` and `input.dat` in the current directory):

    ./gpt2

### Example Output

The above `./gpt2` command prints on Apple M1 Max:
```
$ ./gpt2
Loading the model...
    done.
Model parameters:
n_vocab = 50257
n_ctx   =  1024
n_embd  =   768
n_layer =    12
n_head  =    12

Input parameters:
n_seq                =  19
n_tokens_to_generate =  20

Input tokens:
 36235 39141 18765  1143   326  9061   561   530  1110  1716   845  3665    11   475   772   339   714   407  5967
Decoded input as text:
Alan Turing theorized that computers would one day become very powerful, but even he could not imagine
Running model...
           1         703
           2         484
           3         561
           4         307
           5        1498
           6         284
           7         466
           8         523
           9          13
          10         198
          11         198
          12           1
          13          40
          14         892
          15         326
          16         262
          17         749
          18        1593
          19        1517
          20         318
    done. Time:   0.795s
Output tokens:
   703   484   561   307  1498   284   466   523    13   198   198     1    40   892   326   262   749  1593  1517   318
Decoded output as text:
 how they would be able to do so.

"I think that the most important thing is
```

### Configuration Options

You can turn macOS Accelerate Framework on and off by passing
`-DWITH_ACCELERATE_FRAMEWORK=yes` (and `no`) to `cmake`.

## Benchmarks

On Apple M1 Max, inference of the above input file (20 tokens):

    fastGPT (Accelerate matmul): 0.797s
    PyTorch (conda-forge):       0.873s
    picoGPT (parallel):          1.694s
    picoGPT (serial):            2.429s
    fastGPT (default matmul):    6.449s

Total run:

    fastGPT (./gpt2):       0.828s
    picoGPT:                4.123s
    PyTorch (python pt.py): 5.865s
