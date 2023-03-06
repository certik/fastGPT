# fastGPT

The progression of GPT-2 codes from the original to simpler / better (in some
sense):

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

* `main.f90`: the main driver
* `gpt2.f90`: the actual GPT-2 model and a decoder
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

    cmake .
    make

Create the `model.dat` file from a given GPT-2 model. Supported sizes (and the
corresponding names to be used in `pt.py`): "124M" (`gpt2`), "355M"
(`gpt-medium`), "774M" (`gpt-large`), "1558M" (`gpt-xl`).

    python create_model.py --models_dir "../gpt2/models" --model_size "124M"

Create an input file:

    python encode_input.py \
        "Alan Turing theorized that computers would one day become very powerful, but even he could not imagine" \
        -n 20

Run (requires `model.dat` and `input.dat` in the current directory):

    ./gpt2

### Configuration Options

You can turn macOS Accelerate Framework on and off by passing
`-DWITH_ACCELERATE_FRAMEWORK=yes` (and `no`) to `cmake`.

## Benchmarks

On Apple M1 Max, inference of the above input file (20 tokens):

    fastGPT (Accelerate matmul): 0.797s
    PyTorch (conda-forge):       0.873s
    fastGPT (default matmul):    6.449s

Total run:

    ./gpt2 (Accelerate): 0.828s
    python pt.py:        5.865s
