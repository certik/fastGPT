#!/bin/bash

set -ex
NP=4
FC=gfortran cmake -Bbuild
cmake --build build --parallel
python create_model.py --models_dir "../gpt2/models" --model_size "124M"
# python encode_input.py \
#     "Alan Turing theorized that computers would one day become very powerful, but even he could not imagine" \
#     -n 20
python encode_input.py "" -l 20 -n 20 # Downloads lambada dataset and selects -p prompts for input generation. Outputs are saved in output.txt in order
cafrun -np $NP build/gpt2
> output.txt
for i in $(seq -w 01 $NP); do
    cat output_$i.txt >> output.txt
    rm -rf output_$i.txt
done

         
