#! /bin/bash

eval "$(micromamba shell hook --shell bash)"
micromamba activate /home/ray/micromamba/envs/proxy

BASE_DIR=$(realpath "..")
echo $BASE_DIR

cd $BASE_DIR/dockq-proxy
echo "$PWD"
pip install -e .

cd $BASE_DIR/esm/examples/protein-programming-language
echo "$PWD"
# pip install -e .

# Setup FLEXS
# pip install design-bench==2.0.20
# pip install "tensorflow[and-cuda]==2.12.1" tf-agents==0.7.1 tensorflow_probability==0.19.0
# cd $BASE_DIR
# pip install -e .

# pip install -U deepspeed

# python -c 'import language, programs; print(language, programs)'
# python -c 'import src; print(src)'
# python -c 'import flexs; print(flexs)'

