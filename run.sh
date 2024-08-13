#!/bin/bash 

eval "$(micromamba shell hook --shell bash)"
micromamba activate /home/ray/micromamba/envs/proxy
BASE_DIR=$(realpath ".")
echo $BASE_DIR

cd $BASE_DIR/dockq-proxy
pip install -U -e .

cd $BASE_DIR/esm/examples/protein-programming-language
pip install -U -e .

cd $BASE_DIR
pip install -U -e .
pip install -U deepspeed

python -c 'import language, programs; print(language, programs)'
python -c 'import src; print(src)'
python -c 'import flexs; print(flexs)'
python run_rl.py +experiment=ptm_proxy_dqn program.sequence_length=20
