#!/bin/bash
# Schedule execution of many runs
# Run from root folder with: bash scripts/schedule.sh

python src/train.py trainer.max_epochs=5 logger=csv

python src/train.py trainer.max_epochs=10 logger=csv

# conda
conda activate /home/ray/micromamba/envs/proxy

# Single head, ptm feature
/home/ray/micromamba/envs/proxy/bin/python ~/default/dockq-proxy/src/train.py trainer=gpu experiment=atlas_reg_llm_ptm
/home/ray/micromamba/envs/proxy/bin/python -m pdb -c continue ~/default/dockq-proxy/src/train.py trainer=gpu experiment=atlas_reg_llm_ptm
anyscale job submit /home/ray/default/dockq-proxy/scripts/atlas_reg_llm_ptm.yml

# Single head, ptm feature, dropout
/home/ray/micromamba/envs/proxy/bin/python ~/default/dockq-proxy/src/train.py trainer=gpu experiment=atlas_reg_llm_ptm_dropout
# /home/ray/micromamba/envs/proxy/bin/python -m pdb -c continue ~/default/dockq-proxy/src/train.py trainer=gpu experiment=atlas_reg_llm_ptm_dropout
# anyscale job submit /home/ray/default/dockq-proxy/scripts/atlas_reg_llm_ptm.yml

# Single head, plddt feature
/home/ray/micromamba/envs/proxy/bin/python ~/default/dockq-proxy/src/train.py trainer=gpu experiment=atlas_reg_llm_plddt
/home/ray/micromamba/envs/proxy/bin/python -m pdb -c continue ~/default/dockq-proxy/src/train.py trainer=gpu experiment=atlas_reg_llm_plddt
anyscale job submit /home/ray/default/dockq-proxy/scripts/atlas_reg_llm_plddt.yml

# Two heads, ptm/plddt features, masking
/home/ray/micromamba/envs/proxy/bin/python ~/default/dockq-proxy/src/train.py trainer=gpu experiment=atlas_reg_llm_masking
/home/ray/micromamba/envs/proxy/bin/python -m pdb -c continue ~/default/dockq-proxy/src/train.py trainer=gpu experiment=atlas_reg_llm_masking
anyscale job submit /home/ray/default/dockq-proxy/scripts/atlas_reg_llm_masking.yml

# Two heads, ptm/plddt features, probabilistic masking
/home/ray/micromamba/envs/proxy/bin/python ~/default/dockq-proxy/src/train.py trainer=gpu experiment=atlas_reg_llm_probabilistic_masking
anyscale job submit /home/ray/default/dockq-proxy/scripts/atlas_reg_llm_probabilistic_masking.yml

# Two heads, ptm/plddt features, bootstrap
/home/ray/micromamba/envs/proxy/bin/python ~/default/dockq-proxy/src/train.py trainer=gpu experiment=atlas_reg_llm_bootstrap
anyscale job submit /home/ray/default/dockq-proxy/scripts/atlas_reg_llm_bootstrap.yml

## Mulithead exploration
/home/ray/micromamba/envs/proxy/bin/python ~/default/dockq-proxy/src/train.py trainer=gpu experiment=atlas_reg_llm_bootstrap_multihead model.exploration=bootstrap

/home/ray/micromamba/envs/proxy/bin/python ~/default/dockq-proxy/src/train.py trainer=gpu experiment=atlas_reg_llm_bootstrap_multihead model.exploration=masking

/home/ray/micromamba/envs/proxy/bin/python ~/default/dockq-proxy/src/train.py trainer=gpu experiment=atlas_reg_llm_bootstrap_multihead model.exploration=probabilistic_masking

# Single head, ptm feature, acquisition function (bmdal)
pip install dill catboost

/home/ray/micromamba/envs/proxy/bin/python ~/default/dockq-proxy/src/train.py trainer=gpu experiment=atlas_reg_llm_ptm_acquisition
/home/ray/micromamba/envs/proxy/bin/python -m pdb -c continue ~/default/dockq-proxy/src/train.py trainer=gpu experiment=atlas_reg_llm_ptm_acquisition
anyscale job submit /home/ray/default/dockq-proxy/scripts/atlas_reg_llm_ptm_acquisition.yml


/home/ray/micromamba/envs/proxy/bin/python -m src.train trainer=gpu experiment=atlas_reg_llm_ptm_acquisition


