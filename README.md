# RL for Protein Sequence Design 


## Environment setup

```sh
pip install -r requirements.txt
pip install -e dockq-proxy
pip install -e esm/examples/protein-programming-language
```


## Run MCMC Baseline

```sh
python run_mcmc.py
```


## Run RL (DynaPPO) with Proxy

```sh
python run_rl.py +experiment=dynappo program.sequence_length=30
```

