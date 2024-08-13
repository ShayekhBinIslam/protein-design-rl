#!/bin/bash
# for target in CD3e TrkA MDM2 PD1 PDL1 SARSCoV2
#for target in TrkA MDM2
iptm=10
for target in MDM2
do
  for total_num_steps in 100 1000 3000 10000 30000
  do
    for tmax in 1 10
    do
      python main.py \
        target_name=${target} \
        program=vhh_binder \
        experiment=hparam_sweep \
        name=${target}_vhh \
        chain.T_max=${tmax} \
        wandb.tags=["hparam","v3","num_steps"] \
        program.energy_function_weights=[1.0,1.0,1.0,${iptm},${iptm}] \
        chain.num_chains=5 &
      sleep 5
    done
  done
done
